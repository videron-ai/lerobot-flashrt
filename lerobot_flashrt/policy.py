"""FlashRTPI05Policy — PI05Policy subclass that replaces the forward pass with FlashRT.

Inherits PI05Policy so that all action queue management, reset(), forward()
(training), and config/preprocessor/postprocessor machinery is unchanged.
Only predict_action_chunk() is overridden.

The wrapper only touches the inference path. Training, loss computation, and
push_to_hub are all inherited and work as-is.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# Lazy imports so the wrapper can be imported in the FlashRT env
# (where transformers may be absent) as long as PI05Policy is not instantiated.
try:
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy, resize_with_pad_torch
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
    _LEROBOT_AVAILABLE = True
except ImportError:
    _LEROBOT_AVAILABLE = False
    PI05Policy = object  # type: ignore[misc,assignment]
    PI05Config = None    # type: ignore[assignment]

from .client import FlashRTClient, CANONICAL_VIEW_ORDER

_IMG_H, _IMG_W = 224, 224
_MAX_ACTION_DIM = 32


class FlashRTPI05Policy(PI05Policy):
    """PI05Policy with the action-prediction replaced by a FlashRT server call.

    Everything except predict_action_chunk() is inherited:
    - reset() / _action_queue logic in select_action()
    - forward() training pass (runs on local GPU, not on the FlashRT server)
    - save_pretrained(), push_to_hub(), etc.

    The batch passed to predict_action_chunk must already be preprocessed by
    make_pi05_pre_post_processors's preprocessor pipeline:
    - images normalized to [0,1] float32 (VISUAL IDENTITY mapping)
    - state normalized to [-1,1] float32 (QUANTILES mapping, padded to 32 dims)
    - tokens/masks built from the full "Task:…State:…Action:" prompt

    The wrapper extracts the normalized state and raw images from the batch,
    bypasses the tokenizer output (FlashRT builds its own prompt), and sends
    to the FlashRT server. The returned (30, 32) normalized chunk is handed
    back to select_action() which queues it; the postprocessor
    (UnnormalizerProcessorStep + AbsoluteActionsProcessorStep) is applied
    outside by the caller.
    """

    config_class = PI05Config if PI05Config is not None else type(None)
    name = "flashrt_pi05"

    def __init__(self, config: Any, client: FlashRTClient):
        if not _LEROBOT_AVAILABLE:
            raise ImportError(
                "lerobot must be importable to construct FlashRTPI05Policy. "
                "Install lerobot in this environment."
            )
        super().__init__(config)
        self._client = client

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Replace local GPU inference with a FlashRT server call.

        Args:
            batch: Preprocessed batch from the PolicyProcessorPipeline.
                Expected keys:
                    observation.images.<cam>   float32 [1, C, H, W] in [0,1]
                    observation.state          float32 [1, 32]  normalized
                    observation.language_tokens         (unused — FlashRT tokenizes)
                    observation.language_attention_mask (unused)
                Task text is extracted from the complementary_data stored
                in the tokenized prompt key, or from observation.task if
                available.

        Returns:
            Tensor of shape (1, 30, 32) float32, raw **normalized** actions
            (same space as the postprocessor expects).
        """
        if batch.get("batch_size", 1) != 1 or self._first_batch_dim(batch) != 1:
            raise ValueError(
                "FlashRTPI05Policy requires batch_size=1 (the captured CUDA "
                "graph is B=1). Got batch first-dim != 1."
            )

        images_dict = self._extract_images(batch)
        state = self._extract_state(batch)
        prompt = self._extract_prompt(batch)
        seed = kwargs.pop("seed", None)

        actions_np = self._client.predict(
            images=images_dict,
            prompt=prompt,
            state=state,
            seed=seed,
        )

        return torch.as_tensor(
            actions_np, dtype=torch.float32, device=self.config.device
        ).unsqueeze(0)  # (1, 30, 32)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _first_batch_dim(self, batch: dict[str, Tensor]) -> int:
        for v in batch.values():
            if isinstance(v, Tensor):
                return v.shape[0]
        return 1

    def _extract_images(self, batch: dict[str, Tensor]) -> dict[str, np.ndarray]:
        """Extract, resize-to-224, and convert images to uint8 HWC numpy."""
        result: dict[str, np.ndarray] = {}
        for key in CANONICAL_VIEW_ORDER:
            if key not in batch:
                raise ValueError(
                    f"Camera '{key}' missing from batch. "
                    f"Expected all of CANONICAL_VIEW_ORDER: {CANONICAL_VIEW_ORDER}. "
                    "A missing camera must raise rather than silently shift views."
                )
            img: Tensor = batch[key]  # [1, C, H, W] float32 [0,1]
            if img.shape[0] != 1:
                raise ValueError(f"{key}: expected batch dim 1, got {img.shape[0]}")

            # resize_with_pad_torch handles any input HW → 224×224
            if img.shape[2:] != (_IMG_H, _IMG_W):
                img = resize_with_pad_torch(img, _IMG_H, _IMG_W)

            # [1, C, H, W] → [H, W, C] uint8
            img_hwc = img.squeeze(0).permute(1, 2, 0)  # [H, W, C]
            img_u8 = (img_hwc.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
            result[key] = img_u8.copy()  # contiguous uint8 HWC

        return result

    def _extract_state(self, batch: dict[str, Tensor]) -> np.ndarray:
        """Extract normalized padded state as float32 (32,) numpy."""
        from lerobot.utils.constants import OBS_STATE
        key = OBS_STATE
        if key not in batch:
            raise ValueError(
                f"'{key}' not in batch. "
                "State must be normalized by the preprocessor before reaching "
                "predict_action_chunk."
            )
        state: Tensor = batch[key]  # [1, 32]
        if state.shape != (1, _MAX_ACTION_DIM):
            raise ValueError(
                f"state shape {tuple(state.shape)}: expected (1, {_MAX_ACTION_DIM}). "
                "Pad observation.state to max_state_dim=32 via the preprocessor."
            )
        return state.squeeze(0).float().cpu().numpy()

    def _extract_prompt(self, batch: dict[str, Tensor]) -> str:
        """Extract the raw task text from the batch.

        The preprocessor's Pi05PrepareStateTokenizerProcessorStep replaces the
        task string in complementary_data with the full formatted prompt before
        tokenizing. We need the original task text — look for 'task' in
        complementary_data, or fall back to a stored attribute on the policy.
        """
        # Option 1: task stored as a plain string field in the batch
        if "task" in batch and isinstance(batch["task"], (list, str)):
            t = batch["task"]
            return (t[0] if isinstance(t, list) else t).strip()

        # Option 2: complementary_data.task (present before tokenizer step runs)
        if "complementary_data" in batch:
            cd = batch["complementary_data"]
            if isinstance(cd, dict) and "task" in cd:
                t = cd["task"]
                return (t[0] if isinstance(t, (list, tuple)) else t).strip()

        # Option 3: reuse last prompt if none present (second call in same episode)
        if self._client._last_prompt is not None:
            logger.debug("No task in batch; reusing last prompt")
            return self._client._last_prompt

        raise ValueError(
            "Could not find task text in batch. "
            "Expected 'task' key or 'complementary_data.task'. "
            "Ensure the tokenizer step has not already consumed the raw task string."
        )
