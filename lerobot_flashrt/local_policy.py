"""LocalFlashRTPI05Policy — PI05Policy with in-process FlashRT inference.

Identical contract to FlashRTPI05Policy but removes the HTTP server:
the FlashRT model is loaded directly into the same process and called
from predict_action_chunk().  Use when LeRobot and FlashRT coexist in
the same container or environment.

The calling convention and batch format are identical to FlashRTPI05Policy
so the two variants are drop-in replacements for each other.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

try:
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy, resize_with_pad_torch
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.utils.constants import OBS_STATE
    _LEROBOT_AVAILABLE = True
except ImportError:
    _LEROBOT_AVAILABLE = False
    PI05Policy = object  # type: ignore[misc,assignment]
    PI05Config = None    # type: ignore[assignment]

from .client import CANONICAL_VIEW_ORDER

_IMG_H, _IMG_W = 224, 224
_MAX_ACTION_DIM = 32


class LocalFlashRTPI05Policy(PI05Policy):
    """PI05Policy with predict_action_chunk replaced by direct FlashRT inference.

    Args:
        config: PI05Config loaded from the checkpoint.
        model: flash_rt.VLAModel returned by flash_rt.load_model().
        state_in_prompt_dim: Number of state dims to pass to model.predict().
            Must equal the real DOF count (output_features.action.shape[0])
            so the discretized state bins in the prompt match what LeRobot's
            Pi05PrepareStateTokenizerProcessorStep produces.
    """

    config_class = PI05Config if PI05Config is not None else type(None)
    name = "local_flashrt_pi05"

    def __init__(self, config: Any, model: Any, state_in_prompt_dim: int):
        if not _LEROBOT_AVAILABLE:
            raise ImportError(
                "lerobot must be importable to construct LocalFlashRTPI05Policy."
            )
        super().__init__(config)
        self._model = model
        self._state_in_prompt_dim = state_in_prompt_dim
        self._last_prompt: str | None = None

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Run one inference step via the in-process FlashRT model.

        Args:
            batch: Preprocessed batch from the PolicyProcessorPipeline.
            seed: Optional int — seeds CUDA RNG before inference for
                reproducible ODE noise (used by the parity test).

        Returns:
            Tensor shape (1, Sa, 32) float32, raw normalized actions.
        """
        images_dict = self._extract_images(batch)
        state = self._extract_state(batch)
        prompt = self._extract_prompt(batch)
        seed = kwargs.pop("seed", None)

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        imgs_ordered = [images_dict[k] for k in CANONICAL_VIEW_ORDER]

        actions_np = self._model.predict(
            images=imgs_ordered,
            prompt=prompt,
            state=state[:self._state_in_prompt_dim],
        )

        return torch.as_tensor(
            actions_np, dtype=torch.float32, device=self.config.device
        ).unsqueeze(0)  # (1, Sa, 32)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_images(self, batch: dict[str, Tensor]) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for key in CANONICAL_VIEW_ORDER:
            if key not in batch:
                raise ValueError(
                    f"Camera '{key}' missing from batch. "
                    f"Expected all of CANONICAL_VIEW_ORDER: {CANONICAL_VIEW_ORDER}."
                )
            img: Tensor = batch[key]
            if img.shape[2:] != (_IMG_H, _IMG_W):
                img = resize_with_pad_torch(img, _IMG_H, _IMG_W)
            img_hwc = img.squeeze(0).permute(1, 2, 0)
            result[key] = (img_hwc.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy().copy()
        return result

    def _extract_state(self, batch: dict[str, Tensor]) -> np.ndarray:
        if OBS_STATE not in batch:
            raise ValueError(f"'{OBS_STATE}' not in batch.")
        state: Tensor = batch[OBS_STATE]
        if state.shape != (1, _MAX_ACTION_DIM):
            raise ValueError(
                f"state shape {tuple(state.shape)}: expected (1, {_MAX_ACTION_DIM}). "
                "Pad observation.state to max_state_dim=32 via the preprocessor."
            )
        return state.squeeze(0).float().cpu().numpy()

    def _extract_prompt(self, batch: dict[str, Tensor]) -> str:
        if "task" in batch and isinstance(batch["task"], (list, str)):
            t = batch["task"]
            return (t[0] if isinstance(t, list) else t).strip()
        if "complementary_data" in batch:
            cd = batch["complementary_data"]
            if isinstance(cd, dict) and "task" in cd:
                t = cd["task"]
                return (t[0] if isinstance(t, (list, tuple)) else t).strip()
        if self._last_prompt is not None:
            logger.debug("No task in batch; reusing last prompt")
            return self._last_prompt
        raise ValueError(
            "Could not find task text in batch. "
            "Expected 'task' key or 'complementary_data.task'."
        )
