"""Shared helpers for the FP8 calibration search.

The key trick: the RTX Pi0.5 pipeline reads every FP8 activation scale through
a device pointer (``pipeline.fp8_act_scales[name].ptr``) at graph-replay time,
so a candidate calibration can be *hot-swapped* into an already-captured model
by uploading new values into those buffers. No reload, no recapture — scoring a
candidate costs one pass over the eval frames.

Per-frame amax collection replicates ``Pi05TorchFrontendRtx._calibrate_multi_frame``
(zero scales → eager dynamic-quant forward → read scales), except that the
prompt is set per frame so the state/task tokens match that frame.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from offline_rollout import load_policy_and_processors  # noqa: E402

ACTION_DIM_PAD = 32


def obs_dict(images: np.ndarray, state: np.ndarray) -> dict:
    views = [np.ascontiguousarray(images[v]) for v in range(images.shape[0])]
    obs = {"images": views, "image": views[0], "state": state}
    if len(views) >= 2:
        obs["wrist_image"] = views[1]
    if len(views) >= 3:
        obs["wrist_image_right"] = views[2]
    return obs


class Harness:
    def __init__(self, ckpt: str, precision: str = "fp8", action_horizon: int = 30,
                 rtc_guidance: bool = True, device: str = "cuda"):
        self.ckpt = Path(ckpt)
        self.precision = precision
        self.device = device
        (self.model, self.preprocessor, self.postprocessor, self.action_dim,
         self.view_keys) = load_policy_and_processors(
            self.ckpt, action_horizon, rtc_guidance=rtc_guidance, precision=precision)
        from lerobot.processor import RelativeActionsProcessorStep
        self.relative_step = next(
            (s for s in self.preprocessor.steps
             if isinstance(s, RelativeActionsProcessorStep)), None)

    # ── graph bootstrap ───────────────────────────────────────────────────────
    def bootstrap(self, task: str, images: np.ndarray, state: np.ndarray,
                  calib_obs: list[dict] | None = None, percentile: float = 99.9):
        """Build + capture the graph. ``calib_obs`` → real multi-frame calibrate()."""
        self.model.set_prompt(task, state=state)
        if self.precision == "fp8":
            obs = calib_obs if calib_obs is not None else [obs_dict(images, state)]
            self.model.calibrate(obs, percentile=percentile, verbose=False)
        # Make predict() see the prompt as current.
        self.model._current_prompt = task

    # ── FP8 scale access ──────────────────────────────────────────────────────
    @property
    def pipe(self):
        return self.model._pipe.pipeline

    def scale_names(self) -> list[str]:
        return list(self.pipe.fp8_act_scales.keys())

    def get_scales(self) -> np.ndarray:
        return np.array([float(b.download_new((1,), np.float32)[0])
                         for b in self.pipe.fp8_act_scales.values()], np.float32)

    def set_scales(self, vec: np.ndarray) -> None:
        for buf, v in zip(self.pipe.fp8_act_scales.values(), vec):
            buf.upload(np.array([v], dtype=np.float32))
        torch.cuda.synchronize()

    def frame_amax(self, task: str, images: np.ndarray, state: np.ndarray,
                   seed: int) -> np.ndarray:
        """Per-GEMM dynamic amax for one frame (its own prompt/state tokens)."""
        fe = self.model._pipe
        self.model.set_prompt(task, state=state)
        pipe = fe.pipeline
        with torch.cuda.stream(fe._graph_torch_stream):
            s = fe._graph_torch_stream.cuda_stream
            imgs = fe._stack_images(obs_dict(images, state))
            g = torch.Generator(device="cuda").manual_seed(seed)
            noise = torch.randn(fe.chunk_size, ACTION_DIM_PAD, dtype=torch.bfloat16,
                                device="cuda", generator=g)
            fe._copy_tensor_to_pipeline_buf_stream(imgs, pipe.input_images_buf, s)
            fe._copy_tensor_to_pipeline_buf_stream(noise, pipe.input_noise_buf, s)
            fe._zero_pipeline_scales()
            pipe.fp8_calibrated = False
            try:
                pipe.run_pipeline(stream=s)
                fe._cudart.cudaStreamSynchronize(ctypes.c_void_p(s))
            finally:
                pipe.fp8_calibrated = True
        return self.get_scales()

    # ── inference + postprocessing ────────────────────────────────────────────
    def predict(self, task: str, images: np.ndarray, state: np.ndarray, seed: int,
                warm: bool = False) -> np.ndarray:
        views = [np.ascontiguousarray(images[v]) for v in range(images.shape[0])]
        if warm:  # absorb any lazy capture / prompt rebuild so it can't eat RNG
            self.model.predict(images=views, prompt=task, state=state)
        torch.cuda.manual_seed(seed)
        return self.model.predict(images=views, prompt=task, state=state)

    def to_joint(self, chunk_norm: np.ndarray, state_raw: np.ndarray) -> np.ndarray:
        """(T, 32) normalized relative → (T, action_dim) absolute joint space."""
        if self.relative_step is not None:
            self.relative_step._last_state = torch.from_numpy(state_raw).unsqueeze(0).to(
                self.device)
        c = torch.from_numpy(chunk_norm[:, :self.action_dim]).float()
        with torch.no_grad():
            return self.postprocessor(c.unsqueeze(0)).squeeze(0).cpu().numpy()
