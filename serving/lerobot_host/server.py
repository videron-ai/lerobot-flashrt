#!/usr/bin/env python3
"""
FlashRT LeRobot inference server.

Loads a Pi0.5 model once at startup and serves rollout predictions via HTTP.
Designed for the two-container architecture where LeRobot and FlashRT live in
separate environments; the LeRobot side drives this server through
lerobot_flashrt/client.py.

Usage:
    python serving/lerobot_host/server.py \
        --checkpoint /path/to/lerobot_pi05_checkpoint \
        --action-horizon 30 \
        --num-views 3 \
        --view-order observation.images.left_wrist \
                     observation.images.right_wrist \
                     observation.images.base \
        --state-prompt-mode fixed \
        --port 8000

Gate test (curl):
    # health
    curl http://localhost:8000/health

    # real predict (base64 image + 32-dim state)
    python - <<'EOF'
    import base64, json, numpy as np, requests
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    imgs_b64 = [base64.b64encode(img.tobytes()).decode() for _ in range(3)]
    state = np.zeros(32, dtype=np.float32)
    r = requests.post("http://localhost:8000/predict", json={
        "images": imgs_b64,
        "prompt": "fold the fabric",
        "state": state.tolist(),
    })
    print(r.json()["shape"], r.json()["latency_ms"])
    EOF
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("flashrt.lerobot_server")

# ── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="FlashRT LeRobot inference server")
    p.add_argument("--checkpoint", required=True,
                   help="Path to the LeRobot / openpi-compatible Pi0.5 safetensors checkpoint dir")
    p.add_argument("--action-horizon", type=int, default=30,
                   help="Action chunk size forwarded to load_model (default 30)")
    p.add_argument("--num-views", type=int, default=3,
                   help="Number of camera views (default 3); must equal len(--view-order)")
    p.add_argument("--view-order", nargs="+",
                   default=[
                       "observation.images.left_wrist",
                       "observation.images.right_wrist",
                       "observation.images.base",
                   ],
                   help="Ordered list of camera names; must match CANONICAL_VIEW_ORDER in client")
    p.add_argument("--state-prompt-mode", default="fixed",
                   choices=["fixed", "exact"],
                   help="Pi0.5 state-in-prompt graph strategy (default 'fixed' for rollouts)")
    p.add_argument("--state-prompt-fixed-max-len", type=int, default=None,
                   help="Pi0.5 fixed-mode padded prompt capacity (tokens; default from frontend)")
    p.add_argument("--autotune", type=int, default=3)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--recalibrate", action="store_true",
                   help="Force fresh FP8 calibration (ignore cache)")
    return p.parse_args()


args = _parse_args()

if args.num_views != len(args.view_order):
    raise SystemExit(
        f"--num-views {args.num_views} != len(--view-order) {len(args.view_order)}"
    )

# ── Load model at startup ────────────────────────────────────────────────────

logger.info(
    "Loading Pi0.5 | checkpoint=%s | action_horizon=%d | num_views=%d | view_order=%s",
    args.checkpoint, args.action_horizon, args.num_views, args.view_order,
)
t0 = time.time()

import flash_rt

model = flash_rt.load_model(
    checkpoint=args.checkpoint,
    framework="torch",
    num_views=args.num_views,
    action_horizon=args.action_horizon,
    autotune=args.autotune,
    recalibrate=args.recalibrate,
    state_prompt_mode=args.state_prompt_mode,
    state_prompt_fixed_max_len=args.state_prompt_fixed_max_len,
)

_load_time = time.time() - t0
logger.info(
    "Model ready in %.1fs | class=%s | chunk=%s",
    _load_time,
    type(model._pipe).__name__,
    getattr(model._pipe, "chunk_size", getattr(model._pipe, "Sa", "?")),
)

# Read original_action_dim from checkpoint config so the server only digitizes
# the real DOF dims into the state-in-prompt, not the zero-padding dims.
# This must match what LeRobot's Pi05PrepareStateTokenizerProcessorStep does:
# it receives the normalizer output (16-dim for this checkpoint) and builds a
# prompt with exactly that many bins.  Passing all 32 dims (with 16 trailing
# zeros → bin 127) would produce a different prompt → different conditioning.
_state_in_prompt_dim = 32
try:
    import json as _json
    with open(os.path.join(args.checkpoint, "config.json")) as _f:
        _ckpt_cfg = _json.load(_f)
    _action_shape = (
        _ckpt_cfg
        .get("output_features", {})
        .get("action", {})
        .get("shape", [32])
    )
    _state_in_prompt_dim = int(_action_shape[0])
except Exception as _exc:
    logger.warning("Could not read original_action_dim from config.json (%s); "
                   "using full 32-dim state for prompt", _exc)
logger.info("State-in-prompt dims: %d (of 32)", _state_in_prompt_dim)

# Warmup tracking
_warmup_done = False

# ── FastAPI app ──────────────────────────────────────────────────────────────

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="FlashRT LeRobot Server", version=flash_rt.__version__)

# One GPU, one captured graph — never run two infers concurrently.
_lock = asyncio.Lock()

# ── Request / response schemas ───────────────────────────────────────────────

_IMG_H, _IMG_W, _IMG_C = 224, 224, 3
_IMG_BYTES = _IMG_H * _IMG_W * _IMG_C
_STATE_DIM = 32


class PredictRequest(BaseModel):
    images: List[str]               # exactly num_views base64-encoded (H*W*C) uint8 blobs
    prompt: str                     # raw task text, NOT the full "Task:…State:…Action:" form
    state: List[float]              # 32-dim normalized state (LeRobot preprocessor output)
    seed: Optional[int] = None      # if set, seeds CUDA RNG before inference for reproducible noise


class PredictResponse(BaseModel):
    actions: List[List[float]]      # (chunk, 32) normalized, LeRobot postprocessor unnormalizes
    latency_ms: float
    shape: List[int]


class WarmupRequest(BaseModel):
    images: List[str]               # same format as PredictRequest
    prompt: str
    states: List[List[float]]       # list of representative 32-dim states for bucket warming


class WarmupResponse(BaseModel):
    warmed_lengths: Optional[List[int]]
    calibrated: bool
    duration_ms: float


# ── Helpers ──────────────────────────────────────────────────────────────────

def _decode_images(b64_list: List[str]) -> List[np.ndarray]:
    imgs = []
    for i, b64 in enumerate(b64_list):
        try:
            raw = base64.b64decode(b64)
        except Exception as exc:
            raise ValueError(f"images[{i}]: base64 decode failed: {exc}") from exc
        if len(raw) != _IMG_BYTES:
            raise ValueError(
                f"images[{i}]: expected {_IMG_BYTES} bytes "
                f"({_IMG_H}×{_IMG_W}×{_IMG_C} uint8), got {len(raw)}"
            )
        imgs.append(np.frombuffer(raw, dtype=np.uint8).reshape(_IMG_H, _IMG_W, _IMG_C))
    return imgs


def _validate_predict_request(req: PredictRequest) -> tuple[list, np.ndarray]:
    if len(req.images) != args.num_views:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Expected {args.num_views} images (view_order={args.view_order}), "
                f"got {len(req.images)}. Wrong view count will produce garbage actions."
            ),
        )
    try:
        imgs = _decode_images(req.images)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if len(req.state) != _STATE_DIM:
        raise HTTPException(
            status_code=422,
            detail=f"state must be {_STATE_DIM}-dim float32, got {len(req.state)} elements",
        )
    state = np.array(req.state, dtype=np.float32)
    return imgs, state


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    pipe = model._pipe
    chunk = getattr(pipe, "chunk_size", getattr(pipe, "Sa", None))
    return {
        "status": "ok",
        "warmup_done": _warmup_done,
        "frontend_class": type(pipe).__name__,
        "chunk_size": chunk,
        "action_dim": 32,
        "num_views": args.num_views,
        "view_order": args.view_order,
        "state_prompt_mode": args.state_prompt_mode,
        "framework": model.framework,
        "version": flash_rt.__version__,
        "current_prompt": model.prompt,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    imgs, state = _validate_predict_request(req)

    async with _lock:
        try:
            if req.seed is not None:
                import torch
                torch.manual_seed(req.seed)
                torch.cuda.manual_seed(req.seed)
            t0 = time.perf_counter()
            actions = model.predict(
                images=imgs,
                prompt=req.prompt if req.prompt else None,
                state=state[:_state_in_prompt_dim],
            )
            latency_ms = (time.perf_counter() - t0) * 1000
        except Exception as exc:
            logger.exception("predict() failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    if actions.shape != (args.action_horizon, _STATE_DIM):
        logger.warning(
            "Unexpected actions shape %s (expected (%d, %d))",
            actions.shape, args.action_horizon, _STATE_DIM,
        )

    logger.debug("predict latency=%.1f ms shape=%s", latency_ms, list(actions.shape))
    return PredictResponse(
        actions=actions.tolist(),
        latency_ms=round(latency_ms, 2),
        shape=list(actions.shape),
    )


@app.post("/warmup", response_model=WarmupResponse)
async def warmup(req: WarmupRequest):
    """Calibrate FP8 scales and warm state-prompt graph buckets.

    Call once at startup with real (or representative synthetic) frames before
    starting the rollout loop. The response blocks until both calibration and
    bucket warming complete.
    """
    global _warmup_done

    if len(req.images) != args.num_views:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {args.num_views} images, got {len(req.images)}",
        )
    try:
        imgs = _decode_images(req.images)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not req.states:
        raise HTTPException(status_code=422, detail="states list must be non-empty")

    states_np = []
    for i, s in enumerate(req.states):
        if len(s) != _STATE_DIM:
            raise HTTPException(
                status_code=422,
                detail=f"states[{i}]: expected {_STATE_DIM} floats, got {len(s)}",
            )
        states_np.append(np.array(s, dtype=np.float32))

    async with _lock:
        t0 = time.perf_counter()

        # FP8 calibration (build activation scales from real frames)
        obs = _build_obs(imgs, states_np[0])
        try:
            if hasattr(model._pipe, "calibrate"):
                logger.info("Running FP8 calibration on %d frame(s)", 1)
                model._pipe.calibrate([obs], percentile=99.9, verbose=True)
            elif hasattr(model._pipe, "calibrate_with_real_data"):
                logger.info("Running FP8 calibration (calibrate_with_real_data)")
                model._pipe.calibrate_with_real_data([obs])
        except Exception as exc:
            logger.warning("Calibration failed (may be pre-calibrated): %s", exc)

        # State-prompt bucket warming (exact mode only; fixed mode uses one graph)
        warmed_lengths = None
        if args.state_prompt_mode == "exact" and hasattr(model, "warm_state_prompt_buckets"):
            try:
                logger.info(
                    "Warming state-prompt buckets for %d states", len(states_np)
                )
                warmed_lengths = model.warm_state_prompt_buckets(
                    images=imgs,
                    prompt=req.prompt,
                    states=states_np,
                )
            except Exception as exc:
                logger.warning("warm_state_prompt_buckets failed: %s", exc)

        duration_ms = (time.perf_counter() - t0) * 1000
        _warmup_done = True

    logger.info(
        "Warmup complete (%.0f ms) | calibrated=%s | warmed_lengths=%s",
        duration_ms,
        getattr(model._pipe, "calibrated", "?"),
        warmed_lengths,
    )
    return WarmupResponse(
        warmed_lengths=list(warmed_lengths) if warmed_lengths is not None else None,
        calibrated=bool(getattr(model._pipe, "calibrated", False)),
        duration_ms=round(duration_ms, 1),
    )


def _build_obs(imgs: list[np.ndarray], state: np.ndarray) -> dict:
    """Build the obs dict that FlashRT frontends expect."""
    obs = {
        "images": imgs,
        "image": imgs[0],
        "state": state,
    }
    if len(imgs) >= 2:
        obs["wrist_image"] = imgs[1]
    if len(imgs) >= 3:
        obs["wrist_image_right"] = imgs[2]
    return obs


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logger.info(
        "Starting FlashRT LeRobot server on %s:%d | view_order=%s",
        args.host, args.port, args.view_order,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
