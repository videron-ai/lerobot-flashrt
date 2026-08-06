#!/usr/bin/env python3
"""offline_rollout.py — offline evaluation of FlashRT π₀.₅ on a LeRobot dataset.

FlashRT is used as a drop-in inference backend inside lerobot's standard
pre/post-processing pipeline.  The RTC execution model (ActionQueue,
LatencyTracker, inference_delay) is taken directly from lerobot so the
offline simulation is faithful to what lerobot-rollout does live.

Data flow per prediction:
  dataset frame
    ──► lerobot preprocessor  (normalizes state→16-dim, adds language tokens)
    ──► extract for FlashRT   (images → 224×224 uint8, state → normalized numpy)
    ──► flash_rt.model.predict → (action_horizon, 32) normalized chunk
    ──► slice [:, :action_dim]  → (action_horizon, 16)
    ──► lerobot postprocessor  → (action_horizon, 16) joint-space actions
    ──► ActionQueue.merge()    → sliding window with inference-delay skip
    ──► ActionQueue.get() per frame → one action per dataset step

RTC prefix guidance (``--rtc_guidance``, default on) forwards the queue's
unexecuted tail and the measured inference delay into FlashRT, so each chunk is
conditioned on its predecessor.  Run with ``--rtc_guidance=0`` to A/B it: the
GT-error metrics barely move, but ``splice`` (the joint-space jump at each
re-prediction boundary) is what guidance is actually there to fix.

CLI equivalent:
    lerobot-rollout --inference.type=rtc
                    --inference.rtc.execution_horizon=12
                    --inference.rtc.max_guidance_weight=10.0
                    --inference.rtc.prefix_attention_schedule=EXP
                    --interpolation_multiplier=3

Usage (inside the FlashRT container):
    python examples/offline_rollout.py \\
        --ckpt /openarm/outputs/train/openarm_folding_high_quality_60k/checkpoints/060000/pretrained_model \\
        --dataset videron/rollout_test_20260718_143829 \\
        --task "Fold the T-shirt properly" \\
        --out_dir ./rollout_outputs
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lerobot.policies.common.vla_utils import resize_with_pad_torch


# ── Runtime tuning ────────────────────────────────────────────────────────────

def configure_torch_threads(n: int) -> None:
    """Cap the torch CPU thread pool for the rollout loop.

    Call *after* the dataset is decoded so video decoding keeps the full pool.

    NOTE: the online script's GPU-side observation prep is deliberately not
    mirrored here — it exists to avoid a CPU ``uint8 -> float32`` expansion and
    an HWC->CHW transpose that ``prepare_observation_for_inference`` does on the
    host.  ``LeRobotDataset`` already hands back float32 CHW tensors, so this
    harness's ``preprocess_frame`` is a straight H2D and measures 0.69 ms —
    there is nothing to move to the GPU.
    """
    if n <= 0:
        return
    torch.set_num_threads(n)
    print(f"  torch CPU threads: {torch.get_num_threads()}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        default=os.environ.get(
            "LEROBOT_CKPT",
            "/openarm/outputs/train/openarm_folding_high_quality_60k"
            "/checkpoints/060000/pretrained_model",
        ),
    )
    p.add_argument(
        "--dataset",
        default=os.environ.get("DATASET_REPO", "videron/rollout_test_20260718_143829"),
    )
    p.add_argument("--task", default="Fold the T-shirt properly")
    # RTC parameters — mirror lerobot-rollout CLI names
    p.add_argument("--execution_horizon", type=int, default=12,
                   help="--inference.rtc.execution_horizon: normalises prev_chunk_left_over length")
    p.add_argument("--max_guidance_weight", type=float, default=10.0,
                   help="--inference.rtc.max_guidance_weight")
    p.add_argument("--prefix_attention_schedule", default="EXP",
                   choices=["ZEROS", "ONES", "LINEAR", "EXP"],
                   help="--inference.rtc.prefix_attention_schedule")
    p.add_argument("--action_horizon", type=int, default=30,
                   help="Chunk size passed to flash_rt.load_model()")
    p.add_argument("--interpolation_multiplier", type=int, default=3)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--max_frames", type=int, default=0, help="0 = full episode")
    p.add_argument("--out_dir", default="./rollout_outputs")
    p.add_argument("--no_save", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", default="fp8", choices=["fp8", "fp16"],
                   help="FlashRT numeric path. 'fp16' is the non-quantized "
                        "path: measured 18.7% lower mean per-joint MAE than "
                        "fp8, concentrated in the gripper joints, at the cost "
                        "of exact-length prompt graphs.")
    p.add_argument("--calib_frames", type=int, default=1,
                   help="FP8 activation-scale calibration samples, spread "
                        "evenly across the episode. 1 (default) keeps the "
                        "stock lazy behaviour: scales frozen on frame 0, the "
                        "start pose. Higher values cover the approach and "
                        "grasp, where the scene differs most from frame 0.")
    p.add_argument("--calib_percentile", type=float, default=99.9,
                   help="Percentile for multi-sample amax reduction "
                        "(100.0 = plain max). Only used when calib_frames > 1.")
    p.add_argument("--rtc_guidance", type=int, default=1,
                   help="1 = forward prev_chunk_left_over + inference_delay into "
                        "FlashRT's RTC prefix guidance; 0 = independent chunks")
    p.add_argument("--torch_threads", type=int, default=1,
                   help="Cap the torch CPU thread pool during the rollout loop "
                        "(0 = leave torch's default). Elementwise CPU work here "
                        "is memory-bandwidth-bound and an unbounded pool both "
                        "thrashes and adds jitter: measured 79.9 ms/tick at the "
                        "default 20 threads vs 75.3 ms at 1. Applied after the "
                        "dataset is decoded so loading stays parallel.")
    p.add_argument("--requeue_threshold", type=int, default=-1,
                   help="Re-predict once the queue drops to this many actions. "
                        "-1 = execution_horizon. Must be < action_horizon. The "
                        "real RTCInferenceEngine tops the queue up long before "
                        "it drains (rtc_queue_threshold defaults to the chunk "
                        "size), which is what leaves a leftover tail for the "
                        "prefix to condition on.")
    return p.parse_args()


# ── Setup ─────────────────────────────────────────────────────────────────────

def load_policy_and_processors(ckpt: Path, action_horizon: int, *,
                               rtc_guidance: bool = True,
                               execution_horizon: int = 12,
                               prefix_attention_schedule: str = "EXP",
                               max_guidance_weight: float = 10.0,
                               precision: str = "fp8"):
    """Load FlashRT model + lerobot pre/postprocessors from the same checkpoint.

    ``precision="fp16"`` selects FlashRT's non-quantized full-FP16 path.  It
    supports RTC prefix guidance like the FP8 default, but takes no
    ``state_prompt_mode`` (``load_model`` drops the kwarg rather than erroring,
    so the graph tracks the exact prompt length instead of running one padded
    graph, and recaptures when that length changes).
    """
    import json
    import flash_rt
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    with open(ckpt / "config.json") as f:
        raw = json.load(f)

    config = PI05Config(**{k: v for k, v in raw.items()
                           if k in PI05Config.__dataclass_fields__})

    # action_dim = real DOF count from the checkpoint (16 for this ckpt)
    action_dim = int(
        raw.get("output_features", {}).get("action", {}).get("shape", [16])[0]
    )

    # Camera view keys in the checkpoint's own (training) order, so the image
    # list handed to FlashRT matches what the model was trained on instead of
    # a hardcoded list that silently breaks on a different robot.
    view_keys = [k for k, v in raw.get("input_features", {}).items()
                 if v.get("type") == "VISUAL"]
    if not view_keys:
        raise RuntimeError("Checkpoint has no VISUAL input features")

    # Same preprocessor/postprocessor that lerobot-rollout loads
    preprocessor, postprocessor = make_pre_post_processors(
        config, pretrained_path=str(ckpt)
    )

    # RTC prefix guidance has to be armed before graph capture — the
    # correction is part of the captured denoise loop.
    rtc_kwargs = {}
    if rtc_guidance:
        rtc_kwargs = {
            "rtc_guidance": True,
            "rtc_execution_horizon": execution_horizon,
            "rtc_prefix_attention_schedule": prefix_attention_schedule.lower(),
            "rtc_max_guidance_weight": max_guidance_weight,
        }

    if precision not in ("fp8", "fp16"):
        raise ValueError(f"precision must be 'fp8' or 'fp16', got {precision!r}")
    precision_kwargs = (
        {"use_fp16": True, "use_fp8": False} if precision == "fp16" else {}
    )

    model = flash_rt.load_model(
        checkpoint=str(ckpt),
        framework="torch",
        num_views=len(view_keys),
        action_horizon=action_horizon,
        state_prompt_mode="fixed",
        autotune=3,
        **precision_kwargs,
        **rtc_kwargs,
    )

    print(f"  action_dim={action_dim}  action_horizon={action_horizon}")
    print(f"  views={view_keys}")
    print(f"  precision={precision}")
    print(f"  rtc_guidance={'on' if rtc_guidance else 'off'}")
    return model, preprocessor, postprocessor, action_dim, view_keys


def load_episode_frames(dataset_repo: str, episode_idx: int, max_frames: int):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(dataset_repo)
    indices = [i for i in range(len(ds)) if int(ds[i]["episode_index"]) == episode_idx]
    if not indices:
        raise ValueError(f"No frames for episode {episode_idx}")
    if max_frames > 0:
        indices = indices[:max_frames]
    frames = [ds[i] for i in indices]
    fps = ds.fps
    print(f"  Episode {episode_idx}: {len(frames)} frames @ {fps} fps "
          f"({len(frames)/fps:.1f} s)")
    return frames, fps


# ── Per-frame helpers ─────────────────────────────────────────────────────────

def preprocess_frame(frame: dict, preprocessor, task: str, device: str,
                     view_keys: list[str]) -> dict:
    """Build obs dict from a dataset frame and run lerobot's preprocessor."""
    obs = {}
    for key in view_keys:
        obs[key] = frame[key].unsqueeze(0).to(device)          # (1, C, H, W) float32
    obs["observation.state"] = frame["observation.state"].unsqueeze(0).to(device)  # (1, 16)
    obs["task"] = [task]
    return preprocessor(obs)


def extract_flashrt_inputs(preprocessed: dict, view_keys: list[str]) -> tuple[list, np.ndarray]:
    """Extract uint8 images and normalized state for flash_rt.model.predict().

    The lerobot preprocessor does NOT resize — PI05 resizes inside
    ``PI05Policy._preprocess_images`` with ``resize_with_pad_torch``
    (aspect-preserving + centered black padding), which FlashRT bypasses.  We
    apply the same resize here; a plain bilinear stretch would distort every
    frame relative to training (a 1280×720 wrist camera has to become 224×126
    with 49 px black bars, not a squashed square).

    State is already normalized to the training distribution by the
    preprocessor, which is what FlashRT's state-in-prompt discretizer expects.
    """
    imgs_uint8 = []
    for key in view_keys:
        img = resize_with_pad_torch(preprocessed[key], 224, 224)  # (1, C, 224, 224)
        hwc = img.squeeze(0).permute(1, 2, 0)                     # (224, 224, C)
        # round(), not truncate: values arrive as uint8/255, so x*255 lands at
        # e.g. 199.99997 and a plain cast would bias every pixel down by 1 LSB.
        imgs_uint8.append(
            hwc.mul(255).round().clamp(0, 255).to(torch.uint8).cpu().numpy())

    state_np = preprocessed["observation.state"].squeeze(0).float().cpu().numpy()  # (16,)
    return imgs_uint8, state_np


# ── RTC rollout ───────────────────────────────────────────────────────────────

def calibrate_over_episode(model, frames, preprocessor, task, device, view_keys,
                           *, n_samples: int, percentile: float) -> None:
    """Freeze FP8 activation scales over frames spanning the whole episode.

    FlashRT calibrates lazily on the first ``predict()``, which in this script
    is frame 0 — the start pose, grippers pointed down and nowhere near the
    table.  Those scales then have to cover every later frame, including the
    approach and grasp, where the scene looks nothing like frame 0.  That is
    the leading explanation for the FP8-vs-FP16 gap being concentrated in the
    gripper joints (joint 7: 8.144 vs 4.615 MAE), since FP16 does no
    quantization and therefore has no calibration frame to be wrong about.

    Sampling evenly across the episode and reducing per-sample amax with a
    percentile gives the scales a view of the actual operating range.  A
    percentile below 100 also clips one-off activation spikes that would
    otherwise stretch every scale to cover a single outlier frame.

    No-op for the FP16 path, which has no activation scales to freeze.
    """
    idxs = sorted({int(i) for i in
                   np.linspace(0, len(frames) - 1, num=n_samples, dtype=int)})
    obs_list = []
    for i in idxs:
        pre = preprocess_frame(frames[i], preprocessor, task, device, view_keys)
        imgs, state_np = extract_flashrt_inputs(pre, view_keys)
        obs = {"images": list(imgs), "image": imgs[0], "state": state_np}
        if len(imgs) >= 2:
            obs["wrist_image"] = imgs[1]
        if len(imgs) >= 3:
            obs["wrist_image_right"] = imgs[2]
        obs_list.append(obs)

    # calibrate() requires a prompt to have been set. Use the first sample's
    # state so the state-in-prompt tokens match what inference will produce.
    model.set_prompt(task, state=obs_list[0]["state"])
    shown = ", ".join(str(i) for i in idxs[:6])
    print(f"  calibrating on {len(obs_list)} frames "
          f"[{shown}{', ...' if len(idxs) > 6 else ''}] percentile={percentile}")
    model.calibrate(obs_list, percentile=percentile, verbose=True)


def rtc_rollout(
    frames: list[dict],
    model,
    preprocessor,
    postprocessor,
    action_dim: int,
    view_keys: list[str],
    task: str,
    fps: int,
    action_horizon: int,
    execution_horizon: int,
    max_guidance_weight: float,
    prefix_attention_schedule: str,
    device: str,
    rtc_guidance: bool = True,
    requeue_threshold: int = -1,
    calib_frames: int = 1,
    calib_percentile: float = 99.9,
) -> dict:
    """Simulate lerobot's RTC execution loop over dataset frames.

    Uses lerobot's ActionQueue and LatencyTracker directly so inference-delay
    compensation and prev_chunk_left_over handling match what lerobot-rollout does.

    Re-predicts once the queue drops to ``requeue_threshold`` actions, the
    synchronous equivalent of the background RTC thread firing on
    ``qsize() <= rtc_queue_threshold``.  This must trigger *before* the queue
    drains: ``get_left_over()`` returns a zero-length tensor once it does, which
    ``_normalize_prev_actions_length`` would zero-pad — and guiding toward an
    all-zero prefix pulls the chunk toward zero actions rather than toward its
    predecessor.
    """
    from lerobot.policies.rtc import ActionQueue, LatencyTracker, reanchor_relative_rtc_prefix
    from lerobot.policies.rtc.configuration_rtc import RTCConfig, RTCAttentionSchedule
    from lerobot.processor import NormalizerProcessorStep, RelativeActionsProcessorStep
    from lerobot.rollout.inference.rtc import _normalize_prev_actions_length

    # Processor introspection for relative-action re-anchoring, mirroring
    # RTCInferenceEngine.__init__. For a relative-action checkpoint the queue's
    # leftover is stored in ABSOLUTE joint space, while the policy consumes
    # actions relative to the *current* state — feeding the raw leftover in
    # would guide toward a target expressed in the previous chunk's frame.
    relative_step = next(
        (s for s in preprocessor.steps
         if isinstance(s, RelativeActionsProcessorStep) and s.enabled), None)
    normalizer_step = next(
        (s for s in preprocessor.steps
         if isinstance(s, NormalizerProcessorStep)), None)
    if relative_step is not None:
        print("  relative actions: RTC prefix will be re-anchored per prediction")

    rtc_config = RTCConfig(
        enabled=True,
        execution_horizon=execution_horizon,
        max_guidance_weight=max_guidance_weight,
        prefix_attention_schedule=RTCAttentionSchedule[prefix_attention_schedule],
    )
    queue = ActionQueue(rtc_config)
    latency_tracker = LatencyTracker()
    time_per_frame = 1.0 / fps

    if requeue_threshold < 0:
        requeue_threshold = execution_horizon
    if requeue_threshold >= action_horizon:
        raise ValueError(
            f"requeue_threshold ({requeue_threshold}) must be < action_horizon "
            f"({action_horizon}), otherwise the queue never falls below it and "
            f"every frame re-predicts.")

    N = len(frames)
    predicted  = np.zeros((N, action_dim), dtype=np.float32)
    gt_actions = np.zeros((N, action_dim), dtype=np.float32)
    chunk_at: list[int] = []
    latencies_ms: list[float] = []
    splice_jumps: list[float] = []

    if calib_frames > 1:
        calibrate_over_episode(
            model, frames, preprocessor, task, device, view_keys,
            n_samples=calib_frames, percentile=calib_percentile,
        )

    print(f"\n  RTC | frames={N} exec_h={execution_horizon} "
          f"act_h={action_horizon} guidance={max_guidance_weight} sched={prefix_attention_schedule} "
          f"prefix={'on' if rtc_guidance else 'off'}")

    for t in range(N):
        gt_actions[t] = frames[t]["action"].float().cpu().numpy()

        # Top the queue up before it drains (stand-in for the background RTC
        # thread firing on qsize <= rtc_queue_threshold)
        if queue.qsize() <= requeue_threshold:
            # ── Preprocess ────────────────────────────────────────────────
            preprocessed = preprocess_frame(frames[t], preprocessor, task, device, view_keys)
            imgs, state_np = extract_flashrt_inputs(preprocessed, view_keys)

            # ── RTC bookkeeping (mirrors RTCInferenceEngine._rtc_loop) ────
            idx_before   = queue.get_action_index()
            prev_left    = queue.get_left_over()     # (remaining, action_dim) or None
            # Joint-space action the robot is about to execute from the OLD
            # chunk; compared against the new chunk's replacement below.
            prev_exec    = queue.get_processed_left_over()

            # A drained queue yields a zero-length leftover — treat that as "no
            # previous chunk" rather than zero-padding it into a bogus target.
            if prev_left is not None and prev_left.shape[0] == 0:
                prev_left = None

            # Re-anchor the leftover into the current relative frame. The
            # preprocessor above just cached this frame's raw state, so this
            # matches RTCInferenceEngine._rtc_loop exactly. Skipping it feeds the
            # guidance a target from the previous chunk's coordinate frame, which
            # makes guided rollouts markedly *worse* than unguided ones.
            if prev_left is not None and relative_step is not None:
                raw_state = relative_step.get_cached_state()
                prev_abs = queue.get_processed_left_over()
                if raw_state is not None and prev_abs is not None and prev_abs.numel() > 0:
                    prev_left = reanchor_relative_rtc_prefix(
                        prev_actions_absolute=prev_abs,
                        current_state=raw_state,
                        relative_step=relative_step,
                        normalizer_step=normalizer_step,
                        policy_device=device,
                    )

            # Pad/truncate leftover to execution_horizon length for stable inference
            if prev_left is not None:
                prev_left = _normalize_prev_actions_length(prev_left, execution_horizon)

            # Compute delay from previous inference latency
            max_lat = latency_tracker.max()
            delay   = math.ceil(max_lat / time_per_frame) if max_lat else 0

            # prev_left is already in the same normalized space FlashRT returns
            # (it is ActionQueue's `original` tensor), so it needs no conversion.
            prev_np = None
            if rtc_guidance and prev_left is not None:
                prev_np = prev_left.float().cpu().numpy()

            # ── FlashRT inference ─────────────────────────────────────────
            t0 = time.perf_counter()
            # ``delay`` still drives the ActionQueue merge offset below; it is
            # only withheld from predict() when guidance is off, where the
            # frontend ignores it anyway (``pi05_rtx.py`` stages RTC inputs
            # solely under ``if self._rtc_guidance``).  Passing it
            # unconditionally makes ``api.predict`` take its RTC branch, which
            # hard-fails on frontends without a ``prev_actions`` parameter —
            # notably the FP16 reference path.
            chunk_norm = model.predict(           # (action_horizon, 32)
                images=imgs,
                prompt=task,
                state=state_np,                   # already normalized by preprocessor
                prev_actions=prev_np,
                inference_delay=delay if rtc_guidance else 0,
            )
            new_lat = time.perf_counter() - t0
            latencies_ms.append(new_lat * 1000.0)
            latency_tracker.add(new_lat)
            new_delay = math.ceil(new_lat / time_per_frame)

            # ── Postprocess via lerobot's pipeline ────────────────────────
            # Slice to action_dim (16) — the normalizer was fitted on action_dim dims
            chunk_16 = torch.from_numpy(chunk_norm[:, :action_dim]).float()  # (T, 16)
            with torch.no_grad():
                chunk_post = postprocessor(chunk_16.unsqueeze(0)).squeeze(0)  # (T, 16)

            # merge() skips the first new_delay actions (robot consumed them
            # while FlashRT was running) and resets the queue's read head
            queue.merge(chunk_16, chunk_post, new_delay, idx_before)
            chunk_at.append(t)

            # Splice discontinuity: the joint-space step the robot sees at the
            # handover from the old chunk to the new one. This is what prefix
            # guidance exists to shrink; MAE-vs-GT barely registers it.
            new_exec = queue.get_processed_left_over()
            if (prev_exec is not None and prev_exec.shape[0] > 0
                    and new_exec is not None and new_exec.shape[0] > 0):
                splice_jumps.append(
                    float((new_exec[0] - prev_exec[0]).abs().mean()))

            if len(chunk_at) <= 3 or len(chunk_at) % 20 == 0:
                print(f"    t={t:4d}/{N}  predict #{len(chunk_at):3d}  "
                      f"lat={new_lat*1000:.1f} ms  delay={new_delay}f", flush=True)

        # Pop the next action for this frame
        action = queue.get()
        if action is not None:
            predicted[t] = action.cpu().numpy()
        elif t > 0:
            predicted[t] = predicted[t - 1]     # queue exhausted: hold last action

    lat = np.array(latencies_ms)
    print(f"\n  {len(chunk_at)} predictions | "
          f"median={np.median(lat):.1f} ms  p99={np.percentile(lat, 99):.1f} ms")

    return {
        "predicted_actions": predicted,
        "gt_actions":        gt_actions,
        "chunk_at":          chunk_at,
        "latencies_ms":      lat,
        "rtc_config":        rtc_config,
        "splice_jumps":      np.array(splice_jumps, dtype=np.float32),
        "rtc_guidance":      rtc_guidance,
    }


# ── Stats ─────────────────────────────────────────────────────────────────────

def print_stats(results: dict, fps: int) -> None:
    pred = results["predicted_actions"]
    gt   = results["gt_actions"]
    err  = np.abs(pred - gt)
    nrm  = np.linalg.norm(pred, axis=1) * np.linalg.norm(gt, axis=1) + 1e-12
    cos  = np.einsum("nd,nd->n", pred, gt) / nrm

    # Typical joint-space motion between adjacent timesteps — the scale a
    # splice jump should be compared against.
    step = np.abs(np.diff(pred, axis=0)).mean()
    splice = results.get("splice_jumps", np.array([]))

    print(f"\n{'─'*55}")
    print("  Rollout statistics")
    print(f"{'─'*55}")
    print(f"  Frames:       {len(pred)} ({len(pred)/fps:.1f} s @ {fps} fps)")
    print(f"  Predictions:  {len(results['chunk_at'])}")
    print(f"  RTC guidance: {'on' if results.get('rtc_guidance') else 'off'}")
    print(f"  MAE mean:     {err.mean():.4f}")
    print(f"  MAE per joint:{err.mean(axis=0).round(4)}")
    print(f"  Cosine mean:  {cos.mean():.4f}")
    print(f"  Cosine min:   {cos.min():.4f}")
    print(f"  Step size:    {step:.4f}   (mean |Δaction| between frames)")
    if splice.size:
        print(f"  Splice mean:  {splice.mean():.4f}   ({splice.mean()/max(step,1e-9):.2f}x step size)")
        print(f"  Splice max:   {splice.max():.4f}")
    print(f"  Lat median:   {np.median(results['latencies_ms']):.1f} ms")
    print(f"  Lat p99:      {np.percentile(results['latencies_ms'], 99):.1f} ms")
    print(f"{'─'*55}\n")


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(results: dict, fps: int, out_dir: Path, episode: int,
                 tag: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    pred     = results["predicted_actions"]
    gt       = results["gt_actions"]
    chunk_at = results["chunk_at"]
    N, D     = pred.shape
    t_axis   = np.arange(N) / fps

    # ── Figure 1: per-joint GT vs predicted ───────────────────────────────────
    ncols = 4
    nrows = (D + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 2.8), sharex=True)
    axes = axes.flatten()
    for j in range(D):
        ax = axes[j]
        ax.plot(t_axis, gt[:, j],   color="#2196F3", lw=1.2, alpha=0.9)
        ax.plot(t_axis, pred[:, j], color="#FF5722", lw=1.0, alpha=0.9)
        for ca in chunk_at:
            ax.axvline(ca / fps, color="#4CAF50", lw=0.4, alpha=0.4)
        ax.set_title(f"joint {j}", fontsize=8, pad=2)
        ax.tick_params(labelsize=7)
        if j >= (nrows - 1) * ncols:
            ax.set_xlabel("time (s)", fontsize=7)
    for j in range(D, len(axes)):
        axes[j].set_visible(False)
    fig.legend(
        handles=[mpatches.Patch(color="#2196F3", label="GT"),
                 mpatches.Patch(color="#FF5722", label="FlashRT (RTC)")],
        loc="upper right", fontsize=9,
    )
    rtc = results["rtc_config"]
    fig.suptitle(
        f"Offline RTC Rollout — ep{episode} | {N} frames @ {fps} fps | "
        f"{len(chunk_at)} predictions | exec_h={rtc.execution_horizon} "
        f"guidance={rtc.max_guidance_weight} | "
        f"prefix={'on' if results.get('rtc_guidance') else 'off'}",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = out_dir / f"ep{episode:03d}{tag}_per_joint.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Saved {p}")

    # ── Figure 2: error heatmap + latency histogram ────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 4))
    err = np.abs(pred - gt)
    im  = axes2[0].imshow(err.T, aspect="auto", origin="lower", cmap="hot",
                           extent=[0, N / fps, -0.5, D - 0.5])
    for ca in chunk_at:
        axes2[0].axvline(ca / fps, color="cyan", lw=0.5, alpha=0.5)
    axes2[0].set_xlabel("time (s)"); axes2[0].set_ylabel("joint")
    axes2[0].set_title("|predicted − GT| heatmap")
    fig2.colorbar(im, ax=axes2[0], label="|Δ|")

    lat = results["latencies_ms"]
    axes2[1].hist(lat, bins=20, color="#4CAF50", edgecolor="white", alpha=0.85)
    axes2[1].axvline(np.median(lat), color="red",    lw=1.5, label=f"median {np.median(lat):.1f} ms")
    axes2[1].axvline(np.percentile(lat, 99), color="orange", lw=1.5,
                     label=f"p99 {np.percentile(lat, 99):.1f} ms")
    axes2[1].set_xlabel("latency (ms)"); axes2[1].set_ylabel("count")
    axes2[1].set_title("FlashRT inference latency")
    axes2[1].legend(fontsize=8)
    fig2.suptitle(f"Episode {episode} — Error & Latency", fontsize=11)
    fig2.tight_layout()
    p2 = out_dir / f"ep{episode:03d}{tag}_error_latency.png"
    fig2.savefig(p2, dpi=150); plt.close(fig2)
    print(f"  Saved {p2}")

    # ── Figure 3: RTC re-prediction timeline ──────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(16, 3))
    ax3.plot(t_axis, gt[:, :8].mean(1),   color="#2196F3", lw=1.0, label="GT mean j0-7")
    ax3.plot(t_axis, pred[:, :8].mean(1), color="#FF5722", lw=1.0,
             label="FlashRT mean j0-7")
    for ca in chunk_at:
        ax3.axvline(ca / fps, color="#4CAF50", lw=0.8, alpha=0.55)
    ax3.set_xlabel("time (s)"); ax3.set_ylabel("mean action value")
    ax3.set_title(f"RTC prediction events (green lines = {len(chunk_at)} calls, "
                  f"exec_h={rtc.execution_horizon})")
    ax3.legend(fontsize=8)
    fig3.tight_layout()
    p3 = out_dir / f"ep{episode:03d}{tag}_rtc_timeline.png"
    fig3.savefig(p3, dpi=150); plt.close(fig3)
    print(f"  Saved {p3}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    ckpt    = Path(args.ckpt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Offline FlashRT RTC Rollout ===")
    print(f"  ckpt:              {ckpt}")
    print(f"  dataset:           {args.dataset}")
    print(f"  task:              {args.task!r}")
    print(f"  execution_horizon: {args.execution_horizon}")
    print(f"  max_guidance:      {args.max_guidance_weight}")
    print(f"  sched:             {args.prefix_attention_schedule}")
    print(f"  action_horizon:    {args.action_horizon}")
    print(f"  precision:         {args.precision}")
    print(f"  rtc_guidance:      {bool(args.rtc_guidance)}")

    print("\nLoading FlashRT + lerobot processors...")
    model, preprocessor, postprocessor, action_dim, view_keys = load_policy_and_processors(
        ckpt, args.action_horizon,
        rtc_guidance=bool(args.rtc_guidance),
        execution_horizon=args.execution_horizon,
        prefix_attention_schedule=args.prefix_attention_schedule,
        max_guidance_weight=args.max_guidance_weight,
        precision=args.precision,
    )

    print("\nLoading dataset...")
    frames, fps = load_episode_frames(args.dataset, args.episode, args.max_frames)

    # After decoding, so the dataset load keeps the full thread pool.
    configure_torch_threads(args.torch_threads)

    print("\nRunning RTC rollout...")
    results = rtc_rollout(
        frames=frames,
        model=model,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        action_dim=action_dim,
        view_keys=view_keys,
        task=args.task,
        fps=fps,
        action_horizon=args.action_horizon,
        execution_horizon=args.execution_horizon,
        max_guidance_weight=args.max_guidance_weight,
        prefix_attention_schedule=args.prefix_attention_schedule,
        device=args.device,
        rtc_guidance=bool(args.rtc_guidance),
        requeue_threshold=args.requeue_threshold,
        calib_frames=args.calib_frames,
        calib_percentile=args.calib_percentile,
    )

    print_stats(results, fps)

    # Tag outputs so a guided/unguided A/B doesn't overwrite itself.
    tag = "_guided" if args.rtc_guidance else "_unguided"

    if not args.no_save:
        npz = out_dir / f"ep{args.episode:03d}{tag}_results.npz"
        np.savez(
            npz,
            predicted_actions=results["predicted_actions"],
            gt_actions=results["gt_actions"],
            chunk_at=np.array(results["chunk_at"]),
            latencies_ms=results["latencies_ms"],
            splice_jumps=results["splice_jumps"],
            rtc_guidance=np.array(bool(results["rtc_guidance"])),
        )
        print(f"  Saved {npz}")
        print("\nGenerating plots...")
        plot_results(results, fps, out_dir, args.episode, tag)

    print("\nDone.")


if __name__ == "__main__":
    main()
