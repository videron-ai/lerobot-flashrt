#!/usr/bin/env python3
"""offline_rollout.py — offline evaluation of FlashRT π₀.₅ on a LeRobot dataset.

FlashRT is used as a drop-in inference backend inside lerobot's standard
pre/post-processing pipeline.  The RTC execution model (ActionQueue,
LatencyTracker, inference_delay) is taken directly from lerobot so the
offline simulation is faithful to what lerobot-rollout does live.

Data flow per prediction:
  dataset frame
    ──► lerobot preprocessor  (normalizes state→16-dim, adds language tokens)
    ──► extract for FlashRT   (images → 224×224 uint8, state → 16-dim numpy)
    ──► flash_rt.model.predict → (action_horizon, 32) normalized chunk
    ──► slice [:, :action_dim]  → (action_horizon, 16)
    ──► lerobot postprocessor  → (action_horizon, 16) joint-space actions
    ──► ActionQueue.merge()    → sliding window with inference-delay skip
    ──► ActionQueue.get() per frame → one action per dataset step

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
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_VIEW_KEYS = [
    "observation.images.left_wrist",
    "observation.images.right_wrist",
    "observation.images.base",
]


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
    return p.parse_args()


# ── Setup ─────────────────────────────────────────────────────────────────────

def load_policy_and_processors(ckpt: Path, action_horizon: int):
    """Load FlashRT model + lerobot pre/postprocessors from the same checkpoint."""
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

    # Same preprocessor/postprocessor that lerobot-rollout loads
    preprocessor, postprocessor = make_pre_post_processors(
        config, pretrained_path=str(ckpt)
    )

    model = flash_rt.load_model(
        checkpoint=str(ckpt),
        framework="torch",
        num_views=3,
        action_horizon=action_horizon,
        state_prompt_mode="fixed",
        autotune=3,
    )

    print(f"  action_dim={action_dim}  action_horizon={action_horizon}")
    return model, preprocessor, postprocessor, action_dim


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

def preprocess_frame(frame: dict, preprocessor, task: str, device: str) -> dict:
    """Build obs dict from a dataset frame and run lerobot's preprocessor."""
    obs = {}
    for key in _VIEW_KEYS:
        obs[key] = frame[key].unsqueeze(0).to(device)          # (1, C, H, W) float32
    obs["observation.state"] = frame["observation.state"].unsqueeze(0).to(device)  # (1, 16)
    obs["task"] = [task]
    return preprocessor(obs)


def extract_flashrt_inputs(preprocessed: dict, device: str) -> tuple[list, np.ndarray]:
    """Extract uint8 images and normalized state for flash_rt.model.predict().

    Images are resized to 224×224 here (FlashRT requires uniform size).
    State is already normalized to the training distribution by the preprocessor.
    """
    imgs_uint8 = []
    for key in _VIEW_KEYS:
        img = preprocessed[key]                      # (1, C, H, W) float32 [0,1]
        img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
        hwc = img.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0)
        imgs_uint8.append((hwc * 255).to(torch.uint8).cpu().numpy())

    state_np = preprocessed["observation.state"].squeeze(0).float().cpu().numpy()  # (16,)
    return imgs_uint8, state_np


# ── RTC rollout ───────────────────────────────────────────────────────────────

def rtc_rollout(
    frames: list[dict],
    model,
    preprocessor,
    postprocessor,
    action_dim: int,
    task: str,
    fps: int,
    action_horizon: int,
    execution_horizon: int,
    max_guidance_weight: float,
    prefix_attention_schedule: str,
    device: str,
) -> dict:
    """Simulate lerobot's RTC execution loop over dataset frames.

    Uses lerobot's ActionQueue and LatencyTracker directly so inference-delay
    compensation and prev_chunk_left_over handling match what lerobot-rollout does.

    Re-predicts whenever the ActionQueue is empty (synchronous equivalent of the
    background RTC thread firing when queue.qsize() drops below threshold).
    """
    from lerobot.policies.rtc import ActionQueue, LatencyTracker
    from lerobot.policies.rtc.configuration_rtc import RTCConfig, RTCAttentionSchedule
    from lerobot.rollout.inference.rtc import _normalize_prev_actions_length

    rtc_config = RTCConfig(
        enabled=True,
        execution_horizon=execution_horizon,
        max_guidance_weight=max_guidance_weight,
        prefix_attention_schedule=RTCAttentionSchedule[prefix_attention_schedule],
    )
    queue = ActionQueue(rtc_config)
    latency_tracker = LatencyTracker()
    time_per_frame = 1.0 / fps

    N = len(frames)
    predicted  = np.zeros((N, action_dim), dtype=np.float32)
    gt_actions = np.zeros((N, action_dim), dtype=np.float32)
    chunk_at: list[int] = []
    latencies_ms: list[float] = []

    print(f"\n  RTC | frames={N} exec_h={execution_horizon} "
          f"act_h={action_horizon} guidance={max_guidance_weight} sched={prefix_attention_schedule}")

    for t in range(N):
        gt_actions[t] = frames[t]["action"].float().cpu().numpy()

        # Re-predict when the queue runs dry (synchronous stand-in for the
        # background RTC thread firing when qsize <= queue_threshold)
        if queue.empty():
            # ── Preprocess ────────────────────────────────────────────────
            preprocessed = preprocess_frame(frames[t], preprocessor, task, device)
            imgs, state_np = extract_flashrt_inputs(preprocessed, device)

            # ── RTC bookkeeping (mirrors RTCInferenceEngine._rtc_loop) ────
            idx_before   = queue.get_action_index()
            prev_left    = queue.get_left_over()     # (remaining, action_dim) or None

            # Pad/truncate leftover to execution_horizon length for stable inference
            if prev_left is not None:
                prev_left = _normalize_prev_actions_length(prev_left, execution_horizon)

            # Compute delay from previous inference latency
            max_lat = latency_tracker.max()
            delay   = math.ceil(max_lat / time_per_frame) if max_lat else 0

            # ── FlashRT inference ─────────────────────────────────────────
            t0 = time.perf_counter()
            chunk_norm = model.predict(           # (action_horizon, 32)
                images=imgs,
                prompt=task,
                state=state_np,                   # already normalized by preprocessor
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
    }


# ── Stats ─────────────────────────────────────────────────────────────────────

def print_stats(results: dict, fps: int) -> None:
    pred = results["predicted_actions"]
    gt   = results["gt_actions"]
    err  = np.abs(pred - gt)
    nrm  = np.linalg.norm(pred, axis=1) * np.linalg.norm(gt, axis=1) + 1e-12
    cos  = np.einsum("nd,nd->n", pred, gt) / nrm

    print(f"\n{'─'*55}")
    print("  Rollout statistics")
    print(f"{'─'*55}")
    print(f"  Frames:       {len(pred)} ({len(pred)/fps:.1f} s @ {fps} fps)")
    print(f"  Predictions:  {len(results['chunk_at'])}")
    print(f"  MAE mean:     {err.mean():.4f}")
    print(f"  MAE per joint:{err.mean(axis=0).round(4)}")
    print(f"  Cosine mean:  {cos.mean():.4f}")
    print(f"  Cosine min:   {cos.min():.4f}")
    print(f"  Lat median:   {np.median(results['latencies_ms']):.1f} ms")
    print(f"  Lat p99:      {np.percentile(results['latencies_ms'], 99):.1f} ms")
    print(f"{'─'*55}\n")


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(results: dict, fps: int, out_dir: Path, episode: int) -> None:
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
        ax.plot(t_axis, pred[:, j], color="#FF5722", lw=1.0, alpha=0.9, linestyle="--")
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
        f"guidance={rtc.max_guidance_weight}",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = out_dir / f"ep{episode:03d}_per_joint.png"
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
    p2 = out_dir / f"ep{episode:03d}_error_latency.png"
    fig2.savefig(p2, dpi=150); plt.close(fig2)
    print(f"  Saved {p2}")

    # ── Figure 3: RTC re-prediction timeline ──────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(16, 3))
    ax3.plot(t_axis, gt[:, :8].mean(1),   color="#2196F3", lw=1.0, label="GT mean j0-7")
    ax3.plot(t_axis, pred[:, :8].mean(1), color="#FF5722", lw=1.0,
             linestyle="--", label="FlashRT mean j0-7")
    for ca in chunk_at:
        ax3.axvline(ca / fps, color="#4CAF50", lw=0.8, alpha=0.55)
    ax3.set_xlabel("time (s)"); ax3.set_ylabel("mean action value")
    ax3.set_title(f"RTC prediction events (green lines = {len(chunk_at)} calls, "
                  f"exec_h={rtc.execution_horizon})")
    ax3.legend(fontsize=8)
    fig3.tight_layout()
    p3 = out_dir / f"ep{episode:03d}_rtc_timeline.png"
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

    print("\nLoading FlashRT + lerobot processors...")
    model, preprocessor, postprocessor, action_dim = load_policy_and_processors(
        ckpt, args.action_horizon
    )

    print("\nLoading dataset...")
    frames, fps = load_episode_frames(args.dataset, args.episode, args.max_frames)

    print("\nRunning RTC rollout...")
    results = rtc_rollout(
        frames=frames,
        model=model,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        action_dim=action_dim,
        task=args.task,
        fps=fps,
        action_horizon=args.action_horizon,
        execution_horizon=args.execution_horizon,
        max_guidance_weight=args.max_guidance_weight,
        prefix_attention_schedule=args.prefix_attention_schedule,
        device=args.device,
    )

    print_stats(results, fps)

    if not args.no_save:
        npz = out_dir / f"ep{args.episode:03d}_results.npz"
        np.savez(
            npz,
            predicted_actions=results["predicted_actions"],
            gt_actions=results["gt_actions"],
            chunk_at=np.array(results["chunk_at"]),
            latencies_ms=results["latencies_ms"],
        )
        print(f"  Saved {npz}")
        print("\nGenerating plots...")
        plot_results(results, fps, out_dir, args.episode)

    print("\nDone.")


if __name__ == "__main__":
    main()
