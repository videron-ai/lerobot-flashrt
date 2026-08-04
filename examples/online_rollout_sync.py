#!/usr/bin/env python3
"""online_rollout_sync.py — lerobot-rollout with FlashRT inference, synchronous mode.

Identical to online_rollout.py except uses ``--inference.type=sync`` instead of RTC.

In sync mode PI05Policy maintains an internal action queue: ``select_action`` calls
``predict_action_chunk`` once every ``n_action_steps`` ticks and pops one action per
tick.  The control loop blocks during inference, so no background thread is created.
With FlashRT at ~20 ms and n_action_steps=30 at 50 Hz the inference fits inside a
single 20 ms control tick — there is no latency penalty vs. RTC for this checkpoint.

Use this script when:
  - You want to avoid the RTC background thread entirely
  - You do not need prefix-attention chunk continuity
  - Debugging policy output tick-by-tick (deterministic, single-threaded)

Usage:
    python examples/online_rollout_sync.py \\
        --config_path=/openarm/rollout.yaml \\
        --strategy.type=base \\
        --policy.path=/openarm/outputs/train/openarm_folding_high_quality_60k/checkpoints/060000/pretrained_model \\
        --task="Fold the T-shirt properly" \\
        --interpolation_multiplier=3 \\
        --inference.type=sync \\
        --use_torch_compile=False \\
        --duration=0
"""

from __future__ import annotations

import logging
import os
import sys
import types

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Camera / robot / teleop imports that draccus needs for config resolution ──
from lerobot.cameras.opencv import OpenCVCameraConfig       # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig              # noqa: F401
from lerobot.configs import FeatureType, parser
from lerobot.robots import (                                  # noqa: F401
    Robot, RobotConfig,
    bi_openarm_follower, bi_rebot_b601_follower, bi_so_follower,
    earthrover_mini_plus, hope_jr, koch_follower, lekiwi,
    omx_follower, openarm_follower, reachy2, rebot_b601_follower,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.rollout import RolloutConfig, build_rollout_context, create_strategy
from lerobot.teleoperators import (                           # noqa: F401
    Teleoperator, TeleoperatorConfig,
    bi_openarm_leader, bi_openarm_mini, bi_rebot_102_leader, bi_so_leader,
    homunculus, koch_leader, omx_leader, openarm_leader, openarm_mini,
    reachy2_teleoperator, rebot_102_leader, so_leader,
    unitree_g1,
)
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_visualization, shutdown_visualization

logger = logging.getLogger(__name__)


# ── FlashRT warmup ────────────────────────────────────────────────────────────

def _warmup_flashrt(model, task: str, state_dim: int, num_views: int, n_iters: int = 20) -> None:
    """Run n_iters dummy predict() calls to trigger CUDA graph capture.

    FlashRT captures a static CUDA graph on the first call (can take 1–60 s).
    Warming up before the robot loop ensures that latency is paid upfront
    rather than on the first live control tick.
    """
    import numpy as np

    dummy_img   = np.zeros((224, 224, 3), dtype=np.uint8)
    dummy_imgs  = [dummy_img] * num_views
    dummy_state = np.zeros(state_dim, dtype=np.float32)

    logger.info("Warming up FlashRT (%d iterations, %d views)...", n_iters, num_views)
    for i in range(n_iters):
        model.predict(images=dummy_imgs, prompt=task, state=dummy_state)
        if (i + 1) % 5 == 0:
            logger.info("  warmup %d/%d", i + 1, n_iters)
    torch.cuda.synchronize()
    logger.info("FlashRT warmup complete")


# ── Emergency robot disconnect ────────────────────────────────────────────────

def _emergency_disconnect(ctx) -> None:
    """Disconnect the robot when normal strategy teardown is not available."""
    try:
        robot = ctx.hardware.robot_wrapper.inner
        if robot.is_connected:
            logger.warning("Performing emergency robot disconnect after setup failure")
            robot.disconnect()
    except Exception as exc:
        logger.error("Emergency disconnect failed: %s", exc)
    teleop = ctx.hardware.teleop
    if teleop is not None:
        try:
            if teleop.is_connected:
                teleop.disconnect()
        except Exception as exc:
            logger.error("Emergency teleop disconnect failed: %s", exc)


# ── FlashRT backend installation ──────────────────────────────────────────────

def _install_flashrt_backend(ctx, cfg: RolloutConfig) -> None:
    """Load FlashRT and patch predict_action_chunk on the live policy.

    In sync mode PI05Policy.select_action() calls predict_action_chunk() once
    when its internal action queue is empty, fills the queue with n_action_steps
    actions, then pops one per tick.  Patching predict_action_chunk is sufficient
    — select_action is left unchanged.
    """
    import flash_rt

    policy     = ctx.policy.policy
    action_dim = policy.config.output_features["action"].shape[0]
    chunk_size = getattr(policy.config, "chunk_size", None)
    if chunk_size is None:
        raise RuntimeError(
            "policy.config.chunk_size not found — cannot determine FlashRT action_horizon."
        )
    device = cfg.device

    # Validate task before loading the (expensive) model.
    task = cfg.task or (cfg.dataset.single_task if cfg.dataset else "")
    if not task:
        raise ValueError(
            "Task prompt is empty. Pass --task='<description>' on the command line "
            "or set dataset.single_task in the config."
        )

    # Derive view keys from the checkpoint's input_features in training order.
    view_keys = [
        k for k, v in policy.config.input_features.items()
        if v.type == FeatureType.VISUAL
    ]
    if not view_keys:
        raise RuntimeError(
            "Policy has no VISUAL input features — cannot determine camera view keys."
        )

    # state_in_prompt_dim must match what PI05PrepareStateTokenizerProcessorStep produces.
    state_dim = action_dim

    logger.info(
        "Loading FlashRT model | ckpt=%s  action_dim=%d  chunk=%d  views=%s",
        cfg.policy.pretrained_path, action_dim, chunk_size, view_keys,
    )
    flash_model = flash_rt.load_model(
        checkpoint=cfg.policy.pretrained_path,
        framework="torch",
        num_views=len(view_keys),
        action_horizon=chunk_size,
        state_prompt_mode="fixed",
        autotune=3,
    )

    _warmup_flashrt(flash_model, task, state_dim, num_views=len(view_keys), n_iters=20)

    # Close over everything the patched method needs.
    _model     = flash_model
    _task      = task
    _act_dim   = action_dim
    _state_dim = state_dim
    _device    = device
    _views     = view_keys

    def predict_action_chunk(self, batch: dict, **kwargs) -> torch.Tensor:
        """FlashRT drop-in for PI05Policy.predict_action_chunk.

        Called by select_action() when the internal action queue is empty.
        kwargs are not used in sync mode (no inference_delay, no prev_chunk_left_over).

        Returns:
            Tensor shape (1, chunk_size, action_dim) float32, normalized.
        """
        # Images: (1, C, H, W) float32 [0,1] — lerobot preprocessor already resizes
        # to 224×224, but interpolate is kept as a safety net for raw-resolution inputs.
        imgs = []
        for key in _views:
            img = batch[key]
            if img.shape[-2:] != (224, 224):
                img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
            hwc = img.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0)
            imgs.append((hwc * 255).to(torch.uint8).cpu().numpy())

        # State: already normalized by lerobot's preprocessor → (state_dim,) numpy
        state_np = batch[OBS_STATE].squeeze(0).float().cpu().numpy()

        # FlashRT inference — returns (chunk_size, 32) normalized actions
        with torch.no_grad():
            chunk_np = _model.predict(
                images=imgs,
                prompt=_task,
                state=state_np[:_state_dim],
            )

        # Slice to action_dim, return (1, T, action_dim) on target device
        return (
            torch.from_numpy(chunk_np[:, :_act_dim])
            .float()
            .unsqueeze(0)
            .to(_device)
        )

    # Bind on the policy instance — shadows the class method so select_action
    # picks up the FlashRT implementation when it calls predict_action_chunk.
    policy.predict_action_chunk = types.MethodType(predict_action_chunk, policy)

    logger.info(
        "FlashRT sync backend installed | action_dim=%d  chunk=%d  n_action_steps=%d  "
        "state_dim=%d  views=%s  task=%r",
        action_dim, chunk_size, getattr(policy.config, "n_action_steps", chunk_size),
        state_dim, view_keys, task,
    )


# ── Main rollout ──────────────────────────────────────────────────────────────

@parser.wrap()
def rollout(cfg: RolloutConfig):
    """Entry point — mirrors lerobot-rollout with FlashRT sync inference."""
    init_logging()

    if cfg.display_data:
        logger.info(
            "Initializing %s visualization (ip=%s, port=%s)",
            cfg.display_mode, cfg.display_ip, cfg.display_port,
        )
        init_visualization(
            cfg.display_mode,
            session_name="rollout",
            ip=cfg.display_ip,
            port=cfg.display_port,
        )

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    logger.info("Building rollout context (sync inference)...")
    ctx = build_rollout_context(cfg, shutdown_event)

    # Everything after robot connect is wrapped so the robot is always
    # disconnected — even if FlashRT load or strategy creation fails.
    strategy = None
    try:
        _install_flashrt_backend(ctx, cfg)

        strategy = create_strategy(cfg.strategy)
        logger.info(
            "Strategy: %s | Robot: %s | FPS: %.0f | Duration: %s",
            cfg.strategy.type,
            cfg.robot.type if cfg.robot else "?",
            cfg.fps,
            f"{cfg.duration}s" if cfg.duration > 0 else "infinite",
        )

        strategy.setup(ctx)
        logger.info("Rollout started (FlashRT sync backend)")
        strategy.run(ctx)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        if strategy is not None:
            strategy.teardown(ctx)
        else:
            _emergency_disconnect(ctx)
        if cfg.display_data:
            shutdown_visualization(cfg.display_mode)

    logger.info("Rollout finished")


def main():
    register_third_party_plugins()
    rollout()


if __name__ == "__main__":
    main()
