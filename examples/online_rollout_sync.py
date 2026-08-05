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

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Camera / robot / teleop imports that draccus needs for config resolution ──
from lerobot.cameras.opencv import OpenCVCameraConfig       # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig              # noqa: F401
from lerobot.configs import FeatureType, parser
from lerobot.policies.common.vla_utils import resize_with_pad_torch
from lerobot.policies.utils import prepare_observation_for_inference
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
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_visualization, shutdown_visualization

logger = logging.getLogger(__name__)

# One-shot flag for the state-dimension mismatch warning.
_state_dim_warned = False


# ── Preprocessed batch → FlashRT inputs ───────────────────────────────────────

def _extract_flashrt_inputs(batch: dict, view_keys: list[str]) -> tuple[list, np.ndarray]:
    """Convert a lerobot-preprocessed batch into FlashRT's predict() inputs.

    Images: (1, C, H, W) float32 in [0, 1] → list of (224, 224, 3) uint8.

    The lerobot preprocessor does NOT resize — PI05 resizes inside
    ``PI05Policy._preprocess_images`` with ``resize_with_pad_torch``
    (aspect-preserving + centered black padding), which FlashRT bypasses.
    We therefore apply the *same* resize here.  A plain bilinear stretch
    would distort every frame relative to training (e.g. a 1280×720 wrist
    camera must become 224×126 with 49 px black bars, not a squashed square).

    State: already normalized to [-1, 1] by NormalizerProcessorStep, which is
    what FlashRT's state-in-prompt discretizer expects.

    This helper is shared by warmup/calibration and the live control path so
    FP8 activation scales are calibrated on exactly the tensors inference sees.
    """
    imgs = []
    for key in view_keys:
        img = resize_with_pad_torch(batch[key], 224, 224)        # (1, C, 224, 224)
        hwc = img.squeeze(0).permute(1, 2, 0)                    # (224, 224, C)
        # round(), not truncate: values arrive as uint8/255, so x*255 lands at
        # e.g. 199.99997 and a plain cast would bias every pixel down by 1 LSB.
        imgs.append(hwc.mul(255).round().clamp(0, 255).to(torch.uint8).cpu().numpy())

    state_np = batch[OBS_STATE].squeeze(0).float().cpu().numpy()
    return imgs, state_np


# ── FlashRT warmup / calibration ──────────────────────────────────────────────

def _capture_warmup_inputs(ctx, cfg: RolloutConfig, task: str, view_keys: list[str]):
    """Grab one real robot observation and push it through lerobot's preprocessor.

    FlashRT freezes its FP8 activation scales on the first predict() call
    (``VLAModel.predict`` → ``Pi05TorchFrontendRtx._calibrate_single_frame``),
    and the RTX path does not persist that calibration to disk — it is redone
    on every process start.  Calibrating on synthetic black frames would leave
    every GEMM scaled for a degenerate input, so the warmup must run on a real
    frame at the robot's actual operating point.
    """
    robot = ctx.hardware.robot_wrapper
    obs = robot.get_observation()
    obs_frame = build_dataset_frame(ctx.data.hw_features, obs, prefix="observation")
    obs_batch = prepare_observation_for_inference(
        obs_frame, cfg.device, task, robot.robot_type
    )
    obs_batch["task"] = [task]
    preprocessed = ctx.policy.preprocessor(obs_batch)
    return _extract_flashrt_inputs(preprocessed, view_keys)


def _warmup_flashrt(model, task: str, imgs: list, state, n_iters: int = 20) -> None:
    """Run n_iters predict() calls on a real observation.

    The first call performs FP8 activation calibration and static CUDA graph
    capture (1–60 s); the rest exercise the replay path.  Doing this before the
    robot loop means that cost is paid upfront rather than on the first live
    control tick.
    """
    logger.info("Warming up FlashRT (%d iterations, %d views)...", n_iters, len(imgs))
    for i in range(n_iters):
        model.predict(images=imgs, prompt=task, state=state)
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

    # state_in_prompt_dim must match what Pi05PrepareStateTokenizerProcessorStep
    # produces.  Read it from the checkpoint rather than assuming
    # state_dim == action_dim — robots with velocity channels (e.g. LeKiwi) have
    # a wider state than action.
    state_feature = policy.config.input_features.get(OBS_STATE)
    if state_feature is None:
        raise RuntimeError(
            f"Policy has no '{OBS_STATE}' input feature — cannot determine state dim."
        )
    state_dim = state_feature.shape[0]

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

    # Warmup on a real observation: the first predict() call freezes FlashRT's
    # FP8 activation scales and captures the CUDA graph, so it must see a frame
    # from the robot's actual operating point (see _capture_warmup_inputs).
    warm_imgs, warm_state = _capture_warmup_inputs(ctx, cfg, task, view_keys)
    _warmup_flashrt(flash_model, task, warm_imgs, warm_state, n_iters=20)

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
        global _state_dim_warned

        # Resize + uint8 exactly as the warmup/calibration frame was prepared.
        imgs, state_np = _extract_flashrt_inputs(batch, _views)

        # The full state goes into the prompt — truncating it would produce a
        # token sequence the checkpoint was never trained on.
        if not _state_dim_warned and state_np.shape[0] != _state_dim:
            logger.warning(
                "Runtime state dim (%d) differs from the checkpoint's declared %s dim (%d); "
                "the state-in-prompt tokens will not match training.",
                state_np.shape[0], OBS_STATE, _state_dim,
            )
            _state_dim_warned = True

        # FlashRT inference — returns (chunk_size, 32) normalized actions
        with torch.no_grad():
            chunk_np = _model.predict(
                images=imgs,
                prompt=_task,
                state=state_np,
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
