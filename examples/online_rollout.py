#!/usr/bin/env python3
"""online_rollout.py — lerobot-rollout with FlashRT inference backend.

Identical to ``lerobot-rollout`` in every respect — robot connection,
preprocessors, postprocessors, ActionQueue, RTCInferenceEngine, and
rollout strategy — except that ``predict_action_chunk`` is patched on
the live policy object to call ``flash_rt.model.predict()`` instead of
the PI05 VLA forward pass.

Usage:
    python examples/online_rollout.py \\
        --config_path=/openarm/rollout.yaml \\
        --strategy.type=base \\
        --policy.path=/openarm/outputs/train/openarm_folding_high_quality_60k/checkpoints/060000/pretrained_model \\
        --task="Fold the T-shirt properly" \\
        --interpolation_multiplier=3 \\
        --inference.type=rtc \\
        --inference.rtc.execution_horizon=12 \\
        --inference.rtc.max_guidance_weight=10.0 \\
        --inference.rtc.prefix_attention_schedule=EXP \\
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

# One-shot flag so we only emit the RTC-prefix warning once per process.
_rtc_prefix_warned = False


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
    """Disconnect the robot when normal strategy teardown is not available.

    Called when _install_flashrt_backend or create_strategy raise before
    the strategy object is bound, leaving the robot connected with no
    teardown path through strategy.teardown().
    """
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

    Everything else in the rollout context — the robot, preprocessors,
    postprocessors, ActionQueue, RTCInferenceEngine, and strategy — is
    left completely unchanged.  Only the model forward pass is swapped.
    """
    import flash_rt

    policy     = ctx.policy.policy
    action_dim = policy.config.output_features["action"].shape[0]   # e.g. 16
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
    # This ensures the image list matches what FlashRT was calibrated with and
    # avoids silent mismatches when running a different checkpoint or robot.
    view_keys = [
        k for k, v in policy.config.input_features.items()
        if v.type == FeatureType.VISUAL
    ]
    if not view_keys:
        raise RuntimeError(
            "Policy has no VISUAL input features — cannot determine camera view keys."
        )

    # state_in_prompt_dim: number of state dims the checkpoint was trained with
    # (must match what PI05PrepareStateTokenizerProcessorStep produces)
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

    # Warmup: CUDA graph capture happens on the first few calls; run dummy
    # predictions before the robot loop so the first live tick is fast.
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

        Accepts the same preprocessed batch and kwargs (inference_delay,
        prev_chunk_left_over) that RTCInferenceEngine passes.

        NOTE: prev_chunk_left_over and inference_delay are not forwarded —
        flash_rt.predict() has no API for them.  This means RTC prefix
        conditioning (prefix_attention_schedule) is inactive and each chunk
        is generated independently.  Chunk transitions may be slightly less
        smooth than native PI05 inference.

        Returns:
            Tensor shape (1, chunk_size, action_dim) float32, normalized.
        """
        global _rtc_prefix_warned
        if not _rtc_prefix_warned and kwargs.get("prev_chunk_left_over") is not None:
            logger.warning(
                "FlashRT backend: prev_chunk_left_over is not forwarded to flash_rt.predict() "
                "(no API for it). RTC prefix conditioning (--prefix_attention_schedule) is "
                "inactive — chunk transitions may be slightly less smooth than native PI05."
            )
            _rtc_prefix_warned = True

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

    # Bind and install on the policy instance — shadows the class method so
    # both ctx.policy.policy and ctx.policy.inference._policy see the new impl.
    policy.predict_action_chunk = types.MethodType(predict_action_chunk, policy)

    logger.info(
        "FlashRT backend installed | action_dim=%d  chunk=%d  state_dim=%d  views=%s  task=%r",
        action_dim, chunk_size, state_dim, view_keys, task,
    )


# ── Main rollout (mirrors lerobot_rollout.py exactly) ────────────────────────

@parser.wrap()
def rollout(cfg: RolloutConfig):
    """Entry point — mirrors lerobot-rollout with FlashRT inference."""
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

    # Build the full lerobot rollout context:
    # robot connection, preprocessors, postprocessors, ActionQueue, RTCInferenceEngine
    logger.info("Building rollout context...")
    ctx = build_rollout_context(cfg, shutdown_event)

    # Everything after robot connect is wrapped so the robot is always
    # disconnected — even if FlashRT load or strategy creation fails.
    strategy = None
    try:
        # Swap in FlashRT as the policy inference backend.
        # Must happen before strategy.setup() starts the RTC thread.
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
        logger.info("Rollout started (FlashRT backend)")
        strategy.run(ctx)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        if strategy is not None:
            strategy.teardown(ctx)
        else:
            # FlashRT load or strategy creation failed; robot is still
            # connected and must be disconnected manually.
            _emergency_disconnect(ctx)
        if cfg.display_data:
            shutdown_visualization(cfg.display_mode)

    logger.info("Rollout finished")


def main():
    register_third_party_plugins()
    rollout()


if __name__ == "__main__":
    main()
