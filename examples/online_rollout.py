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
from lerobot.configs import parser
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

_VIEW_KEYS = [
    "observation.images.left_wrist",
    "observation.images.right_wrist",
    "observation.images.base",
]


# ── FlashRT backend installation ──────────────────────────────────────────────

def _install_flashrt_backend(ctx, cfg: RolloutConfig) -> None:
    """Load FlashRT and patch predict_action_chunk on the live policy.

    Everything else in the rollout context — the robot, preprocessors,
    postprocessors, ActionQueue, RTCInferenceEngine, and strategy — is
    left completely unchanged.  Only the model forward pass is swapped.
    """
    import flash_rt

    policy     = ctx.policy.policy
    action_dim = policy.config.output_features["action"].shape[0]   # 16
    chunk_size = getattr(policy.config, "chunk_size", 30)            # 30
    device     = cfg.device

    # state_in_prompt_dim: number of state dims the checkpoint was trained with
    # (must match what PI05PrepareStateTokenizerProcessorStep produces)
    state_dim  = action_dim

    task = cfg.task or (cfg.dataset.single_task if cfg.dataset else "")

    logger.info(
        "Loading FlashRT model | ckpt=%s  action_dim=%d  chunk=%d",
        cfg.policy.pretrained_path, action_dim, chunk_size,
    )
    flash_model = flash_rt.load_model(
        checkpoint=cfg.policy.pretrained_path,
        framework="torch",
        num_views=len(_VIEW_KEYS),
        action_horizon=chunk_size,
        state_prompt_mode="fixed",
        autotune=3,
    )

    # Close over everything the patched method needs
    _model     = flash_model
    _task      = task
    _act_dim   = action_dim
    _state_dim = state_dim
    _device    = device
    _views     = list(_VIEW_KEYS)

    def predict_action_chunk(self, batch: dict, **kwargs) -> torch.Tensor:
        """FlashRT drop-in for PI05Policy.predict_action_chunk.

        Accepts the same preprocessed batch and kwargs (inference_delay,
        prev_chunk_left_over) that RTCInferenceEngine passes; kwargs are
        accepted but not forwarded — FlashRT handles its own RTC-equivalent
        scheduling internally.

        Returns:
            Tensor shape (1, chunk_size, action_dim) float32, normalized.
        """
        # Images: (1, C, H, W) float32 [0,1] → 224×224 uint8 HWC numpy
        imgs = []
        for key in _views:
            img = batch[key]
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
    # both ctx.policy.policy and ctx.policy.inference._policy see the new impl
    policy.predict_action_chunk = types.MethodType(predict_action_chunk, policy)

    logger.info(
        "FlashRT backend installed | action_dim=%d  chunk=%d  state_dim=%d  task=%r",
        action_dim, chunk_size, state_dim, task,
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

    # Swap in FlashRT as the policy inference backend
    _install_flashrt_backend(ctx, cfg)

    strategy = create_strategy(cfg.strategy)
    logger.info(
        "Strategy: %s | Robot: %s | FPS: %.0f | Duration: %s",
        cfg.strategy.type,
        cfg.robot.type if cfg.robot else "?",
        cfg.fps,
        f"{cfg.duration}s" if cfg.duration > 0 else "infinite",
    )

    try:
        strategy.setup(ctx)
        logger.info("Rollout started (FlashRT backend)")
        strategy.run(ctx)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        strategy.teardown(ctx)
        if cfg.display_data:
            shutdown_visualization(cfg.display_mode)

    logger.info("Rollout finished")


def main():
    register_third_party_plugins()
    rollout()


if __name__ == "__main__":
    main()
