"""factory.py — single entry point for building a FlashRT-backed PI05 policy.

Usage:
    from lerobot_flashrt import make_flashrt_policy

    policy, preprocessor, postprocessor = make_flashrt_policy(
        checkpoint_dir="/path/to/lerobot_checkpoint",
        server_endpoint="http://flashrt-host:8000",
    )

    # Rollout loop:
    obs_batch = preprocessor(raw_observation)
    action_chunk = policy.predict_action_chunk(obs_batch)   # (1, 30, 32) normalized
    robot_actions = postprocessor(action_chunk)              # unnorm + absolute
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def make_flashrt_policy(
    checkpoint_dir: str | Path,
    server_endpoint: str = "http://localhost:8000",
    client_timeout_s: float = 10.0,
    device: str = "cpu",
):
    """Build config, preprocessor, postprocessor, and FlashRTPI05Policy.

    Args:
        checkpoint_dir: Path to the LeRobot pretrained model directory
            containing ``config.json``, ``model.safetensors``, and the
            ``policy_*processor*.{json,safetensors}`` files.
        server_endpoint: Base URL of the running FlashRT LeRobot server.
        client_timeout_s: Per-request timeout for the FlashRT client.  Set
            high enough for the first call (graph capture can take 10–60 s).
        device: Torch device for preprocessor output (default "cpu"; the
            FlashRT server runs on its own GPU).

    Returns:
        (policy, preprocessor, postprocessor) tuple where:
            policy       — FlashRTPI05Policy instance (eval mode)
            preprocessor — PolicyProcessorPipeline: raw obs → normalized batch
            postprocessor — PolicyProcessorPipeline: normalized chunk → robot actions
    """
    from lerobot.policies.pi05 import PI05Config, make_pi05_pre_post_processors
    from lerobot.processor.pipeline import DataProcessorPipeline

    from .client import FlashRTClient
    from .policy import FlashRTPI05Policy

    checkpoint_dir = Path(checkpoint_dir)

    # ── Config ────────────────────────────────────────────────────────────────
    # Load via the pipeline's from_pretrained so that all dataclass fields
    # (normalization_mapping, use_relative_actions, etc.) are populated from
    # the saved config.json rather than defaults.
    import json
    with open(checkpoint_dir / "config.json") as f:
        raw = json.load(f)
    config = PI05Config(**{k: v for k, v in raw.items()
                           if k in PI05Config.__dataclass_fields__})
    config.device = device
    config.validate_features()

    # ── Preprocessor + postprocessor ─────────────────────────────────────────
    # Load stats from the checkpoint's saved processor safetensors.
    # DataProcessorPipeline.from_pretrained reads policy_preprocessor.json and
    # the referenced .safetensors file for normalizer stats automatically.
    try:
        preprocessor = DataProcessorPipeline.from_pretrained(
            checkpoint_dir,
            config_filename="policy_preprocessor.json",
        )
        postprocessor = DataProcessorPipeline.from_pretrained(
            checkpoint_dir,
            config_filename="policy_postprocessor.json",
        )
        logger.info("Loaded pre/post processors from %s", checkpoint_dir)
    except Exception as exc:
        logger.warning(
            "Could not load processors via from_pretrained (%s); "
            "falling back to make_pi05_pre_post_processors with loaded stats.",
            exc,
        )
        dataset_stats = _load_stats_from_checkpoint(checkpoint_dir)
        preprocessor, postprocessor = make_pi05_pre_post_processors(
            config, dataset_stats=dataset_stats
        )

    # ── Client + policy ───────────────────────────────────────────────────────
    client = FlashRTClient(server_endpoint, timeout_s=client_timeout_s)

    policy = FlashRTPI05Policy(config, client)
    policy.eval()
    policy.reset()

    logger.info(
        "FlashRTPI05Policy ready | endpoint=%s | chunk=%d | action_dim=%d",
        server_endpoint,
        config.chunk_size,
        config.max_action_dim,
    )

    return policy, preprocessor, postprocessor


def make_local_flashrt_policy(
    checkpoint_dir: str | Path,
    device: str = "cuda",
    action_horizon: int = 30,
    num_views: int = 3,
    state_prompt_mode: str = "fixed",
    autotune: int = 3,
):
    """Build a LocalFlashRTPI05Policy that runs FlashRT inference in-process.

    Use when LeRobot and FlashRT coexist in the same environment.  No HTTP
    server is required — the FlashRT model is loaded directly.

    Args:
        checkpoint_dir: Path to the LeRobot pretrained model directory.
        device: Torch device string for preprocessor output tensors.
        action_horizon: Action chunk length passed to flash_rt.load_model().
        num_views: Number of camera views.
        state_prompt_mode: ``"fixed"`` (one graph, recommended for rollouts)
            or ``"exact"`` (per-length graph, lower latency after warmup).
        autotune: CUDA Graph autotune trials (0 = off, 3 = default).

    Returns:
        (policy, preprocessor, postprocessor) tuple.
    """
    import json
    import flash_rt
    from lerobot.policies.pi05 import make_pi05_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    from .local_policy import LocalFlashRTPI05Policy

    checkpoint_dir = Path(checkpoint_dir)

    with open(checkpoint_dir / "config.json") as f:
        raw = json.load(f)

    config = PI05Config(**{k: v for k, v in raw.items()
                           if k in PI05Config.__dataclass_fields__})
    config.device = device
    config.validate_features()

    state_in_prompt_dim = int(
        raw.get("output_features", {})
           .get("action", {})
           .get("shape", [_MAX_ACTION_DIM])[0]
    )

    model = flash_rt.load_model(
        checkpoint=str(checkpoint_dir),
        framework="torch",
        num_views=num_views,
        action_horizon=action_horizon,
        state_prompt_mode=state_prompt_mode,
        autotune=autotune,
    )

    try:
        from lerobot.processor.pipeline import DataProcessorPipeline
        preprocessor = DataProcessorPipeline.from_pretrained(
            checkpoint_dir, config_filename="policy_preprocessor.json"
        )
        postprocessor = DataProcessorPipeline.from_pretrained(
            checkpoint_dir, config_filename="policy_postprocessor.json"
        )
    except Exception as exc:
        logger.warning(
            "Could not load processors via from_pretrained (%s); "
            "falling back to make_pi05_pre_post_processors.", exc
        )
        dataset_stats = _load_stats_from_checkpoint(checkpoint_dir)
        preprocessor, postprocessor = make_pi05_pre_post_processors(
            config, dataset_stats=dataset_stats
        )

    policy = LocalFlashRTPI05Policy(config, model, state_in_prompt_dim)
    policy.eval()
    policy.reset()

    logger.info(
        "LocalFlashRTPI05Policy ready | state_in_prompt_dim=%d | chunk=%d",
        state_in_prompt_dim,
        action_horizon,
    )
    return policy, preprocessor, postprocessor


_MAX_ACTION_DIM = 32


def _load_stats_from_checkpoint(checkpoint_dir: Path) -> dict:
    """Load dataset statistics from the saved normalizer safetensors.

    Returns a dict shaped for make_pi05_pre_post_processors:
        {"action": {"q01": tensor, "q99": tensor, ...},
         "observation.state": {...}}
    """
    import torch
    from safetensors.torch import load_file

    result: dict = {}
    for fname in checkpoint_dir.iterdir():
        if "normalizer_processor" not in fname.name:
            continue
        tensors = load_file(str(fname))
        for key, val in tensors.items():
            # key format: "<feature_name>.<stat>" e.g. "action.q01"
            if "." not in key:
                continue
            parts = key.rsplit(".", 1)
            if len(parts) != 2:
                continue
            feat, stat = parts
            if feat not in result:
                result[feat] = {}
            result[feat][stat] = val
    return result
