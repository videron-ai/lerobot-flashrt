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

# NOTE: no ``from __future__ import annotations`` here. ``lerobot``'s
# ``parser.wrap`` reads the raw ``__annotations__`` of ``rollout`` to find the
# config class; under PEP 563 it would get the string ``"RolloutConfig"`` and
# draccus would fail with "must be called with a dataclass type or instance".

import logging
import os
import sys
import time
import types
from contextlib import contextmanager, nullcontext
from functools import wraps

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


# ── Host-side observation prep ────────────────────────────────────────────────

def _fast_prepare_observation_for_inference(observation, device, task=None, robot_type=None):
    """Drop-in for ``lerobot.policies.utils.prepare_observation_for_inference``.

    Uploads camera frames as uint8 and does the ``/255`` and the HWC→CHW
    transpose on the GPU.  The stock version expands to float32 on the CPU
    *before* the transfer, so it moves 4x the bytes over PCIe and runs a
    cache-hostile transpose of ~25 MiB on the host.

    Output matches the stock function to within **1 ULP** of float32 (max abs
    diff 6e-8) — the CPU and GPU float32 dividers round the ``/255`` slightly
    differently; both land within 1 ULP of the exact value.  For scale, these
    images are then quantized to uint8 for FlashRT, where 1 LSB is 3.9e-3, so
    the difference is ~65,000x below the quantization step.

    Measured on GB10 with two 1280x720 + one 640x480 camera:

        stock  15.4 / 18.5 / 19.7 ms  (min / median / p90), and highly
               sensitive to the torch CPU thread pool — 5 ms to 21 ms
               depending on ``torch.set_num_threads``
        this    0.37 / 0.39 / 0.40 ms

    The jitter matters as much as the mean: this runs inside the RTC inference
    thread, so it feeds straight into ``inference_delay``, which is what
    ``ActionQueue.merge`` uses to decide how many actions to discard.

    Pinned staging was tried and is *slower* (~1.2 ms) — the extra host memcpy
    costs more than the 6 MiB transfer saves.
    """
    for name in observation:
        tensor = torch.as_tensor(observation[name])
        if "image" in name:
            # Upload first (6 MiB uint8, not 25 MiB float32), convert on device.
            tensor = tensor.to(device)
            if tensor.dtype == torch.uint8:
                tensor = tensor.float().div_(255)
            tensor = tensor.permute(2, 0, 1).contiguous().unsqueeze(0)
        else:
            tensor = tensor.unsqueeze(0).to(device)
        observation[name] = tensor

    observation["task"] = task if task else ""
    observation["robot_type"] = robot_type if robot_type else ""
    return observation


def _install_fast_observation_prep() -> None:
    """Patch the inference engines' bound reference to the observation prep.

    Both engines do ``from lerobot.policies.utils import
    prepare_observation_for_inference`` at import time, so the name has to be
    rebound in each consuming module — patching only ``policies.utils`` would
    have no effect.

    Set ``FLASHRT_FAST_OBS_PREP=0`` to keep lerobot's stock implementation.
    """
    if os.environ.get("FLASHRT_FAST_OBS_PREP", "1") == "0":
        logger.info("Fast observation prep disabled via FLASHRT_FAST_OBS_PREP=0")
        return

    import lerobot.policies.utils as _utils
    from lerobot.rollout.inference import rtc as _rtc
    from lerobot.rollout.inference import sync as _sync

    patched = []
    for mod in (_utils, _rtc, _sync):
        if getattr(mod, "prepare_observation_for_inference", None) is not None:
            mod.prepare_observation_for_inference = _fast_prepare_observation_for_inference
            patched.append(mod.__name__)
    logger.info("Installed GPU-side observation prep in: %s", ", ".join(patched))


def _read_state_dict_without_mmap(filename, device="cpu"):
    """Drop-in for ``safetensors.torch.load_file`` that reads instead of mmaps.

    ``load_file`` mmaps the checkpoint and returns in ~10 ms, but that only
    defers the work: every subsequent byte touched is a minor page fault.  The
    9.35 GB PI05 checkpoint is ~2.3 M pages, and it is read exactly once, so
    the faults buy nothing and cost ~50 s — first in ``load_state_dict``, then
    again in ``policy.to(cuda)``.  A plain sequential ``read()`` of the same
    file off the same NVMe runs at 13.4 GB/s.

    Only the ``device="cpu"`` case is handled here; anything else falls back to
    the stock loader, which knows how to place tensors directly.
    """
    if device not in ("cpu", None):
        from safetensors.torch import load_file
        return load_file(filename, device=device)

    from safetensors.torch import load as _load_bytes
    with open(filename, "rb") as f:
        return _load_bytes(f.read())


def _install_fast_policy_load() -> None:
    """Cut PI05 policy load from ~150 s to ~4 s.  Three independent stalls:

    1. **Random init.**  ``from_pretrained`` builds the module with
       ``cls(config)`` and *then* overwrites every parameter from the
       checkpoint.  That constructor runs PaliGemma's full random init over
       4.1 B params on the CPU — 98 s on GB10 — and every value it produces is
       discarded a second later.  ``no_init_weights`` no-ops torch's init
       functions for the duration.  Tensors are still really allocated, so
       buffers, ``.to()``, ``.device()`` and ``load_state_dict`` behave exactly
       as before.  Measured: 98 s -> 2.9 s.

    2. **mmap page faults.**  See :func:`_read_state_dict_without_mmap`.
       Measured: ``load_state_dict`` 55.0 s -> 0.01 s, ``.to(cuda)``
       49.8 s -> 0.18 s.

    3. **The state-dict copy.**  With the tensors already resident on the heap,
       ``assign=True`` rebinds them into the module instead of copying 9.35 GB
       into the freshly allocated parameters.  The allocated-but-unwritten
       parameters from step 1 are then dropped without ever being touched.

    Only the *timing* changes; the checkpoint's weights land in the model
    exactly as before (verified: ``strict=True``, 0 missing, 0 unexpected).
    The one behavioural difference is that a parameter *absent* from the
    checkpoint would hold uninitialized memory rather than a random draw —
    ``from_pretrained`` prints any such missing keys, and with FlashRT owning
    the forward pass none of these tensors are read at all.

    Set ``FLASHRT_FAST_POLICY_LOAD=0`` to keep lerobot's stock load path.
    """
    if os.environ.get("FLASHRT_FAST_POLICY_LOAD", "1") == "0":
        logger.info("Fast policy load disabled via FLASHRT_FAST_POLICY_LOAD=0")
        return

    try:
        # transformers >= 5 moved this out of modeling_utils.
        from transformers.initialization import no_init_weights
    except ImportError:
        try:
            from transformers.modeling_utils import no_init_weights
        except ImportError:
            from contextlib import nullcontext as no_init_weights
            logger.warning(
                "no_init_weights not found in transformers — policy construction "
                "will run the full random init."
            )

    import safetensors.torch as _st

    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    if getattr(PI05Policy.from_pretrained.__func__, "_flashrt_fast_load", False):
        return

    original = PI05Policy.from_pretrained.__func__

    @wraps(original)
    def fast_from_pretrained(cls, *args, **kwargs):
        # ``from_pretrained`` imports load_file and calls load_state_dict at
        # call time, so both patches have to be live across the call and are
        # restored immediately after — they are not safe to leave installed.
        stock_load_file = _st.load_file
        stock_load_state_dict = PI05Policy.load_state_dict

        def assigning_load_state_dict(self, state_dict, strict=True, assign=False):
            return stock_load_state_dict(self, state_dict, strict=strict, assign=True)

        _st.load_file = _read_state_dict_without_mmap
        PI05Policy.load_state_dict = assigning_load_state_dict
        t0 = time.perf_counter()
        try:
            with no_init_weights():
                model = original(cls, *args, **kwargs)
        finally:
            _st.load_file = stock_load_file
            PI05Policy.load_state_dict = stock_load_state_dict

        logger.info("Policy loaded in %.1f s (fast path)", time.perf_counter() - t0)
        return model

    fast_from_pretrained._flashrt_fast_load = True
    PI05Policy.from_pretrained = classmethod(fast_from_pretrained)
    logger.info("Installed fast policy load (no random init, no mmap, assign)")


class _FlashRTModelStub(torch.nn.Module):
    """Parameter-free stand-in for ``PI05Pytorch`` when FlashRT owns the forward.

    Accepts the same constructor arguments so ``PI05Policy.__init__`` runs
    unchanged, and exposes the attributes the policy touches outside the
    forward pass (``rtc_processor``, ``gradient_checkpointing_enable``).  Any
    attempt to actually run the PyTorch model fails loudly instead of silently
    producing garbage from unloaded weights.
    """

    def __init__(self, config, rtc_processor=None):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor

    def gradient_checkpointing_enable(self):
        pass

    def gradient_checkpointing_disable(self):
        pass

    def _unavailable(self, *args, **kwargs):
        raise RuntimeError(
            "The PyTorch PI05 model was not loaded (FlashRT weightless policy load). "
            "predict_action_chunk must be served by FlashRT; set "
            "FLASHRT_WEIGHTLESS_POLICY=0 to load the full PyTorch model."
        )

    forward = sample_actions = denoise_step = embed_prefix = embed_suffix = _unavailable


def _install_weightless_policy_load() -> bool:
    """Build ``PI05Policy`` without the PyTorch model it would never run.

    FlashRT loads its own copy of the weights from the checkpoint and serves
    ``predict_action_chunk``, so the ~9 GB ``PI05Pytorch`` submodule is dead
    weight — and with ``--policy.device=cuda`` it lands in GPU memory next to
    FlashRT's copy.  This swaps it for :class:`_FlashRTModelStub` during
    construction and skips the state-dict load entirely, so the policy object
    keeps everything the rollout stack actually uses (config, ``reset()``,
    RTC wiring, the ``predict_action_chunk`` hook) while the preprocessor,
    tokenizer outputs, normalizer and postprocessor all run on the GPU.

    Returns True if installed.  Set ``FLASHRT_WEIGHTLESS_POLICY=0`` to load the
    full PyTorch model instead (falls back to :func:`_install_fast_policy_load`).
    PEFT checkpoints always take the full load, since the adapter has to wrap
    real modules.
    """
    if os.environ.get("FLASHRT_WEIGHTLESS_POLICY", "1") == "0":
        logger.info("Weightless policy load disabled via FLASHRT_WEIGHTLESS_POLICY=0")
        return False

    from lerobot.policies.pi05 import modeling_pi05
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    if getattr(PI05Policy.from_pretrained.__func__, "_flashrt_weightless", False):
        return True

    original = PI05Policy.from_pretrained.__func__

    @wraps(original)
    def weightless_from_pretrained(cls, pretrained_name_or_path, *args, config=None, **kwargs):
        if config is None or getattr(config, "use_peft", False):
            logger.warning("Weightless policy load needs a non-PEFT config; using the full load")
            return original(cls, pretrained_name_or_path, *args, config=config, **kwargs)

        stock_model_cls = modeling_pi05.PI05Pytorch
        modeling_pi05.PI05Pytorch = _FlashRTModelStub
        try:
            policy = cls(config)
        finally:
            modeling_pi05.PI05Pytorch = stock_model_cls
        logger.info(
            "PI05Policy built without PyTorch weights (%d parameters); FlashRT serves inference",
            sum(p.numel() for p in policy.parameters()),
        )
        return policy

    weightless_from_pretrained._flashrt_weightless = True
    PI05Policy.from_pretrained = classmethod(weightless_from_pretrained)
    logger.info("Installed weightless policy load (PyTorch PI05 model not materialized)")
    return True


def _is_viz_image(value) -> bool:
    """True for values ``log_rerun_data`` would treat as an image.

    Mirrors its check: a 3-D ndarray, CHW or HWC.  Everything else there is a
    scalar or a 1-D array and costs almost nothing to log.
    """
    return isinstance(value, np.ndarray) and value.ndim == 3


def _install_throttled_telemetry() -> None:
    """Log camera frames at the policy rate instead of the control rate.

    ``RolloutStrategy._log_telemetry`` runs once per control tick, which with
    ``interpolation_multiplier=3`` is ``fps * 3`` = 90 Hz.  But
    ``_process_observation_and_notify`` is throttled to policy ticks and
    returns the *cached* ``obs_processed`` in between, so two out of every
    three calls re-log byte-identical frames.

    That is not free.  With two 1280x720 cameras plus one 640x480 it is ~6.4 MB
    of image data per call — ~575 MB/s of transpose-and-serialize on the
    control thread at 90 Hz, and worse with ``display_compressed_images=True``,
    which JPEG-encodes synchronously in the same thread.  The budget per
    iteration is ``1 / (fps * multiplier)`` = 11.1 ms; overruns show up as
    lerobot's "Record loop is running slower" warning (which misreports the
    target as ``cfg.fps``, ignoring the multiplier — 62.8 Hz measured against
    an apparent 30 Hz target is really 62.8 Hz against 90 Hz).

    So: images go out at most every ``1/fps`` seconds, while scalars and the
    action still go every tick — those are cheap, and the interpolated action
    genuinely changes between policy ticks.  Since the dropped frames are the
    cached duplicates, no information is lost.

    The throttle is time-based rather than a tick counter because the
    interpolator's ``needs_new_action()`` has already been consumed by
    ``_process_observation_and_notify`` by the time this runs.

    ``FLASHRT_VIZ_IMAGE_HZ`` overrides the image rate; ``0`` disables the
    throttle and restores stock per-tick logging.
    """
    from lerobot.rollout.strategies.core import RolloutStrategy
    from lerobot.utils.visualization_utils import log_visualization_data

    if getattr(RolloutStrategy._log_telemetry, "_flashrt_throttled", False):
        return

    last_image_log = [0.0]

    def throttled_log_telemetry(obs_processed, action_dict, runtime_ctx) -> None:
        cfg = runtime_ctx.cfg
        if not cfg.display_data:
            return

        override = os.environ.get("FLASHRT_VIZ_IMAGE_HZ")
        image_hz = float(override) if override is not None else float(cfg.fps or 0.0)

        observation = obs_processed
        if image_hz > 0 and observation is not None:
            now = time.perf_counter()
            if now - last_image_log[0] >= 1.0 / image_hz:
                last_image_log[0] = now
            else:
                observation = {
                    k: v for k, v in observation.items() if not _is_viz_image(v)
                }

        log_visualization_data(
            cfg.display_mode,
            observation=observation,
            action=action_dict,
            compress_images=cfg.display_compressed_images,
        )

    throttled_log_telemetry._flashrt_throttled = True
    RolloutStrategy._log_telemetry = staticmethod(throttled_log_telemetry)
    logger.info("Installed throttled telemetry (images at policy rate)")


def _install_episode_engine_pause() -> None:
    """Pause the inference engine the moment a recording episode ends.

    ``EpisodicStrategy`` calls ``self._engine.resume()`` at the start of every
    episode (``episodic.py``) and never calls ``pause()`` anywhere — only
    ``dagger.py`` does.  So from the first episode onward the RTC thread keeps
    inferring continuously through the reset phase, ``save_episode()``, video
    encoding and the hub upload, stopping only at ``engine.stop()`` in
    teardown.

    Measured on a 10 s episode: the thread ran for ~40 s, ~19 s of it
    overlapping SVT-AV1 encoding and a 58 MB upload — GPU and CPU contention
    at exactly the point the dataset is being written, and it hands
    ``stop()`` a thread that is mid-inference to join.

    ``pause()`` only clears the engine's ``_policy_active`` event, dropping the
    thread into its idle sleep, and the ``resume()`` already present at the
    next episode start undoes it.  Nothing to unwind, so wrapping
    ``_policy_loop`` is enough.

    Set ``FLASHRT_EPISODE_PAUSE=0`` to keep lerobot's stock behaviour.
    """
    if os.environ.get("FLASHRT_EPISODE_PAUSE", "1") == "0":
        logger.info("Episode-end engine pause disabled via FLASHRT_EPISODE_PAUSE=0")
        return

    from lerobot.rollout.strategies.episodic import EpisodicStrategy

    original = EpisodicStrategy._policy_loop
    if getattr(original, "_flashrt_episode_pause", False):
        return

    @wraps(original)
    def policy_loop_then_pause(self, *args, **kwargs):
        try:
            return original(self, *args, **kwargs)
        finally:
            # In `finally`, not after the call: an episode aborted by an
            # exception or by the shutdown event must not leave the thread
            # inferring into teardown either.
            engine = getattr(self, "_engine", None)
            if engine is not None:
                engine.pause()

    policy_loop_then_pause._flashrt_episode_pause = True
    EpisodicStrategy._policy_loop = policy_loop_then_pause
    logger.info("Installed episode-end inference pause (EpisodicStrategy)")


def _install_record_decimation(ctx, cfg: RolloutConfig) -> None:
    """Record one dataset frame per camera frame, not per control tick.

    ``--interpolation_multiplier=N`` raises the control loop to ``fps * N``
    (``ActionInterpolator.get_control_interval`` returns ``1/(fps*N)``) so the
    arm gets smoothed commands, while the policy is still consumed at ``fps``
    — the interpolator buffers N blends and only pulls a new action when that
    buffer empties.  That part is correct.  The recording is not: the strategy
    writes a dataset frame on *every* control tick, but the dataset is created
    with ``cfg.dataset.fps`` (``rollout/context.py``), and nothing reconciles
    the two.

    With ``fps=30, N=3`` against 30 fps cameras that gives a 10 s episode 881
    frames declared at 30 fps — 29 s of playback, where 2 of every 3 frames are
    the same camera capture, and the image writer does 3x the work *during* the
    episode.

    Forwarding only every Nth ``add_frame`` keeps the dataset at the camera
    rate while control stays at ``fps * N``.  Decimation is uniform rather than
    aligned to the interpolator's policy-action boundary, because regular
    spacing is what a fixed ``dataset.fps`` actually claims; each kept frame
    still pairs the observation of that tick with the action sent on it.

    Set ``FLASHRT_RECORD_DECIMATE=0`` to record every tick.
    """
    n = int(getattr(cfg, "interpolation_multiplier", 1) or 1)
    if n <= 1:
        return
    if os.environ.get("FLASHRT_RECORD_DECIMATE", "1") == "0":
        logger.info("Record decimation disabled via FLASHRT_RECORD_DECIMATE=0")
        return

    dataset = getattr(getattr(ctx, "data", None), "dataset", None)
    if dataset is None:
        return          # strategy records nothing (e.g. base)

    original = dataset.add_frame
    if getattr(original, "_flashrt_decimated", False):
        return

    state = {"tick": 0}

    @wraps(original)
    def decimated_add_frame(*args, **kwargs):
        tick = state["tick"]
        state["tick"] = tick + 1
        if tick % n:
            return None
        return original(*args, **kwargs)

    decimated_add_frame._flashrt_decimated = True
    dataset.add_frame = decimated_add_frame
    logger.info(
        "Recording 1 frame per %d control ticks (interpolation_multiplier=%d): "
        "control %.0f Hz, dataset %.0f Hz",
        n, n, cfg.fps * n, cfg.fps,
    )


def _configure_torch_threads() -> None:
    """Bound the torch CPU thread pool for the control loop.

    Elementwise CPU work in the observation path is memory-bandwidth-bound, and
    an unbounded pool both thrashes and competes with the RTC inference thread —
    measured swings of 5 ms to 21 ms on the same workload purely from thread
    count.  Override with ``FLASHRT_TORCH_THREADS``; ``0`` leaves torch alone.
    """
    requested = os.environ.get("FLASHRT_TORCH_THREADS")
    if requested is None:
        return
    n = int(requested)
    if n <= 0:
        logger.info("Leaving torch thread pool at default (%d)", torch.get_num_threads())
        return
    torch.set_num_threads(n)
    logger.info("torch CPU threads set to %d", torch.get_num_threads())


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


def _calibration_cache_path(cfg, n_frames: int) -> "os.PathLike | str":
    """Where the prepared calibration frames live.

    Defaults next to the checkpoint, keyed by frame count, because the cached
    states are normalized by *that* checkpoint's normalizer and the cached
    images are resized for its view set — a cache from another checkpoint would
    be silently wrong.  Override with ``FLASHRT_CALIB_CACHE``.
    """
    override = os.environ.get("FLASHRT_CALIB_CACHE")
    if override:
        return override
    return os.path.join(
        str(cfg.policy.pretrained_path), f"flashrt_calib_{n_frames}.npz"
    )


def _build_calibration_frames(cfg, ctx, task: str, view_keys: list[str],
                              n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Prepare N calibration frames spread across a recorded dataset episode.

    Returns ``(images, states)`` where images is ``(N, V, 224, 224, 3)`` uint8
    and states is ``(N, S)`` float32 — exactly the tensors ``predict()`` is fed,
    so the frozen scales match live inference rather than an approximation.

    Runs the dataset frames through the same lerobot preprocessor and the same
    :func:`_extract_flashrt_inputs` the control loop uses.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo_id = os.environ.get("FLASHRT_CALIB_DATASET")
    if not repo_id:
        raise ValueError(
            "FLASHRT_CALIB_FRAMES > 1 requires FLASHRT_CALIB_DATASET=<repo_id> "
            "on the first run (the prepared frames are cached afterwards). "
            "The robot cannot tour its own operating range before it has moved, "
            "so the calibration frames come from a recorded episode."
        )
    episode = int(os.environ.get("FLASHRT_CALIB_EPISODE", "0"))

    logger.info("Building calibration frames from %s episode %d...", repo_id, episode)
    ds = LeRobotDataset(repo_id)
    indices = [i for i in range(len(ds)) if int(ds[i]["episode_index"]) == episode]
    if not indices:
        raise ValueError(f"No frames for episode {episode} in {repo_id}")

    picks = sorted({int(i) for i in
                    np.linspace(0, len(indices) - 1, num=n_frames, dtype=int)})
    imgs_out, states_out = [], []
    for p in picks:
        frame = ds[indices[p]]
        obs = {k: frame[k].unsqueeze(0).to(cfg.device) for k in view_keys}
        obs[OBS_STATE] = frame[OBS_STATE].unsqueeze(0).to(cfg.device)
        obs["task"] = [task]
        imgs, state_np = _extract_flashrt_inputs(ctx.policy.preprocessor(obs), view_keys)
        imgs_out.append(np.stack(imgs))
        states_out.append(state_np)

    logger.info("Prepared %d calibration frames (episode indices %s)",
                len(picks), picks[:6])
    return np.stack(imgs_out), np.stack(states_out).astype(np.float32)


def _load_or_build_calibration(cfg, ctx, task: str, view_keys: list[str],
                               n_frames: int) -> list[dict]:
    """Prepared calibration observations, from cache when possible.

    Preparing them means decoding a dataset episode and running the lerobot
    preprocessor, which is far too slow to repeat on every robot start.  The
    cache stores the finished uint8 images and normalized states, so subsequent
    runs just load an array.  It is invalidated on view-set or shape mismatch.
    """
    path = _calibration_cache_path(cfg, n_frames)
    images = states = None

    if os.path.exists(path):
        try:
            blob = np.load(path, allow_pickle=False)
            images, states = blob["images"], blob["states"]
            cached_views = [str(v) for v in blob["view_keys"]]
            if cached_views != list(view_keys):
                logger.warning(
                    "Calibration cache %s was built for views %s but this run "
                    "uses %s — rebuilding.", path, cached_views, view_keys)
                images = states = None
            elif images.shape[0] != n_frames:
                logger.warning(
                    "Calibration cache %s holds %d frames, %d requested — "
                    "rebuilding.", path, images.shape[0], n_frames)
                images = states = None
            else:
                logger.info("Loaded %d calibration frames from %s",
                            images.shape[0], path)
        except Exception as e:
            logger.warning("Could not read calibration cache %s (%s) — rebuilding.",
                           path, e)
            images = states = None

    if images is None:
        images, states = _build_calibration_frames(
            cfg, ctx, task, view_keys, n_frames)
        try:
            np.savez_compressed(
                path, images=images, states=states,
                view_keys=np.array(view_keys, dtype=object).astype("U"),
            )
            logger.info("Saved calibration frames to %s", path)
        except Exception as e:
            logger.warning("Could not write calibration cache %s (%s); "
                           "frames will be rebuilt next run.", path, e)

    observations = []
    for i in range(images.shape[0]):
        views = [np.ascontiguousarray(images[i, v]) for v in range(images.shape[1])]
        obs = {"images": views, "image": views[0], "state": states[i]}
        if len(views) >= 2:
            obs["wrist_image"] = views[1]
        if len(views) >= 3:
            obs["wrist_image_right"] = views[2]
        observations.append(obs)
    return observations


def _calibrate_flashrt(model, observations: list[dict], task: str) -> None:
    """Freeze FP8 activation scales over the prepared frames.

    Left to itself FlashRT calibrates lazily on the first ``predict()`` — which
    here is the robot's home pose, grippers pointed down and away from the
    table.  Every later activation then has to fit scales fitted to a scene the
    policy never revisits, and the cost lands hardest on the gripper channels.

    Measured offline on ep000, single frame -> 16 frames: mean per-joint MAE
    3.376 -> 2.362, gripper joint 7 8.144 -> 3.880 (-52%), joint 15
    5.676 -> 3.516 (-38%), median latency 122 ms -> 85 ms.  That is better than
    the non-quantized FP16 path scores (2.563 / 4.615 / 4.063), so this is the
    fix for gripper accuracy — not switching precision.

    Sample count plateaus hard: 32 and 64 frames land within ~3% of 16, and
    percentile between 99.0 and 99.9 makes no measurable difference.
    """
    percentile = float(os.environ.get("FLASHRT_CALIB_PERCENTILE", "99.9"))
    # calibrate() requires a prompt; use the first sample's state so the
    # state-in-prompt tokens match what inference will produce.
    model.set_prompt(task, state=observations[0]["state"])
    logger.info("Calibrating FlashRT on %d frames (percentile=%.1f)...",
                len(observations), percentile)
    t0 = time.perf_counter()
    model.calibrate(observations, percentile=percentile, verbose=True)
    logger.info("Calibration complete in %.1f s", time.perf_counter() - t0)


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


# ── Per-chunk latency profiler ────────────────────────────────────────────────

class _LatencyProfiler:
    """Times each stage of one RTC inference chunk, from observation to actions.

    A chunk record is opened when the RTC thread starts converting an
    observation (``build_dataset_frame``) and closed when the postprocessor
    returns, which is the moment the chunk is ready to merge into the queue.
    Stages:

        obs_wait         robot observation handed to the engine -> picked up
        build_frame      raw robot dict -> dataset-shaped frame
        obs_prep         upload + /255 + CHW (GPU)
        preprocess       normalize, state tokens, tokenizer, device move
        flashrt_inputs   resize-with-pad + uint8 quantize + GPU->host copy
        flashrt_predict  FlashRT forward (VLM prefix + action expert)
        to_device        numpy chunk -> torch on the policy device
        postprocess      unnormalize (GPU) + move to CPU
        other            RTC bookkeeping between the stages above

    ``inference`` is the sum of every stage but ``obs_wait`` (the latency
    RTC's delay estimate sees); ``e2e`` adds ``obs_wait`` on top.

    CUDA work is asynchronous, so each stage boundary synchronizes the device
    — otherwise GPU time is billed to whichever later stage happens to block.
    The syncs cost tens of microseconds per stage.

    ``FLASHRT_LATENCY_LOG=0`` disables profiling entirely (no syncs).
    ``FLASHRT_LATENCY_EVERY=N`` logs every Nth chunk (default 1; 0 = never).
    ``FLASHRT_LATENCY_SUMMARY=N`` logs mean/p50/p90/max every N chunks
    (default 100; 0 = never).
    """

    STAGES = ("obs_wait", "build_frame", "obs_prep", "preprocess", "flashrt_inputs",
              "flashrt_predict", "to_device", "postprocess", "other")

    def __init__(self, device, every: int, summary_every: int):
        self._sync = torch.device(device).type == "cuda"
        self._every = every
        self._summary_every = summary_every
        self._rec = None
        self._owner = None
        self._t0 = 0.0
        self._count = 0
        self._history = {k: [] for k in self.STAGES + ("inference", "e2e")}
        self._obs_stamp = (None, 0.0)   # (id(obs), perf_counter at hand-off)

    def _now(self) -> float:
        if self._sync:
            torch.cuda.synchronize()
        return time.perf_counter()

    def stamp_observation(self, obs) -> None:
        self._obs_stamp = (id(obs), time.perf_counter())

    def begin(self, obs) -> None:
        import threading
        self._owner = threading.get_ident()
        self._t0 = self._now()
        obs_id, t_obs = self._obs_stamp
        self._rec = {"obs_wait": (self._t0 - t_obs) if obs_id == id(obs) else 0.0}

    @contextmanager
    def step(self, name: str):
        import threading
        if self._rec is None or threading.get_ident() != self._owner:
            yield
            return
        t = self._now()
        try:
            yield
        finally:
            self._rec[name] = self._rec.get(name, 0.0) + (self._now() - t)

    def end(self) -> None:
        import threading
        rec = self._rec
        if rec is None or threading.get_ident() != self._owner:
            return
        self._rec = None
        total = time.perf_counter() - self._t0
        rec["other"] = max(0.0, total - sum(v for k, v in rec.items() if k != "obs_wait"))
        rec["inference"] = total
        rec["e2e"] = total + rec["obs_wait"]
        self._count += 1
        for k, v in rec.items():
            self._history[k].append(v)

        if self._every and self._count % self._every == 0:
            parts = [f"{k} {_fmt_ms_hz(rec.get(k, 0.0))}" for k in self.STAGES]
            logger.info(
                "[latency] chunk %d | %s || inference %s | e2e %s",
                self._count, " | ".join(parts),
                _fmt_ms_hz(rec["inference"]), _fmt_ms_hz(rec["e2e"]),
            )
        if self._summary_every and self._count % self._summary_every == 0:
            self._log_summary()

    def _log_summary(self) -> None:
        lines = [f"[latency] summary over last {self._summary_every} chunks "
                 f"({self._count} total)",
                 f"  {'stage':<16}{'mean ms':>9}{'p50 ms':>9}{'p90 ms':>9}{'max ms':>9}{'mean Hz':>10}"]
        for k in self.STAGES + ("inference", "e2e"):
            v = np.asarray(self._history[k][-self._summary_every:]) * 1e3
            if v.size == 0:
                continue
            mean = float(v.mean())
            hz = f"{1e3 / mean:10.1f}" if mean > 0 else f"{'-':>10}"
            lines.append(f"  {k:<16}{mean:9.2f}{np.percentile(v, 50):9.2f}"
                         f"{np.percentile(v, 90):9.2f}{v.max():9.2f}{hz}")
        logger.info("\n".join(lines))
        for k in self._history:
            del self._history[k][:-self._summary_every]


def _fmt_ms_hz(seconds: float) -> str:
    ms = seconds * 1e3
    return f"{ms:.2f}ms ({1e3 / ms:.0f}Hz)" if ms > 0 else f"{ms:.2f}ms (-)"


class _TimedProcessor:
    """Proxy that times ``__call__`` and forwards everything else (``reset`` …)."""

    def __init__(self, inner, profiler: _LatencyProfiler, name: str, closes_chunk: bool = False):
        self._inner = inner
        self._profiler = profiler
        self._name = name
        self._closes_chunk = closes_chunk

    def __call__(self, *args, **kwargs):
        with self._profiler.step(self._name):
            out = self._inner(*args, **kwargs)
        if self._closes_chunk:
            self._profiler.end()
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)


_PROFILER: "_LatencyProfiler | None" = None


def _prof_step(name: str):
    """Profiler stage context for code that runs inside a chunk; no-op when off."""
    return _PROFILER.step(name) if _PROFILER is not None else nullcontext()


def _install_latency_profiler(ctx, cfg: RolloutConfig) -> None:
    """Hook the RTC engine's per-chunk stages into :class:`_LatencyProfiler`.

    Must run after :func:`_install_fast_observation_prep` (it wraps whichever
    observation prep is live) and before ``strategy.setup()`` starts the RTC
    thread.  Only the engine's own references are wrapped — calibration and
    warmup go through ``ctx.policy.preprocessor`` and stay untimed.
    """
    global _PROFILER
    if os.environ.get("FLASHRT_LATENCY_LOG", "1") == "0":
        logger.info("Latency profiler disabled via FLASHRT_LATENCY_LOG=0")
        return

    from lerobot.rollout.inference import rtc as _rtc

    engine = ctx.policy.inference
    if not isinstance(engine, _rtc.RTCInferenceEngine):
        logger.warning("Latency profiler only supports --inference.type=rtc; skipping")
        return

    prof = _LatencyProfiler(
        cfg.device,
        every=int(os.environ.get("FLASHRT_LATENCY_EVERY", "1")),
        summary_every=int(os.environ.get("FLASHRT_LATENCY_SUMMARY", "100")),
    )

    stock_notify = engine.notify_observation

    def notify_observation(obs):
        prof.stamp_observation(obs)
        return stock_notify(obs)

    engine.notify_observation = notify_observation

    stock_build = _rtc.build_dataset_frame
    stock_prep = _rtc.prepare_observation_for_inference

    def timed_build_dataset_frame(features, values, *args, **kwargs):
        prof.begin(values)
        with prof.step("build_frame"):
            return stock_build(features, values, *args, **kwargs)

    def timed_prepare_observation(*args, **kwargs):
        with prof.step("obs_prep"):
            return stock_prep(*args, **kwargs)

    _rtc.build_dataset_frame = timed_build_dataset_frame
    _rtc.prepare_observation_for_inference = timed_prepare_observation
    engine._preprocessor = _TimedProcessor(engine._preprocessor, prof, "preprocess")
    engine._postprocessor = _TimedProcessor(
        engine._postprocessor, prof, "postprocess", closes_chunk=True)

    _PROFILER = prof
    logger.info(
        "Installed latency profiler (cuda sync=%s, log every %d chunk(s), summary every %d)",
        prof._sync, prof._every, prof._summary_every,
    )


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
    # (must match what Pi05PrepareStateTokenizerProcessorStep produces).
    # Read it from the checkpoint rather than assuming state_dim == action_dim —
    # robots with velocity channels (e.g. LeKiwi) have a wider state than action.
    state_feature = policy.config.input_features.get(OBS_STATE)
    if state_feature is None:
        raise RuntimeError(
            f"Policy has no '{OBS_STATE}' input feature — cannot determine state dim."
        )
    state_dim = state_feature.shape[0]

    # RTC prefix guidance: forward lerobot's --inference.rtc.* settings into
    # FlashRT so each chunk is conditioned on the previous chunk's unexecuted
    # tail. Must be armed at load time — the correction is part of the
    # captured CUDA graph.
    rtc = getattr(cfg.inference, "rtc", None)
    rtc_enabled = rtc is not None and getattr(rtc, "enabled", False)
    rtc_kwargs = {}
    if rtc_enabled:
        rtc_kwargs = {
            "rtc_guidance": True,
            "rtc_execution_horizon": rtc.execution_horizon,
            "rtc_prefix_attention_schedule": rtc.prefix_attention_schedule.name.lower(),
            "rtc_max_guidance_weight": rtc.max_guidance_weight,
        }

    # ``FLASHRT_PRECISION=fp16`` selects the non-quantized full-FP16 path.
    # Measured on ep000 against the FP8 default: mean per-joint MAE improves
    # 18.7%, concentrated in the gripper channels — joint 7 by 43.3%
    # (8.144 -> 4.615) and joint 15 by 28.4%.  Action chatter is unchanged
    # (0.415 vs 0.373 direction reversals per step, against 0.126 for the
    # demonstrations), so this buys grasp precision, not smoothness.
    # Costs: no ``state_prompt_mode="fixed"`` support, so the graph tracks the
    # exact prompt length and recaptures when it changes; and arming RTC
    # disables the fused action-update kernel.
    precision = os.environ.get("FLASHRT_PRECISION", "fp8").lower()
    if precision not in ("fp8", "fp16"):
        raise ValueError(
            f"FLASHRT_PRECISION must be 'fp8' or 'fp16', got {precision!r}"
        )
    precision_kwargs = (
        {"use_fp16": True, "use_fp8": False} if precision == "fp16" else {}
    )
    # state_prompt_mode is only honoured by the FP8 frontend; load_model drops
    # it for FP16 rather than erroring, so pass it only where it does something.
    mode_kwargs = {} if precision == "fp16" else {"state_prompt_mode": "fixed"}

    logger.info(
        "Loading FlashRT model | ckpt=%s  action_dim=%d  chunk=%d  views=%s  "
        "precision=%s  rtc=%s",
        cfg.policy.pretrained_path, action_dim, chunk_size, view_keys,
        precision, rtc_kwargs or "off",
    )
    flash_model = flash_rt.load_model(
        checkpoint=cfg.policy.pretrained_path,
        framework="torch",
        num_views=len(view_keys),
        action_horizon=chunk_size,
        autotune=3,
        **mode_kwargs,
        **precision_kwargs,
        **rtc_kwargs,
    )

    # Calibrate before anything else touches predict(). FlashRT freezes its FP8
    # activation scales on the first inference, so whatever runs first decides
    # them — and one frame of the robot's home pose is a poor choice (see
    # _calibrate_flashrt for the measured cost). Explicit calibration clears
    # the lazy-bootstrap latch, so the warmup below no longer sets the scales.
    n_calib = int(os.environ.get("FLASHRT_CALIB_FRAMES", "16"))
    if n_calib > 1:
        observations = _load_or_build_calibration(
            cfg, ctx, task, view_keys, n_calib)
        _calibrate_flashrt(flash_model, observations, task)
    else:
        logger.warning(
            "FLASHRT_CALIB_FRAMES=%d — falling back to single-frame "
            "calibration on the robot's current pose.", n_calib)

    # Warmup on a real observation. With calibration already done this exercises
    # the replay path and pays the graph-capture cost upfront rather than on the
    # first live control tick; it still runs on a real frame at the robot's
    # actual operating point (see _capture_warmup_inputs).
    warm_imgs, warm_state = _capture_warmup_inputs(ctx, cfg, task, view_keys)
    _warmup_flashrt(flash_model, task, warm_imgs, warm_state, n_iters=20)

    # Close over everything the patched method needs.
    _model     = flash_model
    _task      = task
    _act_dim   = action_dim
    _state_dim = state_dim
    _device    = device
    _views     = view_keys
    _rtc_on    = rtc_enabled

    def predict_action_chunk(self, batch: dict, **kwargs) -> torch.Tensor:
        """FlashRT drop-in for PI05Policy.predict_action_chunk.

        Accepts the same preprocessed batch and kwargs (inference_delay,
        prev_chunk_left_over) that RTCInferenceEngine passes, and forwards
        both to FlashRT's RTC prefix guidance when --inference.type=rtc.

        Returns:
            Tensor shape (1, chunk_size, action_dim) float32, normalized.
        """
        global _state_dim_warned

        # prev_chunk_left_over is already in the same normalized space this
        # function returns (it is the `original` tensor from ActionQueue.merge),
        # so it can go straight to FlashRT with no conversion.
        with _prof_step("flashrt_inputs"):
            prev_np = None
            if _rtc_on:
                prev = kwargs.get("prev_chunk_left_over")
                if prev is not None:
                    if prev.dim() == 3:
                        prev = prev.squeeze(0)
                    prev_np = prev[:, :_act_dim].float().cpu().numpy()

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
        with torch.no_grad(), _prof_step("flashrt_predict"):
            chunk_np = _model.predict(
                images=imgs,
                prompt=_task,
                state=state_np,
                prev_actions=prev_np,
                inference_delay=int(kwargs.get("inference_delay") or 0),
            )

        # Slice to action_dim, return (1, T, action_dim) on target device
        with _prof_step("to_device"):
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

    _configure_torch_threads()
    _install_fast_observation_prep()
    if not _install_weightless_policy_load():
        _install_fast_policy_load()
    _install_throttled_telemetry()
    _install_episode_engine_pause()

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
        # Keep the dataset at the camera rate when interpolation raises the
        # control rate. Must happen before the strategy starts recording.
        _install_record_decimation(ctx, cfg)

        # Swap in FlashRT as the policy inference backend.
        # Must happen before strategy.setup() starts the RTC thread.
        _install_flashrt_backend(ctx, cfg)

        # Per-chunk stage timings (ms / Hz) for the RTC inference thread.
        # Must happen before strategy.setup() starts the RTC thread.
        _install_latency_profiler(ctx, cfg)

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
