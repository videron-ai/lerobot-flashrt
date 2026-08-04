"""Local latency gate — LocalFlashRTPI05Policy (in-process FlashRT).

Gate criteria (same as Phase 5 HTTP gate):
    1. N_SETTLE calls are executed and discarded (CUDA graph capture window).
    2. N_GATE consecutive predict_action_chunk calls all complete within
       3× the warm-window median.

No HTTP server required — FlashRT runs in the same process.

Usage:
    LEROBOT_CKPT=/path/to/checkpoint \\
    pytest tests/test_local_latency.py -v -s

Environment variables:
    LEROBOT_CKPT          Path to the LeRobot pretrained model directory.
    LATENCY_N_SETTLE      Settle calls to discard (default 5).
    LATENCY_N_GATE        Gated calls to measure (default 200).
    LATENCY_SEED          RNG seed (default 42).
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np
import pytest
import torch

from lerobot_flashrt.client import CANONICAL_VIEW_ORDER
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, OBS_STATE

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

CKPT = os.environ.get(
    "LEROBOT_CKPT",
    "/home/videron/Desktop/openarm/outputs/train/openarm_folding_high_quality_60k"
    "/checkpoints/060000/pretrained_model",
)
N_SETTLE = int(os.environ.get("LATENCY_N_SETTLE", "5"))
N_GATE   = int(os.environ.get("LATENCY_N_GATE",   "200"))
SEED     = int(os.environ.get("LATENCY_SEED",     "42"))

_CHUNK            = 30
_DIM              = 32
_ORIGINAL_ACTION_DIM = 16   # output_features.action.shape[0] for this checkpoint

# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def local_policy():
    from lerobot_flashrt import make_local_flashrt_policy
    policy, _, _ = make_local_flashrt_policy(
        CKPT,
        device="cuda",
        action_horizon=_CHUNK,
        num_views=3,
        state_prompt_mode="fixed",
    )
    return policy


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_images(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Fresh set of 3 random uint8 images, one per view."""
    return {
        key: rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
        for key in CANONICAL_VIEW_ORDER
    }


def _make_state(rng: np.random.Generator, perturb: float = 0.3) -> np.ndarray:
    """32-dim state: real joint values in dims 0:_ORIGINAL_ACTION_DIM, zeros beyond."""
    s = np.zeros(_DIM, dtype=np.float32)
    s[:_ORIGINAL_ACTION_DIM] = np.clip(
        rng.uniform(-perturb, perturb, _ORIGINAL_ACTION_DIM), -1.0, 1.0
    ).astype(np.float32)
    return s


def _make_batch(images: dict[str, np.ndarray], state: np.ndarray,
                prompt: str, device: str) -> dict[str, torch.Tensor]:
    """Build a minimal predict_action_chunk-compatible batch from raw numpy inputs."""
    batch: dict[str, torch.Tensor] = {}
    for key, img in images.items():
        batch[key] = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    batch[OBS_STATE] = torch.from_numpy(state).unsqueeze(0).to(device)
    batch["task"] = [prompt]
    batch[OBS_LANGUAGE_TOKENS] = torch.zeros(1, 200, dtype=torch.long, device=device)
    batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones(1, 200, dtype=torch.bool, device=device)
    return batch


def _timed_predict(policy, batch: dict[str, torch.Tensor]) -> float:
    """Return wall-clock latency of one predict_action_chunk call in ms."""
    t0 = time.perf_counter()
    with torch.no_grad():
        policy.predict_action_chunk(batch)
    return (time.perf_counter() - t0) * 1000.0


def _print_latency_report(latencies_ms: list[float], n_gate: int) -> tuple[float, list]:
    lat_arr      = np.array(latencies_ms)
    median_ms    = float(np.median(lat_arr))
    mean_ms      = float(np.mean(lat_arr))
    p95_ms       = float(np.percentile(lat_arr, 95))
    p99_ms       = float(np.percentile(lat_arr, 99))
    threshold_ms = median_ms * 3.0

    print(
        f"\nLatency over {n_gate} calls:\n"
        f"  median={median_ms:.1f} ms  mean={mean_ms:.1f} ms  "
        f"p95={p95_ms:.1f} ms  p99={p99_ms:.1f} ms\n"
        f"  threshold (3× median)={threshold_ms:.1f} ms"
    )

    hist, edges = np.histogram(lat_arr, bins=10)
    print("  Histogram:")
    for count, lo, hi in zip(hist, edges[:-1], edges[1:]):
        bar = "#" * int(count * 40 / max(hist, default=1))
        print(f"    {lo:6.1f}–{hi:6.1f} ms | {bar} ({count})")

    slow_calls = [(i, lat) for i, lat in enumerate(latencies_ms) if lat > threshold_ms]
    if slow_calls:
        slow_info = ", ".join(f"call {i}: {lat:.1f} ms" for i, lat in slow_calls[:10])
        extra = " (showing first 10)" if len(slow_calls) > 10 else ""
        print(f"\n  SLOW CALLS: {slow_info}{extra}")

    return threshold_ms, slow_calls


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not os.path.exists(CKPT),
    reason=f"Checkpoint not found at {CKPT}",
)
def test_local_predict_shape(local_policy):
    """Smoke test: predict_action_chunk returns (1, chunk, 32) on a single call."""
    rng = np.random.default_rng(SEED)
    batch = _make_batch(_make_images(rng), _make_state(rng), "fold the fabric",
                        local_policy.config.device)
    with torch.no_grad():
        out = local_policy.predict_action_chunk(batch)
    assert out.shape == (1, _CHUNK, _DIM), f"unexpected shape {out.shape}"
    assert out.dtype == torch.float32


@pytest.mark.skipif(
    not os.path.exists(CKPT),
    reason=f"Checkpoint not found at {CKPT}",
)
def test_local_latency_stability(local_policy):
    """Gate: N_GATE predict_action_chunk calls all within 3× warm median.

    Failure modes:
        all calls slow      → CUDA graph not captured; check state_prompt_mode=fixed
        occasional spikes   → graph re-capture on state length change (exact mode)
                              → switch to fixed mode
        first N slow then ok→ settle window too short; increase LATENCY_N_SETTLE
        random spikes       → OS jitter / GPU preemption; re-run to confirm
    """
    rng    = np.random.default_rng(SEED)
    prompt = "fold the fabric"
    device = local_policy.config.device

    # Pre-generate all batches so batch construction time is not included.
    all_batches = [
        _make_batch(_make_images(rng), _make_state(rng), prompt, device)
        for _ in range(N_SETTLE + N_GATE)
    ]

    print(f"\n=== Local latency gate | settle={N_SETTLE} gate={N_GATE} ===")
    print("Settling ... ", end="", flush=True)
    for i in range(N_SETTLE):
        t = _timed_predict(local_policy, all_batches[i])
        print(f"{t:.0f}ms ", end="", flush=True)
    print()

    latencies_ms: list[float] = []
    for i in range(N_GATE):
        latencies_ms.append(_timed_predict(local_policy, all_batches[N_SETTLE + i]))

    threshold_ms, slow_calls = _print_latency_report(latencies_ms, N_GATE)

    logger.info(
        "Local latency gate | median=%.1f ms | p99=%.1f ms | slow=%d/%d | threshold=%.1f ms",
        float(np.median(latencies_ms)), float(np.percentile(latencies_ms, 99)),
        len(slow_calls), N_GATE, threshold_ms,
    )

    slow_info = ", ".join(f"call {i}: {lat:.1f} ms" for i, lat in slow_calls[:10])
    assert not slow_calls, (
        f"{len(slow_calls)}/{N_GATE} calls exceeded 3× median ({threshold_ms:.1f} ms).\n"
        "\nTriage:\n"
        "  All calls slow (>3×)      → CUDA graph not captured; check state_prompt_mode=fixed\n"
        "  Periodic spikes (every N) → graph re-capture in exact mode → switch to fixed\n"
        "  First few slow, then OK   → settle window too short; increase LATENCY_N_SETTLE\n"
        "  Random isolated spikes    → OS jitter / GPU preemption; re-run to confirm\n"
        f"\nSlow calls: {slow_info or 'none'}"
    )


@pytest.mark.skipif(
    not os.path.exists(CKPT),
    reason=f"Checkpoint not found at {CKPT}",
)
def test_local_vs_http_latency():
    """Compare local and HTTP latency if FLASHRT_SERVER is set (informational, not gated)."""
    server = os.environ.get("FLASHRT_SERVER")
    if not server:
        pytest.skip("FLASHRT_SERVER not set; skipping local-vs-HTTP latency comparison.")

    from lerobot_flashrt import make_local_flashrt_policy, make_flashrt_policy
    from lerobot_flashrt.client import FlashRTClient

    policy_local, _, _ = make_local_flashrt_policy(CKPT, action_horizon=_CHUNK)
    client_http = FlashRTClient(server, timeout_s=60.0)

    rng    = np.random.default_rng(SEED)
    prompt = "fold the fabric"
    device = policy_local.config.device
    n      = 50

    # Settle both paths
    for _ in range(N_SETTLE):
        images = _make_images(rng)
        state  = _make_state(rng)
        _timed_predict(policy_local, _make_batch(images, state, prompt, device))
        client_http.predict(images, prompt, state)

    local_lats, http_lats = [], []
    for _ in range(n):
        images = _make_images(rng)
        state  = _make_state(rng)
        local_lats.append(_timed_predict(policy_local, _make_batch(images, state, prompt, device)))
        t0 = time.perf_counter()
        client_http.predict(images, prompt, state)
        http_lats.append((time.perf_counter() - t0) * 1000.0)

    print(f"\n=== Local vs HTTP latency ({n} calls each) ===")
    print(f"  Local  — median={np.median(local_lats):.1f} ms  p99={np.percentile(local_lats, 99):.1f} ms")
    print(f"  HTTP   — median={np.median(http_lats):.1f} ms  p99={np.percentile(http_lats, 99):.1f} ms")
    print(f"  Overhead (HTTP − local): {np.median(http_lats) - np.median(local_lats):.1f} ms median")
