"""Phase 5 gate — warmup sequence and latency stability after warmup.

Gate criteria:
    1. /warmup accepts valid frames and states, returns calibrated=True.
    2. N_SETTLE calls are executed after /warmup and discarded (allow any
       remaining graph captures to complete).
    3. N_GATE consecutive predict calls all report client-side latency
       ≤ 3× the warm-window median.  A CUDA graph capture event adds
       100–500 ms per occurrence; the 3× threshold (~45–60 ms for a
       15–20 ms warm median) catches any capture without false-positives
       from normal jitter.

Run against a live FlashRT server (Phase 5 gate proper):
    FLASHRT_SERVER=http://flashrt-host:8000 \\
    pytest tests/test_latency.py -v -s

Smoke-test against mock server only (no GPU needed):
    pytest tests/test_latency.py -v -k "mock"

Environment variables:
    FLASHRT_SERVER    FlashRT server base URL.  If unset, real-server tests
                      are skipped and only mock tests run.
    LATENCY_N_SETTLE  Calls to discard before gated window (default 5).
    LATENCY_N_GATE    Gated predict calls (default 200).
    LATENCY_SEED      RNG seed for state variation (default 42).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from lerobot_flashrt.client import CANONICAL_VIEW_ORDER, FlashRTClient

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

REAL_SERVER = os.environ.get("FLASHRT_SERVER", None)
N_SETTLE = int(os.environ.get("LATENCY_N_SETTLE", "5"))
N_GATE = int(os.environ.get("LATENCY_N_GATE", "200"))
SEED = int(os.environ.get("LATENCY_SEED", "42"))

_CHUNK = 30
_DIM = 32
# Real DOF count for the checkpoint (output_features.action.shape[0]).
# Only dims 0:_ORIGINAL_ACTION_DIM carry meaningful normalized joint values;
# dims _ORIGINAL_ACTION_DIM:_DIM are zero-padding that the server ignores
# when building the state-in-prompt (it truncates to _ORIGINAL_ACTION_DIM).
_ORIGINAL_ACTION_DIM = 16

# ── Mock server ───────────────────────────────────────────────────────────────

_MOCK_ACTIONS = np.random.default_rng(0).standard_normal((_CHUNK, _DIM)).astype(np.float32)

_MOCK_HEALTH = {
    "status": "ok",
    "warmup_done": False,
    "frontend_class": "Pi05TorchFrontendThor",
    "chunk_size": _CHUNK,
    "action_dim": _DIM,
    "num_views": len(CANONICAL_VIEW_ORDER),
    "view_order": CANONICAL_VIEW_ORDER,
    "state_prompt_mode": "fixed",
    "framework": "torch",
    "version": "test",
    "current_prompt": None,
}


class _MockHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send_json(self, code: int, body: dict):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, _MOCK_HEALTH)
        else:
            self._send_json(404, {"detail": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))

        if self.path == "/predict":
            if len(body.get("images", [])) != len(CANONICAL_VIEW_ORDER):
                self._send_json(422, {"detail": "wrong view count"})
                return
            if len(body.get("state", [])) != _DIM:
                self._send_json(422, {"detail": "wrong state dim"})
                return
            self._send_json(200, {
                "actions": _MOCK_ACTIONS.tolist(),
                "latency_ms": 15.0,
                "shape": [_CHUNK, _DIM],
            })
        elif self.path == "/warmup":
            n_imgs = len(body.get("images", []))
            if n_imgs != len(CANONICAL_VIEW_ORDER):
                self._send_json(422, {
                    "detail": f"Expected {len(CANONICAL_VIEW_ORDER)} images, got {n_imgs}"
                })
                return
            n_states = len(body.get("states", []))
            if n_states == 0:
                self._send_json(422, {"detail": "states list must be non-empty"})
                return
            self._send_json(200, {
                "warmed_lengths": [117, 118],
                "calibrated": True,
                "duration_ms": 4200.0,
            })
        else:
            self._send_json(404, {"detail": "not found"})


@pytest.fixture(scope="module")
def mock_server():
    server = HTTPServer(("127.0.0.1", 0), _MockHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.fixture(scope="module")
def mock_client(mock_server):
    return FlashRTClient(mock_server, timeout_s=5.0)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_images() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    return {
        key: rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
        for key in CANONICAL_VIEW_ORDER
    }


def _make_state(rng: np.random.Generator | None = None, perturb: float = 0.0) -> np.ndarray:
    """Return a 32-dim normalized state as a real client would send it.

    Only the first _ORIGINAL_ACTION_DIM dims carry meaningful joint values;
    dims _ORIGINAL_ACTION_DIM:_DIM are zero-padding.  This matches what
    LeRobot's preprocessor produces (NormalizerProcessorStep outputs only
    the real DOF count; the client zero-pads to 32 before sending).
    """
    s = np.zeros(_DIM, dtype=np.float32)
    if rng is not None and perturb > 0.0:
        s[:_ORIGINAL_ACTION_DIM] = np.clip(
            rng.uniform(-perturb, perturb, _ORIGINAL_ACTION_DIM), -1.0, 1.0
        ).astype(np.float32)
    return s


def _timed_predict(client: FlashRTClient, images: dict, prompt: str,
                   state: np.ndarray) -> float:
    """Return wall-clock predict latency in milliseconds."""
    t0 = time.perf_counter()
    client.predict(images, prompt, state)
    return (time.perf_counter() - t0) * 1000.0


# ── Mock tests (always run — no GPU required) ─────────────────────────────────

def test_warmup_response_schema_mock(mock_client):
    """Warmup returns calibrated=True and expected schema fields."""
    result = mock_client.warmup(
        _make_images(),
        "fold the fabric",
        [_make_state(), _make_state()],
    )
    assert result["calibrated"] is True
    assert isinstance(result["warmed_lengths"], list)
    assert isinstance(result["duration_ms"], float)


def test_warmup_then_predict_mock(mock_client):
    """Predict after warmup returns correct shape."""
    mock_client.warmup(
        _make_images(),
        "fold the fabric",
        [_make_state()],
    )
    actions = mock_client.predict(_make_images(), "fold the fabric", _make_state())
    assert actions.shape == (_CHUNK, _DIM)
    assert actions.dtype == np.float32


def test_warmup_wrong_image_count_rejected_mock(mock_server):
    """Server rejects warmup with wrong number of images (422)."""
    import requests
    client = FlashRTClient.__new__(FlashRTClient)
    client._endpoint = mock_server.rstrip("/")
    client._timeout = 5.0
    client._session = requests.Session()

    payload = {
        "images": [
            base64.b64encode(np.zeros((224, 224, 3), dtype=np.uint8).tobytes()).decode()
        ],  # only 1 image
        "prompt": "test",
        "states": [np.zeros(32, dtype=np.float32).tolist()],
    }
    r = client._session.post(f"{mock_server}/warmup", json=payload, timeout=5.0)
    assert r.status_code == 422


def test_warmup_empty_states_rejected_mock(mock_server):
    """Server rejects warmup with empty states list (422)."""
    import requests
    imgs_b64 = [
        base64.b64encode(np.zeros((224, 224, 3), dtype=np.uint8).tobytes()).decode()
        for _ in CANONICAL_VIEW_ORDER
    ]
    payload = {
        "images": imgs_b64,
        "prompt": "test",
        "states": [],
    }
    r = requests.post(f"{mock_server}/warmup", json=payload, timeout=5.0)
    assert r.status_code == 422


def test_predict_latency_mock_shape_correct(mock_client):
    """Rapid successive predicts against mock all return correct shape (no state issues)."""
    rng = np.random.default_rng(SEED)
    images = _make_images()
    for _ in range(20):
        state = _make_state(rng, perturb=0.5)
        actions = mock_client.predict(images, "fold the fabric", state)
        assert actions.shape == (_CHUNK, _DIM)


# ── Real-server latency gate (requires FLASHRT_SERVER) ───────────────────────

@pytest.fixture(scope="module")
def real_client():
    if REAL_SERVER is None:
        pytest.skip(
            "FLASHRT_SERVER not set — set it to run the Phase 5 latency gate. "
            "Example: FLASHRT_SERVER=http://localhost:8000 pytest tests/test_latency.py -v -s"
        )
    try:
        return FlashRTClient(REAL_SERVER, timeout_s=60.0)
    except RuntimeError as exc:
        pytest.fail(f"Cannot connect to FlashRT server at {REAL_SERVER}: {exc}")


def test_warmup_calibrated(real_client):
    """Gate 5.1: /warmup returns calibrated=True on real server."""
    rng = np.random.default_rng(SEED)
    images = _make_images()
    states = [
        np.zeros(_DIM, dtype=np.float32),                   # reset pose
        _make_state(rng, perturb=0.3),                       # mid-rollout
        _make_state(rng, perturb=0.5),                       # near-goal
    ]
    result = real_client.warmup(images, "fold the fabric", states)
    logger.info(
        "Warmup result: calibrated=%s warmed_lengths=%s duration_ms=%.0f",
        result.get("calibrated"),
        result.get("warmed_lengths"),
        result.get("duration_ms", 0),
    )
    assert result["calibrated"] is True, (
        f"/warmup returned calibrated=False: {result}. "
        "FP8 calibration did not complete. Check server logs for calibrate() errors."
    )


def test_latency_stability(real_client):
    """Gate 5.2: 200 predict calls after warmup all within 3× warm median.

    Failure modes:
        all calls slow      → CUDA graph not captured; check state_prompt_mode
        occasional spikes   → graph re-capture; state permutation crossing bucket
                             boundary in exact mode → switch to fixed mode
        first 5 slow then ok→ graph capture during settle window (expected; OK)
    """
    rng = np.random.default_rng(SEED)
    images = _make_images()
    prompt = "fold the fabric"

    # Vary state across calls to exercise different digitized bins.
    # perturb=0.3 changes several bins each call; 32-dim uniform variation
    # covers the representative space without going outside [-1, 1].
    all_states = [_make_state(rng, perturb=0.3) for _ in range(N_SETTLE + N_GATE)]

    # Settle window — discard these latencies; graph captures are expected here.
    print(f"\n=== Phase 5 latency gate | settle={N_SETTLE} gate={N_GATE} ===")
    print("Settling ... ", end="", flush=True)
    for i in range(N_SETTLE):
        t = _timed_predict(real_client, images, prompt, all_states[i])
        print(f"{t:.0f}ms ", end="", flush=True)
    print()

    # Gated window — all calls must be within 3× median.
    latencies_ms: list[float] = []
    for i in range(N_GATE):
        state = all_states[N_SETTLE + i]
        lat = _timed_predict(real_client, images, prompt, state)
        latencies_ms.append(lat)

    lat_arr = np.array(latencies_ms)
    median_ms = float(np.median(lat_arr))
    mean_ms = float(np.mean(lat_arr))
    p95_ms = float(np.percentile(lat_arr, 95))
    p99_ms = float(np.percentile(lat_arr, 99))
    threshold_ms = median_ms * 3.0

    print(
        f"Latency over {N_GATE} calls:\n"
        f"  median={median_ms:.1f} ms  mean={mean_ms:.1f} ms  "
        f"p95={p95_ms:.1f} ms  p99={p99_ms:.1f} ms\n"
        f"  threshold (3× median)={threshold_ms:.1f} ms"
    )

    # Print simple histogram (10 buckets)
    hist, edges = np.histogram(lat_arr, bins=10)
    print("  Histogram:")
    for count, lo, hi in zip(hist, edges[:-1], edges[1:]):
        bar = "#" * int(count * 40 / max(hist, default=1))
        print(f"    {lo:6.1f}–{hi:6.1f} ms | {bar} ({count})")

    slow_calls = [(i, lat) for i, lat in enumerate(latencies_ms) if lat > threshold_ms]
    if slow_calls:
        slow_info = ", ".join(f"call {i}: {lat:.1f} ms" for i, lat in slow_calls[:10])
        extra = f" (showing first 10)" if len(slow_calls) > 10 else ""
        print(f"\n  SLOW CALLS: {slow_info}{extra}")

    logger.info(
        "Latency gate | median=%.1f ms | p99=%.1f ms | slow=%d/%d | threshold=%.1f ms",
        median_ms, p99_ms, len(slow_calls), N_GATE, threshold_ms,
    )

    assert not slow_calls, (
        f"{len(slow_calls)}/{N_GATE} calls exceeded 3× median ({threshold_ms:.1f} ms).\n"
        "\nTriage:\n"
        "  All calls slow (>3×)     → CUDA graph not captured; check state_prompt_mode=fixed\n"
        "  Periodic spikes (every N) → graph re-capture; state crossing exact-mode bucket edge\n"
        "                              → switch server to --state-prompt-mode fixed\n"
        "  First few slow, then OK  → settle window too short; increase LATENCY_N_SETTLE\n"
        "  Random isolated spikes   → OS jitter / GPU preemption; re-run to confirm\n"
        f"\nSlow calls: {slow_info if slow_calls else 'none'}"
    )
