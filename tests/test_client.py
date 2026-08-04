"""Phase 2 gate — unit tests for FlashRTClient.

Tests run against a mock server so they can execute without a GPU.
All negative cases (wrong dtype, wrong state dim, missing camera, two images
instead of three) must raise — not silently degrade.

Run with:
    pytest tests/test_client.py -v
"""

from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from lerobot_flashrt.client import CANONICAL_VIEW_ORDER, FlashRTClient

# ── Mock server ──────────────────────────────────────────────────────────────

_CHUNK = 30
_DIM = 32
_MOCK_ACTIONS = np.random.randn(_CHUNK, _DIM).astype(np.float32)

_MOCK_HEALTH = {
    "status": "ok",
    "warmup_done": True,
    "frontend_class": "Pi05TorchFrontendThor",
    "chunk_size": _CHUNK,
    "action_dim": _DIM,
    "num_views": 3,
    "view_order": CANONICAL_VIEW_ORDER,
    "state_prompt_mode": "fixed",
    "framework": "torch",
    "version": "test",
    "current_prompt": None,
}


class _MockHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence access log
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
            # Validate server-side: 3 images, 32-dim state
            if len(body.get("images", [])) != 3:
                self._send_json(422, {"detail": "wrong view count"})
                return
            if len(body.get("state", [])) != _DIM:
                self._send_json(422, {"detail": "wrong state dim"})
                return
            self._send_json(200, {
                "actions": _MOCK_ACTIONS.tolist(),
                "latency_ms": 12.3,
                "shape": [_CHUNK, _DIM],
            })
        elif self.path == "/warmup":
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
def client(mock_server):
    return FlashRTClient(mock_server, timeout_s=5.0)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_images() -> dict[str, np.ndarray]:
    return {
        key: np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        for key in CANONICAL_VIEW_ORDER
    }


def _make_state() -> np.ndarray:
    return np.zeros(32, dtype=np.float32)


# ── Happy path ───────────────────────────────────────────────────────────────

def test_health(client):
    h = client.health()
    assert h["status"] == "ok"
    assert h["num_views"] == 3
    assert h["view_order"] == CANONICAL_VIEW_ORDER


def test_predict_shape(client):
    actions = client.predict(_make_images(), "fold the fabric", _make_state())
    assert actions.shape == (_CHUNK, _DIM)
    assert actions.dtype == np.float32


def test_predict_values(client):
    actions = client.predict(_make_images(), "fold the fabric", _make_state())
    np.testing.assert_allclose(actions, _MOCK_ACTIONS, rtol=1e-5)


def test_warmup(client):
    result = client.warmup(
        _make_images(),
        "fold the fabric",
        [_make_state(), _make_state() + 0.1],
    )
    assert result["calibrated"] is True
    assert isinstance(result["warmed_lengths"], list)


# ── Negative cases — all must raise, not silently degrade ────────────────────

def test_wrong_view_count_raises(client):
    images = {k: np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
              for k in CANONICAL_VIEW_ORDER[:2]}  # only 2
    with pytest.raises(ValueError, match="Missing camera"):
        client.predict(images, "test", _make_state())


def test_missing_camera_raises(client):
    images = _make_images()
    del images[CANONICAL_VIEW_ORDER[1]]  # drop right_wrist
    with pytest.raises(ValueError, match="Missing camera"):
        client.predict(images, "test", _make_state())


def test_wrong_image_dtype_raises(client):
    images = _make_images()
    images[CANONICAL_VIEW_ORDER[0]] = images[CANONICAL_VIEW_ORDER[0]].astype(np.float32)
    with pytest.raises(TypeError, match="uint8"):
        client.predict(images, "test", _make_state())


def test_wrong_image_shape_raises(client):
    images = _make_images()
    images[CANONICAL_VIEW_ORDER[0]] = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="224"):
        client.predict(images, "test", _make_state())


def test_wrong_state_dim_raises(client):
    with pytest.raises(ValueError, match="32"):
        client.predict(_make_images(), "test", np.zeros(16, dtype=np.float32))


def test_wrong_state_dtype_raises(client):
    with pytest.raises(TypeError, match="float32"):
        client.predict(_make_images(), "test", np.zeros(32, dtype=np.float64))


def test_state_not_array_raises(client):
    with pytest.raises(TypeError, match="np.ndarray"):
        client.predict(_make_images(), "test", [0.0] * 32)


def test_view_order_mismatch_raises(mock_server, monkeypatch):
    """Client must raise at construction if server view_order differs."""
    wrong_health = dict(_MOCK_HEALTH, view_order=["cam_a", "cam_b", "cam_c"])

    class BadHandler(_MockHandler):
        def do_GET(self):
            if self.path == "/health":
                self._send_json(200, wrong_health)
            else:
                super().do_GET()

    server = HTTPServer(("127.0.0.1", 0), BadHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        with pytest.raises(RuntimeError, match="view_order"):
            FlashRTClient(f"http://127.0.0.1:{port}", timeout_s=2.0)
    finally:
        server.shutdown()


def test_server_unavailable_raises():
    with pytest.raises(RuntimeError, match="connect"):
        FlashRTClient("http://127.0.0.1:19999", timeout_s=1.0)
