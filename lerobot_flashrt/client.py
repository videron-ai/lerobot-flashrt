"""FlashRTClient — HTTP client for the FlashRT LeRobot inference server.

The client serializes a named dict of camera images in CANONICAL_VIEW_ORDER,
sends a 32-dim normalized state and a task-text prompt to the server, and
returns a raw normalized (30, 32) action chunk.

LeRobot's postprocessor (UnnormalizerProcessorStep + AbsoluteActionsProcessorStep)
must be applied to the returned chunk before sending actions to the robot.
"""

from __future__ import annotations

import base64
import logging
from typing import Sequence

import numpy as np
import requests

logger = logging.getLogger(__name__)

# ── Canonical view ordering ──────────────────────────────────────────────────
# Single source of truth.  Must match the server's --view-order and the
# training config's input_features insertion order (JSON key order in config.json):
#   observation.images.left_wrist  → index 0
#   observation.images.right_wrist → index 1
#   observation.images.base        → index 2
# See FINDINGS.md §0.1 for derivation.

CANONICAL_VIEW_ORDER: list[str] = [
    "observation.images.left_wrist",
    "observation.images.right_wrist",
    "observation.images.base",
]

_IMG_H, _IMG_W, _IMG_C = 224, 224, 3
_STATE_DIM = 32


class FlashRTClient:
    """HTTP client for the FlashRT LeRobot inference server.

    Args:
        endpoint: Base URL of the server, e.g. "http://localhost:8000".
        timeout_s: Per-request timeout in seconds.  Rollout budget is ~33 ms at
            30 Hz; set high enough for graph capture on first call.
    """

    def __init__(self, endpoint: str, timeout_s: float = 10.0):
        self._endpoint = endpoint.rstrip("/")
        self._timeout = timeout_s
        self._session = requests.Session()
        self._last_prompt: str | None = None

        # Fail fast at construction — assert server agreement on view order.
        h = self.health()
        server_order = h.get("view_order", [])
        if server_order != CANONICAL_VIEW_ORDER:
            raise RuntimeError(
                f"Server view_order {server_order} != client CANONICAL_VIEW_ORDER "
                f"{CANONICAL_VIEW_ORDER}. "
                "Fix the server --view-order or update CANONICAL_VIEW_ORDER here."
            )
        server_nv = h.get("num_views")
        if server_nv is not None and server_nv != len(CANONICAL_VIEW_ORDER):
            raise RuntimeError(
                f"Server num_views={server_nv} != len(CANONICAL_VIEW_ORDER)="
                f"{len(CANONICAL_VIEW_ORDER)}"
            )
        logger.info(
            "FlashRTClient connected to %s | frontend=%s | chunk=%s | view_order=%s",
            self._endpoint,
            h.get("frontend_class"),
            h.get("chunk_size"),
            server_order,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(
        self,
        images: dict[str, np.ndarray],
        prompt: str,
        state: np.ndarray,
        seed: int | None = None,
    ) -> np.ndarray:
        """Run one inference step.

        Args:
            images: Dict keyed by camera name (must contain all keys in
                CANONICAL_VIEW_ORDER). Each value is (224, 224, 3) uint8 RGB.
                Extra keys are silently ignored after validation passes.
            prompt: Raw task text string, e.g. "fold the fabric".
                Do NOT include the "Task: …, State: …; Action:" prefix —
                the server builds that internally.
            state: Normalized robot state, shape (32,) float32.
                This is the NormalizerProcessorStep output, not raw joint angles.

        Returns:
            np.ndarray of shape (30, 32) float32, raw **normalized** actions.
            Pass through LeRobot's postprocessor to unnormalize and convert
            relative→absolute before sending to the robot.
        """
        imgs_ordered = self._pack_images(images)
        self._validate_state(state)

        payload = {
            "images": [self._encode_image(img) for img in imgs_ordered],
            "prompt": prompt,
            "state": state.tolist(),
        }
        if seed is not None:
            payload["seed"] = seed
        self._last_prompt = prompt
        resp = self._post("/predict", payload)
        actions = np.array(resp["actions"], dtype=np.float32)
        return actions

    def warmup(
        self,
        images: dict[str, np.ndarray],
        prompt: str,
        states: Sequence[np.ndarray],
    ) -> dict:
        """Calibrate FP8 scales and warm state-prompt graph buckets.

        Blocks until both calibration and bucket warming are complete on the
        server.  Call once before starting the rollout loop.

        Args:
            images: Same format as predict().
            prompt: Representative task string.
            states: List of representative normalized 32-dim states (reset
                pose, mid-rollout, near-goal).  Used for bucket warming in
                exact state-prompt mode; passed to FP8 calibration as well.

        Returns:
            Server WarmupResponse dict with warmed_lengths, calibrated, duration_ms.
        """
        imgs_ordered = self._pack_images(images)
        for s in states:
            self._validate_state(s)

        payload = {
            "images": [self._encode_image(img) for img in imgs_ordered],
            "prompt": prompt,
            "states": [s.tolist() for s in states],
        }
        return self._post("/warmup", payload, timeout=300.0)

    def health(self) -> dict:
        """Return the server health dict."""
        try:
            r = self._session.get(
                f"{self._endpoint}/health", timeout=self._timeout
            )
            r.raise_for_status()
            return r.json()
        except requests.Timeout as exc:
            raise RuntimeError(
                f"FlashRT server at {self._endpoint} did not respond within "
                f"{self._timeout}s. Is the server running?"
            ) from exc
        except requests.ConnectionError as exc:
            raise RuntimeError(
                f"Could not connect to FlashRT server at {self._endpoint}. "
                "Start serving/lerobot_host/server.py first."
            ) from exc

    # ── Internals ─────────────────────────────────────────────────────────────

    def _pack_images(self, images: dict[str, np.ndarray]) -> list[np.ndarray]:
        """Validate and serialize camera dict → ordered list."""
        missing = [k for k in CANONICAL_VIEW_ORDER if k not in images]
        if missing:
            raise ValueError(
                f"Missing camera(s): {missing}. "
                f"All of CANONICAL_VIEW_ORDER must be present: {CANONICAL_VIEW_ORDER}. "
                "A missing camera at rollout time (unplugged USB, dropped frame) must "
                "raise rather than silently shift remaining views into wrong slots."
            )
        ordered = []
        for key in CANONICAL_VIEW_ORDER:
            img = images[key]
            if not isinstance(img, np.ndarray):
                raise TypeError(f"{key}: expected np.ndarray, got {type(img)}")
            if img.dtype != np.uint8:
                raise TypeError(
                    f"{key}: expected uint8, got {img.dtype}. "
                    "Images must be raw pixels before sending to FlashRT; "
                    "LeRobot's [0,1] normalization and resize happen inside "
                    "the server."
                )
            if img.shape != (_IMG_H, _IMG_W, _IMG_C):
                raise ValueError(
                    f"{key}: expected ({_IMG_H},{_IMG_W},{_IMG_C}), got {img.shape}. "
                    "Apply resize_with_pad before calling predict()."
                )
            ordered.append(img)
        return ordered

    @staticmethod
    def _validate_state(state: np.ndarray) -> None:
        if not isinstance(state, np.ndarray):
            raise TypeError(f"state must be np.ndarray, got {type(state)}")
        if state.dtype != np.float32:
            raise TypeError(
                f"state must be float32, got {state.dtype}. "
                "Pass the NormalizerProcessorStep output, not raw joint angles."
            )
        if state.shape != (_STATE_DIM,):
            raise ValueError(
                f"state must be shape ({_STATE_DIM},), got {state.shape}. "
                "Pad to max_state_dim=32 before calling predict()."
            )

    @staticmethod
    def _encode_image(img: np.ndarray) -> str:
        return base64.b64encode(img.tobytes()).decode("ascii")

    def _post(self, path: str, payload: dict, timeout: float | None = None) -> dict:
        url = f"{self._endpoint}{path}"
        try:
            r = self._session.post(
                url, json=payload, timeout=timeout or self._timeout
            )
        except requests.Timeout as exc:
            raise RuntimeError(
                f"FlashRT server {url} timed out after {timeout or self._timeout}s. "
                "The first call may trigger graph capture — increase timeout_s."
            ) from exc
        if r.status_code == 422:
            raise ValueError(f"FlashRT server validation error: {r.json().get('detail', r.text)}")
        if not r.ok:
            raise RuntimeError(
                f"FlashRT server {url} returned {r.status_code}: {r.text[:200]}"
            )
        return r.json()
