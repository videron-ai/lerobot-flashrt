"""OpenArm KER as a bimanual LeRobot teleoperator.

The KER is one USB device carrying both arms (16 channels), so this is a single
teleoperator whose action keys match `bi_openarm_follower`:
`left_joint_1.pos` ... `left_gripper.pos`, `right_joint_1.pos` ... `right_gripper.pos`,
all in degrees.

Stock KER channel layout (channel = encoder module ID, as shown by openarm-ker-cli):
    CH1-7   right J1-J7     CH8   right gripper trigger
    CH9-15  left  J1-J7     CH16  left  gripper trigger

By default the plugin drives an OpenArm 1.0 follower from an unmodified (2.0-order)
KER in software: the J6/J7 channels are swapped (follower J6 <- CH7/CH15, J7 <-
CH6/CH14) and the wrist is re-solved (see _wrist_v2_to_v1). A mapping JSON with
"wrist_remap": {"mode": "none"} turns that off and restores the stock layout; its
"channels" can then remap individual joints (e.g. after physically swapping the
wrist encoder modules without reflashing their IDs).
"""

import json
import logging
import math
import sys
import threading
import time
from collections import deque
from functools import lru_cache
from typing import Any

import numpy as np

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from openarm_ker.ker_stream import CMD_STANDBY, CMD_STREAM, KERStream

from .config_openarm_ker import OpenArmKERConfig
from .park import DoubleSqueeze, ParkController

logger = logging.getLogger(__name__)

JOINTS = [f"joint_{i}" for i in range(1, 8)]
SIDES = ("left", "right")
CHANNEL_BASE = {"right": 0, "left": 8}
GRIPPER_CHANNEL = {"right": 7, "left": 15}
# Trigger direction as the KER reports it: the right trigger goes negative when
# squeezed (mech range -90..5.73), the left goes positive (-5.73..60).
DEFAULT_GRIPPER_DIRECTION = {"right": -1.0, "left": 1.0}

# URDF-derived default for the software wrist remap (see _wrist_v2_to_v1): +1 if the
# follower's positive J7 rotation is about (J5 axis x J6 axis), else -1. In the v1.0
# URDF that is +1 for the right arm and -1 for the left.
DEFAULT_WRIST_HANDEDNESS = {"right": 1.0, "left": -1.0}
DEFAULT_WRIST_REMAP = "v2_to_v1"

FIRST_FRAME_TIMEOUT_S = 2.0
STALL_LOG_INTERVAL_S = 1.0


def _rx(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rz(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _n6_max(a7: float, sin_lim: float) -> float:
    """Largest |n6| (rad) that keeps the follower's |p6| <= limit at this n7 (rad)."""
    c = math.cos(a7)
    return math.pi / 2 if c <= sin_lim else math.asin(sin_lim / c)


def _n6_max_slope(a7: float, sin_lim: float) -> float:
    c, s = math.cos(a7), math.sin(a7)
    x = sin_lim / c
    return (sin_lim * s / (c * c)) / math.sqrt(max(1e-12, 1.0 - x * x))


@lru_cache(maxsize=32)
def _taper_knee(j6_limit_deg: float, taper_slope: float) -> tuple[float, float]:
    """|n7| (rad) where the exact bound gets steeper than taper_slope, and the bound there."""
    sin_lim = math.sin(math.radians(j6_limit_deg))
    lo, hi = 0.0, math.acos(sin_lim) - 1e-9
    for _ in range(60):
        mid = (lo + hi) / 2
        if _n6_max_slope(mid, sin_lim) < taper_slope:
            lo = mid
        else:
            hi = mid
    return lo, _n6_max(lo, sin_lim)


def _n6_bound(a7: float, j6_limit_deg: float, taper_slope: float) -> float:
    """Saturation bound for |n6| (rad): exact limit, tapered where it would get steep.

    The exact bound rises steeply as n7 approaches the KER J6 stop. Clamping to it
    there turns a small n7 motion into a large n6 change, i.e. the follower hand
    rotates several times as much as the KER (up to ~5x at a 44 deg limit). Past the
    point where its slope reaches taper_slope, the bound continues as a straight
    line instead. That line lies below the exact bound (it is convex), so the
    follower stays within its limit, just a few degrees short of it near the stop.
    """
    knee, bound_at_knee = _taper_knee(j6_limit_deg, taper_slope)
    a = abs(a7)
    if a <= knee:
        return _n6_max(a, math.sin(math.radians(j6_limit_deg)))
    return min(math.pi / 2, bound_at_knee + taper_slope * (a - knee))


def _wrist_v2_to_v1(
    n5: float, n6: float, n7: float, h: float, j6_limit_deg: float, taper_slope: float = 1.0
) -> tuple[float, float, float]:
    """Re-solve the wrist so an OpenArm 1.0 follower matches an unmodified (2.0-order) KER.

    Inputs are already in follower joint space (degrees) after channel swap, signs and
    offsets: n5 = forearm roll, n6 = rotation about the follower's J6 axis (measured by
    the KER's J7), n7 = rotation about the follower's J7 axis (measured by the KER's J6).
    The KER applies those two rotations in the opposite order to the follower, so:

        KER:      R = Rz(n5) Ry(h*n7) Rx(n6)
        follower: R = Rz(p5) Rx(p6) Ry(h*p7)      -> solve for p5, p6, p7

    The result is exact whenever it is within the follower's limits, so the follower
    hand rotates exactly as the KER hand does. Single-axis motions (n6 == 0 or
    n7 == 0) map straight through unchanged.

    The follower's J6 can't reach the KER's full +-90 on that axis, and the solution is
    singular at 90, so n6 is saturated (see _n6_bound) to keep |p6| <= j6_limit_deg.
    The output stays continuous, and while saturating, the follower hand rotates at
    most ~1.7x as much as the KER's (taper_slope 1.0); elsewhere it is exactly 1x.
    """
    a5, a6, a7 = math.radians(n5), math.radians(n6), math.radians(n7)
    bound = _n6_bound(a7, j6_limit_deg, taper_slope)
    if abs(a6) > bound:
        a6 = math.copysign(bound, a6)

    r = _rz(a5) @ _ry(h * a7) @ _rx(a6)
    p6 = math.asin(max(-1.0, min(1.0, r[2, 1])))
    p5 = math.atan2(-r[0, 1], r[1, 1])
    p7 = h * math.atan2(-r[2, 0], r[2, 2])
    return math.degrees(p5), math.degrees(p6), math.degrees(p7)


def _default_channels(wrist_remap: str) -> dict[str, int]:
    """1-based channel numbers (= encoder module IDs) for every joint and gripper."""
    channels = {f"{s}_{j}": CHANNEL_BASE[s] + i + 1 for s in SIDES for i, j in enumerate(JOINTS)}
    channels.update({f"{s}_gripper": GRIPPER_CHANNEL[s] + 1 for s in SIDES})
    if wrist_remap == "v2_to_v1":
        # Pair each KER wrist channel with the follower axis that points the same way.
        for s in SIDES:
            channels[f"{s}_joint_6"], channels[f"{s}_joint_7"] = channels[f"{s}_joint_7"], channels[f"{s}_joint_6"]
    return channels


class OneEuroFilter:
    """Speed-adaptive low-pass filter over a vector of angles (Casiez et al., CHI 2012).

    The cutoff frequency rises with the (smoothed) speed of each channel, so a nearly
    still hand gets heavy smoothing and a moving one gets little lag. Units: degrees,
    seconds; beta is in Hz per deg/s.
    """

    def __init__(self, min_cutoff_hz: float, beta: float, d_cutoff_hz: float = 1.0):
        if min_cutoff_hz <= 0 or d_cutoff_hz <= 0 or beta < 0:
            raise ValueError("one_euro filter needs min_cutoff_hz > 0, d_cutoff_hz > 0, beta >= 0")
        self.min_cutoff = min_cutoff_hz
        self.beta = beta
        self.d_cutoff = d_cutoff_hz
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None

    @staticmethod
    def _alpha(cutoff, dt: float):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._x = self._dx = None

    def __call__(self, x: np.ndarray, dt: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x is None or self._x.shape != x.shape:
            self._x, self._dx = x.copy(), np.zeros_like(x)
            return self._x.copy()
        dt = max(dt, 1e-3)
        dx = (x - self._x) / dt
        self._dx = self._dx + self._alpha(self.d_cutoff, dt) * (dx - self._dx)
        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        self._x = self._x + self._alpha(cutoff, dt) * (x - self._x)
        return self._x.copy()


class OpenArmKER(Teleoperator):
    """Motorless OpenArm KER leader arm (both arms, one device)."""

    config_class = OpenArmKERConfig
    name = "openarm_ker"

    def __init__(self, config: OpenArmKERConfig):
        super().__init__(config)
        self.config = config
        if config.on_stall not in ("hold", "raise"):
            raise ValueError(f"on_stall must be 'hold' or 'raise', got {config.on_stall!r}")

        self._stream: KERStream | None = None
        self._last_marker: Any = None
        self._last_advance_t = 0.0
        self._last_action: dict[str, float] | None = None
        self._last_stall_log_t = 0.0
        self._warned_channels: set[int] = set()

        self._wrist_remap = DEFAULT_WRIST_REMAP
        self._channels = _default_channels(self._wrist_remap)
        self._signs = {f"{s}_{j}": 1.0 for s in SIDES for j in JOINTS}
        self._offsets = {f"{s}_{j}": 0.0 for s in SIDES for j in JOINTS}
        self._gripper_dir = dict(DEFAULT_GRIPPER_DIRECTION)
        self._wrist_handedness = dict(DEFAULT_WRIST_HANDEDNESS)
        self._wrist_j6_limit_deg = 40.0
        self._wrist_taper_slope = 1.0
        # Park poses in follower joint space (action keys). Rest is the calibration pose.
        self._park_rest = {k: 0.0 for k in self.action_features}
        self._park_ready_path: list[dict[str, float]] = []
        if config.mapping_path is not None:
            self._load_mapping(config.mapping_path)

        self._park: ParkController | None = None
        self._park_listener: threading.Thread | None = None
        if config.park:
            if not self._park_ready_path:
                raise ValueError('park=true needs a "park": {"ready_path": [...]} block in the mapping file')
            self._park = ParkController(
                self._park_rest,
                self._park_ready_path,
                config.park_speed_deg_s,
                config.park_engage_tolerance_deg,
                config.park_engage_hold_s,
                config.park_engage_joints,
                config.park_go_delay_s,
            )
        self._park_gesture: DoubleSqueeze | None = None
        if config.park_gesture:
            if self._park is None:
                raise ValueError("park_gesture=true needs park=true")
            self._park_gesture = DoubleSqueeze(config.park_gesture_window_s)

        if config.filter == "none":
            self._filter: OneEuroFilter | None = None
        elif config.filter == "one_euro":
            self._filter = OneEuroFilter(
                config.filter_min_cutoff_hz, config.filter_beta, config.filter_d_cutoff_hz
            )
        else:
            raise ValueError(f"filter must be 'none' or 'one_euro', got {config.filter!r}")
        self._last_filter_t: float | None = None
        self._source: _KERTargetSource | None = None

    def _load_mapping(self, path) -> None:
        with open(path, encoding="utf-8") as f:
            mapping = json.load(f)
        for key, value in mapping.get("joint_signs", {}).items():
            if key not in self._signs:
                raise KeyError(f"Unknown joint '{key}' in joint_signs (expected e.g. 'right_joint_3')")
            if value not in (1, -1, 1.0, -1.0):
                raise ValueError(f"joint_signs['{key}'] must be 1 or -1, got {value}")
            self._signs[key] = float(value)
        for key, value in mapping.get("joint_offsets", {}).items():
            if key not in self._offsets:
                raise KeyError(f"Unknown joint '{key}' in joint_offsets (expected e.g. 'left_joint_1')")
            self._offsets[key] = float(value)
        wrist = mapping.get("wrist_remap")
        if wrist is not None:
            mode = wrist.get("mode", DEFAULT_WRIST_REMAP)
            if mode not in ("none", "v2_to_v1"):
                raise ValueError(f"wrist_remap.mode must be 'none' or 'v2_to_v1', got {mode!r}")
            self._wrist_remap = mode
            self._channels = _default_channels(mode)  # the mode picks the base layout
            for side, value in wrist.get("handedness", {}).items():
                if side not in self._wrist_handedness or value not in (1, -1):
                    raise ValueError(f"wrist_remap.handedness must map 'left'/'right' to 1 or -1, got {side}: {value}")
                self._wrist_handedness[side] = float(value)
            self._wrist_j6_limit_deg = float(wrist.get("j6_limit_deg", self._wrist_j6_limit_deg))
            if not 0.0 < self._wrist_j6_limit_deg < 45.0:
                raise ValueError("wrist_remap.j6_limit_deg must be between 0 and 45")
            self._wrist_taper_slope = float(wrist.get("taper_slope", self._wrist_taper_slope))
            if not 0.2 <= self._wrist_taper_slope <= 5.0:
                raise ValueError("wrist_remap.taper_slope must be between 0.2 and 5")
        for key, value in mapping.get("channels", {}).items():
            if key not in self._channels:
                raise KeyError(f"Unknown joint '{key}' in channels (expected e.g. 'right_joint_6', 'left_gripper')")
            if not isinstance(value, int) or not 1 <= value <= 16:
                raise ValueError(f"channels['{key}'] must be an integer 1-16, got {value!r}")
            self._channels[key] = value
        used = sorted(self._channels.values())
        if used != list(range(1, 17)):
            raise ValueError(f"channels must use each of CH1-CH16 exactly once, got {used}")
        for side, value in mapping.get("gripper_direction", {}).items():
            if side not in self._gripper_dir:
                raise KeyError(f"gripper_direction keys must be 'left'/'right', got '{side}'")
            self._gripper_dir[side] = float(value)
        park = mapping.get("park")
        if park is not None:
            self._park_rest = self._park_pose(park.get("rest", {}), self._park_rest, "park.rest")
            pose = self._park_rest
            self._park_ready_path = []
            for i, waypoint in enumerate(park.get("ready_path", [])):
                # Joints a waypoint doesn't mention keep their value from the previous one.
                pose = self._park_pose(waypoint, pose, f"park.ready_path[{i}]")
                self._park_ready_path.append(pose)

    @staticmethod
    def _park_pose(values: dict, base: dict[str, float], where: str) -> dict[str, float]:
        pose = dict(base)
        for key, value in values.items():
            if f"{key}.pos" not in pose:
                raise KeyError(f"Unknown joint '{key}' in {where} (expected e.g. 'left_joint_4', 'right_gripper')")
            pose[f"{key}.pos"] = float(value)
        return pose

    # ------------------------------------------------------------------ features
    @property
    def action_features(self) -> dict[str, type]:
        return {f"{s}_{m}.pos": float for s in SIDES for m in (*JOINTS, "gripper")}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}  # motorless: nothing to feed back

    @property
    def is_connected(self) -> bool:
        return self._stream is not None and self._stream.is_connected

    @property
    def is_calibrated(self) -> bool:
        # Zeroing lives on the M5 (jig + "Zero Reset" on the touchscreen, stored in NVS).
        return True

    # ---------------------------------------------------------------- lifecycle
    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        c = self.config
        self._stream = KERStream(transport=c.transport, port=c.port, baud=c.baud)
        self._stream.connect()  # pings and fetches the field schema
        logger.info(f"KER metadata: {self._stream.metadata}")

        # The M5 boots in STANDBY; START also clears a latched jump-detection stop.
        self._stream.send_command(CMD_STREAM)

        deadline = time.monotonic() + FIRST_FRAME_TIMEOUT_S
        while self._stream.latest() is None:
            if time.monotonic() > deadline:
                self.disconnect()
                raise TimeoutError(
                    "Connected to the KER but no frames arrived. Check the M5 screen "
                    "(it should say STREAMING) and run `openarm-ker-cli stream`."
                )
            time.sleep(0.01)
        self._last_advance_t = time.monotonic()
        if c.stream_to_follower:
            try:
                from lerobot_robot_openarm_streaming.streaming import register_target_source
            except ImportError as e:
                self.disconnect()
                raise ImportError(
                    "stream_to_follower=true needs the lerobot_robot_openarm_streaming plugin installed"
                ) from e
            self._source = _KERTargetSource(self)
            register_target_source(self._source)
            logger.info("Streaming KER targets directly to the follower's control loop")
        self.configure()
        logger.info(f"{self} connected.")

    def calibrate(self) -> None:
        logger.info(
            "OpenArm KER is zeroed on the device: mount it in the calibration jig and use "
            "'Zero Reset' on the M5 touchscreen. Nothing to do here."
        )

    def configure(self) -> None:
        pass

    # --------------------------------------------------------------------- park
    @property
    def park_state(self) -> str | None:
        """rest, to_ready, ready, countdown, to_rest, engaging or teleop; None when parking is off."""
        return self._park.state if self._park is not None else None

    def park_command(self, command: str) -> None:
        """Same as typing it in the terminal: "ready", "rest", "go" or "toggle" (bare ENTER)."""
        if self._park is None:
            raise RuntimeError("Parking is off; start with --teleop.park=true")
        self._park.request(command)

    def _start_park_listener(self) -> None:
        # Started on the first get_action(), after the robot's connect() prompts are done.
        if sys.stdin is None or not sys.stdin.isatty():
            logger.warning("park: no interactive terminal for typed commands; call park_command() instead")
            return

        def listen() -> None:
            for line in sys.stdin:
                word = line.strip().lower() or "toggle"
                if word in ("ready", "rest", "go", "toggle"):
                    self._park.request(word)
                else:
                    logger.info(f"park: unknown command '{word}'. Type ready, go or rest, then ENTER.")

        self._park_listener = threading.Thread(target=listen, name="ker-park-commands", daemon=True)
        self._park_listener.start()
        logger.info(
            "park: holding the rest pose. Type 'ready' + ENTER to move the followers above the "
            "work surface, 'go' + ENTER to take over with the KER, 'rest' + ENTER to bring them back down."
        )

    def _apply_park(self, action: dict[str, float], now: float) -> dict[str, float]:
        if self._park is None:
            return action
        if self._park_gesture is not None:
            # The triggers are read from the mapped KER pose, so the gesture also works
            # while the followers are parked and ignoring the KER.
            c = self.config
            span = c.gripper_closed_deg - c.gripper_open_deg
            squeeze = {s: (action[f"{s}_gripper.pos"] - c.gripper_open_deg) / span for s in SIDES}
            if self._park_gesture.update(squeeze["left"], squeeze["right"], now):
                logger.info("park: double squeeze")
                self._park.request("toggle")
        return self._park.step(action, now)

    # ------------------------------------------------------------------- action
    @staticmethod
    def _raw_angles(data: dict[str, Any]) -> np.ndarray:
        angles = data.get("angles")
        if angles is None and data.get("angles_cd") is not None:  # compact firmware build
            angles = [v / 100.0 for v in data["angles_cd"]]
        if angles is None or len(angles) < 16:
            raise RuntimeError(f"Unexpected KER frame layout: fields {list(data)}")
        return np.asarray(angles[:16], dtype=np.float64)

    def _map(self, data: dict[str, Any], now: float | None = None) -> dict[str, float]:
        """Filter (at the caller's rate) and map one KER frame. Used when not streaming."""
        angles = self._raw_angles(data)
        if self._filter is not None:
            now = time.monotonic() if now is None else now
            dt = 0.0 if self._last_filter_t is None else now - self._last_filter_t
            self._last_filter_t = now
            angles = self._filter(angles, dt)
        return self._map_angles(angles)

    def _map_angles(self, angles: np.ndarray) -> dict[str, float]:
        c = self.config
        action: dict[str, float] = {}
        for side in SIDES:
            for joint in JOINTS:
                key = f"{side}_{joint}"
                raw = float(angles[self._channels[key] - 1])
                action[f"{key}.pos"] = self._signs[key] * raw + self._offsets[key]

            if self._wrist_remap == "v2_to_v1":
                p5, p6, p7 = _wrist_v2_to_v1(
                    action[f"{side}_joint_5.pos"],
                    action[f"{side}_joint_6.pos"],
                    action[f"{side}_joint_7.pos"],
                    self._wrist_handedness[side],
                    self._wrist_j6_limit_deg,
                    self._wrist_taper_slope,
                )
                action[f"{side}_joint_5.pos"] = p5
                action[f"{side}_joint_6.pos"] = p6
                action[f"{side}_joint_7.pos"] = p7

            raw = float(angles[self._channels[f"{side}_gripper"] - 1])
            squeeze = min(max(raw * self._gripper_dir[side] / c.gripper_travel_deg, 0.0), 1.0)
            action[f"{side}_gripper.pos"] = (
                c.gripper_open_deg + (c.gripper_closed_deg - c.gripper_open_deg) * squeeze
            )
        return action

    def _check_errors(self, data: dict[str, Any]) -> None:
        mask = data.get("error_mask")
        if mask is None:
            mask = sum(1 << i for i, e in enumerate(data.get("errors") or []) if e)
        for ch in range(16):
            if mask >> ch & 1 and ch not in self._warned_channels:
                self._warned_channels.add(ch)
                logger.warning(f"KER channel {ch + 1} reports no data; its angle is frozen.")

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        now = time.monotonic()
        if self._park is not None and self._park_listener is None:
            self._park_listener = threading.current_thread()  # placeholder if no listener starts
            self._start_park_listener()
        if self._source is not None:
            # Streaming: the follower's control loop samples the same source at its
            # own rate; this returns (and records) the same target it is tracking.
            self._source.note_poll(now)
            sample = self._source.sample(now)
            self._last_advance_t = self._source.last_advance_t
            if sample is not None and (now - self._last_advance_t) <= self.config.stale_timeout_s:
                self._last_action = dict(sample.positions)
                return dict(self._last_action)
            return self._stalled(now)

        data = self._stream.latest()

        # Frame counter ("seq") proves the stream is live; fall back to the timestamp.
        marker = data.get("seq", data.get("timestamp"))
        if marker != self._last_marker:
            self._last_marker = marker
            self._last_advance_t = now

        if (now - self._last_advance_t) > self.config.stale_timeout_s:
            return self._stalled(now)

        self._check_errors(data)
        action = self._apply_park(self._map(data, now), now)
        self._last_action = action
        return action

    def _stalled(self, now: float) -> dict[str, float]:
        msg = (
            f"KER stream stalled for {now - self._last_advance_t:.1f}s "
            f"(link_up={self._stream.is_link_up}). Likely jump detection or STOP on the M5; "
            "fix the cause and press START on the M5 to resume."
        )
        if self.config.on_stall == "raise" or self._last_action is None:
            raise RuntimeError(msg)
        if now - self._last_stall_log_t > STALL_LOG_INTERVAL_S:
            self._last_stall_log_t = now
            logger.warning(msg + " Holding last action; re-record this episode.")
        return dict(self._last_action)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass  # no actuators on the KER

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._source is not None:
            from lerobot_robot_openarm_streaming.streaming import unregister_target_source

            unregister_target_source(self._source)
            self._source = None
        try:
            self._stream.send_command(CMD_STANDBY)
        except Exception:
            pass
        self._stream.close()
        self._stream = None
        logger.info(f"{self} disconnected.")


class _KERTargetSource:
    """Live target stream for the streaming follower (lerobot_robot_openarm_streaming).

    Every new KER frame (500 Hz) is filtered and mapped once, whoever asks first;
    the follower's per-arm control loops and get_action() all read the same result.
    Velocities for MIT feed-forward are the low-passed derivative of the mapped
    targets, so they include the wrist remap and gripper mapping.

    The stream only counts as active while get_action() is being polled (two calls
    within ACTIVE_WINDOW_S). A policy rollout with this teleop attached for
    interventions therefore keeps control until the human takes over.
    """

    VELOCITY_CUTOFF_HZ = 15.0
    ACTIVE_WINDOW_S = 0.5

    def __init__(self, teleop: "OpenArmKER"):
        self._t = teleop
        self._lock = threading.Lock()
        self._marker: Any = None
        self._last_update: float | None = None
        self.last_advance_t = time.monotonic()
        self._keys: list[str] | None = None
        self._pos: np.ndarray | None = None
        self._vel: np.ndarray | None = None
        self._polls: deque[float] = deque(maxlen=2)

    def note_poll(self, now: float) -> None:
        with self._lock:
            self._polls.append(now)

    def _polled(self, now: float) -> bool:
        p = self._polls
        return len(p) == 2 and now - p[-1] <= self.ACTIVE_WINDOW_S and p[-1] - p[0] <= self.ACTIVE_WINDOW_S

    def sample(self, now: float):
        from lerobot_robot_openarm_streaming.streaming import StreamSample

        with self._lock:
            stream = self._t._stream
            data = stream.latest() if stream is not None else None
            if data is not None:
                marker = data.get("seq", data.get("timestamp"))
                if marker != self._marker:
                    self._marker = marker
                    self.last_advance_t = now
                    dt = 0.0 if self._last_update is None else now - self._last_update
                    self._last_update = now
                    self._t._check_errors(data)
                    angles = self._t._raw_angles(data)
                    if self._t._filter is not None:
                        angles = self._t._filter(angles, dt)
                    mapped = self._t._apply_park(self._t._map_angles(angles), now)
                    if self._keys is None:
                        self._keys = list(mapped)
                    pos = np.array([mapped[k] for k in self._keys])
                    if self._pos is None or dt <= 0.0:
                        self._vel = np.zeros_like(pos)
                    else:
                        raw_vel = (pos - self._pos) / dt
                        alpha = dt / (dt + 1.0 / (2.0 * math.pi * self.VELOCITY_CUTOFF_HZ))
                        self._vel = self._vel + alpha * (raw_vel - self._vel)
                    self._pos = pos
            if self._pos is None:
                return None
            fresh = (now - self.last_advance_t) <= self._t.config.stale_timeout_s
            return StreamSample(
                positions=dict(zip(self._keys, self._pos.tolist())),
                velocities=dict(zip(self._keys, self._vel.tolist())),
                active=fresh and self._polled(now),
                stamp=self._last_update,
            )
