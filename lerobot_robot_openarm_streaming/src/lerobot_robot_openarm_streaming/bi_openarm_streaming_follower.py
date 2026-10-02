"""Bimanual OpenArm follower with its own MIT control loop per arm.

Stock LeRobot sends each arm one MIT command per recorded frame (30 Hz) and the
motor driver holds it until the next one, so the arm moves in 33 ms steps. Here a
background thread per arm sends MIT commands at `control_hz` (250 Hz by default):

* In command mode it ramps between successive send_action() targets and sends the
  ramp's velocity as MIT velocity feed-forward. This is what policies use.
* In stream mode it samples a live teleop source (the OpenArm KER plugin with
  `stream_to_follower=true`) on every tick, so the arm follows the operator at the
  loop rate instead of the recording rate.

Each loop is the only user of its CAN bus once started. Motor states come back in the
replies to its MIT commands, and get_observation() returns the latest of those
instead of polling the bus. Safety handled in the loop on every tick: joint-limit
clipping, speed caps on the commanded target, a slow "approach" phase after connect,
stalls and source changes, and holding position (zero velocity) when the input
goes stale.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any

from lerobot.robots.bi_openarm_follower import BiOpenArmFollower
from lerobot.utils.decorators import check_if_not_connected

from .config_bi_openarm_streaming_follower import BiOpenArmStreamingFollowerConfig
from .streaming import get_target_source

logger = logging.getLogger(__name__)

SIDES = ("left", "right")
MOTOR_INDEX = {"joint_1": 0, "joint_2": 1, "joint_3": 2, "joint_4": 3, "joint_5": 4, "joint_6": 5, "joint_7": 6, "gripper": 7}
STARTUP_TIMEOUT_S = 2.0
BUILTIN_REPLY_WINDOW_S = 0.001  # what DamiaoMotorsBus._mit_control_batch waits by itself
MISSING_REPLY_WARN_S = 0.2
LOG_INTERVAL_S = 2.0


def _gain(value: Any, motor: str) -> float:
    if isinstance(value, (list, tuple)):
        return float(value[MOTOR_INDEX[motor]])
    return float(value)


class _ArmLoop:
    """MIT control loop for one arm. Owns the arm's CAN bus while running."""

    def __init__(self, side: str, arm, cfg: BiOpenArmStreamingFollowerConfig, initial_states: dict):
        self._thread = threading.Thread(target=self._run, name=f"openarm-{side}-{cfg.control_hz:.0f}hz", daemon=True)
        self.side = side
        self.arm = arm
        self.bus = arm.bus
        self.cfg = cfg
        self.period = 1.0 / cfg.control_hz
        window = min(cfg.reply_window_ms / 1e3, 0.7 * self.period)
        self.extra_reply_s = max(0.0, window - BUILTIN_REPLY_WINDOW_S)
        self.motors = list(self.bus.motors)
        self.kp = {m: _gain(arm.config.position_kp, m) for m in self.motors}
        self.kd = {m: _gain(arm.config.position_kd, m) for m in self.motors}
        self.limits = dict(arm.config.joint_limits or {})

        start = {m: float(initial_states[m]["position"]) for m in self.motors}
        self._q_cmd = dict(start)  # last commanded target per motor
        self._lock = threading.Lock()
        self._cmd_prev: tuple[float, dict[str, float]] | None = None
        self._cmd_cur: tuple[float, dict[str, float]] | None = None
        self._states = {m: dict(initial_states[m]) for m in self.motors}
        self._approach = True
        self._approach_requested = False
        self._caught_up_t: float | None = None
        self._kind: str | None = None

        self._stop_event = threading.Event()
        self.error: BaseException | None = None
        self.ticks = 0
        self.overruns = 0
        self.max_work_ms = 0.0
        self.missed_replies = 0
        self._miss_since: dict[str, float | None] = {m: None for m in self.motors}
        self._last_miss_log = 0.0
        self._rate_mark = (0.0, 0)
        self.rate_hz = 0.0

    # ------------------------------------------------------------- main thread
    def set_command(self, positions: dict[str, float], now: float) -> dict[str, float]:
        """Store a send_action() target. Returns it clipped to joint limits.

        A command that arrives after a pause, or that jumps faster than the speed cap
        allows (a new target rather than the next point of a trajectory, e.g. a human
        taking over from a policy), sends the arm into the slow approach phase.
        """
        clipped = {}
        with self._lock:
            base = dict(self._cmd_cur[1]) if self._cmd_cur else dict(self._q_cmd)
            for m in self.motors:
                if m in positions:
                    base[m] = self._clip(m, float(positions[m]))
                clipped[m] = base[m]
            if self._cmd_cur is not None:
                t_cur, q_cur = self._cmd_cur
                interval = now - t_cur
                if interval > self.cfg.command_gap_s:
                    self._approach_requested = True
                else:
                    interval = max(interval, self.period)
                    for m in self.motors:
                        cap = self.cfg.max_gripper_speed_deg_s if m == "gripper" else self.cfg.max_speed_deg_s
                        if abs(base[m] - q_cur[m]) / interval > 1.5 * cap:
                            self._approach_requested = True
                            break
            self._cmd_prev, self._cmd_cur = self._cmd_cur, (now, base)
        return clipped

    def snapshot(self) -> dict[str, dict[str, float]]:
        with self._lock:
            return {m: dict(s) for m, s in self._states.items()}

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    # ------------------------------------------------------------- loop thread
    def _clip(self, motor: str, value: float) -> float:
        lim = self.limits.get(motor)
        return value if lim is None else max(lim[0], min(lim[1], value))

    def _compute_target(self, now: float) -> tuple[dict | None, dict | None, str, dict | None]:
        """(target now, its velocity, source kind, final target to judge catching up against)."""
        cfg = self.cfg
        if cfg.target_source in ("auto", "stream"):
            src = get_target_source()
            sample = src.sample(now) if src is not None else None
            if sample is not None and sample.active:
                pre = f"{self.side}_"
                pos = {m: sample.positions[f"{pre}{m}.pos"] for m in self.motors}
                vel = {m: sample.velocities.get(f"{pre}{m}.pos", 0.0) for m in self.motors}
                return pos, vel, "stream", pos
            if cfg.target_source == "stream":
                return None, None, "stream-idle", None

        with self._lock:
            prev, cur = self._cmd_prev, self._cmd_cur
        if cur is None:
            return None, None, "none", None
        t_cur, q_cur = cur
        zeros = {m: 0.0 for m in self.motors}
        if prev is None or cfg.command_interpolation == "hold":
            return q_cur, zeros, "commands", q_cur
        t_prev, q_prev = prev
        interval = t_cur - t_prev
        if interval > cfg.command_gap_s:
            return q_cur, zeros, "commands", q_cur  # don't ramp from a stale target
        interval = max(interval, self.period)
        u = (now - t_cur) / interval
        if u >= 1.0:
            return q_cur, zeros, "commands", q_cur
        u = max(u, 0.0)
        pos = {m: q_prev[m] + (q_cur[m] - q_prev[m]) * u for m in self.motors}
        vel = {m: (q_cur[m] - q_prev[m]) / interval for m in self.motors}
        return pos, vel, "commands", q_cur

    def _drain_late_replies(self) -> None:
        """Process replies that arrived after the previous tick stopped waiting.

        Left in the socket buffer, they would be taken for this tick's replies and
        the state would stay one tick old from then on.
        """
        bus = self.bus
        while (msg := bus.canbus.recv(timeout=0)) is not None:
            motor = bus._recv_id_to_motor.get(msg.arbitration_id)
            if motor is not None:
                bus._process_response(motor, msg)

    def _step(self, now: float, dt: float) -> None:
        cfg = self.cfg
        target, vel, kind, final = self._compute_target(now)
        with self._lock:
            requested, self._approach_requested = self._approach_requested, False
        if requested and kind == "commands":
            if not self._approach:
                logger.info(f"{self.side} arm: command jumped or resumed after a pause; approaching slowly")
            self._approach = True
        if kind != self._kind:
            if self._kind is not None:
                logger.info(f"{self.side} arm: target source {self._kind} -> {kind}; approaching slowly")
            self._kind = kind
            self._approach = True
        if target is None:
            target, vel = dict(self._q_cmd), {m: 0.0 for m in self.motors}
        if final is None:
            final = target

        dt = min(max(dt, 1e-4), 3 * self.period)
        commands = {}
        within = True
        # After catching up, raise the speed cap gradually so the last few degrees
        # aren't closed in one full-speed burst.
        if self._approach or self._caught_up_t is None:
            ramp = 0.0
        else:
            ramp = min(1.0, (now - self._caught_up_t) / max(cfg.approach_ramp_s, 1e-6))
        for m in self.motors:
            gripper = m == "gripper"
            slow = cfg.approach_gripper_speed_deg_s if gripper else cfg.approach_speed_deg_s
            fast = cfg.max_gripper_speed_deg_s if gripper else cfg.max_speed_deg_s
            speed = slow if self._approach else slow + (fast - slow) * ramp
            goal = self._clip(m, float(target[m]))
            prev = self._q_cmd[m]
            delta = goal - prev
            step = speed * dt
            if abs(delta) > step:
                q = prev + math.copysign(step, delta)
                v = math.copysign(speed, delta)
            else:
                q = goal
                v = max(-speed, min(speed, float(vel.get(m, 0.0))))
            if abs(self._clip(m, float(final[m])) - q) > cfg.approach_tolerance_deg:
                within = False
            use_vff = cfg.gripper_velocity_feedforward if gripper else cfg.velocity_feedforward
            self._q_cmd[m] = q
            commands[m] = (self.kp[m], self.kd[m], q, v if use_vff else 0.0, 0.0)
        if self._approach and within:
            self._approach = False
            self._caught_up_t = now
            logger.info(f"{self.side} arm: caught up with target, ramping to full speed")

        self._drain_late_replies()
        before = {m: self.bus._last_known_states[m] for m in self.motors}
        self.bus._mit_control_batch(commands)
        states = self.bus._last_known_states
        if self.extra_reply_s > 0:
            # Keep waiting for the motors that haven't replied yet (see reply_window_ms).
            late = {self.bus._get_motor_recv_id(m): m for m in self.motors if states[m] is before[m]}
            if late:
                for recv_id, msg in self.bus._recv_all_responses(list(late), timeout=self.extra_reply_s).items():
                    self.bus._process_response(late[recv_id], msg)
        for m in self.motors:
            if states[m] is before[m]:  # no reply decoded this tick
                self.missed_replies += 1
                if self._miss_since[m] is None:
                    self._miss_since[m] = now
            else:
                self._miss_since[m] = None
        with self._lock:
            self._states = {m: dict(states[m]) for m in self.motors}
        stuck = [m for m, t in self._miss_since.items() if t is not None and now - t > MISSING_REPLY_WARN_S]
        if stuck and now - self._last_miss_log > LOG_INTERVAL_S:
            self._last_miss_log = now
            logger.warning(f"{self.side} arm: no replies from {stuck} for >{MISSING_REPLY_WARN_S}s; their state is stale")

    def _run(self) -> None:
        next_t = time.monotonic()
        last_t = next_t
        try:
            while not self._stop_event.is_set():
                now = time.monotonic()
                work_start = now
                self._step(now, now - last_t)
                last_t = now
                self.ticks += 1
                done = time.monotonic()
                self.max_work_ms = max(self.max_work_ms, (done - work_start) * 1e3)
                mark_t, mark_n = self._rate_mark
                if done - mark_t >= 1.0:
                    if mark_t:
                        self.rate_hz = (self.ticks - mark_n) / (done - mark_t)
                    self._rate_mark = (done, self.ticks)
                next_t += self.period
                if done > next_t + 0.5 * self.period:
                    self.overruns += 1
                    next_t = done  # fell behind: don't try to catch up
                else:
                    time.sleep(max(0.0, next_t - time.monotonic()))
        except BaseException as e:  # surfaced to the main thread on its next call
            self.error = e
            logger.exception(f"{self.side} arm control loop stopped")

    def stats(self) -> dict[str, Any]:
        return {
            "rate_hz": round(self.rate_hz, 1),
            "ticks": self.ticks,
            "overruns": self.overruns,
            "max_work_ms": round(self.max_work_ms, 2),
            "missed_replies": self.missed_replies,
            "approaching": self._approach,
            "source": self._kind,
        }


class BiOpenArmStreamingFollower(BiOpenArmFollower):
    """bi_openarm_follower with a background MIT control loop per arm."""

    config_class = BiOpenArmStreamingFollowerConfig
    # Datasets record robot.name as robot_type; keep it identical to the stock
    # follower so new data merges with existing bi_openarm_follower datasets.
    name = "bi_openarm_follower"

    def __init__(self, config: BiOpenArmStreamingFollowerConfig):
        super().__init__(config)
        self.config = config
        if config.target_source not in ("auto", "commands", "stream"):
            raise ValueError(f"target_source must be auto, commands or stream, got {config.target_source!r}")
        if config.command_interpolation not in ("interpolate", "hold"):
            raise ValueError(f"command_interpolation must be interpolate or hold, got {config.command_interpolation!r}")
        if not 20.0 <= config.control_hz <= 1000.0:
            raise ValueError("control_hz must be between 20 and 1000")
        if config.reply_window_ms < 1.0:
            raise ValueError("reply_window_ms must be at least 1 (LeRobot's built-in wait)")
        if config.reply_window_ms / 1e3 > 0.7 / config.control_hz:
            logger.warning(
                f"reply_window_ms={config.reply_window_ms} is capped at 70% of the control period "
                f"({700.0 / config.control_hz:.2f} ms at {config.control_hz:.0f} Hz)"
            )
        for name in ("max_speed_deg_s", "max_gripper_speed_deg_s", "approach_speed_deg_s", "approach_gripper_speed_deg_s"):
            if getattr(config, name) <= 0:
                raise ValueError(f"{name} must be positive")
        self._loops: dict[str, _ArmLoop] = {}
        self._warned_custom_gains = False

    def _arms(self):
        return (("left", self.left_arm), ("right", self.right_arm))

    def connect(self, calibrate: bool = True) -> None:
        super().connect(calibrate)
        try:
            self._start_loops()
        except BaseException:
            self._stop_loops()
            super().disconnect()
            raise

    def _start_loops(self) -> None:
        for side, arm in self._arms():
            if arm.config.max_relative_target is not None:
                logger.warning(
                    f"{side} arm: max_relative_target is ignored by bi_openarm_streaming_follower; "
                    "use max_speed_deg_s / approach_speed_deg_s instead"
                )
            # Fresh states (the bus handshake leaves placeholder values in its cache).
            states = arm.bus.sync_read_all_states()
            self._loops[side] = _ArmLoop(side, arm, self.config, states)
        for loop in self._loops.values():
            loop.start()
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            self._raise_if_failed()
            if all(loop.ticks >= 5 for loop in self._loops.values()):
                logger.info(f"{self} control loops running at {self.config.control_hz:.0f} Hz")
                return
            time.sleep(0.01)
        raise RuntimeError("Control loops did not start in time")

    def _stop_loops(self) -> None:
        for loop in self._loops.values():
            loop.stop()
        for side, loop in self._loops.items():
            if loop.is_alive():
                loop.join(timeout=1.0)
            logger.info(f"{side} arm control loop stats: {loop.stats()}")
        self._loops = {}

    def _raise_if_failed(self) -> None:
        for side, loop in self._loops.items():
            if loop.error is not None:
                raise RuntimeError(f"{side} arm control loop stopped: {loop.error!r}") from loop.error

    def streaming_stats(self) -> dict[str, dict[str, Any]]:
        return {side: loop.stats() for side, loop in self._loops.items()}

    @check_if_not_connected
    def get_observation(self) -> dict[str, Any]:
        self._raise_if_failed()
        obs: dict[str, Any] = {}
        for side, arm in self._arms():
            states = self._loops[side].snapshot()
            for motor in arm.bus.motors:
                st = states[motor]
                obs[f"{side}_{motor}.pos"] = st["position"]
                if arm.config.use_velocity_and_torque:
                    obs[f"{side}_{motor}.vel"] = st["velocity"]
                    obs[f"{side}_{motor}.torque"] = st["torque"]
            for cam_key, cam in arm.cameras.items():
                key = cam_key if (side == "left" and cam_key in self._top_level_cam_keys) else f"{side}_{cam_key}"
                if getattr(cam, "use_rgb", True):
                    obs[key] = cam.read_latest()
                if getattr(cam, "use_depth", False):
                    obs[f"{key}_depth"] = cam.read_latest_depth()
        return obs

    @check_if_not_connected
    def send_action(self, action: dict[str, Any], custom_kp=None, custom_kd=None) -> dict[str, Any]:
        self._raise_if_failed()
        if (custom_kp or custom_kd) and not self._warned_custom_gains:
            self._warned_custom_gains = True
            logger.warning("custom_kp/custom_kd are ignored; set position_kp/position_kd in the arm configs")
        now = time.monotonic()
        sent: dict[str, Any] = {}
        for side, _arm in self._arms():
            prefix = f"{side}_"
            positions = {
                key[len(prefix):].removesuffix(".pos"): value
                for key, value in action.items()
                if key.startswith(prefix) and key.endswith(".pos")
            }
            for motor, value in self._loops[side].set_command(positions, now).items():
                sent[f"{prefix}{motor}.pos"] = value
        return sent

    def disconnect(self) -> None:
        self._stop_loops()
        super().disconnect()
