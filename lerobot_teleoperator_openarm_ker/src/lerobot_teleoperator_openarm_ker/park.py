"""Park controller: drives the followers between a rest pose and a ready pose without the KER.

The teleoperator normally outputs the mapped KER pose. With parking enabled it starts
by holding the rest pose (the followers' calibration pose: arms hanging) and ignores
the KER. The "ready" command then plays a fixed joint-space path to the ready pose
above the work surface and holds there. Control passes to the KER once the operator has
brought its shoulder and elbow joints roughly to the ready pose, or on the "go" command
after a short countdown; either way the followers close the remaining gap at the park
speed instead of jumping. The "rest" command plays the path backwards to the rest pose.

    rest --ready--> to_ready --> ready --KER matches--> engaging --> teleop
                                 ready --go--> countdown --> engaging --> teleop
    ready --rest--> to_rest --> rest
    teleop --ready--> to_ready (direct)       teleop --rest--> to_rest (via ready)

"toggle" picks the usual next move: ready from rest or KER control, rest from ready.

Everything is in follower joint space (degrees, LeRobot action keys), after the KER
mapping, so the same output feeds get_action() and the streaming follower.
"""

import logging
import math
import threading

logger = logging.getLogger(__name__)

MIN_SEGMENT_S = 0.2
ENGAGED_TOLERANCE_DEG = 1.0
GRIPPER_SPEED_SCALE = 4.0  # grippers travel 160 deg, so they get a higher speed limit
HINT_INTERVAL_S = 2.0
MAX_DT_S = 0.1


class DoubleSqueeze:
    """Detects both gripper triggers squeezed together twice in quick succession.

    A press counts when both triggers go past `press` after both were below `release`,
    so holding one gripper closed and tapping the other never counts, and neither does
    one long squeeze. Two presses within `window_s` fire once; the detector then ignores
    presses for `refractory_s`.
    """

    def __init__(self, window_s: float, press: float = 0.7, release: float = 0.3, refractory_s: float = 1.5):
        if window_s <= 0:
            raise ValueError("park_gesture_window_s must be positive")
        self.window_s = window_s
        self.press = press
        self.release = release
        self.refractory_s = refractory_s
        self._armed = False  # both triggers have been released since the last press
        self._last_press_t: float | None = None
        self._quiet_until = 0.0

    def update(self, left: float, right: float, now: float) -> bool:
        """Feed the two squeeze values (0 released .. 1 squeezed). True when the gesture completes."""
        if left < self.release and right < self.release:
            self._armed = True
            return False
        if not (self._armed and left > self.press and right > self.press):
            return False
        self._armed = False
        if now < self._quiet_until:
            return False
        if self._last_press_t is not None and now - self._last_press_t <= self.window_s:
            self._last_press_t = None
            self._quiet_until = now + self.refractory_s
            return True
        self._last_press_t = now
        return False


class ParkController:
    def __init__(
        self,
        rest: dict[str, float],
        ready_path: list[dict[str, float]],
        speed_deg_s: float,
        engage_tolerance_deg: float,
        engage_hold_s: float,
        engage_joints: list[int],
        go_delay_s: float,
    ):
        if not ready_path:
            raise ValueError("park needs at least one ready_path waypoint")
        if speed_deg_s <= 0 or engage_tolerance_deg <= 0 or engage_hold_s < 0:
            raise ValueError("park speed and engage tolerance must be positive")
        self.rest = dict(rest)
        self.ready_path = [dict(w) for w in ready_path]
        self.ready = self.ready_path[-1]
        self.speed = speed_deg_s
        self.engage_tolerance = engage_tolerance_deg
        self.engage_hold_s = engage_hold_s
        self.go_delay_s = go_delay_s
        # Joints the KER has to match before it takes over. The wrist is left out by
        # default: with the software wrist remap its angles are hard to judge by eye.
        self._match_keys = [k for k in rest if any(k.endswith(f"_joint_{j}.pos") for j in engage_joints)]
        if not self._match_keys:
            raise ValueError("park_engage_joints must name at least one of joints 1-7")

        self.state = "rest"
        self._out = dict(rest)
        self._lock = threading.Lock()
        self._request: str | None = None
        self._last_t: float | None = None
        self._segments: list[tuple[dict[str, float], dict[str, float], float]] = []
        self._segment_t0 = 0.0
        self._after_move = "rest"
        self._matched_since: float | None = None
        self._last_hint_t = 0.0
        self._go_t = 0.0

    def request(self, command: str) -> None:
        """Ask for a park move: "ready", "rest", "go" or "toggle". Safe to call from any thread."""
        if command not in ("ready", "rest", "go", "toggle"):
            raise ValueError(f"park command must be ready, rest, go or toggle, got {command!r}")
        with self._lock:
            self._request = command

    def _limit(self, key: str) -> float:
        return self.speed * (GRIPPER_SPEED_SCALE if "gripper" in key else 1.0)

    def _plan(self, waypoints: list[dict[str, float]], after: str, now: float) -> None:
        """Queue a move from the current output through the waypoints.

        All joints of a segment start and finish together, so the path is a straight
        line in joint space, eased in and out. The duration puts the fastest joint's
        peak speed at its limit.
        """
        self._segments = []
        start = dict(self._out)
        for end in waypoints:
            longest = max(abs(end[k] - start[k]) / self._limit(k) for k in start)
            self._segments.append((start, dict(end), max(MIN_SEGMENT_S, 0.5 * math.pi * longest)))
            start = dict(end)
        self._segment_t0 = now
        self._after_move = after

    def _handle_request(self, command: str, now: float) -> None:
        if self.state in ("to_ready", "to_rest"):
            logger.info(f"park: a move is already in progress, '{command}' ignored")
            return
        if command == "go":
            if self.state in ("ready", "countdown"):
                self.state = "countdown"
                self._go_t = now + self.go_delay_s
                logger.info(
                    f"park: KER takes over in {self.go_delay_s:.0f} s. Hold it where the arms should go; "
                    "they will move there from the ready pose."
                )
            else:
                logger.info("park: 'go' only works at the ready pose")
            return
        if command == "toggle":
            command = "rest" if self.state in ("ready", "countdown") else "ready"
        if self.state == "countdown":
            self.state = "ready"  # a new move cancels the countdown
        if command == self.state:
            logger.info(f"park: already at the {command} pose")
            return
        if self.state in ("engaging", "teleop"):
            logger.info("park: KER released")
        back_down = [*reversed(self.ready_path[:-1]), self.rest]
        if self.state == "rest":
            self._plan(self.ready_path, "ready", now)
        elif self.state == "ready":
            self._plan(back_down, "rest", now)
        elif command == "ready":  # from KER control: straight to the ready pose, already over the table
            self._plan([self.ready], "ready", now)
        else:  # from KER control: to the ready pose first, then down the usual way
            self._plan([self.ready, *back_down], "rest", now)
        self.state = f"to_{command}"
        logger.info(f"park: moving to the {command} pose")

    def _move(self, now: float) -> None:
        while self._segments:
            start, end, duration = self._segments[0]
            u = (now - self._segment_t0) / duration
            if u < 1.0:
                s = 0.5 - 0.5 * math.cos(math.pi * max(u, 0.0))
                self._out = {k: start[k] + (end[k] - start[k]) * s for k in start}
                return
            self._out = dict(end)
            self._segments.pop(0)
            self._segment_t0 += duration
        self.state = self._after_move
        self._matched_since = None
        if self.state == "ready":
            logger.info("park: at the ready pose. Bring the KER to the same pose to take over.")
        else:
            logger.info("park: at the rest pose")

    def _wait_for_match(self, ker: dict[str, float], now: float) -> None:
        errors = {k: ker[k] - self.ready[k] for k in self._match_keys}
        worst = max(abs(e) for e in errors.values())
        if worst > self.engage_tolerance:
            self._matched_since = None
            if now - self._last_hint_t > HINT_INTERVAL_S:
                self._last_hint_t = now
                off = sorted(errors.items(), key=lambda kv: -abs(kv[1]))[:3]
                hint = ", ".join(f"{k.removesuffix('.pos')} {e:+.0f}" for k, e in off)
                logger.info(
                    f"park: waiting for the KER near the ready pose (KER minus ready, deg: {hint}). "
                    "Or type go."
                )
            return
        if self._matched_since is None:
            self._matched_since = now
        if now - self._matched_since >= self.engage_hold_s:
            self.state = "engaging"
            logger.info("park: KER matches the ready pose, taking over")

    def _engage(self, ker: dict[str, float], dt: float) -> None:
        done = True
        for k, target in ker.items():
            step = self._limit(k) * dt
            delta = target - self._out[k]
            if abs(delta) > step:
                self._out[k] += math.copysign(step, delta)
            else:
                self._out[k] = target
            if abs(target - self._out[k]) > ENGAGED_TOLERANCE_DEG:
                done = False
        if done:
            self.state = "teleop"
            logger.info("park: KER in control")

    def step(self, ker: dict[str, float], now: float) -> dict[str, float]:
        """Return the pose to command, given the mapped KER pose."""
        dt = 0.0 if self._last_t is None else min(max(now - self._last_t, 0.0), MAX_DT_S)
        self._last_t = now
        with self._lock:
            command, self._request = self._request, None
        if command is not None:
            self._handle_request(command, now)

        if self.state in ("to_ready", "to_rest"):
            self._move(now)
        elif self.state == "ready":
            self._wait_for_match(ker, now)
        elif self.state == "countdown":
            if now >= self._go_t:
                self.state = "engaging"
                logger.info("park: taking over")
        elif self.state == "engaging":
            self._engage(ker, dt)
        elif self.state == "teleop":
            self._out = dict(ker)
        return dict(self._out)
