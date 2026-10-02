"""Park controller: drives the followers between a rest pose and a ready pose without the KER.

The teleoperator normally outputs the mapped KER pose. With parking enabled it starts
by holding the rest pose (the followers' calibration pose: arms hanging) and ignores
the KER. The "ready" command then plays a fixed joint-space path to the ready pose
above the work surface and holds there. Control passes to the KER once the operator has
brought its shoulder and elbow joints roughly to the ready pose, or on the "go" command
after a short countdown; either way the followers ease from the ready pose onto the
KER pose instead of jumping. The "rest" command plays the path backwards to the rest pose.

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
import sys
import threading

logger = logging.getLogger(__name__)

MIN_SEGMENT_S = 0.2
MIN_TAKEOVER_S = 1.0
GRIPPER_SPEED_SCALE = 4.0  # grippers travel 160 deg, so they get a higher speed limit
HINT_INTERVAL_S = 2.0
LIVE_HINT_INTERVAL_S = 0.2


def alignment_hint(errors: dict[str, float], tolerance: float) -> str:
    """Say how to move the KER to reach the ready pose, e.g. "L: upper arm forward 35, bend elbow 20 | R: ok".

    `errors` is KER minus ready per matched joint, in follower joint space. Directions
    follow the OpenArm conventions: forward shoulder pitch is negative joint 1 on the
    left arm and positive on the right, outward is negative joint 2 on the left and
    positive on the right, and positive joint 4 bends the elbow.
    """
    parts = []
    for side in ("left", "right"):
        mirror = -1.0 if side == "left" else 1.0
        moves = []
        for key, error in errors.items():
            if not key.startswith(side) or abs(error) <= tolerance:
                continue
            joint = key.removeprefix(f"{side}_").removesuffix(".pos")
            need = -error  # how far the KER joint still has to move
            if joint == "joint_1":
                moves.append(f"upper arm {'forward' if need * mirror > 0 else 'back'} {abs(need):.0f}")
            elif joint == "joint_2":
                moves.append(f"upper arm {'out' if need * mirror > 0 else 'in'} {abs(need):.0f}")
            elif joint == "joint_4":
                moves.append(f"{'bend' if need > 0 else 'straighten'} elbow {abs(need):.0f}")
            elif joint == "joint_3":
                moves.append(f"twist upper arm {need:+.0f}")
            else:
                moves.append(f"{joint} {need:+.0f}")
        parts.append(f"{side[0].upper()}: {', '.join(moves) if moves else 'ok'}")
    return " | ".join(parts)


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
        blend: float = 0.0,
    ):
        if not ready_path:
            raise ValueError("park needs at least one ready_path waypoint")
        if speed_deg_s <= 0 or engage_tolerance_deg <= 0 or engage_hold_s < 0:
            raise ValueError("park speed and engage tolerance must be positive")
        if not 0.0 <= blend <= 0.5:
            raise ValueError("park blend must be between 0 and 0.5")
        self.blend = blend
        self.rest = dict(rest)
        self.ready_path = [dict(w) for w in ready_path]
        self.ready = self.ready_path[-1]
        self.speed = speed_deg_s
        self.engage_tolerance = engage_tolerance_deg
        self.engage_hold_s = engage_hold_s
        self.go_delay_s = go_delay_s
        # Joints the KER has to match before it takes over. The wrist is left out by
        # default: with the software wrist remap its angles are hard to judge by eye.
        self.match_keys = [k for k in rest if any(k.endswith(f"_joint_{j}.pos") for j in engage_joints)]
        if not self.match_keys:
            raise ValueError("park_engage_joints must name at least one of joints 1-7")
        self.arm_keys = [k for k in rest if "gripper" not in k]
        # Guided recording turns this off between episodes, so a takeover only starts
        # once LeRobot is waiting to record.
        self.engage_enabled = True

        self.state = "rest"
        self._out = dict(rest)
        self._lock = threading.Lock()
        self._request: str | None = None
        self._segments: list[tuple[dict[str, float], dict[str, float], float]] = []
        self._segment_t0 = 0.0
        self._pieces: list[tuple] = []  # blended path: lines and rounded corners, each with its cost
        self._path_cost = 0.0
        self._path_t0 = 0.0
        self._path_s = 0.0
        self._after_move = "rest"
        self._matched_since: float | None = None
        self._last_hint_t = 0.0
        self._go_t = 0.0
        self._blend_from: dict[str, float] = {}
        self._blend_t0 = 0.0
        self._blend_s = MIN_TAKEOVER_S

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
        self._segments, self._pieces = [], []
        self._after_move = after
        if self.blend > 0 and len(waypoints) > 1:
            self._plan_blended([dict(self._out), *waypoints], now)
            return
        start = dict(self._out)
        for end in waypoints:
            longest = max(abs(end[k] - start[k]) / self._limit(k) for k in start)
            self._segments.append((start, dict(end), max(MIN_SEGMENT_S, 0.5 * math.pi * longest)))
            start = dict(end)
        self._segment_t0 = now

    def _cost(self, a: dict[str, float], b: dict[str, float]) -> float:
        """Seconds the move a -> b takes with its fastest joint at its speed limit."""
        return max(abs(b[k] - a[k]) / self._limit(k) for k in a)

    def _plan_blended(self, points: list[dict[str, float]], now: float) -> None:
        """One continuous move through all the waypoints, with the corners rounded.

        Around each intermediate waypoint the path leaves the incoming line early and
        joins the outgoing one late, on a quadratic curve that has the waypoint as its
        control point. The cut starts `blend` of the shorter neighbouring segment
        before the waypoint and ends the same travel time after it, so joint speeds
        are continuous and never exceed their limits. The whole path is eased in and
        out once, instead of stopping at every waypoint.
        """
        points = [p for i, p in enumerate(points) if i == 0 or self._cost(points[i - 1], p) > 1e-6]
        costs = [self._cost(a, b) for a, b in zip(points, points[1:])]
        cuts = [0.0] + [self.blend * min(costs[i - 1], costs[i]) for i in range(1, len(costs))] + [0.0]

        def along(a, b, t):  # the point a fraction t of the way from a to b
            return {k: a[k] + (b[k] - a[k]) * t for k in a}

        for i, cost in enumerate(costs):
            a, b = points[i], points[i + 1]
            line_start = along(a, b, cuts[i] / cost)
            line_end = along(a, b, 1.0 - cuts[i + 1] / cost)
            self._pieces.append(("line", line_start, line_end, cost - cuts[i] - cuts[i + 1]))
            if cuts[i + 1] > 0:
                after_corner = along(b, points[i + 2], cuts[i + 1] / costs[i + 1])
                self._pieces.append(("curve", line_end, b, after_corner, 2.0 * cuts[i + 1]))
        self._path_cost = sum(piece[-1] for piece in self._pieces)
        self._path_t0 = now
        self._path_s = max(MIN_SEGMENT_S, 0.5 * math.pi * self._path_cost)

    def _move_blended(self, now: float) -> bool:
        """Advance along the blended path. True once the end is reached."""
        u = (now - self._path_t0) / self._path_s
        if u >= 1.0:
            self._out = dict(self._pieces[-1][2])
            self._pieces = []
            return True
        travelled = (0.5 - 0.5 * math.cos(math.pi * max(u, 0.0))) * self._path_cost
        for piece in self._pieces:
            cost = piece[-1]
            if travelled <= cost or piece is self._pieces[-1]:
                t = min(max(travelled / cost, 0.0), 1.0) if cost > 0 else 1.0
                if piece[0] == "line":
                    _, a, b, _ = piece
                    self._out = {k: a[k] + (b[k] - a[k]) * t for k in a}
                else:
                    _, a, corner, b, _ = piece
                    self._out = {
                        k: (1 - t) ** 2 * a[k] + 2 * t * (1 - t) * corner[k] + t**2 * b[k] for k in a
                    }
                return False
            travelled -= cost
        return False

    def _handle_request(self, command: str, now: float) -> None:
        if self.state in ("to_ready", "to_rest"):
            logger.info(f"park: a move is already in progress, '{command}' ignored")
            return
        if command == "go":
            if not self.engage_enabled:
                logger.info("park: not ready to record yet, 'go' ignored")
            elif self.state in ("ready", "countdown"):
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
        if self._pieces and not self._move_blended(now):
            return
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
        if not self.engage_enabled:
            self._matched_since = None
            return
        errors = {k: ker[k] - self.ready[k] for k in self.match_keys}
        worst = max(abs(e) for e in errors.values())
        live = sys.stdout is not None and sys.stdout.isatty()
        if worst > self.engage_tolerance:
            self._matched_since = None
            if now - self._last_hint_t > (LIVE_HINT_INTERVAL_S if live else HINT_INTERVAL_S):
                self._last_hint_t = now
                hint = alignment_hint(errors, self.engage_tolerance)
                if live:  # one line, redrawn in place, so it can be followed while moving the KER
                    sys.stdout.write(f"\rpark: move the KER (deg) -> {hint}\033[K")
                    sys.stdout.flush()
                else:
                    logger.info(f"park: waiting for the KER near the ready pose. Move it (deg) -> {hint}")
            return
        if self._matched_since is None:
            self._matched_since = now
            if live:
                sys.stdout.write("\rpark: in position, hold it\033[K")
                sys.stdout.flush()
        if now - self._matched_since >= self.engage_hold_s:
            if live:
                sys.stdout.write("\n")
            logger.info("park: KER matches the ready pose, taking over")
            self._start_engage(ker, now)

    def _start_engage(self, ker: dict[str, float], now: float) -> None:
        """Begin the takeover: an eased blend from the held pose to the live KER pose.

        Every joint starts and finishes together, and the output ends exactly on the
        KER with no step, however the KER moves meanwhile. The blend is long enough
        that the joint with the largest gap peaks at its speed limit.
        """
        self._blend_from = dict(self._out)
        self._blend_t0 = now
        longest = max(abs(ker[k] - self._out[k]) / self._limit(k) for k in ker)
        self._blend_s = max(MIN_TAKEOVER_S, 0.5 * math.pi * longest)
        self.state = "engaging"

    def _engage(self, ker: dict[str, float], now: float) -> None:
        u = (now - self._blend_t0) / self._blend_s
        if u >= 1.0:
            self._out = dict(ker)
            self.state = "teleop"
            logger.info("park: KER in control")
            return
        s = 0.5 - 0.5 * math.cos(math.pi * max(u, 0.0))
        self._out = {k: self._blend_from[k] + (ker[k] - self._blend_from[k]) * s for k in ker}

    def step(self, ker: dict[str, float], now: float) -> dict[str, float]:
        """Return the pose to command, given the mapped KER pose."""
        with self._lock:
            command, self._request = self._request, None
        if command is not None:
            self._handle_request(command, now)

        if self.state in ("to_ready", "to_rest"):
            self._move(now)
        elif self.state == "ready":
            self._wait_for_match(ker, now)
        elif self.state == "countdown":
            if not self.engage_enabled:
                self.state = "ready"
            elif now >= self._go_t:
                logger.info("park: taking over")
                self._start_engage(ker, now)
        elif self.state == "engaging":
            self._engage(ker, now)
        elif self.state == "teleop":
            self._out = dict(ker)
        return dict(self._out)
