"""Guided recording: park, takeover and episode boundaries driven by the KER.

LeRobot's record loop does not tell a teleoperator when an episode starts or ends, so
the `lerobot-ker-record` launcher (record.py) wraps that loop and calls in here. One
session then runs hands-free:

    start        the followers rise to the ready pose by themselves
    takeover     the operator brings the KER near the ready pose; recording starts the
                 moment the takeover completes, so no park move or blend is recorded
    episode end  the operator brings the KER back near the ready pose and holds still
    reset        the followers return to the ready pose and hold; the episode is saved
    ...          takeover again starts the next episode
    session end  the followers return to the rest pose before torque is cut

A double squeeze during an episode discards that take and records it again.
"""

import logging
import sys
import time

logger = logging.getLogger(__name__)

STILL_DEG = 3.0  # arm joints moving less than this count as holding still
AWAY_FACTOR = 1.25  # the KER must leave the ready zone by this margin before an episode can end
HINT_INTERVAL_S = 10.0

# Set by the launcher before LeRobot builds the teleoperator.
launcher_active = False


class GuidedSession:
    def __init__(self, teleop):
        self._t = teleop
        self._park = teleop._park
        self.robot = None
        self.events: dict | None = None
        self.fps = 30
        self.phase: str | None = None  # "record" while an episode is being recorded, "reset" in between
        self.episodes_started = 0
        self._was_away = False
        self._still_ref: dict[str, float] | None = None
        self._still_since = 0.0
        self._ended = False

    # ------------------------------------------------------------ called by the launcher
    def attach(self, robot, events: dict, fps: int) -> None:
        self.robot, self.events, self.fps = robot, events, fps

    def _pump(self) -> None:
        """One step of LeRobot's loop without recording: keeps the park moves reaching the robot."""
        self.robot.send_action(self._t.get_action())
        time.sleep(1.0 / self.fps)

    def wait_for_takeover(self) -> bool:
        """Before an episode: block until the KER is in control. False if the session was stopped."""
        cfg = self._t.config
        self.phase = None
        self._park.engage_enabled = True
        auto_ready_t = None
        if self._park.state == "rest" and self.episodes_started == 0 and cfg.park_auto_ready_s >= 0:
            auto_ready_t = time.monotonic() + cfg.park_auto_ready_s
            logger.info(f"park: the followers rise to the ready pose in {cfg.park_auto_ready_s:.0f} s")
        last_hint = time.monotonic()
        while self._park.state != "teleop":
            if self.events["stop_recording"]:
                return False
            self.events["exit_early"] = False  # "next" has nothing to skip while waiting
            now = time.monotonic()
            if auto_ready_t is not None and now >= auto_ready_t:
                auto_ready_t = None
                self._park.request("ready")
            if self._park.state == "rest" and auto_ready_t is None and now - last_hint > HINT_INTERVAL_S:
                last_hint = now
                logger.info("park: at the rest pose. Double-squeeze both triggers to rise to the ready pose.")
            self._pump()
        self.phase = "record"
        self.episodes_started += 1
        self._was_away = False
        self._still_ref = None
        self._ended = False
        sys.stdout.write("\a")
        sys.stdout.flush()
        logger.info("park: RECORDING. Bring the KER back to the ready pose and hold still to end the episode.")
        return True

    def start_reset(self) -> None:
        """After an episode: release the KER and hold the followers at the ready pose."""
        self.phase = "reset"
        self._park.engage_enabled = False  # the next takeover waits until LeRobot is ready to record
        if self._park.state in ("teleop", "engaging", "countdown"):
            self._park.request("ready")

    def end_phase(self) -> None:
        self.phase = None

    def rest_before_disconnect(self) -> None:
        """Session over: bring the followers down to the rest pose before torque is cut."""
        if not self._t.config.park_rest_on_exit or self._park.state == "rest":
            return
        logger.info("park: session over, returning to the rest pose (Ctrl-C to skip)")
        self.phase = None
        self._park.engage_enabled = False
        deadline = time.monotonic() + 60.0
        while self._park.state != "rest" and time.monotonic() < deadline:
            if self._park.state not in ("to_ready", "to_rest"):
                self._park.request("rest")
            self._pump()

    # ------------------------------------------------- called for every mapped KER pose
    def on_gesture(self) -> bool:
        """Double squeeze. True if handled here, False to fall back to the plain park toggle."""
        if self.phase == "record" and self._park.state == "teleop":
            logger.info("park: double squeeze, discarding this take; it will be recorded again")
            self.events["rerecord_episode"] = True
            self.events["exit_early"] = True
            self._ended = True
            return True
        return self._park.state not in ("rest", "ready")  # parked: toggle rest <-> ready as usual

    def on_sample(self, ker: dict[str, float], now: float) -> None:
        """End the episode once the KER has come back to the ready pose and is held still."""
        hold_s = self._t.config.park_end_hold_s
        if self.phase != "record" or self._ended or hold_s <= 0 or self._park.state != "teleop":
            return
        ready, tolerance = self._park.ready, self._t.config.park_end_tolerance_deg
        off = max(abs(ker[k] - ready[k]) for k in self._park.match_keys)
        if off > AWAY_FACTOR * tolerance:
            self._was_away = True
        arm = {k: ker[k] for k in self._park.arm_keys}
        if self._still_ref is None or max(abs(arm[k] - self._still_ref[k]) for k in arm) > STILL_DEG:
            self._still_ref, self._still_since = arm, now
        if self._was_away and off <= tolerance and now - self._still_since >= hold_s:
            self._ended = True
            self.events["exit_early"] = True
            logger.info("park: KER back at the ready pose and still, episode ended")
