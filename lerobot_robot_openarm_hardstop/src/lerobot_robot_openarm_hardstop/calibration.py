# SPDX-License-Identifier: Apache-2.0
"""Hard-stop zero calibration for OpenArm arms (follower or leader) driven by Damiao motors.

Each joint is driven slowly into its mechanical stop through a soft position spring (MIT mode). Contact
is detected when the joint has stopped moving while pushing on the stop for several consecutive samples.
The spring target is then held where it was, so the joint keeps pressing the stop with the torque it
stalled at (including any gravity load), and its position is averaged. The joint's zero offset is the
measured stop position minus the angle at which that stop sits in the joint's zero frame.

The joint order, clearance moves and stop angles follow Enactic's `openarm-can-zero-position-calibration`
tool (enactic/openarm_can, setup/zero_position_calibration.py). Two things differ:

- The motors' stored zero is never rewritten. The offsets are saved in the calibration file and applied
  in software by `OffsetDamiaoMotorsBus`, so the result does not depend on where the arm physically ends
  up, and repeated calibrations do not wear the motors' flash.
- Each joint gets a travel limit, a time limit, a push-torque limit and a debounced stall test, and its
  stop position is averaged while held still.

To check how repeatable the stops are on a given arm without saving anything, run:

    openarm-stop-repeatability --port can0 --side right --runs 5
"""

from __future__ import annotations

import argparse
import logging
import math
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol

from lerobot.motors import MotorCalibration

from .offset_bus import HOMING_OFFSET_UNITS_PER_DEGREE

logger = logging.getLogger(__name__)

ARM_MOTORS = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", "gripper")

# Joint ranges in degrees, in each joint's zero frame, for the right arm. The left arm mirrors joints 1 and 2.
# Working range: LeRobot's default joint limits for the OpenArm follower.
_RIGHT_WORKING_RANGE_DEG = {
    "joint_1": (-75.0, 75.0),
    "joint_2": (-9.0, 90.0),
    "joint_3": (-85.0, 85.0),
    "joint_4": (0.0, 135.0),
    "joint_5": (-85.0, 85.0),
    "joint_6": (-40.0, 40.0),
    "joint_7": (-80.0, 80.0),
    "gripper": (-65.0, 0.0),
}
# Mechanical range: Enactic's mechanical limits, widened to include the working range.
_RIGHT_MECHANICAL_RANGE_DEG = {
    "joint_1": (-80.0, 200.0),
    "joint_2": (-10.0, 100.0),
    "joint_3": (-90.0, 90.0),
    "joint_4": (0.0, 140.0),
    "joint_5": (-90.0, 90.0),
    "joint_6": (-45.0, 45.0),
    "joint_7": (-90.0, 90.0),
    "gripper": (-65.0, 0.0),
}
_MIRRORED = ("joint_1", "joint_2")

# The motors report a single-turn absolute position after power-up, so a joint's raw reading has to stay
# within ±180° of the motor's stored zero over its working range. Keep this much margin.
MAX_RAW_ANGLE_DEG = 175.0
# A joint reading further than this outside its mechanical range means the calibration doesn't match.
PLAUSIBILITY_MARGIN_DEG = 10.0


def _for_side(right_ranges: dict[str, tuple[float, float]], side: str) -> dict[str, tuple[float, float]]:
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    if side == "right":
        return dict(right_ranges)
    return {m: ((-hi, -lo) if m in _MIRRORED else (lo, hi)) for m, (lo, hi) in right_ranges.items()}


def working_range_deg(side: str) -> dict[str, tuple[float, float]]:
    return _for_side(_RIGHT_WORKING_RANGE_DEG, side)


def mechanical_range_deg(side: str) -> dict[str, tuple[float, float]]:
    return _for_side(_RIGHT_MECHANICAL_RANGE_DEG, side)


def offset_problems(offsets_deg: dict[str, float], side: str) -> list[str]:
    """Joints whose raw readings would leave ±MAX_RAW_ANGLE_DEG somewhere in their working range."""
    problems = []
    for motor, offset in offsets_deg.items():
        lo, hi = working_range_deg(side).get(motor, (0.0, 0.0))
        worst = max(abs(lo + offset), abs(hi + offset))
        if worst > MAX_RAW_ANGLE_DEG:
            problems.append(
                f"{motor}: its motor's zero is {offset:+.1f}° from the joint's zero, so raw readings would reach "
                f"{worst:.0f}° in its working range"
            )
    return problems


def position_problems(
    positions_deg: dict[str, float],
    side: str,
    joint_limits: dict[str, tuple[float, float]] | None = None,
) -> list[str]:
    """Joints whose calibrated position lies clearly outside their mechanical range.

    `joint_limits` are the arm's configured limits, if the user set any; a joint whose limits reach past the
    nominal mechanical range (e.g. a different gripper) is allowed that far.
    """
    problems = []
    for motor, position in positions_deg.items():
        if motor not in mechanical_range_deg(side):
            continue
        lo, hi = mechanical_range_deg(side)[motor]
        if joint_limits and motor in joint_limits:
            lo, hi = min(lo, joint_limits[motor][0]), max(hi, joint_limits[motor][1])
        if position < lo - PLAUSIBILITY_MARGIN_DEG or position > hi + PLAUSIBILITY_MARGIN_DEG:
            problems.append(
                f"{motor} reads {position:.1f}°, outside its mechanical range {lo:.0f}°..{hi:.0f}°"
            )
    return problems


# Gains used to hold every joint where it is while another joint moves (Enactic's values).
DEFAULT_HOLD_KP = {
    "joint_1": 300.0,
    "joint_2": 300.0,
    "joint_3": 150.0,
    "joint_4": 150.0,
    "joint_5": 40.0,
    "joint_6": 40.0,
    "joint_7": 30.0,
    "gripper": 10.0,
}
DEFAULT_HOLD_KD = {
    "joint_1": 2.5,
    "joint_2": 2.5,
    "joint_3": 2.5,
    "joint_4": 2.5,
    "joint_5": 0.8,
    "joint_6": 0.8,
    "joint_7": 0.8,
    "gripper": 0.9,
}


class CalibrationAbortedError(RuntimeError):
    """Raised when a joint does not behave as expected during hard-stop calibration."""


class _CalibrationBus(Protocol):
    """The subset of `DamiaoMotorsBus` used here (positions in degrees, velocities in deg/s, torques in Nm)."""

    calibration: dict[str, MotorCalibration]

    def sync_read_all_states(self, motors: str | list[str] | None = None) -> dict[str, dict[str, float]]: ...

    def _mit_control_batch(self, commands: dict[str, tuple[float, float, float, float, float]]) -> None: ...

    def enable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None: ...


@dataclass(frozen=True)
class StopSeek:
    """Drive one joint into its mechanical stop and measure where the stop is."""

    motor: str
    # Direction of travel toward the stop: +1 or -1, in the motor's own frame.
    direction: int
    # Angle of this stop in the joint's zero frame, in degrees.
    stop_deg: float
    # Abort if the joint travels further than this from where the seek started without finding the stop.
    max_travel_deg: float
    kp: float = 45.0
    kd: float = 1.2
    # Stall test: joint slower than this while the motor pushes harder than `torque_threshold_nm`.
    # The velocity threshold must stay well below the seek speed.
    vel_threshold_deg_s: float = math.degrees(0.1)
    torque_threshold_nm: float = 2.0
    # Abort if the motor ever pushes harder than this. Defaults to 4x `torque_threshold_nm`; raise it for a
    # joint that carries a large gravity load on its way to the stop.
    max_push_torque_nm: float | None = None

    def __post_init__(self) -> None:
        if self.direction not in (1, -1):
            raise ValueError(f"direction must be +1 or -1, got {self.direction}")

    @property
    def push_limit_nm(self) -> float:
        return (
            self.max_push_torque_nm if self.max_push_torque_nm is not None else 4 * self.torque_threshold_nm
        )


@dataclass(frozen=True)
class RelativeMove:
    """Move one joint by a relative angle, to clear the way for the next seek or to back off a stop."""

    motor: str
    delta_deg: float
    duration_s: float = 2.0
    kp: float = 52.0
    kd: float = 1.5


Step = StopSeek | RelativeMove


@dataclass
class HardStopCalibrationParams:
    seek_speed_deg_s: float = 15.0
    control_period_s: float = 0.005
    # Consecutive samples that must pass the stall test before contact is accepted.
    stall_samples: int = 5
    # Extra time allowed on top of max_travel_deg / seek_speed_deg_s before a seek is aborted.
    seek_timeout_margin_s: float = 5.0
    # Time spent pressed against the stop before measuring, and the averaging window.
    settle_s: float = 0.3
    measure_s: float = 0.2
    # Abort if the joint moves more than this while it is held against the stop and measured.
    max_measure_spread_deg: float = 0.5
    # Pause between steps.
    pause_s: float = 0.3
    hold_kp: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_HOLD_KP))
    hold_kd: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_HOLD_KD))


@dataclass(frozen=True)
class StopMeasurement:
    motor: str
    # Stop position in the motor's own frame (no software offset), averaged over the measurement window.
    stop_raw_deg: float
    stop_std_deg: float
    stop_deg: float
    travel_deg: float
    contact_torque_nm: float

    @property
    def offset_deg(self) -> float:
        """Software zero offset: the reading at the joint's zero, in the motor's own frame."""
        return self.stop_raw_deg - self.stop_deg


def openarm_sequence(side: str, stop_trim_deg: dict[str, float] | None = None) -> list[Step]:
    """Calibration steps for one OpenArm arm, starting from the arm hanging straight down.

    Mirrors Enactic's sequence: the gripper and elbow first, then the shoulder rotation with the shoulder
    nudged out of the way, then the wrist with the elbow bent for clearance, then the shoulder joints.
    Joints 1 and 2 are mirrored between the left and right arm, and joint 7 is sought toward opposite stops.

    `stop_trim_deg` corrects the stop angles for one particular arm: each value is added to that joint's
    nominal stop angle (see stop_trim.py).
    """
    if stop_trim_deg:
        return [
            replace(step, stop_deg=step.stop_deg + stop_trim_deg.get(step.motor, 0.0))
            if isinstance(step, StopSeek)
            else step
            for step in openarm_sequence(side)
        ]
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    s = 1 if side == "right" else -1

    # Stall thresholds are Enactic's, except joint_2's torque threshold and the gripper's velocity
    # threshold: theirs (0.3 rad/s) was paired with a faster approach and would sit above the seek speed
    # used here.
    return [
        StopSeek("gripper", +1, 0.0, max_travel_deg=75.0, torque_threshold_nm=0.3),
        StopSeek("joint_4", -1, 0.0, max_travel_deg=45.0),
        RelativeMove("joint_2", +5.0 * s, duration_s=0.4),
        StopSeek("joint_3", +1, 90.0, max_travel_deg=120.0),
        RelativeMove("joint_3", -90.0, duration_s=1.0),
        RelativeMove("joint_2", -5.0 * s, duration_s=0.4),
        RelativeMove("joint_4", +90.0),
        StopSeek("joint_5", +1, 90.0, max_travel_deg=120.0),
        RelativeMove("joint_5", -90.0),
        StopSeek("joint_6", +1, 45.0, max_travel_deg=75.0),
        RelativeMove("joint_6", -45.0),
        # Enactic seeks +90° on both arms. That turns the right wrist up into the wrist camera's cable, so
        # the right arm uses the opposite stop and both wrists turn down.
        StopSeek("joint_7", -s, -90.0 * s, max_travel_deg=120.0),
        RelativeMove("joint_7", +90.0 * s),
        # Raised from Enactic's 2 Nm: cable drag alone reached that a few degrees before the stop.
        StopSeek("joint_2", -s, -10.0 * s, max_travel_deg=40.0, torque_threshold_nm=5.0),
        RelativeMove("joint_2", +10.0 * s, duration_s=0.9, kp=180.0, kd=2.0),
        StopSeek(
            "joint_1",
            -s,
            -80.0 * s,
            max_travel_deg=110.0,
            kp=180.0,
            kd=2.1,
            vel_threshold_deg_s=math.degrees(0.0125),
            torque_threshold_nm=5.0,
            # The shoulder may carry much of the arm's weight near this stop.
            max_push_torque_nm=30.0,
        ),
        RelativeMove("joint_1", +80.0 * s),
        RelativeMove("joint_4", -90.0),
    ]


class HardStopCalibrator:
    """Runs a list of steps on a bus and returns the measured stop positions.

    The bus must report positions in the motors' own frame, i.e. carry no software offsets. `sleep` and
    `clock` can be replaced to run against a simulated arm.
    """

    def __init__(
        self,
        bus: _CalibrationBus,
        params: HardStopCalibrationParams | None = None,
        *,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
    ):
        self.bus = bus
        self.params = params if params is not None else HardStopCalibrationParams()
        self._sleep = sleep if sleep is not None else time.sleep
        self._clock = clock if clock is not None else time.monotonic
        # Last command sent to each motor: (kp, kd, target_deg).
        self._commands: dict[str, tuple[float, float, float]] = {}

    def run(self, steps: Sequence[Step]) -> dict[str, StopMeasurement]:
        if any(c.homing_offset for c in self.bus.calibration.values()):
            raise ValueError("Clear the bus's software offsets before running hard-stop calibration.")

        motors = sorted({step.motor for step in steps})
        self.bus.enable_torque(motors)
        self._hold_current(motors)

        measurements: dict[str, StopMeasurement] = {}
        try:
            for step in steps:
                if isinstance(step, StopSeek):
                    measurement = self._seek(step)
                    measurements[step.motor] = measurement
                    logger.info(
                        f"{step.motor}: stop at {measurement.stop_raw_deg:.3f}° raw "
                        f"(±{measurement.stop_std_deg:.3f}°, {measurement.contact_torque_nm:.2f} Nm) "
                        f"→ offset {measurement.offset_deg:.3f}°"
                    )
                else:
                    self._move(step)
                self._wait(self.params.pause_s)
        except BaseException:
            # Stop whatever was moving and hold the arm where it is; the caller decides what comes next.
            try:
                self._hold_current(motors)
            except Exception:
                logger.exception("Could not command the arm to hold its position after an error.")
            raise
        return measurements

    # Motion primitives

    def _read(self, motor: str) -> dict[str, float]:
        return self.bus.sync_read_all_states([motor])[motor]

    def _command(self, motor: str, kp: float, kd: float, target_deg: float) -> None:
        self.bus._mit_control_batch({motor: (kp, kd, target_deg, 0.0, 0.0)})
        self._commands[motor] = (kp, kd, target_deg)

    def _hold_current(self, motors: list[str]) -> None:
        states = self.bus.sync_read_all_states(motors)
        for m in motors:
            kp, kd = self.params.hold_kp.get(m, 30.0), self.params.hold_kd.get(m, 1.0)
            self._commands[m] = (kp, kd, states[m]["position"])
        self.bus._mit_control_batch({m: (kp, kd, q, 0.0, 0.0) for m, (kp, kd, q) in self._commands.items()})

    def _wait(self, duration_s: float) -> None:
        """Wait while re-sending each motor's last command, as the control loop would."""
        end = self._clock() + duration_s
        while self._clock() < end:
            if self._commands:
                self.bus._mit_control_batch(
                    {m: (kp, kd, q, 0.0, 0.0) for m, (kp, kd, q) in self._commands.items()}
                )
            self._sleep(self.params.control_period_s)

    def _move(self, step: RelativeMove) -> None:
        start = self._read(step.motor)["position"]
        goal = start + step.delta_deg
        t0 = self._clock()
        while True:
            alpha = min(1.0, (self._clock() - t0) / step.duration_s) if step.duration_s > 0 else 1.0
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            self._command(step.motor, step.kp, step.kd, start + (goal - start) * smooth)
            if alpha >= 1.0:
                return
            self._sleep(self.params.control_period_s)

    def _seek(self, step: StopSeek) -> StopMeasurement:
        p = self.params
        if step.vel_threshold_deg_s > 0.5 * p.seek_speed_deg_s:
            raise ValueError(
                f"{step.motor}: velocity threshold {step.vel_threshold_deg_s:.1f}°/s must be at most half the "
                f"seek speed ({p.seek_speed_deg_s:.1f}°/s), or free motion looks like a stall."
            )
        state = self._read(step.motor)
        start = state["position"]
        target = start
        # Spring stretch (deg) at which the commanded spring alone reaches the torque threshold.
        lead_at_threshold = math.degrees(step.torque_threshold_nm / step.kp)
        timeout = step.max_travel_deg / p.seek_speed_deg_s + p.seek_timeout_margin_s

        t0 = last = self._clock()
        stalled = 0
        while True:
            now = self._clock()
            # The target keeps advancing while contact is being confirmed, so a joint that only stuck for
            # a moment breaks free instead of being taken for stopped. The push limit bounds the force.
            target += step.direction * p.seek_speed_deg_s * (now - last)
            last = now
            self._command(step.motor, step.kp, step.kd, target)
            self._sleep(p.control_period_s)
            state = self._read(step.motor)

            travel = step.direction * (state["position"] - start)
            if travel > step.max_travel_deg:
                raise CalibrationAbortedError(
                    f"{step.motor} travelled {travel:.1f}° without reaching its stop "
                    f"(limit {step.max_travel_deg:.0f}°). Was the arm hanging straight down at the start?"
                )
            if abs(state["torque"]) > step.push_limit_nm:
                raise CalibrationAbortedError(
                    f"{step.motor} pushed {abs(state['torque']):.1f} Nm (limit {step.push_limit_nm:.1f} Nm) "
                    f"after {travel:.1f}° without a confirmed stop."
                )
            if self._clock() - t0 > timeout:
                raise CalibrationAbortedError(f"{step.motor} did not reach its stop within {timeout:.1f} s.")

            # Requiring the spring to be stretched toward the stop keeps a joint that is still getting
            # started (or held by gravity before it moves) from passing for stalled, while a joint that
            # starts against its stop is detected right away.
            lead = step.direction * (target - state["position"])
            is_stalled = (
                abs(state["velocity"]) < step.vel_threshold_deg_s
                and abs(state["torque"]) > step.torque_threshold_nm
                and lead > 0.5 * lead_at_threshold
            )
            stalled = stalled + 1 if is_stalled else 0
            if stalled >= p.stall_samples:
                break

        # Keep the target where contact was confirmed, so the joint goes on pressing the stop with the push it
        # stalled at (including any gravity load); let it settle, then average.
        self._command(step.motor, step.kp, step.kd, target)
        self._wait(p.settle_s)

        positions: list[float] = []
        torques: list[float] = []
        end = self._clock() + p.measure_s
        while self._clock() < end or not positions:
            self._command(step.motor, step.kp, step.kd, target)
            self._sleep(p.control_period_s)
            sample = self._read(step.motor)
            positions.append(sample["position"])
            torques.append(sample["torque"])

        if max(positions) - min(positions) > p.max_measure_spread_deg:
            raise CalibrationAbortedError(
                f"{step.motor} was not steady against its stop "
                f"({max(positions) - min(positions):.2f}° of motion while measuring)."
            )
        stop_raw = statistics.fmean(positions)
        return StopMeasurement(
            motor=step.motor,
            stop_raw_deg=stop_raw,
            stop_std_deg=statistics.pstdev(positions),
            stop_deg=step.stop_deg,
            travel_deg=step.direction * (stop_raw - start),
            contact_torque_nm=statistics.fmean(torques),
        )


def offsets_to_calibration(
    offsets_deg: dict[str, float], motor_ids: dict[str, int], range_deg: tuple[int, int] = (-90, 90)
) -> dict[str, MotorCalibration]:
    """Build calibration entries carrying the software offsets.

    `range_min`/`range_max` are kept at the same placeholder values as the manual calibration; joint limits
    come from the robot config.
    """
    return {
        motor: MotorCalibration(
            id=motor_ids[motor],
            drive_mode=0,
            homing_offset=round(offset * HOMING_OFFSET_UNITS_PER_DEGREE),
            range_min=range_deg[0],
            range_max=range_deg[1],
        )
        for motor, offset in offsets_deg.items()
    }


def run_openarm_hard_stop_calibration(
    bus,
    side: str,
    *,
    params: HardStopCalibrationParams | None = None,
    prompt: Callable[[str], str] | None = None,
    stop_trim_deg: dict[str, float] | None = None,
) -> dict[str, MotorCalibration]:
    """Interactive hard-stop calibration of one OpenArm arm. Returns the new calibration.

    `stop_trim_deg` is this arm's correction to the nominal stop angles, if it has one.

    Clears the bus's software offsets, runs the sequence, and installs the new calibration on the bus.
    Nothing is saved to disk here. On failure the arm is held in place, then torque is released once the
    user confirms they are supporting it, and the error is re-raised. The previous calibration is put back
    on the bus if the new one is rejected.
    """
    if prompt is None:
        prompt = input
    steps = openarm_sequence(side, stop_trim_deg)
    missing = {step.motor for step in steps} - set(bus.motors)
    if missing:
        raise ValueError(f"Hard-stop calibration needs motors {sorted(missing)}, which are not on this bus.")

    prompt(
        "\nHard-stop calibration\n"
        "The arm WILL MOVE ON ITS OWN, pressing each joint gently into its mechanical stop.\n"
        "  - Clear the workspace around the arm and keep the emergency stop in reach\n"
        "  - Let the arm hang straight down, elbow straight, gripper closed\n"
        "Press ENTER to start (Ctrl+C at any time stops the arm and holds it in place)..."
    )

    previous = dict(bus.calibration or {})
    bus.write_calibration({})
    calibrator = HardStopCalibrator(bus, params)
    try:
        measurements = calibrator.run(steps)
    except BaseException as e:
        logger.error(f"Hard-stop calibration stopped: {e!r}")
        bus.write_calibration(previous)
        try:
            prompt("The arm is holding its position. Support it, then press ENTER to release torque...")
        finally:
            bus.disable_torque()
        raise

    offsets = {motor: m.offset_deg for motor, m in measurements.items()}
    problems = offset_problems(offsets, side)
    if problems:
        bus.write_calibration(previous)
        raise CalibrationAbortedError(
            "Some motors' stored zero is too far from their joint's zero for software offsets to work after a "
            "power cycle:\n  - "
            + "\n  - ".join(problems)
            + "\nRe-zero the motors with the arm hanging straight "
            "down (Enactic's openarm-can-zero-position-calibration, or LeRobot's stock manual calibration), "
            "then run this calibration again."
        )
    calibration = offsets_to_calibration(offsets, {name: motor.id for name, motor in bus.motors.items()})
    bus.write_calibration(calibration)

    print("\nMeasured stops (raw) and resulting offsets:")
    for motor in ARM_MOTORS:
        if motor in measurements:
            m = measurements[motor]
            print(
                f"  {motor:<8} stop {m.stop_raw_deg:9.3f}° ±{m.stop_std_deg:.3f}  "
                f"→ offset {m.offset_deg:8.3f}°  ({m.contact_torque_nm:+.2f} Nm)"
            )
    return calibration


def measure_repeatability(
    bus, side: str, runs: int, *, params: HardStopCalibrationParams | None = None
) -> dict[str, list[float]]:
    """Run the stop-seeking sequence several times and return each joint's raw stop positions."""
    if bus.calibration and any(c.homing_offset for c in bus.calibration.values()):
        raise ValueError("Clear the bus's software offsets before measuring repeatability.")
    steps = openarm_sequence(side)
    calibrator = HardStopCalibrator(bus, params)
    stops: dict[str, list[float]] = {}
    for i in range(runs):
        logger.info(f"Repeatability run {i + 1}/{runs}")
        for motor, m in calibrator.run(steps).items():
            stops.setdefault(motor, []).append(m.stop_raw_deg)
    return stops


def main() -> None:
    """`openarm-stop-repeatability`: measure how repeatable an arm's hard stops are."""
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.damiao import DamiaoMotorsBus
    from lerobot.robots.openarm_follower.config_openarm_follower import OpenArmFollowerConfigBase

    parser = argparse.ArgumentParser(
        description="Measure how repeatable an OpenArm's hard stops are. Moves the arm; saves nothing."
    )
    parser.add_argument("--port", required=True, help="CAN interface, e.g. can0")
    parser.add_argument("--side", required=True, choices=["left", "right"])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--can-interface", default="socketcan")
    parser.add_argument("--no-can-fd", action="store_true")
    args = parser.parse_args()
    # LeRobot installs a root handler on import, so without force this call would be a no-op.
    logging.basicConfig(level=logging.INFO, force=True)

    config = OpenArmFollowerConfigBase(port=args.port)
    motors = {}
    for name, (send_id, recv_id, motor_type) in config.motor_config.items():
        motors[name] = Motor(
            send_id, motor_type, MotorNormMode.DEGREES, motor_type_str=motor_type, recv_id=recv_id
        )
    bus = DamiaoMotorsBus(
        port=args.port,
        motors=motors,
        can_interface=args.can_interface,
        use_can_fd=not args.no_can_fd,
        bitrate=config.can_bitrate,
        data_bitrate=None if args.no_can_fd else config.can_data_bitrate,
    )
    bus.connect()
    try:
        input(
            "The arm WILL MOVE ON ITS OWN. Clear the workspace, let it hang straight down with the gripper "
            "closed, and press ENTER to start..."
        )
        stops = measure_repeatability(bus, args.side, args.runs)
    finally:
        input("Support the arm, then press ENTER to release torque...")
        bus.disconnect(disable_torque=True)

    print(f"\nStop repeatability over {args.runs} runs (degrees):")
    print(f"  {'motor':<8} {'mean':>10} {'std':>7} {'spread':>7}")
    for motor in ARM_MOTORS:
        if motor in stops:
            values = stops[motor]
            spread = max(values) - min(values)
            print(
                f"  {motor:<8} {statistics.fmean(values):10.3f} {statistics.pstdev(values):7.3f} {spread:7.3f}"
            )


if __name__ == "__main__":
    main()
