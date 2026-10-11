# SPDX-License-Identifier: Apache-2.0
"""Per-arm correction to the nominal stop angles, measured once at a reference pose.

Hard-stop calibration places a joint's zero at `stop reading - stop angle`. The stop reading is very
repeatable, but the stop angle is a nominal value from the arm's design. On a real arm the stop can sit a
few degrees from it: the stop and the structure give under the push, parts are mounted slightly differently,
and the two arms of a pair differ. The zero is then off by that amount, the same amount every time.

The trim fixes that. With the arm at a pose where the joint is known to be at zero (hanging freely for the
shoulder joints), whatever the calibrated joint reads is the error. `openarm-zero-trim` measures it, shifts
the saved calibration by it and stores it in `<id>.stop_trim.json` next to the calibration file, as degrees
added to the nominal stop angle. With `--gravity` the error is measured from the motors' own torque instead
of from a pose a person set up (see gravity_zero.py). Later hard-stop calibrations of that arm read the file, so they land on the
same corrected zero without the reference pose being needed again.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from pathlib import Path

from .calibration import ARM_MOTORS
from .offset_bus import HOMING_OFFSET_UNITS_PER_DEGREE

logger = logging.getLogger(__name__)

# The shoulder joints hang vertical under gravity, which makes the free-hanging pose a reference for them.
DEFAULT_TRIM_JOINTS = ("joint_1", "joint_2")
# A reading larger than this at the reference pose means the arm is not at it (or the calibration is
# stale), not that a stop is slightly off.
MAX_TRIM_STEP_DEG = 10.0
READ_SECONDS = 2.0
READ_PERIOD_S = 0.05


def stop_trim_path(calibration_fpath: Path) -> Path:
    return calibration_fpath.with_suffix(".stop_trim.json")


def load_stop_trim(calibration_fpath: Path) -> dict[str, float]:
    """This arm's stop-angle corrections in degrees, or {} if it has none."""
    path = stop_trim_path(calibration_fpath)
    if not path.is_file():
        return {}
    trim = {motor: float(value) for motor, value in json.loads(path.read_text()).items()}
    unknown = set(trim) - set(ARM_MOTORS)
    if unknown:
        raise ValueError(f"{path} names unknown joints: {sorted(unknown)}")
    return trim


def save_stop_trim(calibration_fpath: Path, trim_deg: dict[str, float]) -> None:
    stop_trim_path(calibration_fpath).write_text(json.dumps(trim_deg, indent=4) + "\n")


def read_mean_positions(bus, seconds: float = READ_SECONDS) -> dict[str, float]:
    samples: dict[str, list[float]] = {motor: [] for motor in bus.motors}
    deadline = time.monotonic() + seconds
    while True:
        for motor, state in bus.sync_read_all_states().items():
            samples[motor].append(state["position"])
        if time.monotonic() >= deadline:
            break
        time.sleep(READ_PERIOD_S)
    return {motor: statistics.fmean(values) for motor, values in samples.items()}


def apply_zero_trim(arm, readings_deg: dict[str, float]) -> dict[str, float]:
    """Make the joints in `readings_deg` read zero where they currently read those values.

    Updates the arm's calibration and its stop-trim file. Returns the arm's full stop trim.
    """
    too_large = {m: r for m, r in readings_deg.items() if abs(r) > MAX_TRIM_STEP_DEG}
    if too_large:
        raise ValueError(
            f"Readings of {too_large} at the reference pose are more than {MAX_TRIM_STEP_DEG:.0f}° from zero. "
            "Check the pose, or recalibrate with lerobot-calibrate first."
        )
    trim = load_stop_trim(arm.calibration_fpath)
    for motor, reading in readings_deg.items():
        # calibrated = raw - offset, so raising the offset by the reading makes this pose read zero. The
        # stop itself has not moved, so its angle in the corrected frame drops by the same amount.
        arm.calibration[motor].homing_offset += round(reading * HOMING_OFFSET_UNITS_PER_DEGREE)
        trim[motor] = round(trim.get(motor, 0.0) - reading, 3)
    arm.bus.write_calibration(arm.calibration)
    arm._save_calibration()
    save_stop_trim(arm.calibration_fpath, trim)
    return trim


def main() -> None:
    """`openarm-zero-trim`: correct one arm's zero at a reference pose. Nothing moves."""
    from .follower import OpenArmFollowerHardstop, OpenArmFollowerHardstopConfig

    parser = argparse.ArgumentParser(
        description="Correct a hard-stop calibrated OpenArm's zero at a reference pose. The arm goes limp; "
        "nothing is commanded to move."
    )
    parser.add_argument("--port", required=True, help="CAN interface, e.g. can0")
    parser.add_argument("--side", required=True, choices=["left", "right"])
    parser.add_argument("--id", required=True, help="The arm's calibration id, e.g. my_bimanual_follower_left")
    parser.add_argument("--calibration-dir", type=Path, default=None)
    parser.add_argument("--no-can-fd", action="store_true")
    parser.add_argument(
        "--joints",
        nargs="+",
        default=None,
        choices=ARM_MOTORS,
        help="Joints to correct. Default: joint_1 joint_2 (they hang vertical under gravity), plus joint_4 "
        "with --gravity",
    )
    parser.add_argument(
        "--gravity",
        action="store_true",
        help="Find the zero from gravity torque: the arm sweeps each joint slowly, forward and back. "
        "The ARM MOVES. Supports joint_1, joint_2 and joint_4.",
    )
    parser.add_argument("--passes", type=int, default=2, help="With --gravity: times each joint is swept")
    parser.add_argument(
        "--shift",
        nargs="+",
        metavar="JOINT=DEG",
        help="Instead of reading a reference pose, move joints' zero by hand: joint_1=1.5 makes every commanded "
        "pose sit 1.5° further in joint_1's positive direction. Nothing is read from or sent to the arm.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)

    arm = OpenArmFollowerHardstop(
        OpenArmFollowerHardstopConfig(
            port=args.port,
            side=args.side,
            id=args.id,
            calibration_dir=args.calibration_dir,
            use_can_fd=not args.no_can_fd,
        )
    )
    if not arm.calibration:
        raise SystemExit(f"No calibration at {arm.calibration_fpath}. Run lerobot-calibrate first.")

    if args.shift:
        try:
            shifts = {motor: float(value) for motor, value in (item.split("=", 1) for item in args.shift)}
        except ValueError:
            parser.error("--shift takes JOINT=DEG pairs, e.g. --shift joint_1=1.5 joint_4=-0.5")
        if unknown := set(shifts) - set(ARM_MOTORS):
            parser.error(f"--shift names unknown joints: {sorted(unknown)}")
        # A pose that reads +x where it should read zero and a zero that should move by +x are the same
        # correction.
        trim = apply_zero_trim(arm, shifts)
        print(f"Saved {arm.calibration_fpath} and {stop_trim_path(arm.calibration_fpath)}")
        print("Stop trim is now: " + ", ".join(f"{m} {v:+.2f}°" for m, v in trim.items()))
        return

    if args.gravity:
        _gravity_trim(arm, args)
        return
    if args.joints is None:
        args.joints = list(DEFAULT_TRIM_JOINTS)

    arm.bus.connect()
    try:
        arm.bus.disable_torque()
        input(
            f"\nThe {args.side} arm is limp. Bring {', '.join(args.joints)} to zero: let the arm hang "
            "straight down and jiggle it so it settles where gravity puts it.\nPress ENTER to read..."
        )
        positions = read_mean_positions(arm.bus)
        readings = {motor: positions[motor] for motor in args.joints}
        old_trim = load_stop_trim(arm.calibration_fpath)
        print(f"\n  {'joint':<8} {'reads':>8} {'stop trim':>10} {'new trim':>9}")
        for motor, reading in readings.items():
            old = old_trim.get(motor, 0.0)
            print(f"  {motor:<8} {reading:+8.2f} {old:+10.2f} {old - reading:+9.2f}")
        if input("\nMake this pose zero for those joints? Type 'y' and press ENTER: ").strip().lower() != "y":
            print("Nothing changed.")
            return
        apply_zero_trim(arm, readings)
        after = read_mean_positions(arm.bus, 0.5)
        print(f"Saved {arm.calibration_fpath} and {stop_trim_path(arm.calibration_fpath)}")
        print("They now read: " + ", ".join(f"{m} {after[m]:+.2f}°" for m in args.joints))
    finally:
        arm.bus.disconnect(disable_torque=True)


def _gravity_trim(arm, args) -> None:
    arm.bus.connect()
    try:
        run_gravity_zero(arm, args.side, args.joints, args.passes)
    finally:
        arm.bus.disconnect(disable_torque=True)


def run_gravity_zero(arm, side: str, joints: list[str] | None = None, passes: int = 2) -> dict[str, float]:
    """Find the joints' zero from gravity torque on a connected, calibrated arm, and save it once the user
    confirms. The arm moves. Returns the corrections applied (empty if none were).

    Torque is released at the end, also after an error, which is re-raised.
    """
    from .gravity_zero import GravityZeroFinder, gravity_sweeps

    sweeps = gravity_sweeps(side, joints)
    names = ", ".join(sweep.motor for sweep in sweeps)
    apply = False
    readings: dict[str, float] = {}
    try:
        input(
            f"\nGravity zero for {names} on the {side} arm.\n"
            "The arm WILL MOVE ON ITS OWN: each shoulder joint swings a few degrees either side of hanging, "
            "the elbow bends to 90 degrees and back, and then the upper arm tilts 30 degrees back while the "
            "elbow moves between 10 and 60 degrees.\n"
            "  - Clear the space in front of, beside and behind the arm and keep the emergency stop in reach\n"
            "  - Let the arm hang straight down, elbow straight\n"
            "Press ENTER to start (Ctrl+C at any time stops the arm and holds it in place)..."
        )
        fits = GravityZeroFinder(arm.bus).find(sweeps, passes)

        old_trim = load_stop_trim(arm.calibration_fpath)
        print(f"\n  {'joint':<8} {'zero is at':>10} {'±':>6} {'pass diff':>10} {'gravity':>9} {'friction':>9} {'new trim':>9}")
        for motor, passes in fits.items():
            fit = passes[-1]
            diff = f"{fit.zero_deg - passes[-2].zero_deg:+.2f}" if len(passes) > 1 else "-"
            if fit.trusted:
                readings[motor] = fit.zero_deg
                new_trim = f"{old_trim.get(motor, 0.0) - fit.zero_deg:+9.2f}"
            else:
                new_trim = "  not used"
            print(
                f"  {motor:<8} {fit.zero_deg:+10.2f} {fit.stderr_deg:6.2f} {diff:>10} {fit.amplitude_nm:8.2f}N "
                f"{fit.friction_nm:8.2f}N {new_trim}"
            )
        print("The arm is holding that zero now. 'pass diff' is how far the last pass moved it.")
        for motor, passes in fits.items():
            fit = passes[-1]
            if fit.two_sided_zero_deg is not None:
                print(
                    f"{motor}: fitted constant torque {fit.constant_nm:+.3f} Nm. The tilted sweeps alone, with no "
                    f"constant fitted, put the zero at {fit.two_sided_zero_deg:+.2f}°."
                )
        if len(readings) < len(fits):
            print("Joints marked 'not used' had too uncertain a fit and are left as they are.")
        if readings:
            apply = input("\nMake it the joints' zero? Type 'y' and press ENTER: ").strip().lower() == "y"
    except BaseException as e:
        logger.error(f"Gravity zero stopped: {e!r}")
        raise
    finally:
        try:
            input("The arm is holding its position. Support it, then press ENTER to release torque...")
        finally:
            arm.bus.disable_torque()
            if apply:
                apply_zero_trim(arm, readings)
                print(f"Saved {arm.calibration_fpath} and {stop_trim_path(arm.calibration_fpath)}")
            else:
                print("Nothing changed.")
    return readings if apply else {}


if __name__ == "__main__":
    main()
