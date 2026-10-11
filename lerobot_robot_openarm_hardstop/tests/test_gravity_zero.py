# SPDX-License-Identifier: Apache-2.0
"""Gravity zero: joint zeros found from the torque needed to hold the arm against its own weight."""

import math

import numpy as np
import pytest
from conftest import HIDDEN_ZERO
from lerobot.robots.utils import make_robot_from_config
from sim_openarm import MOTORS, STOPS

from lerobot_robot_openarm_hardstop import (
    CalibrationAbortedError,
    GravityZeroError,
    OpenArmFollowerHardstopConfig,
)
from lerobot_robot_openarm_hardstop.calibration import offsets_to_calibration
from lerobot_robot_openarm_hardstop.gravity_zero import (
    GravityZeroFinder,
    fit_gravity_segments,
    fit_gravity_zero,
    gravity_sweeps,
)
from lerobot_robot_openarm_hardstop.stop_trim import apply_zero_trim, load_stop_trim, read_mean_positions

# Roughly the rig's arm: about 13 Nm to hold a shoulder joint horizontal, 2.5 Nm for the forearm.
ARM_GRAVITY = np.array([13.0, 13.0, 0.0, 2.5, 0.0, 0.3, 0.0, 0.0])


def synthetic_sweep(zero_deg, amplitude, friction, constant, lo, hi, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    q = np.arange(lo, hi, 0.02)
    gravity = amplitude * np.sin(np.radians(q - zero_deg)) + constant
    forward = list(zip(q, gravity + friction + rng.normal(0, noise, len(q))))
    backward = list(zip(q, gravity - friction + rng.normal(0, noise, len(q))))
    return forward, backward


def test_fit_recovers_the_zero_through_friction():
    forward, backward = synthetic_sweep(2.7, 13.0, 0.8, 0.0, -12, 12)
    fit = fit_gravity_zero("joint_1", forward, backward)
    assert fit.zero_deg == pytest.approx(2.7, abs=0.05)
    assert fit.amplitude_nm == pytest.approx(13.0, rel=0.05)
    assert fit.friction_nm == pytest.approx(0.8, abs=0.05)
    assert fit.stderr_deg < 0.05


def test_fit_is_indifferent_to_the_torque_sign_convention():
    forward, backward = synthetic_sweep(-1.9, 13.0, 0.5, 0.0, -16, 4)
    flipped = lambda samples: [(q, -tau) for q, tau in samples]  # noqa: E731
    assert fit_gravity_zero("joint_2", flipped(backward), flipped(forward)).zero_deg == pytest.approx(-1.9, abs=0.05)


def test_one_sided_fit_separates_a_constant_torque_from_the_zero():
    forward, backward = synthetic_sweep(1.2, 2.5, 0.3, 0.15, 3, 90)
    with_constant = fit_gravity_zero("joint_4", forward, backward, fit_constant=True)
    assert with_constant.zero_deg == pytest.approx(1.2, abs=0.15)
    assert with_constant.constant_nm == pytest.approx(0.15, abs=0.02)
    # Without the constant term the same data puts the zero in the wrong place.
    assert abs(fit_gravity_zero("joint_4", forward, backward).zero_deg - 1.2) > 1.0


def test_a_tilted_sweep_tightens_the_fit_that_separates_a_constant_torque():
    """One-sided sweep alone against the same sweep plus one with the upper arm tilted 30 degrees back."""
    one_sided = synthetic_sweep(1.2, 4.4, 0.8, 0.15, 3, 90, noise=0.15)
    # Tilted 30 back, the forearm hangs vertical with the elbow at 31.2.
    tilted = synthetic_sweep(31.2, 4.4, 0.8, 0.15, 10, 60, noise=0.15, seed=1)
    alone = fit_gravity_segments("joint_4", [(*one_sided, 0.0)], fit_constant=True)
    both = fit_gravity_segments("joint_4", [(*one_sided, 0.0), (*tilted, -30.0)], fit_constant=True)
    assert both.zero_deg == pytest.approx(1.2, abs=0.3)
    assert both.constant_nm == pytest.approx(0.15, abs=0.05)
    assert both.stderr_deg < 0.7 * alone.stderr_deg
    # The tilted sweep by itself is the least scattered but absorbs the constant torque into the zero.
    tilted_only = fit_gravity_segments("joint_4", [(*tilted, -30.0)])
    assert tilted_only.stderr_deg < 0.5 * both.stderr_deg
    assert tilted_only.zero_deg - 1.2 == pytest.approx(-math.degrees(0.15 / 4.4), abs=0.3)


def test_too_little_overlap_is_refused():
    forward, backward = synthetic_sweep(0.0, 13.0, 0.5, 0.0, -2, 2)
    with pytest.raises(CalibrationAbortedError, match="overlap"):
        fit_gravity_zero("joint_1", forward, backward)


@pytest.mark.parametrize("side", ["right", "left"])
def test_finder_measures_and_trim_corrects_a_calibration_that_is_off(world, tmp_path, side):
    """Calibrated zeros a few degrees off the true ones, as the nominal stop angles leave them on the rig."""
    error = {"joint_1": 2.5, "joint_2": 3.0, "joint_4": 1.0}
    if side == "left":
        error = {"joint_1": -1.6, "joint_2": -1.4, "joint_4": 0.4}
    sim = world.add(side=side, hidden_zero_deg=HIDDEN_ZERO, gravity=ARM_GRAVITY, start_deg={})
    # The upper arm's pitch carries the elbow: back is negative joint 1 on the right arm, positive on the left.
    sim.forearm_pitch_gain = 1.0 if side == "right" else -1.0
    sim.torque_bias[MOTORS.index("joint_4")] = 0.12
    arm = make_robot_from_config(
        OpenArmFollowerHardstopConfig(port="can0", side=side, id="sim", calibration_dir=tmp_path)
    )
    offsets = {m: HIDDEN_ZERO[m] - error.get(m, 0.0) for m in MOTORS}
    arm.calibration = offsets_to_calibration(offsets, {name: motor.id for name, motor in arm.bus.motors.items()})
    arm.bus.calibration = arm.calibration
    arm._save_calibration()
    arm.bus.connect()

    fits = GravityZeroFinder(arm.bus).find(gravity_sweeps(side))
    for motor, expected in error.items():
        assert fits[motor][-1].zero_deg == pytest.approx(expected, abs=0.3), motor
        assert fits[motor][-1].trusted, motor
    # The elbow's torque reading was 0.12 Nm off; the fit found that instead of moving the zero.
    assert fits["joint_4"][-1].constant_nm == pytest.approx(0.12, abs=0.06)
    assert fits["joint_4"][-1].two_sided_zero_deg is not None
    # The arm is left held at the zero it found: hanging true.
    for motor in error:
        assert sim.calibrated_deg(motor) == pytest.approx(0.0, abs=0.6), motor

    apply_zero_trim(arm, {motor: passes[-1].zero_deg for motor, passes in fits.items()})
    after = read_mean_positions(arm.bus, 0.1)
    for motor in error:
        assert after[motor] == pytest.approx(sim.calibrated_deg(motor), abs=0.25), motor
    arm.bus.disconnect(True)


def test_finder_stops_when_a_joint_is_blocked(world, tmp_path):
    sim = world.add(side="right", hidden_zero_deg=HIDDEN_ZERO, gravity=ARM_GRAVITY, start_deg={})
    sim.hi[MOTORS.index("joint_1")] = math.radians(5.0)  # something in the way 5 degrees forward
    arm = make_robot_from_config(
        OpenArmFollowerHardstopConfig(port="can0", side="right", id="sim", calibration_dir=tmp_path)
    )
    arm.calibration = offsets_to_calibration(HIDDEN_ZERO, {name: motor.id for name, motor in arm.bus.motors.items()})
    arm.bus.calibration = arm.calibration
    arm.bus.connect()
    with pytest.raises(CalibrationAbortedError, match="in its way"):
        GravityZeroFinder(arm.bus).find(gravity_sweeps("right", ["joint_1"]))
    arm.bus.disconnect(True)


def test_calibration_ends_with_the_gravity_zero(world, tmp_path, monkeypatch):
    """A first connect on an arm whose shoulder stops are not at their nominal angles: the hard-stop
    calibration alone would leave the zero off by the stops' error; the gravity step that follows fixes it."""
    stops = {**STOPS["right"], "joint_1": (-82.5, 200.0), "joint_2": (-13.0, 100.0)}
    sim = world.add(side="right", hidden_zero_deg=HIDDEN_ZERO, stops=stops, gravity=ARM_GRAVITY, start_deg={})
    sim.forearm_pitch_gain = 1.0
    world.run_gravity_zero = True
    prompts = []
    monkeypatch.setattr("builtins.input", lambda p="": prompts.append(p) or ("y" if "Make it" in p else ""))
    arm = make_robot_from_config(
        OpenArmFollowerHardstopConfig(port="can0", side="right", id="sim", calibration_dir=tmp_path)
    )
    arm.connect()

    assert world.gravity_zero_calls == ["right"]
    assert any("Hard-stop calibration" in p for p in prompts) and any("Gravity zero" in p for p in prompts)
    # The saved zero is the true one, and the stops' error is on file for later hard-stop calibrations.
    for motor in MOTORS:
        assert arm.calibration[motor].homing_offset / 1000 == pytest.approx(HIDDEN_ZERO[motor], abs=0.4), motor
    trim = load_stop_trim(arm.calibration_fpath)
    assert trim["joint_1"] == pytest.approx(-2.5, abs=0.4)
    assert trim["joint_2"] == pytest.approx(-3.0, abs=0.4)
    positions = read_mean_positions(arm.bus, 0.1)
    for motor in ("joint_1", "joint_2", "joint_4"):
        assert positions[motor] == pytest.approx(sim.calibrated_deg(motor), abs=0.4), motor
    arm.disconnect()


def test_calibration_says_so_when_the_gravity_zero_fails(world, tmp_path):
    world.add(side="right", hidden_zero_deg=HIDDEN_ZERO, gravity=np.zeros(len(MOTORS)), start_deg={})
    world.run_gravity_zero = True
    arm = make_robot_from_config(
        OpenArmFollowerHardstopConfig(port="can0", side="right", id="sim", calibration_dir=tmp_path)
    )
    with pytest.raises(GravityZeroError, match="hard-stop calibration .* is saved"):
        arm.connect()
    # The hard-stop calibration is kept and no correction was written.
    assert arm.calibration_fpath.is_file()
    assert load_stop_trim(arm.calibration_fpath) == {}
    arm.bus.disconnect(True)
