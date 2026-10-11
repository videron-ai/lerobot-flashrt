# SPDX-License-Identifier: Apache-2.0
"""Stop trim: an arm whose stops are not at their nominal angles, corrected at a reference pose."""

import json

import numpy as np
import pytest
from conftest import HIDDEN_ZERO
from lerobot.robots.utils import make_robot_from_config
from sim_openarm import MOTORS, STOPS

from lerobot_robot_openarm_hardstop import OpenArmFollowerHardstopConfig
from lerobot_robot_openarm_hardstop.stop_trim import main as stop_trim_main
from lerobot_robot_openarm_hardstop.stop_trim import (
    apply_zero_trim,
    load_stop_trim,
    read_mean_positions,
    stop_trim_path,
)

# This arm's shoulder stops sit past their nominal -80° and -10°.
REAL_STOPS = {**STOPS["right"], "joint_1": (-82.5, 200.0), "joint_2": (-13.0, 100.0)}
STOP_ERROR = {"joint_1": 2.5, "joint_2": 3.0}


def make_arm(tmp_path):
    return make_robot_from_config(
        OpenArmFollowerHardstopConfig(port="can0", side="right", id="sim", calibration_dir=tmp_path)
    )


def hang(sim):
    sim.enabled[:] = False
    sim.q[:] = 0.0
    sim.dq[:] = 0.0


def test_trim_corrects_the_zero_and_later_calibrations_keep_it(world, tmp_path, monkeypatch):
    sim = world.add(hidden_zero_deg=HIDDEN_ZERO, stops=REAL_STOPS, gravity=np.zeros(len(MOTORS)))
    arm = make_arm(tmp_path)
    arm.connect()  # hard-stop calibration against the nominal stop angles

    # Hanging at its true zero, the arm reads the stops' error on the shoulder joints.
    hang(sim)
    readings = {m: v for m, v in read_mean_positions(arm.bus, 0.1).items() if m in STOP_ERROR}
    for motor, error in STOP_ERROR.items():
        assert readings[motor] == pytest.approx(error, abs=0.3)

    trim = apply_zero_trim(arm, readings)
    for motor, error in STOP_ERROR.items():
        assert trim[motor] == pytest.approx(-error, abs=0.3)
    after = read_mean_positions(arm.bus, 0.1)
    for motor in MOTORS:
        assert after[motor] == pytest.approx(0.0, abs=0.3), motor
    arm.disconnect()

    # A fresh hard-stop calibration reads the trim file and lands on the corrected zero by itself.
    answers = iter(["c"])  # recalibrate instead of reusing the file; ENTER for the prompts after it
    monkeypatch.setattr("builtins.input", lambda *_: next(answers, ""))
    arm = make_arm(tmp_path)
    arm.connect(calibrate=False)
    arm.calibrate()
    saved = json.loads((tmp_path / "sim.json").read_text())
    for motor in MOTORS:
        assert saved[motor]["homing_offset"] / 1000 == pytest.approx(HIDDEN_ZERO[motor], abs=0.3), motor
    assert load_stop_trim(arm.calibration_fpath) == trim
    arm.disconnect()


def test_a_reading_far_from_zero_is_refused(world, tmp_path):
    world.add(hidden_zero_deg=HIDDEN_ZERO)
    arm = make_arm(tmp_path)
    arm.connect()
    before = json.loads((tmp_path / "sim.json").read_text())
    with pytest.raises(ValueError, match="reference pose"):
        apply_zero_trim(arm, {"joint_2": 25.0})
    assert json.loads((tmp_path / "sim.json").read_text()) == before
    assert not stop_trim_path(arm.calibration_fpath).exists()
    arm.disconnect()


def test_shift_moves_where_a_commanded_pose_sits(world, tmp_path, monkeypatch):
    sim = world.add(hidden_zero_deg=HIDDEN_ZERO)
    arm = make_arm(tmp_path)
    arm.connect()
    arm.disconnect()

    args = ["openarm-zero-trim", "--port=can0", "--side=right", "--id=sim", f"--calibration-dir={tmp_path}"]
    monkeypatch.setattr("sys.argv", [*args, "--shift", "joint_3=1.5"])
    stop_trim_main()
    assert load_stop_trim(arm.calibration_fpath) == {"joint_3": -1.5}

    arm = make_arm(tmp_path)
    arm.connect()
    arm.send_action({"joint_3.pos": 10.0})
    world.sleep(2.0)
    assert sim.calibrated_deg("joint_3") == pytest.approx(11.5, abs=0.5)
    arm.disconnect()
