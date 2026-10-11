# SPDX-License-Identifier: Apache-2.0
"""The plugin's device types, end to end, through LeRobot's own factories and CLI."""

import importlib.metadata
import json
import math
import os
import subprocess
import sys

import pytest
from conftest import HIDDEN_ZERO
from lerobot.robots.utils import make_robot_from_config
from lerobot.teleoperators.utils import make_teleoperator_from_config
from sim_openarm import MOTORS

from lerobot_robot_openarm_hardstop import calibration as hsc
from lerobot_robot_openarm_hardstop import (
    BiOpenArmFollowerHardstop,
    BiOpenArmFollowerHardstopConfig,
    BiOpenArmLeaderHardstopConfig,
    CalibrationMismatchError,
    OpenArmFollowerHardstop,
    OpenArmFollowerHardstopConfig,
    OpenArmLeaderHardstop,
    OpenArmLeaderHardstopConfig,
)


def follower_config(tmp_path, side="right", port="can0", **kwargs):
    return OpenArmFollowerHardstopConfig(port=port, side=side, id="sim", calibration_dir=tmp_path, **kwargs)


def assert_reads_true_angles(readings, sim, prefix=""):
    for motor in MOTORS:
        assert readings[f"{prefix}{motor}.pos"] == pytest.approx(sim.calibrated_deg(motor), abs=0.2), motor


def test_factory_builds_plugin_types(tmp_path):
    robot = make_robot_from_config(follower_config(tmp_path))
    assert type(robot) is OpenArmFollowerHardstop
    leader = make_teleoperator_from_config(
        OpenArmLeaderHardstopConfig(port="can1", side="left", id="sim", calibration_dir=tmp_path)
    )
    assert type(leader) is OpenArmLeaderHardstop


def test_follower_calibrates_on_first_connect_and_reuses(world, tmp_path):
    sim = world.add(hidden_zero_deg=HIDDEN_ZERO)
    robot = make_robot_from_config(follower_config(tmp_path))
    robot.connect()

    saved = json.loads((tmp_path / "sim.json").read_text())
    for motor in MOTORS:
        assert saved[motor]["homing_offset"] / 1000 == pytest.approx(HIDDEN_ZERO[motor], abs=0.15)
    assert sim.set_zero_count == 0
    assert_reads_true_angles(robot.get_observation(), sim)

    # Commanding the calibrated zero pose puts every joint at its true zero.
    robot.send_action({"joint_3.pos": 0.0, "joint_5.pos": 0.0})
    world.sleep(1.5)
    assert abs(sim.calibrated_deg("joint_3")) < 0.5
    assert abs(sim.calibrated_deg("joint_5")) < 0.5
    robot.disconnect()

    # Power off and on: readings are unchanged and a fresh instance reuses the file without moving.
    sim.power_cycle()
    robot = make_robot_from_config(follower_config(tmp_path))
    robot.connect()
    assert_reads_true_angles(robot.get_observation(), sim)
    assert sim.set_zero_count == 0
    robot.disconnect()


def test_follower_refuses_calibration_that_no_longer_matches(world, tmp_path):
    sim = world.add(hidden_zero_deg=HIDDEN_ZERO)
    robot = make_robot_from_config(follower_config(tmp_path))
    robot.connect()
    robot.disconnect()

    # Someone re-zeroes joint 4 with the elbow bent at 100°.
    sim.bias[MOTORS.index("joint_4")] = math.radians(-100.0)

    robot = make_robot_from_config(follower_config(tmp_path))
    with pytest.raises(CalibrationMismatchError, match="joint_4"):
        robot.connect()
    assert not sim.enabled.any()

    # lerobot-calibrate (connect without calibrating, then calibrate) goes straight to recalibrating.
    robot = make_robot_from_config(follower_config(tmp_path))
    robot.connect(calibrate=False)
    robot.calibrate()
    assert_reads_true_angles(robot.get_observation(), sim)
    robot.disconnect()


def test_leader_never_rezeros_and_stays_limp(world, tmp_path):
    sim = world.add(side="left", hidden_zero_deg=HIDDEN_ZERO)
    config = OpenArmLeaderHardstopConfig(port="can0", side="left", id="leader", calibration_dir=tmp_path)
    for _ in range(2):  # first connect calibrates, second reuses the file
        leader = make_teleoperator_from_config(config)
        leader.connect()
        assert sim.set_zero_count == 0
        assert not sim.enabled.any()  # manual_control leaves the leader free to move by hand
        assert_reads_true_angles(leader.get_action(), sim)
        leader.disconnect()


def test_requires_side(tmp_path):
    with pytest.raises(ValueError, match="side"):
        OpenArmFollowerHardstop(OpenArmFollowerHardstopConfig(port="can0", calibration_dir=tmp_path))
    with pytest.raises(ValueError, match="side"):
        OpenArmLeaderHardstop(OpenArmLeaderHardstopConfig(port="can0", calibration_dir=tmp_path))


def test_custom_joint_limits_are_kept(tmp_path):
    custom = {**OpenArmFollowerHardstopConfig(port="can0").joint_limits, "gripper": [-160.0, 0.0]}
    arm = OpenArmFollowerHardstop(
        OpenArmFollowerHardstopConfig(port="can0", side="left", calibration_dir=tmp_path, joint_limits=custom)
    )
    assert arm.config.joint_limits["gripper"] == (-160.0, 0.0)
    # A wide-open gripper is inside the configured limits, so it is no sign of a stale calibration.
    assert hsc.position_problems({"gripper": -150.0}, "left", arm.config.joint_limits) == []
    assert hsc.position_problems({"gripper": -150.0}, "left") != []

    preset = OpenArmFollowerHardstop(
        OpenArmFollowerHardstopConfig(port="can0", side="left", calibration_dir=tmp_path)
    )
    assert preset.config.joint_limits["gripper"] == (-65.0, 0.0)


def test_bimanual_follower_keeps_custom_joint_limits(tmp_path):
    config = _bi_follower_config(tmp_path)
    config.left_arm_config.joint_limits = {**config.left_arm_config.joint_limits, "gripper": (-160.0, 0.0)}
    robot = make_robot_from_config(config)
    assert robot.left_arm.config.joint_limits["gripper"] == (-160.0, 0.0)
    assert robot.right_arm.config.joint_limits["gripper"] == (-65.0, 0.0)


def _bi_follower_config(tmp_path, left_side="left"):
    from lerobot.robots.openarm_follower import OpenArmFollowerConfigBase

    return BiOpenArmFollowerHardstopConfig(
        id="bi",
        calibration_dir=tmp_path,
        left_arm_config=OpenArmFollowerConfigBase(port="can0", side=left_side),
        right_arm_config=OpenArmFollowerConfigBase(port="can1", side="right"),
    )


def test_bimanual_follower(world, tmp_path):
    left = world.add("can0", side="left", hidden_zero_deg=HIDDEN_ZERO)
    right_zero = {m: -v for m, v in HIDDEN_ZERO.items()}
    right = world.add("can1", side="right", hidden_zero_deg=right_zero, seed=1)

    robot = make_robot_from_config(_bi_follower_config(tmp_path))
    assert type(robot) is BiOpenArmFollowerHardstop
    assert type(robot.left_arm) is OpenArmFollowerHardstop
    robot.connect()

    obs = robot.get_observation()
    assert_reads_true_angles(obs, left, "left_")
    assert_reads_true_angles(obs, right, "right_")
    assert {p.name for p in tmp_path.iterdir()} == {"bi_left.json", "bi_right.json"}
    assert left.set_zero_count == right.set_zero_count == 0
    robot.disconnect()


def test_bimanual_follower_requires_sides(tmp_path):
    with pytest.raises(ValueError, match="left_arm_config.side=left"):
        make_robot_from_config(_bi_follower_config(tmp_path, left_side=None))


def test_bimanual_leader(world, tmp_path):
    from lerobot.teleoperators.openarm_leader import OpenArmLeaderConfigBase

    left = world.add("can2", side="left", hidden_zero_deg=HIDDEN_ZERO)
    right = world.add("can3", side="right", hidden_zero_deg=HIDDEN_ZERO, seed=2)
    config = BiOpenArmLeaderHardstopConfig(
        id="bil",
        calibration_dir=tmp_path,
        left_arm_config=OpenArmLeaderConfigBase(port="can2"),
        right_arm_config=OpenArmLeaderConfigBase(port="can3"),
    )
    leader = make_teleoperator_from_config(config)
    leader.connect()

    action = leader.get_action()
    assert_reads_true_angles(action, left, "left_")
    assert_reads_true_angles(action, right, "right_")
    assert left.set_zero_count == right.set_zero_count == 0
    leader.disconnect()


def test_lerobot_calibrate_cli(world, tmp_path, monkeypatch):
    from lerobot.scripts import lerobot_calibrate

    world.add(hidden_zero_deg=HIDDEN_ZERO)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lerobot-calibrate",
            "--robot.type=openarm_follower_hardstop",
            "--robot.port=can0",
            "--robot.side=right",
            "--robot.id=cli",
            f"--robot.calibration_dir={tmp_path}",
        ],
    )
    lerobot_calibrate.main()

    saved = json.loads((tmp_path / "cli.json").read_text())
    assert saved["joint_6"]["homing_offset"] / 1000 == pytest.approx(HIDDEN_ZERO["joint_6"], abs=0.15)


def _plugin_installed() -> bool:
    try:
        importlib.metadata.distribution("lerobot_robot_openarm_hardstop")
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


@pytest.mark.skipif(not _plugin_installed(), reason="plugin is not pip-installed")
def test_lerobot_discovers_installed_plugin():
    code = (
        "from lerobot.utils.import_utils import register_third_party_plugins\n"
        "register_third_party_plugins()\n"
        "from lerobot.robots.config import RobotConfig\n"
        "from lerobot.teleoperators.config import TeleoperatorConfig\n"
        "print(RobotConfig.get_choice_class('openarm_follower_hardstop').__name__)\n"
        "print(TeleoperatorConfig.get_choice_class('bi_openarm_leader_hardstop').__name__)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=os.environ.copy(), check=True
    ).stdout.split()
    assert out[-2:] == ["OpenArmFollowerHardstopConfig", "BiOpenArmLeaderHardstopConfig"]


def test_gravity_zero_follows_follower_calibration_unless_turned_off(world, tmp_path):
    world.add(hidden_zero_deg=HIDDEN_ZERO)
    robot = make_robot_from_config(follower_config(tmp_path / "on"))
    robot.connect()
    assert world.gravity_zero_calls == ["right"]
    robot.disconnect()

    # A saved calibration is reused without it.
    robot = make_robot_from_config(follower_config(tmp_path / "on"))
    robot.connect()
    assert world.gravity_zero_calls == ["right"]
    robot.disconnect()

    robot = make_robot_from_config(follower_config(tmp_path / "off", gravity_zero=False))
    robot.connect()
    assert world.gravity_zero_calls == ["right"]
    robot.disconnect()


def test_leader_calibration_has_no_gravity_zero_by_default(world, tmp_path):
    world.add(port="can1", side="left", hidden_zero_deg=HIDDEN_ZERO)
    leader = make_teleoperator_from_config(
        OpenArmLeaderHardstopConfig(port="can1", side="left", id="sim", calibration_dir=tmp_path)
    )
    leader.connect()
    assert world.gravity_zero_calls == []
    leader.disconnect()
