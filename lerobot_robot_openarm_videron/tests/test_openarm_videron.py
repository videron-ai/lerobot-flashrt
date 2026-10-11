"""openarm_videron against simulated arms: LeRobot's real Damiao bus, the streaming follower's real control
loops and the hard-stop plugin's real calibration, with no hardware."""

import json
import time
from dataclasses import dataclass

import draccus
import pytest
from lerobot.robots.config import RobotConfig
from lerobot.robots.utils import make_robot_from_config
from lerobot_robot_openarm_hardstop import CalibrationMismatchError, OpenArmFollowerHardstop
from lerobot_robot_openarm_hardstop.calibration import offsets_to_calibration
from lerobot_robot_openarm_streaming import StreamSample, register_target_source, unregister_target_source
from sim_openarm import MOTORS

from lerobot_robot_openarm_videron import OpenArmVideron, OpenArmVideronConfig

HZ = 100
SIDES = ("left", "right")
HIDDEN_ZERO = {
    "left": {"joint_1": 17.3, "joint_2": -6.2, "joint_3": 24.9, "joint_4": -11.1,
             "joint_5": 3.3, "joint_6": -28.4, "joint_7": 31.2, "gripper": 7.7},
    "right": {"joint_1": -2.1, "joint_2": -8.0, "joint_3": -1.5, "joint_4": -1.5,
              "joint_5": 1.8, "joint_6": 0.2, "joint_7": -1.9, "gripper": 1.6},
}  # fmt: skip


@dataclass
class _Cli:
    robot: RobotConfig


def make_robot(tmp_path, *extra_args):
    """Parse CLI args the way lerobot-record does and build the robot through plugin discovery."""
    args = [
        "--robot.type=openarm_videron",
        "--robot.id=rig",
        f"--robot.calibration_dir={tmp_path}",
        "--robot.left_arm_config.port=can_left",
        "--robot.right_arm_config.port=can_right",
        f"--robot.control_hz={HZ}",
        *extra_args,
    ]
    return make_robot_from_config(draccus.parse(_Cli, args=args).robot)


def add_arms(world):
    return {
        side: world.add(f"can_{side}", side=side, hidden_zero_deg=HIDDEN_ZERO[side], seed=i)
        for i, side in enumerate(SIDES)
    }


def save_true_calibration(robot):
    """Write the calibration files hard-stop calibration would produce, without running it."""
    for side, arm in (("left", robot.left_arm), ("right", robot.right_arm)):
        ids = {name: motor.id for name, motor in arm.bus.motors.items()}
        arm.calibration = offsets_to_calibration(HIDDEN_ZERO[side], ids)
        arm.bus.calibration = arm.calibration
        arm._save_calibration()


def assert_observes_true_angles(robot, sims, tolerance_deg=0.3):
    obs = robot.get_observation()
    for side in SIDES:
        for motor in MOTORS:
            assert obs[f"{side}_{motor}.pos"] == pytest.approx(sims[side].calibrated_deg(motor), abs=tolerance_deg)


def drive(robot, action, sims, timeout_s=10.0, tolerance_deg=1.5):
    """Send `action` at 30 Hz, like a record loop, until the simulated joints are really there."""
    deadline = time.monotonic() + timeout_s
    errors = {}
    while time.monotonic() < deadline:
        robot.send_action(action)
        time.sleep(1 / 30)
        errors = {
            key: sims[key.split("_", 1)[0]].calibrated_deg(key.split("_", 1)[1].removesuffix(".pos")) - target
            for key, target in action.items()
        }
        if all(abs(e) < tolerance_deg for e in errors.values()):
            return
    raise AssertionError(f"joints did not reach their targets (errors in deg: {errors})")


def test_built_through_plugin_discovery_with_hardstop_arms(tmp_path):
    robot = make_robot(
        tmp_path,
        "--robot.left_arm_config.joint_limits={'joint_1': [-200, 80], 'gripper': [-160, 0]}",
    )
    assert type(robot) is OpenArmVideron
    assert robot.name == "openarm_videron"
    # Sides follow from the arm's position, and limits given in the config survive setting a side.
    assert (robot.left_arm.config.side, robot.right_arm.config.side) == ("left", "right")
    assert type(robot.left_arm) is type(robot.right_arm) is OpenArmFollowerHardstop
    assert robot.left_arm.config.joint_limits["gripper"] == (-160.0, 0.0)
    assert robot.right_arm.config.joint_limits["gripper"] == (-65.0, 0.0)  # the side's preset
    assert set(robot.action_features) == {f"{s}_{m}.pos" for s in SIDES for m in MOTORS}


def test_wrong_side_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="left_arm_config.side=left"):
        make_robot(tmp_path, "--robot.left_arm_config.side=right")


def test_first_connect_calibrates_then_streams_in_the_calibrated_frame(world, tmp_path):
    sims = add_arms(world)
    robot = make_robot(tmp_path)
    robot.connect()
    try:
        for side in SIDES:
            saved = json.loads((tmp_path / f"rig_{side}.json").read_text())
            for motor in MOTORS:
                assert saved[motor]["homing_offset"] / 1000 == pytest.approx(HIDDEN_ZERO[side][motor], abs=0.2)
            assert sims[side].set_zero_count == 0  # the motors' own zero is never rewritten
        # Each arm's hard-stop calibration was followed by its gravity zero step.
        assert world.gravity_zero_calls == list(SIDES)
        assert all(s["ticks"] > 0 for s in robot.streaming_stats().values())
        assert_observes_true_angles(robot, sims)

        drive(robot, {"left_joint_4.pos": 40.0, "right_joint_3.pos": -25.0, "right_joint_5.pos": 30.0}, sims)
        assert_observes_true_angles(robot, sims)
    finally:
        robot.disconnect()
    assert not any(sim.enabled.any() for sim in sims.values())


def test_saved_calibration_is_reused_and_a_teleop_stream_lands_on_true_angles(world, tmp_path):
    sims = add_arms(world)
    save_true_calibration(make_robot(tmp_path))
    for sim in sims.values():
        sim.power_cycle()

    targets = {f"{side}_{motor}.pos": 0.0 for side in SIDES for motor in MOTORS}
    targets.update({"left_joint_4.pos": 35.0, "right_joint_4.pos": 50.0, "right_joint_7.pos": -40.0})

    class Source:
        def sample(self, now):
            return StreamSample(positions=targets, velocities={}, active=True, stamp=now)

    source = Source()
    register_target_source(source)
    robot = make_robot(tmp_path)
    try:
        robot.connect()
        assert world.prompts == []  # no calibration prompt: the saved files were used
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            errors = {
                key: sims[key.split("_", 1)[0]].calibrated_deg(key.split("_", 1)[1].removesuffix(".pos")) - target
                for key, target in targets.items()
            }
            if all(abs(e) < 1.5 for e in errors.values()):
                break
            time.sleep(0.05)
        else:
            raise AssertionError(f"the stream's targets were not reached (errors in deg: {errors})")
        assert {s["source"] for s in robot.streaming_stats().values()} == {"stream"}
        assert_observes_true_angles(robot, sims)
    finally:
        unregister_target_source(source)
        if robot.is_connected:
            robot.disconnect()


def test_lerobot_calibrate_flow_recalibrates_a_connected_robot(world, tmp_path):
    sims = add_arms(world)
    robot = make_robot(tmp_path)
    # The files on disk are wrong for the right arm by a few degrees, as after a motor was re-seated.
    save_true_calibration(robot)
    stale = json.loads((tmp_path / "rig_right.json").read_text())
    stale["joint_3"]["homing_offset"] += 4000
    (tmp_path / "rig_right.json").write_text(json.dumps(stale))

    robot = make_robot(tmp_path)
    robot.connect(calibrate=False)  # what lerobot-calibrate does; the control loops are now running
    try:
        loops_before = dict(robot._loops)
        world.answers = ["c", "", "c", ""]  # per arm: recalibrate, then confirm the arm is clear
        robot.calibrate()

        assert all(not loop.is_alive() for loop in loops_before.values())
        assert all(loop.is_alive() for loop in robot._loops.values())
        saved = json.loads((tmp_path / "rig_right.json").read_text())
        assert saved["joint_3"]["homing_offset"] / 1000 == pytest.approx(HIDDEN_ZERO["right"]["joint_3"], abs=0.2)
        assert_observes_true_angles(robot, sims)
    finally:
        robot.disconnect()


def test_stale_calibration_refuses_to_connect(world, tmp_path):
    sims = add_arms(world)
    save_true_calibration(make_robot(tmp_path))
    sims["left"].bias[MOTORS.index("joint_4")] -= 1.2  # the motor was re-zeroed ~70 degrees away

    robot = make_robot(tmp_path)
    with pytest.raises(CalibrationMismatchError, match="joint_4"):
        robot.connect()
    assert robot._loops == {}
    assert not sims["left"].enabled.any()


def test_ker_stream_drives_the_calibrated_arms(world, tmp_path, monkeypatch):
    """The OpenArm KER teleoperator streaming straight into openarm_videron's control loops."""
    km = pytest.importorskip("lerobot_teleoperator_openarm_ker.openarm_ker")
    from lerobot_teleoperator_openarm_ker import OpenArmKER, OpenArmKERConfig

    class FakeKERStream:
        """Stands in for openarm_ker.KERStream: 500 Hz frames holding a fixed pose."""

        metadata = {"fw": "sim"}
        is_connected = True
        is_link_up = True

        def __init__(self, **_):
            self.angles = [0.0] * 16
            self.angles[3] = 40.0  # right J4 channel
            self.angles[8] = -20.0  # left J1 channel

        def connect(self):
            pass

        def send_command(self, command):
            pass

        def close(self):
            pass

        def latest(self):
            seq = int(time.monotonic() * 500)
            return {"timestamp": seq, "seq": seq, "angles": list(self.angles), "errors": [False] * 16}

    monkeypatch.setattr(km, "KERStream", FakeKERStream)
    sims = add_arms(world)
    save_true_calibration(make_robot(tmp_path))
    # The KER's gripper range is the rig's 0 to -160 deg, so the follower needs the rig's gripper limit.
    limits = "{'gripper': [-160, 0], 'joint_1': [-200, 200], 'joint_4': [0, 140]}"
    robot = make_robot(
        tmp_path,
        f"--robot.left_arm_config.joint_limits={limits}",
        f"--robot.right_arm_config.joint_limits={limits}",
    )
    ker = OpenArmKER(OpenArmKERConfig(stream_to_follower=True))
    ker.connect()
    try:
        robot.connect()
        arm_keys = [k for k in ker.action_features if "gripper" not in k]
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline:
            action = ker.get_action()  # polling it is what marks the stream as active
            robot.send_action(action)
            time.sleep(1 / 30)
            errors = {
                key: sims[key.split("_", 1)[0]].calibrated_deg(key.split("_", 1)[1].removesuffix(".pos")) - action[key]
                for key in arm_keys
            }
            if all(abs(e) < 1.5 for e in errors.values()):
                break
        else:
            raise AssertionError(f"the followers did not reach the KER pose (errors in deg: {errors})")
        assert action["right_joint_4.pos"] == pytest.approx(40.0, abs=0.5)
        assert {s["source"] for s in robot.streaming_stats().values()} == {"stream"}
        obs = robot.get_observation()
        for key in arm_keys:
            assert obs[key] == pytest.approx(action[key], abs=1.5), key
    finally:
        if robot.is_connected:
            robot.disconnect()
        ker.disconnect()
