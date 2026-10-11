# SPDX-License-Identifier: Apache-2.0
import pytest
from conftest import HIDDEN_ZERO
from sim_openarm import GRAVITY, MOTORS, SEND_IDS, STOPS
from test_offset_bus import make_bus

from lerobot_robot_openarm_hardstop import calibration as hsc

# Starting poses: as instructed (elbow straight and gripper closed, i.e. already at their stops), or a bit off.
AT_STOPS = {"joint_4": 0.0, "gripper": 0.0}
SLIGHTLY_OFF = {"joint_4": 3.0, "gripper": -2.0, "joint_3": -4.0}


@pytest.mark.parametrize(
    ("side", "shoulder_gravity_nm", "start_deg"),
    [
        ("right", 0.0, AT_STOPS),
        ("left", 12.0, AT_STOPS),
        ("right", 12.0, SLIGHTLY_OFF),
        ("left", 0.0, SLIGHTLY_OFF),
    ],
)
def test_recovers_hidden_zero(world, side, shoulder_gravity_nm, start_deg):
    # Joint 1 may carry much of the arm's weight near its stop, pulling it away from the stop.
    gravity = GRAVITY.copy()
    gravity[0] = shoulder_gravity_nm
    sim = world.add(side=side, hidden_zero_deg=HIDDEN_ZERO, gravity=gravity, start_deg=start_deg)
    bus = make_bus()
    bus.connect()

    measurements = hsc.HardStopCalibrator(bus).run(hsc.openarm_sequence(side))

    assert set(measurements) == set(MOTORS)
    for motor, m in measurements.items():
        assert m.offset_deg == pytest.approx(HIDDEN_ZERO[motor], abs=0.15), motor
        assert m.stop_std_deg < 0.05, motor
    # The sequence ends with the arm back near its starting pose.
    for motor in MOTORS[:-1]:
        assert abs(sim.calibrated_deg(motor)) < 5.0, motor


def test_interactive_calibration_installs_offsets(world):
    sim = world.add(hidden_zero_deg=HIDDEN_ZERO)
    bus = make_bus()
    bus.connect()

    calibration = hsc.run_openarm_hard_stop_calibration(bus, "right", prompt=lambda _: "")

    assert bus.calibration is calibration
    for motor in MOTORS:
        assert calibration[motor].homing_offset / 1000 == pytest.approx(HIDDEN_ZERO[motor], abs=0.15)
    positions = bus.sync_read("Present_Position")
    for motor in MOTORS:
        assert positions[motor] == pytest.approx(sim.calibrated_deg(motor), abs=0.2), motor


def test_missing_stop_aborts_holds_and_restores(world):
    stops = {**STOPS["right"], "joint_5": (-90.0, 400.0)}  # no stop where joint 5 expects one
    sim = world.add(stops=stops)
    previous = hsc.offsets_to_calibration(dict.fromkeys(MOTORS, 1.0), SEND_IDS)
    bus = make_bus(previous)
    bus.connect()
    prompts = []

    with pytest.raises(hsc.CalibrationAbortedError, match="joint_5"):
        hsc.run_openarm_hard_stop_calibration(bus, "right", prompt=lambda text: prompts.append(text) or "")

    # Joint 5 stopped within its travel limit, the user was asked to support the arm, then torque was cut.
    assert sim.calibrated_deg("joint_5") < 125.0
    assert "Support it" in prompts[-1]
    assert not sim.enabled.any()
    assert bus.calibration == previous


def test_rejects_offsets_that_would_wrap_after_power_cycle(world):
    world.add(hidden_zero_deg={**HIDDEN_ZERO, "joint_4": 60.0})
    previous = hsc.offsets_to_calibration(dict.fromkeys(MOTORS, 1.0), SEND_IDS)
    bus = make_bus(previous)
    bus.connect()

    with pytest.raises(hsc.CalibrationAbortedError, match="joint_4") as e:
        hsc.run_openarm_hard_stop_calibration(bus, "right", prompt=lambda _: "")

    assert "Re-zero the motors" in str(e.value)
    assert bus.calibration == previous


def test_push_limit_aborts(world):
    sim = world.add()
    bus = make_bus()
    bus.connect()
    never_stalls = hsc.StopSeek("joint_5", +1, 90.0, max_travel_deg=120.0, vel_threshold_deg_s=0.0)

    with pytest.raises(hsc.CalibrationAbortedError, match="pushed"):
        hsc.HardStopCalibrator(bus).run([never_stalls])

    assert abs(sim.tau_motor[MOTORS.index("joint_5")]) < never_stalls.push_limit_nm + 1.0


def test_rejects_velocity_threshold_above_seek_speed(world):
    world.add()
    bus = make_bus()
    bus.connect()
    too_loose = hsc.StopSeek("joint_5", +1, 90.0, max_travel_deg=120.0, vel_threshold_deg_s=10.0)
    with pytest.raises(ValueError, match="seek speed"):
        hsc.HardStopCalibrator(bus).run([too_loose])


def test_repeatability(world):
    world.add(hidden_zero_deg=HIDDEN_ZERO, seed=3)
    bus = make_bus()
    bus.connect()

    stops = hsc.measure_repeatability(bus, "right", runs=3)

    for motor, values in stops.items():
        assert len(values) == 3
        assert max(values) - min(values) < 0.1, motor


def test_rejects_bus_with_offsets(world):
    world.add()
    bus = make_bus(hsc.offsets_to_calibration({"joint_1": 5.0}, SEND_IDS))
    bus.connect()
    with pytest.raises(ValueError, match="offsets"):
        hsc.HardStopCalibrator(bus).run(hsc.openarm_sequence("right"))


def test_range_checks_mirror_for_left_arm():
    assert hsc.working_range_deg("left")["joint_2"] == (-90.0, 9.0)
    assert hsc.mechanical_range_deg("left")["joint_1"] == (-200.0, 80.0)
    # Joint 2 works on opposite sides of zero on the two arms, so the same offset is fine on one only.
    assert hsc.offset_problems({"joint_2": 100.0}, "left") == []
    assert hsc.offset_problems({"joint_2": 100.0}, "right") != []
    assert hsc.offset_problems({"joint_2": -100.0}, "left") != []
    assert hsc.position_problems({"joint_4": -30.0, "joint_3": 50.0}, "right") == [
        "joint_4 reads -30.0°, outside its mechanical range 0°..140°"
    ]


def test_repeatability_command(world, monkeypatch, capsys):
    world.add(hidden_zero_deg=HIDDEN_ZERO, seed=5)
    monkeypatch.setattr(
        "sys.argv", ["openarm-stop-repeatability", "--port", "can0", "--side", "right", "--runs", "2"]
    )

    hsc.main()

    table = capsys.readouterr().out.split("Stop repeatability over 2 runs")[1].splitlines()
    rows = {line.split()[0]: [float(x) for x in line.split()[1:]] for line in table[2:] if line.strip()}
    assert set(rows) == set(MOTORS)
    # Raw stop position = the motor's reading at the joint's zero + the stop's angle.
    stop_deg = {s.motor: s.stop_deg for s in hsc.openarm_sequence("right") if isinstance(s, hsc.StopSeek)}
    for motor, (mean, _std, spread) in rows.items():
        assert mean == pytest.approx(HIDDEN_ZERO[motor] + stop_deg[motor], abs=0.15), motor
        assert spread < 0.1, motor
