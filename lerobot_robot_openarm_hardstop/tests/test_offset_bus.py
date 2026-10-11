# SPDX-License-Identifier: Apache-2.0
import can
import pytest
from conftest import HIDDEN_ZERO, make_motors
from lerobot.motors.damiao import DamiaoMotorsBus
from sim_openarm import MOTORS, SEND_IDS

from lerobot_robot_openarm_hardstop.calibration import offsets_to_calibration
from lerobot_robot_openarm_hardstop.offset_bus import (
    ControlModeError,
    IncompatibleLeRobotError,
    OffsetDamiaoMotorsBus,
)


def make_bus(calibration=None, port="can0"):
    return OffsetDamiaoMotorsBus(
        port=port, motors=make_motors(), calibration=calibration, can_interface="socketcan"
    )


def test_offsets_applied_once_on_every_path(world):
    sim = world.add(hidden_zero_deg=HIDDEN_ZERO, start_deg={"joint_3": 12.0, "joint_5": -20.0})
    bus = make_bus(offsets_to_calibration(HIDDEN_ZERO, SEND_IDS))
    bus.connect()

    positions = bus.sync_read("Present_Position")
    for motor in MOTORS:
        assert positions[motor] == pytest.approx(sim.calibrated_deg(motor), abs=0.05), motor

    bus._mit_control_batch({"joint_3": (60.0, 1.5, 0.0, 0.0, 0.0)})
    world.sleep(1.0)
    assert sim.calibrated_deg("joint_3") == pytest.approx(0.0, abs=0.2)

    bus.sync_write("Kp", {"joint_5": 30.0})
    bus.sync_write("Kd", {"joint_5": 1.0})
    bus.sync_write("Goal_Position", {"joint_5": 25.0})
    world.sleep(1.0)
    assert sim.calibrated_deg("joint_5") == pytest.approx(25.0, abs=0.3)

    bus.write("Goal_Position", "joint_5", -15.0)
    world.sleep(1.0)
    assert sim.calibrated_deg("joint_5") == pytest.approx(-15.0, abs=0.3)
    bus.disconnect()


def test_failed_decode_keeps_position(world):
    world.add(hidden_zero_deg=HIDDEN_ZERO)
    bus = make_bus(offsets_to_calibration(HIDDEN_ZERO, SEND_IDS))
    bus.connect()
    before = bus.sync_read("Present_Position")["joint_3"]

    bus._process_response("joint_3", can.Message(arbitration_id=0x13, data=[1, 2, 3], is_extended_id=False))

    assert bus._last_known_states["joint_3"]["position"] == before
    bus.disconnect()


def test_refuses_to_rezero_with_offsets(world):
    sim = world.add()
    bus = make_bus(offsets_to_calibration(HIDDEN_ZERO, SEND_IDS))
    bus.connect()
    with pytest.raises(RuntimeError, match="re-zero"):
        bus.set_zero_position()
    assert sim.set_zero_count == 0
    bus.disconnect()


def test_self_test_leaves_bus_untouched():
    calibration = offsets_to_calibration(HIDDEN_ZERO, SEND_IDS)
    bus = make_bus(calibration)
    assert bus.calibration == calibration
    assert bus.canbus is None
    assert not bus.is_connected


def test_self_test_catches_lerobot_applying_offsets_on_read(monkeypatch):
    original = DamiaoMotorsBus._process_response

    def native(self, motor, msg):
        original(self, motor, msg)
        cal = self.calibration.get(motor)
        if cal:
            self._last_known_states[motor] = {
                **self._last_known_states[motor],
                "position": self._last_known_states[motor]["position"] - cal.homing_offset / 1000,
            }

    monkeypatch.setattr(DamiaoMotorsBus, "_process_response", native)
    with pytest.raises(IncompatibleLeRobotError, match="read"):
        make_bus()


def test_self_test_catches_lerobot_applying_offsets_on_write(monkeypatch):
    original = DamiaoMotorsBus._mit_control_batch

    def native(self, commands):
        original(
            self,
            {
                m: (kp, kd, q + self.calibration[m].homing_offset / 1000, dq, t)
                for m, (kp, kd, q, dq, t) in commands.items()
            },
        )

    monkeypatch.setattr(DamiaoMotorsBus, "_mit_control_batch", native)
    with pytest.raises(IncompatibleLeRobotError, match="_mit_control_batch"):
        make_bus()


def test_self_test_catches_changed_internals(monkeypatch):
    monkeypatch.delattr(DamiaoMotorsBus, "_uint_to_float")
    with pytest.raises(IncompatibleLeRobotError):
        make_bus()


def test_connect_switches_motors_to_mit_mode(world, caplog):
    sim = world.add(hidden_zero_deg=HIDDEN_ZERO, start_deg={"joint_3": 12.0})
    sim.flashed_mode = 2  # position-velocity, as some OpenArms ship
    sim.power_cycle()
    sim.ctrl_mode[MOTORS.index("joint_5")] = 1
    bus = make_bus(offsets_to_calibration(HIDDEN_ZERO, SEND_IDS))
    with caplog.at_level("INFO"):
        bus.connect()

    assert list(sim.ctrl_mode) == [1] * len(MOTORS)
    assert sim.enabled.all()
    assert "switched" in caplog.text and "joint_5" not in caplog.text
    # The motors now follow MIT commands, and positions still read in the calibrated frame.
    bus._mit_control_batch({"joint_3": (60.0, 1.5, 0.0, 0.0, 0.0)})
    world.sleep(1.0)
    assert sim.calibrated_deg("joint_3") == pytest.approx(0.0, abs=0.2)
    assert bus.sync_read("Present_Position")["joint_3"] == pytest.approx(0.0, abs=0.2)
    bus.disconnect()

    # Already in MIT mode: nothing is written and no motor is disabled.
    sent = []
    send = sim.send
    sim.send = lambda msg, timeout=None: (sent.append(bytes(msg.data)), send(msg, timeout))[1]
    bus.connect()
    assert not any(d[2] == 0x55 or d == b"\xff" * 7 + b"\xfd" for d in sent)
    bus.disconnect()


def test_connect_fails_when_a_motor_stays_out_of_mit_mode(world):
    sim = world.add()
    sim.flashed_mode = 2
    sim.power_cycle()
    send = sim.send

    def drop_mode_writes(msg, timeout=None):
        if msg.arbitration_id == 0x7FF and msg.data[2] == 0x55:
            return
        send(msg, timeout)

    sim.send = drop_mode_writes
    bus = make_bus()
    with pytest.raises(ControlModeError, match="joint_1 is in POS_VEL mode"):
        bus.connect()
    assert not bus.is_connected
    assert not sim.enabled.any()


def test_motor_that_does_not_report_its_mode_is_left_alone(world, caplog):
    sim = world.add()
    sim.answers_mode_query = False
    bus = make_bus()
    bus.connect()
    assert "did not report its control mode" in caplog.text
    assert sim.enabled.all()
    bus.disconnect()
