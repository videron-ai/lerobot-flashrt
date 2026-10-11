# SPDX-License-Identifier: Apache-2.0
"""A Damiao motor bus that applies per-motor zero offsets in software.

Stock LeRobot stores a `homing_offset` in each motor's calibration but the Damiao bus ignores it. This bus
applies it: every position read from a motor has the offset subtracted, and every position sent to a motor
has it added back. The offset is stored as an integer number of millidegrees because LeRobot's calibration
files only accept integers there.

The bus relies on a few internal methods of LeRobot's `DamiaoMotorsBus`. To make sure a LeRobot update can
never cause offsets to be applied twice or not at all, every instance runs a self-test when it is created:
it feeds a synthetic state frame through the read path and captures the CAN frames produced by each write
path, without touching any hardware. If the result is not exactly "offset applied once", construction fails
with `IncompatibleLeRobotError`.
"""

from __future__ import annotations

import logging
import math
import time

import can
from lerobot.motors import MotorCalibration
from lerobot.motors.damiao import DamiaoMotorsBus
from lerobot.motors.damiao.tables import (
    CAN_CMD_DISABLE,
    CAN_CMD_ENABLE,
    CAN_CMD_QUERY_PARAM,
    CAN_CMD_WRITE_PARAM,
    CAN_PARAM_ID,
    MOTOR_LIMIT_PARAMS,
)

logger = logging.getLogger(__name__)

HOMING_OFFSET_UNITS_PER_DEGREE = 1000

# The motor's control mode register. Written to RAM only (never saved to flash), so a motor returns to its
# flashed mode at the next power cycle.
CTRL_MODE_REGISTER = 10
CONTROL_MODES = {1: "MIT", 2: "POS_VEL", 3: "VEL", 4: "POS_FORCE"}
MIT_MODE = 1
_CTRL_MODE_READ_TIMEOUT_S = 0.3

_SELF_TEST_OFFSET_DEG = 7.0
_SELF_TEST_RAW_DEG = 20.0
_SELF_TEST_TARGET_DEG = 10.0
# One step of the 16-bit position encoding is about 0.022 degrees.
_SELF_TEST_TOLERANCE_DEG = 0.05


class IncompatibleLeRobotError(RuntimeError):
    """The installed LeRobot's Damiao bus does not behave the way this plugin expects."""


class ControlModeError(ConnectionError):
    """A motor could not be switched to MIT mode."""


class _CaptureBus:
    """Stands in for the CAN bus during the self-test: records sent frames, never answers."""

    def __init__(self) -> None:
        self.sent: list[can.Message] = []

    def send(self, msg, timeout=None) -> None:
        self.sent.append(msg)

    def recv(self, timeout=None):
        return None

    def shutdown(self) -> None:
        pass


class OffsetDamiaoMotorsBus(DamiaoMotorsBus):
    """`DamiaoMotorsBus` that applies `MotorCalibration.homing_offset` (in millidegrees) in software.

    calibrated_position_deg = raw_position_deg - homing_offset / 1000

    Connecting also puts every motor in MIT mode, the only mode LeRobot commands them in. Motors flashed
    with another mode (position-velocity, for example) come back in it after each power cycle and would
    ignore MIT commands.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._run_self_test()

    @classmethod
    def from_bus(cls, bus: DamiaoMotorsBus) -> OffsetDamiaoMotorsBus:
        """Build an offset bus with the same settings as an existing, not yet connected, bus."""
        if bus.is_connected:
            raise RuntimeError("Replace the bus before connecting it.")
        return cls(
            port=bus.port,
            motors=bus.motors,
            calibration=bus.calibration,
            can_interface=bus.can_interface,
            use_can_fd=bus.use_can_fd,
            bitrate=bus.bitrate,
            data_bitrate=bus.data_bitrate,
        )

    # Control mode

    def connect(self, handshake: bool = True) -> None:
        super().connect(handshake)
        if not handshake:
            return
        try:
            self.ensure_mit_mode()
        except Exception:
            self.disconnect(disable_torque=True)
            raise

    def ensure_mit_mode(self) -> list[str]:
        """Switch every motor that is in another control mode to MIT. Returns the motors switched.

        The mode can only be changed while a motor is disabled, so a switched motor goes limp for a moment
        and is then enabled again, as the handshake left it. A motor that does not answer the mode query is
        left alone.
        """
        switched = []
        for motor in self.motors:
            mode = self._read_control_mode(motor)
            if mode == MIT_MODE:
                continue
            if mode is None:
                logger.warning(f"{self.port}: {motor} did not report its control mode; assuming MIT.")
                continue
            self._send_simple_command(motor, CAN_CMD_DISABLE)
            time.sleep(0.02)
            self._send_control_mode_frame(motor, CAN_CMD_WRITE_PARAM, MIT_MODE)
            time.sleep(0.05)
            now = self._read_control_mode(motor)
            if now != MIT_MODE:
                raise ControlModeError(
                    f"{self.port}: {motor} is in {CONTROL_MODES[mode]} mode and did not switch to MIT "
                    f"(it now reports {CONTROL_MODES.get(now, 'nothing')}). Check power and CAN wiring."
                )
            self._send_simple_command(motor, CAN_CMD_ENABLE)
            switched.append(motor)
        if switched:
            logger.info(f"{self.port}: switched {', '.join(switched)} to MIT mode (until the next power cycle).")
        return switched

    def _send_control_mode_frame(self, motor: str, command: int, value: int = 0) -> None:
        motor_id = self._get_motor_id(motor)
        data = [motor_id & 0xFF, (motor_id >> 8) & 0xFF, command, CTRL_MODE_REGISTER, *value.to_bytes(4, "little")]
        self.canbus.send(
            can.Message(arbitration_id=CAN_PARAM_ID, data=data, is_extended_id=False, is_fd=self.use_can_fd)
        )

    def _read_control_mode(self, motor: str) -> int | None:
        """The motor's control mode, or None if it does not answer."""
        motor_id = self._get_motor_id(motor)
        self._send_control_mode_frame(motor, CAN_CMD_QUERY_PARAM)
        deadline = time.monotonic() + _CTRL_MODE_READ_TIMEOUT_S
        while (remaining := deadline - time.monotonic()) > 0:
            msg = self.canbus.recv(timeout=remaining)
            if msg is None:
                return None
            data = bytes(msg.data)
            # Matched by content: state frames arrive on the same id. A state frame cannot pass for a
            # reply because its last bytes (temperatures) do not decode to a valid mode.
            if (
                len(data) >= 8
                and data[0] == motor_id & 0xFF
                and data[1] == (motor_id >> 8) & 0xFF
                and data[2] in (CAN_CMD_QUERY_PARAM, CAN_CMD_WRITE_PARAM)
                and data[3] == CTRL_MODE_REGISTER
                and (mode := int.from_bytes(data[4:8], "little")) in CONTROL_MODES
            ):
                return mode
        return None

    def offset_deg(self, motor: str) -> float:
        calibration = self.calibration.get(motor) if self.calibration else None
        return 0.0 if calibration is None else calibration.homing_offset / HOMING_OFFSET_UNITS_PER_DEGREE

    @property
    def has_offsets(self) -> bool:
        return any(c.homing_offset for c in (self.calibration or {}).values())

    # Read path

    def _process_response(self, motor, msg) -> None:
        before = self._last_known_states.get(motor)
        super()._process_response(motor, msg)
        after = self._last_known_states.get(motor)
        # The parent stores a fresh state dict only when the frame decoded; on failure it keeps the old one,
        # which already has the offset applied.
        if after is not None and after is not before:
            after["position"] -= self.offset_deg(motor)

    # Write paths

    def _mit_control(self, motor, kp, kd, position_degrees, velocity_deg_per_sec, torque) -> None:
        offset = self.offset_deg(self._get_motor_name(motor))
        super()._mit_control(motor, kp, kd, position_degrees + offset, velocity_deg_per_sec, torque)

    def _mit_control_batch(self, commands) -> None:
        shifted = {
            motor: (kp, kd, position + self.offset_deg(self._get_motor_name(motor)), velocity, torque)
            for motor, (kp, kd, position, velocity, torque) in commands.items()
        }
        super()._mit_control_batch(shifted)

    def sync_write(self, data_name, values) -> None:
        if data_name == "Goal_Position":
            values = {motor: float(value) + self.offset_deg(motor) for motor, value in values.items()}
        super().sync_write(data_name, values)

    def set_zero_position(self, motors=None) -> None:
        if self.has_offsets:
            raise RuntimeError(
                "Refusing to re-zero motors that have hard-stop calibration offsets: the offsets are relative "
                "to the motors' current zero and would no longer be valid."
            )
        super().set_zero_position(motors)

    # Self-test

    def _run_self_test(self) -> None:
        motor = next(iter(self.motors))
        motor_type = self._motor_types[motor]
        pmax = MOTOR_LIMIT_PARAMS[motor_type][0]
        saved = (self.calibration, self.canbus, self._is_connected, dict(self._last_known_states))
        capture = _CaptureBus()

        def last_sent_target_deg() -> float:
            frames = [m for m in capture.sent if m.arbitration_id == self.motors[motor].id]
            if not frames:
                raise IncompatibleLeRobotError("No command frame was sent.")
            q_uint = (frames[-1].data[0] << 8) | frames[-1].data[1]
            return math.degrees(self._uint_to_float(q_uint, -pmax, pmax, 16))

        try:
            self.calibration = {
                motor: MotorCalibration(
                    id=self.motors[motor].id,
                    drive_mode=0,
                    homing_offset=round(_SELF_TEST_OFFSET_DEG * HOMING_OFFSET_UNITS_PER_DEGREE),
                    range_min=-90,
                    range_max=90,
                )
            }
            self.canbus = capture
            self._is_connected = True

            q_uint = self._float_to_uint(math.radians(_SELF_TEST_RAW_DEG), -pmax, pmax, 16)
            zero_12 = self._float_to_uint(0.0, -1.0, 1.0, 12)
            frame = can.Message(
                arbitration_id=self.motors[motor].recv_id,
                data=[
                    self.motors[motor].id,
                    q_uint >> 8,
                    q_uint & 0xFF,
                    zero_12 >> 4,
                    ((zero_12 & 0xF) << 4) | (zero_12 >> 8),
                    zero_12 & 0xFF,
                    25,
                    25,
                ],
                is_extended_id=False,
            )
            self._process_response(motor, frame)
            results = {"read": self._last_known_states[motor]["position"]}

            self._mit_control(motor, 1.0, 0.1, _SELF_TEST_TARGET_DEG, 0.0, 0.0)
            results["_mit_control"] = last_sent_target_deg()
            self._mit_control_batch({motor: (1.0, 0.1, _SELF_TEST_TARGET_DEG, 0.0, 0.0)})
            results["_mit_control_batch"] = last_sent_target_deg()
            self.sync_write("Goal_Position", {motor: _SELF_TEST_TARGET_DEG})
            results["sync_write"] = last_sent_target_deg()
        except IncompatibleLeRobotError:
            raise
        except Exception as e:
            raise IncompatibleLeRobotError(
                f"This version of LeRobot's Damiao motor bus is not compatible with lerobot_robot_openarm_hardstop "
                f"(self-test failed: {e!r})."
            ) from e
        finally:
            self.calibration, self.canbus, self._is_connected, states = saved
            self._last_known_states = states

        expected = {
            "read": _SELF_TEST_RAW_DEG - _SELF_TEST_OFFSET_DEG,
            "_mit_control": _SELF_TEST_TARGET_DEG + _SELF_TEST_OFFSET_DEG,
            "_mit_control_batch": _SELF_TEST_TARGET_DEG + _SELF_TEST_OFFSET_DEG,
            "sync_write": _SELF_TEST_TARGET_DEG + _SELF_TEST_OFFSET_DEG,
        }
        wrong = {
            path: f"{results[path]:.2f}° (expected {value:.2f}°)"
            for path, value in expected.items()
            if abs(results[path] - value) > _SELF_TEST_TOLERANCE_DEG
        }
        if wrong:
            raise IncompatibleLeRobotError(
                "The installed LeRobot does not apply calibration offsets the way lerobot_robot_openarm_hardstop "
                f"expects ({wrong}). It may now handle offsets itself, which would apply them twice. "
                "Update the plugin before using it with this LeRobot version."
            )
