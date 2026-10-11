# SPDX-License-Identifier: Apache-2.0
"""A simulated OpenArm on a CAN bus, for testing calibration without hardware.

It sits behind python-can: it decodes the MIT, refresh, enable/disable and set-zero frames that LeRobot's
`DamiaoMotorsBus` sends, integrates simple joint dynamics with compliant mechanical stops, gravity, friction
and noise, and answers with state frames encoded the way the motors encode them. Each motor has a hidden
bias (what it reads at its joint's true zero) that calibration should recover.
"""

import math

import can
import numpy as np
from lerobot.motors.damiao.tables import (
    CAN_CMD_DISABLE,
    CAN_CMD_ENABLE,
    CAN_CMD_QUERY_PARAM,
    CAN_CMD_REFRESH,
    CAN_CMD_SET_ZERO,
    CAN_CMD_WRITE_PARAM,
    CAN_PARAM_ID,
    MIT_KD_RANGE,
    MIT_KP_RANGE,
    MOTOR_LIMIT_PARAMS,
    MotorType,
)

CTRL_MODE_REGISTER = 10

MOTORS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", "gripper"]
SEND_IDS = {m: i + 1 for i, m in enumerate(MOTORS)}
RECV_IDS = {m: i + 0x11 for i, m in enumerate(MOTORS)}
TYPES = {
    "joint_1": MotorType.DM8009,
    "joint_2": MotorType.DM8009,
    "joint_3": MotorType.DM4340,
    "joint_4": MotorType.DM4340,
    "joint_5": MotorType.DM4310,
    "joint_6": MotorType.DM4310,
    "joint_7": MotorType.DM4310,
    "gripper": MotorType.DM4310,
}

# Mechanical stops in the joint's zero frame (degrees), per side (from Enactic's mechanical limits).
STOPS = {
    "right": {
        "joint_1": (-80.0, 200.0),
        "joint_2": (-10.0, 100.0),
        "joint_3": (-90.0, 90.0),
        "joint_4": (0.0, 140.0),
        "joint_5": (-90.0, 90.0),
        "joint_6": (-45.0, 45.0),
        "joint_7": (-90.0, 90.0),
        "gripper": (-60.0, 0.0),
    },
}
STOPS["left"] = {**STOPS["right"], "joint_1": (-200.0, 80.0), "joint_2": (-100.0, 10.0)}

# Rough joint properties: inertia (kg m^2), viscous and Coulomb friction, gravity torque amplitude,
# stop stiffness (Nm/rad).
INERTIA = np.array([0.6, 0.5, 0.08, 0.05, 0.01, 0.01, 0.005, 0.002])
VISCOUS = np.array([0.5, 0.5, 0.2, 0.2, 0.05, 0.05, 0.05, 0.02])
COULOMB = np.array([0.4, 0.4, 0.2, 0.2, 0.08, 0.08, 0.05, 0.05])
# Gravity torque amplitude per joint (Nm), as tau = -amplitude * sin(angle).
GRAVITY = np.array([0.0, 1.5, 0.0, 1.0, 0.0, 0.3, 0.0, 0.0])
STOP_STIFFNESS = np.array([8000.0, 8000.0, 4000.0, 4000.0, 2000.0, 2000.0, 2000.0, 400.0])
STOP_DAMPING = np.array([20.0, 20.0, 8.0, 8.0, 3.0, 3.0, 3.0, 1.0])

SUBSTEP_S = 2.5e-4


def _float_to_uint(x, x_min, x_max, bits):
    x = min(max(x, x_min), x_max)
    return int((x - x_min) / (x_max - x_min) * ((1 << bits) - 1))


def _uint_to_float(x, x_min, x_max, bits):
    return x / ((1 << bits) - 1) * (x_max - x_min) + x_min


class SimulatedOpenArm:
    """One OpenArm on a CAN bus. Implements the parts of `can.BusABC` the Damiao bus uses."""

    def __init__(self, side="right", hidden_zero_deg=None, start_deg=None, stops=None, seed=0, gravity=None):
        self.t = 0.0
        self.gravity = GRAVITY if gravity is None else gravity
        self.rng = np.random.default_rng(seed)
        stops = stops if stops is not None else STOPS[side]
        self.lo = np.radians([stops[m][0] for m in MOTORS])
        self.hi = np.radians([stops[m][1] for m in MOTORS])
        # Joint angle in the joint's zero frame; the motor reports q + bias, so bias is what the motor
        # reads at the joint's true zero (the offset calibration should recover).
        start = start_deg if start_deg is not None else {"joint_4": 3.0, "gripper": -2.0}
        self.q = np.radians([start.get(m, 0.0) for m in MOTORS])
        self.dq = np.zeros(len(MOTORS))
        hidden = hidden_zero_deg if hidden_zero_deg is not None else {}
        self.bias = np.radians([hidden.get(m, 0.0) for m in MOTORS])
        self.enabled = np.zeros(len(MOTORS), dtype=bool)
        # Last MIT command per motor: kp, kd, q_des (motor frame, rad), dq_des, tau_ff.
        self.cmd = np.zeros((len(MOTORS), 5))
        self.tau_motor = np.zeros(len(MOTORS))
        self.pmax = np.array([MOTOR_LIMIT_PARAMS[TYPES[m]][0] for m in MOTORS])
        self.tmax = np.array([MOTOR_LIMIT_PARAMS[TYPES[m]][2] for m in MOTORS])
        self.inbox = []
        self.set_zero_count = 0
        # Control mode per motor (1 = MIT). A motor in another mode ignores MIT commands, and returns to
        # its flashed mode at a power cycle.
        self.flashed_mode = 1
        self.ctrl_mode = np.full(len(MOTORS), 1)
        self.answers_mode_query = True
        # Optional realism for gravity measurements: the forearm's angle from vertical is joint 4 plus this
        # many times joint 1 (the upper arm's pitch carries the elbow with it), and a constant error in each
        # motor's reported torque.
        self.forearm_pitch_gain = 0.0
        self.torque_bias = np.zeros(len(MOTORS))

    # Physics

    def advance(self, duration_s):
        n = max(1, round(duration_s / SUBSTEP_S))
        for _ in range(n):
            q_motor = self.q + self.bias
            kp, kd, q_des, dq_des, tau_ff = self.cmd.T
            tau_m = np.clip(kp * (q_des - q_motor) + kd * (dq_des - self.dq) + tau_ff, -self.tmax, self.tmax)
            tau_m = np.where(self.enabled, tau_m, 0.0)
            tau_stop = np.where(
                self.q < self.lo, STOP_STIFFNESS * (self.lo - self.q) - STOP_DAMPING * self.dq, 0.0
            )
            tau_stop += np.where(
                self.q > self.hi, -STOP_STIFFNESS * (self.q - self.hi) - STOP_DAMPING * self.dq, 0.0
            )
            hang = self.q.copy()
            hang[MOTORS.index("joint_4")] += self.forearm_pitch_gain * self.q[MOTORS.index("joint_1")]
            tau_grav = -self.gravity * np.sin(hang)
            tau_fric = VISCOUS * self.dq + COULOMB * np.tanh(self.dq / 0.01)
            noise = self.rng.normal(0.0, 0.02, len(MOTORS))
            ddq = (tau_m + tau_stop + tau_grav - tau_fric + noise) / INERTIA
            self.dq += ddq * SUBSTEP_S
            self.q += self.dq * SUBSTEP_S
            self.tau_motor = tau_m
        self.t += n * SUBSTEP_S

    # CAN side

    def send(self, msg, timeout=None):
        data = bytes(msg.data)
        if msg.arbitration_id == CAN_PARAM_ID:
            i = MOTORS.index(_name_for_id(data[0] | (data[1] << 8)))
            if data[2] == CAN_CMD_REFRESH:
                self._reply(i)
            elif data[2] in (CAN_CMD_QUERY_PARAM, CAN_CMD_WRITE_PARAM) and data[3] == CTRL_MODE_REGISTER:
                if data[2] == CAN_CMD_WRITE_PARAM and not self.enabled[i]:
                    self.ctrl_mode[i] = int.from_bytes(data[4:8], "little")
                if self.answers_mode_query:
                    reply = [*data[:4], *int(self.ctrl_mode[i]).to_bytes(4, "little")]
                    self.inbox.append(
                        can.Message(arbitration_id=RECV_IDS[MOTORS[i]], data=reply, is_extended_id=False)
                    )
            return
        i = MOTORS.index(_name_for_id(msg.arbitration_id))
        if data[:7] == b"\xff" * 7 and data[7] in (CAN_CMD_ENABLE, CAN_CMD_DISABLE, CAN_CMD_SET_ZERO):
            if data[7] == CAN_CMD_ENABLE:
                self.enabled[i] = True
            elif data[7] == CAN_CMD_DISABLE:
                self.enabled[i] = False
                self.cmd[i] = 0.0
            else:
                self.bias[i] = -self.q[i]
                self.set_zero_count += 1
        elif self.ctrl_mode[i] == 1:
            pmax, vmax, tmax = MOTOR_LIMIT_PARAMS[TYPES[MOTORS[i]]]
            q_uint = (data[0] << 8) | data[1]
            dq_uint = (data[2] << 4) | (data[3] >> 4)
            kp_uint = ((data[3] & 0xF) << 8) | data[4]
            kd_uint = (data[5] << 4) | (data[6] >> 4)
            tau_uint = ((data[6] & 0xF) << 8) | data[7]
            self.cmd[i] = (
                _uint_to_float(kp_uint, *MIT_KP_RANGE, 12),
                _uint_to_float(kd_uint, *MIT_KD_RANGE, 12),
                _uint_to_float(q_uint, -pmax, pmax, 16),
                _uint_to_float(dq_uint, -vmax, vmax, 12),
                _uint_to_float(tau_uint, -tmax, tmax, 12),
            )
        self._reply(i)

    def _reply(self, i):
        pmax, vmax, tmax = MOTOR_LIMIT_PARAMS[TYPES[MOTORS[i]]]
        q_uint = _float_to_uint(self.q[i] + self.bias[i], -pmax, pmax, 16)
        dq_uint = _float_to_uint(self.dq[i], -vmax, vmax, 12)
        tau_uint = _float_to_uint(self.tau_motor[i] + self.torque_bias[i], -tmax, tmax, 12)
        data = [
            SEND_IDS[MOTORS[i]],
            q_uint >> 8,
            q_uint & 0xFF,
            dq_uint >> 4,
            ((dq_uint & 0xF) << 4) | (tau_uint >> 8),
            tau_uint & 0xFF,
            30,
            30,
        ]
        self.inbox.append(can.Message(arbitration_id=RECV_IDS[MOTORS[i]], data=data, is_extended_id=False))

    def recv(self, timeout=None):
        return self.inbox.pop(0) if self.inbox else None

    def shutdown(self):
        pass

    # Helpers for assertions

    def calibrated_deg(self, motor):
        return math.degrees(self.q[MOTORS.index(motor)])

    def hidden_zero_deg(self, motor):
        return math.degrees(self.bias[MOTORS.index(motor)])

    def power_cycle(self):
        """Motors come back up reporting a single-turn absolute position, within ±180° of their zero."""
        raw = self.q + self.bias
        wrapped = (raw + math.pi) % (2 * math.pi) - math.pi
        self.bias = wrapped - self.q
        self.enabled[:] = False
        self.cmd[:] = 0.0
        self.ctrl_mode[:] = self.flashed_mode


def _name_for_id(send_id):
    return MOTORS[send_id - 1]
