# SPDX-License-Identifier: Apache-2.0
import sys
import types
from pathlib import Path

import can
import pytest
from lerobot.motors import Motor, MotorNormMode

import lerobot_robot_openarm_hardstop.calibration as calibration_module
import lerobot_robot_openarm_hardstop.hardstop_arm as hardstop_arm_module

sys.path.insert(0, str(Path(__file__).parent))

from sim_openarm import MOTORS, RECV_IDS, SEND_IDS, TYPES, SimulatedOpenArm  # noqa: E402

HIDDEN_ZERO = {
    "joint_1": 17.3,
    "joint_2": -6.2,
    "joint_3": 24.9,
    "joint_4": -11.1,
    "joint_5": 3.3,
    "joint_6": -28.4,
    "joint_7": 31.2,
    "gripper": 7.7,
}


class SimWorld:
    """Simulated arms keyed by CAN port, all advancing on one simulated clock."""

    def __init__(self):
        self.arms: dict[str, SimulatedOpenArm] = {}
        self.t = 0.0
        # Sides whose calibration reached the gravity zero step, which is skipped unless
        # `run_gravity_zero` is set (most tests' arms have too little gravity for it).
        self.gravity_zero_calls: list[str] = []
        self.run_gravity_zero = False

    def add(self, port="can0", **kwargs) -> SimulatedOpenArm:
        self.arms[port] = SimulatedOpenArm(**kwargs)
        return self.arms[port]

    def sleep(self, dt):
        for arm in self.arms.values():
            arm.advance(dt)
        self.t += dt

    def monotonic(self):
        return self.t


@pytest.fixture
def world(monkeypatch, tmp_path):
    """Route python-can to simulated arms, run calibration on simulated time, answer prompts with ENTER."""
    w = SimWorld()
    monkeypatch.setattr(can.interface, "Bus", lambda **kw: w.arms[kw["channel"]])
    monkeypatch.setattr(
        calibration_module, "time", types.SimpleNamespace(sleep=w.sleep, monotonic=w.monotonic)
    )
    monkeypatch.setattr("builtins.input", lambda *_: "")
    real_gravity_zero = hardstop_arm_module.run_gravity_zero

    def gravity_zero(arm, side, *args, **kwargs):
        w.gravity_zero_calls.append(side)
        return real_gravity_zero(arm, side, *args, **kwargs) if w.run_gravity_zero else {}

    monkeypatch.setattr(hardstop_arm_module, "run_gravity_zero", gravity_zero)
    return w


def make_motors():
    return {
        m: Motor(
            SEND_IDS[m], TYPES[m].name.lower(), MotorNormMode.DEGREES, TYPES[m].name.lower(), RECV_IDS[m]
        )
        for m in MOTORS
    }
