"""Simulated rig for the tests: the hard-stop plugin's simulated arms (mechanical stops, hidden motor
zeros, gravity, friction) driven in real time while a control loop is running."""

import sys
import time
import types
from pathlib import Path

import can
import pytest

# The simulated arm lives with the hard-stop plugin's tests, next to this package in the repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lerobot_robot_openarm_hardstop" / "tests"))

import lerobot_robot_openarm_hardstop.calibration as calibration_module  # noqa: E402
import lerobot_robot_openarm_hardstop.hardstop_arm as hardstop_arm_module  # noqa: E402
from lerobot_robot_openarm_streaming import bi_openarm_streaming_follower as streaming_module  # noqa: E402
from sim_openarm import SUBSTEP_S, SimulatedOpenArm  # noqa: E402


class RealtimeArm(SimulatedOpenArm):
    """Advances on the simulated clock during calibration, and with the wall clock while `realtime` is set
    (the control loops talk to it from their own threads at their own pace)."""

    realtime = False
    _last_send: float | None = None

    def send(self, msg, timeout=None):
        if self.realtime:
            now = time.monotonic()
            if self._last_send is None:
                self._last_send = now
            elif now - self._last_send >= SUBSTEP_S:
                self.advance(min(now - self._last_send, 0.05))
                self._last_send = now
        else:
            self._last_send = None
        super().send(msg, timeout)


class SimWorld:
    def __init__(self):
        self.arms: dict[str, RealtimeArm] = {}
        self.t = 0.0
        self.answers: list[str] = []  # typed answers to prompts, in order; ENTER once they run out
        self.prompts: list[str] = []
        # Sides whose calibration reached the gravity zero step. The step itself is skipped: it has its
        # own tests in the hard-stop plugin and these arms carry too little gravity for it.
        self.gravity_zero_calls: list[str] = []

    def add(self, port, **kwargs) -> RealtimeArm:
        self.arms[port] = RealtimeArm(**kwargs)
        return self.arms[port]

    def sleep(self, dt):
        for arm in self.arms.values():
            arm.advance(dt)
        self.t += dt

    def monotonic(self):
        return self.t

    def input(self, prompt=""):
        self.prompts.append(str(prompt))
        return self.answers.pop(0) if self.answers else ""

    def set_realtime(self, on: bool):
        for arm in self.arms.values():
            arm.realtime = on


@pytest.fixture
def world(monkeypatch):
    w = SimWorld()
    monkeypatch.setattr(can.interface, "Bus", lambda **kw: w.arms[kw["channel"]])
    monkeypatch.setattr(
        calibration_module, "time", types.SimpleNamespace(sleep=w.sleep, monotonic=w.monotonic)
    )
    monkeypatch.setattr("builtins.input", w.input)
    monkeypatch.setattr(
        hardstop_arm_module, "run_gravity_zero", lambda arm, side: w.gravity_zero_calls.append(side) or {}
    )

    follower = streaming_module.BiOpenArmStreamingFollower
    start, stop = follower._start_loops, follower._stop_loops

    def start_loops(self):
        w.set_realtime(True)
        start(self)

    def stop_loops(self):
        stop(self)
        w.set_realtime(False)

    monkeypatch.setattr(follower, "_start_loops", start_loops)
    monkeypatch.setattr(follower, "_stop_loops", stop_loops)
    return w
