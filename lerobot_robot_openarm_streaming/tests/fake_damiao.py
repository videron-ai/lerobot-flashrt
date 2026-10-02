"""Simulated Damiao MIT motors on a python-can virtual bus.

Lets the streaming follower run against LeRobot's real DamiaoMotorsBus code with no
hardware: each simulated arm listens on a virtual CAN channel, decodes MIT and
enable/disable/refresh frames exactly as Damiao drivers receive them, integrates a
rigid-joint model (inertia + viscous friction, PD computed from the latest MIT
command, like the driver's inner loop), and replies with Damiao feedback frames.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field

import can

from lerobot.motors.damiao.tables import MIT_KD_RANGE, MIT_KP_RANGE, MOTOR_LIMIT_PARAMS, MotorType

CMD_ENABLE, CMD_DISABLE, CMD_ZERO, CMD_REFRESH = 0xFC, 0xFD, 0xFE, 0xCC
PARAM_READ, PARAM_WRITE, PARAM_SAVE = 0x33, 0x55, 0xAA
PARAM_ID = 0x7FF

# OpenArm default motor layout: name -> (send id, recv id, type, inertia kg m^2, viscous Nm s/rad)
OPENARM_MOTORS = {
    "joint_1": (0x01, 0x11, MotorType.DM8009, 0.30, 0.05),
    "joint_2": (0x02, 0x12, MotorType.DM8009, 0.30, 0.05),
    "joint_3": (0x03, 0x13, MotorType.DM4340, 0.08, 0.02),
    "joint_4": (0x04, 0x14, MotorType.DM4340, 0.08, 0.02),
    "joint_5": (0x05, 0x15, MotorType.DM4310, 0.002, 0.005),
    "joint_6": (0x06, 0x16, MotorType.DM4310, 0.006, 0.005),
    "joint_7": (0x07, 0x17, MotorType.DM4310, 0.006, 0.005),
    "gripper": (0x08, 0x18, MotorType.DM4310, 0.002, 0.005),
}


def _u2f(x, lo, hi, bits):
    return x / ((1 << bits) - 1) * (hi - lo) + lo


def _f2u(x, lo, hi, bits):
    x = max(lo, min(hi, x))
    return int((x - lo) / (hi - lo) * ((1 << bits) - 1))


@dataclass
class SimMotor:
    name: str
    send_id: int
    recv_id: int
    mtype: MotorType
    inertia: float
    viscous: float
    q: float = 0.0  # rad
    v: float = 0.0
    enabled: bool = False
    kp: float = 0.0
    kd: float = 0.0
    q_des: float = 0.0
    v_des: float = 0.0
    tau_ff: float = 0.0
    tau: float = 0.0
    # (time, q_des deg, v_des deg/s, q deg) per MIT command received
    log: list = field(default_factory=list)
    # register id -> uint32 value (RAM), and what was last saved to flash
    params: dict = field(default_factory=lambda: {9: 0})
    saved: dict = field(default_factory=dict)

    def step(self, dt: float) -> None:
        _, _, tmax = MOTOR_LIMIT_PARAMS[self.mtype]
        if self.enabled:
            tau = self.kp * (self.q_des - self.q) + self.kd * (self.v_des - self.v) + self.tau_ff
            tau = max(-tmax, min(tmax, tau))
        else:
            tau = 0.0
        self.tau = tau
        a = (tau - self.viscous * self.v) / self.inertia
        self.v += a * dt
        self.q += self.v * dt

    def feedback(self) -> bytes:
        pmax, vmax, tmax = MOTOR_LIMIT_PARAMS[self.mtype]
        qu = _f2u(self.q, -pmax, pmax, 16)
        vu = _f2u(self.v, -vmax, vmax, 12)
        tu = _f2u(self.tau, -tmax, tmax, 12)
        return bytes([(1 if self.enabled else 0) << 4 | (self.send_id & 0x0F), qu >> 8, qu & 0xFF, vu >> 4,
                      ((vu & 0xF) << 4) | (tu >> 8), tu & 0xFF, 30, 30])


class SimArm(threading.Thread):
    def __init__(
        self,
        channel: str,
        initial_deg: dict[str, float] | None = None,
        substep_s: float = 0.0005,
        reply_delay_s: float = 0.0,
    ):
        super().__init__(name=f"sim-{channel}", daemon=True)
        self.channel = channel
        self.substep = substep_s
        self.motors = {}
        for name, (sid, rid, mtype, inertia, visc) in OPENARM_MOTORS.items():
            m = SimMotor(name, sid, rid, mtype, inertia, visc)
            m.q = math.radians((initial_deg or {}).get(name, 0.0))
            m.q_des = m.q
            self.motors[name] = m
        self.by_send = {m.send_id: m for m in self.motors.values()}
        self.bus = can.interface.Bus(interface="virtual", channel=channel)
        self._halt = threading.Event()
        self.lock = threading.Lock()
        self.disable_count = 0
        self.mit_frames = 0
        # Feedback frames are held back this long, like replies queuing on a classic CAN bus.
        self.reply_delay_s = reply_delay_s
        self._pending: list[tuple[float, can.Message]] = []

    def stop(self):
        self._halt.set()
        self.join(timeout=1)
        self.bus.shutdown()

    def _reply(self, m: SimMotor) -> None:
        msg = can.Message(arbitration_id=m.recv_id, data=m.feedback(), is_extended_id=False)
        if self.reply_delay_s > 0:
            self._pending.append((time.monotonic() + self.reply_delay_s, msg))
        else:
            self.bus.send(msg)

    def _handle(self, msg: can.Message, now: float) -> None:
        d = bytes(msg.data)
        if msg.arbitration_id == PARAM_ID and len(d) >= 3 and d[2] == CMD_REFRESH:
            m = self.by_send.get(d[0] | (d[1] << 8))
            if m:
                self._reply(m)
            return
        if msg.arbitration_id == PARAM_ID and len(d) >= 8 and d[2] in (PARAM_READ, PARAM_WRITE, PARAM_SAVE):
            m = self.by_send.get(d[0] | (d[1] << 8))
            if m is None:
                return
            if d[2] == PARAM_SAVE:
                m.saved = dict(m.params)
                return
            if d[2] == PARAM_WRITE:
                m.params[d[3]] = int.from_bytes(d[4:8], "little")
            value = m.params.get(d[3], 0)
            self.bus.send(can.Message(arbitration_id=m.recv_id, is_extended_id=False,
                                      data=bytes(d[:4]) + value.to_bytes(4, "little")))
            return
        m = self.by_send.get(msg.arbitration_id)
        if m is None or len(d) < 8:
            return
        if d[:7] == b"\xff" * 7 and d[7] in (CMD_ENABLE, CMD_DISABLE, CMD_ZERO):
            if d[7] == CMD_ENABLE:
                if not m.enabled:
                    m.q_des, m.v_des, m.kp, m.kd, m.tau_ff = m.q, 0.0, 0.0, 0.0, 0.0
                m.enabled = True
            elif d[7] == CMD_DISABLE:
                m.enabled = False
                self.disable_count += 1
            else:
                m.q = 0.0
            self._reply(m)
            return
        pmax, vmax, tmax = MOTOR_LIMIT_PARAMS[m.mtype]
        m.q_des = _u2f((d[0] << 8) | d[1], -pmax, pmax, 16)
        m.v_des = _u2f((d[2] << 4) | (d[3] >> 4), -vmax, vmax, 12)
        m.kp = _u2f(((d[3] & 0xF) << 8) | d[4], *MIT_KP_RANGE, 12)
        m.kd = _u2f((d[5] << 4) | (d[6] >> 4), *MIT_KD_RANGE, 12)
        m.tau_ff = _u2f(((d[6] & 0xF) << 8) | d[7], -tmax, tmax, 12)
        m.log.append((now, math.degrees(m.q_des), math.degrees(m.v_des), math.degrees(m.q)))
        self.mit_frames += 1
        self._reply(m)

    def run(self) -> None:
        last = time.monotonic()
        while not self._halt.is_set():
            msg = self.bus.recv(timeout=0.0002)
            now = time.monotonic()
            with self.lock:
                while last < now:
                    dt = min(self.substep, now - last)
                    for m in self.motors.values():
                        m.step(dt)
                    last += dt
                if msg is not None:
                    self._handle(msg, now)
                while self._pending and self._pending[0][0] <= time.monotonic():
                    self.bus.send(self._pending.pop(0)[1])

    def positions_deg(self) -> dict[str, float]:
        with self.lock:
            return {n: math.degrees(m.q) for n, m in self.motors.items()}
