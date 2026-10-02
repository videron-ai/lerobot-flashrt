"""End-to-end tests for bi_openarm_streaming_follower against simulated Damiao motors.

Runs LeRobot's real config parsing, plugin discovery and DamiaoMotorsBus code on
python-can virtual buses (see fake_damiao.py). No hardware needed:

    python tests/test_streaming_follower.py        # or: pytest tests/

The KER stream tests also need lerobot_teleoperator_openarm_ker installed.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from fake_damiao import SimArm  # noqa: E402

import draccus  # noqa: E402
from lerobot.motors import MotorCalibration  # noqa: E402
from lerobot.robots.config import RobotConfig  # noqa: E402
from lerobot.robots.utils import make_robot_from_config  # noqa: E402
from lerobot.teleoperators.config import TeleoperatorConfig  # noqa: E402
from lerobot.utils.import_utils import register_third_party_plugins  # noqa: E402

register_third_party_plugins()

HZ = 250.0
_counter = [0]


@dataclass
class _Cli:
    robot: RobotConfig


def make_robot(extra_args=(), initial=None, reply_delay_s=0.0):
    """Parse CLI args exactly like lerobot-record, build via plugin discovery, connect to sim arms."""
    _counter[0] += 1
    left_ch, right_ch = f"sim_left_{_counter[0]}", f"sim_right_{_counter[0]}"
    sims = {
        "left": SimArm(left_ch, initial, reply_delay_s=reply_delay_s),
        "right": SimArm(right_ch, initial, reply_delay_s=reply_delay_s),
    }
    for s in sims.values():
        s.start()
    args = [
        "--robot.type=bi_openarm_streaming_follower",
        "--robot.id=test_follower",
        f"--robot.left_arm_config.port={left_ch}",
        "--robot.left_arm_config.side=left",
        "--robot.left_arm_config.can_interface=virtual",
        f"--robot.right_arm_config.port={right_ch}",
        "--robot.right_arm_config.side=right",
        "--robot.right_arm_config.can_interface=virtual",
        f"--robot.control_hz={HZ}",
        *extra_args,
    ]
    cfg = draccus.parse(_Cli, args=args).robot
    robot = make_robot_from_config(cfg)
    for arm in (robot.left_arm, robot.right_arm):  # pretend calibration exists so connect() won't prompt
        cal = {m: MotorCalibration(id=i, drive_mode=0, homing_offset=0, range_min=0, range_max=0)
               for i, m in enumerate(arm.bus.motors)}
        arm.calibration = cal
        arm.bus.calibration = cal
    robot.connect()
    return robot, sims


def teardown(robot, sims):
    try:
        if robot.is_connected:
            robot.disconnect()
    finally:
        for s in sims.values():
            s.stop()


def log_since(sim, motor, t0):
    return [e for e in sim.motors[motor].log if e[0] >= t0]


def max_rate(entries, span=25):
    """Largest commanded speed (deg/s), averaged over `span` control ticks.

    Per-tick rates are meaningless here: the simulator timestamps frames when it
    happens to get the CPU, so two frames can land microseconds apart.
    Windows whose timestamps are squeezed together (the simulator was starved of CPU
    and then drained a backlog of frames at once) are skipped; a real overspeed
    still shows up in every normally spaced window around it.
    """
    r = 0.0
    nominal = span / HZ
    for (t0, q0, _, _), (t1, q1, _, _) in zip(entries, entries[span:]):
        if t1 - t0 >= 0.7 * nominal:
            r = max(r, abs(q1 - q0) / (t1 - t0))
    return r


def drive_until_caught_up(robot, act_fn, side="right", timeout=8.0, settle=False):
    """Keep a record-style loop running until the follower leaves its approach phase.

    With settle=True, keep going until the post-approach speed ramp has finished too,
    so the follower is tracking at full speed.
    """
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        robot.send_action(act_fn())
        time.sleep(1 / 30)
        if not robot.streaming_stats()[side]["approaching"]:
            break
    else:
        raise AssertionError("follower never caught up with the target")
    if settle:
        t1 = time.monotonic()
        while time.monotonic() - t1 < robot.config.approach_ramp_s + 0.3:
            robot.send_action(act_fn())
            time.sleep(1 / 30)


# ----------------------------------------------------------------------------- tests
def test_loop_rate_and_observation():
    robot, sims = make_robot(initial={"joint_4": 30.0})
    try:
        time.sleep(1.5)
        stats = robot.streaming_stats()
        obs = robot.get_observation()
        assert set(obs) == set(robot.observation_features), "observation keys differ from features"
        assert abs(obs["left_joint_4.pos"] - 30.0) < 1.0, obs["left_joint_4.pos"]
        assert robot.name == "bi_openarm_follower"
        for side in ("left", "right"):
            assert stats[side]["rate_hz"] > 0.8 * HZ, stats
        return f"loops at {stats['left']['rate_hz']:.0f}/{stats['right']['rate_hz']:.0f} Hz, worst cycle {max(s['max_work_ms'] for s in stats.values()):.2f} ms, obs keys match"
    finally:
        teardown(robot, sims)


def test_main_thread_never_touches_bus():
    robot, sims = make_robot()
    try:
        offenders = []
        for arm in (robot.left_arm, robot.right_arm):
            bus = arm.canbus if hasattr(arm, "canbus") else arm.bus.canbus
            orig = bus.send
            def guarded(msg, *a, _orig=orig, **k):
                if threading.current_thread() is threading.main_thread():
                    offenders.append(msg.arbitration_id)
                return _orig(msg, *a, **k)
            bus.send = guarded
        for _ in range(30):
            robot.get_observation()
            robot.send_action({k: 0.0 for k in robot.action_features})
            time.sleep(1 / 30)
        assert not offenders, f"main thread sent CAN frames: {offenders[:5]}"
        return "get_observation/send_action never touch the CAN bus"
    finally:
        teardown(robot, sims)


def test_startup_approach_then_full_speed():
    robot, sims = make_robot(initial={"joint_4": 30.0})
    try:
        t0 = time.monotonic()
        target = {k: 0.0 for k in robot.action_features}
        target["left_joint_4.pos"] = 10.0  # inside limits, 20 deg away
        while time.monotonic() - t0 < 2.0:
            robot.send_action(target)
            time.sleep(1 / 30)
        log = log_since(sims["left"], "joint_4", t0)
        approach = [e for e in log if abs(e[1] - 10.0) > 2.0]  # until within the 2 deg tolerance
        handover = max_rate(log)
        assert max_rate(approach) <= 20.0 * 1.1, ("startup approach", max_rate(approach))
        assert handover <= 60.0, ("handover", handover)  # smooth ramp, not a full-speed burst
        assert abs(log[-1][1] - 10.0) < 0.03, log[-1]
        # keep commanding (a pause would count as a gap and restart the approach)
        # until the post-approach speed ramp has finished
        t_hold = time.monotonic()
        while time.monotonic() - t_hold < robot.config.approach_ramp_s + 0.2:
            robot.send_action(target)
            time.sleep(1 / 30)
        # a fast but continuous trajectory (240 deg/s, 30 Hz commands): followed, capped at 180 deg/s
        t1 = time.monotonic()
        while time.monotonic() - t1 < 0.5:
            target["left_joint_4.pos"] = min(100.0, 10.0 + 240.0 * (time.monotonic() - t1))
            robot.send_action(target)
            time.sleep(1 / 30)
        fast = log_since(sims["left"], "joint_4", t1)
        rate = max_rate(fast)
        assert 150 <= rate <= 180 * 1.1, ("fast trajectory", rate)
        # a sudden 90 deg jump is a new target, not a trajectory: slow approach again
        t2 = time.monotonic()
        target["left_joint_4.pos"] = 10.0
        while time.monotonic() - t2 < 1.0:
            robot.send_action(target)
            time.sleep(1 / 30)
        jump_rate = max_rate(log_since(sims["left"], "joint_4", t2))
        assert jump_rate <= 20.0 * 1.1, ("jump", jump_rate)
        return (f"startup approach {max_rate(approach):.1f} deg/s (peak {handover:.0f} while ramping up); "
                f"fast trajectory capped at {rate:.0f} deg/s; sudden 90 deg jump approached at {jump_rate:.1f} deg/s")
    finally:
        teardown(robot, sims)


def test_joint_limits_enforced():
    robot, sims = make_robot()
    try:
        t0 = time.monotonic()
        target = {k: 0.0 for k in robot.action_features}
        target["right_joint_6.pos"] = 80.0  # right J6 limit is +40
        target["right_joint_4.pos"] = -50.0  # right J4 limit is [0, 135]
        sent = robot.send_action(target)
        while time.monotonic() - t0 < 3.0:
            robot.send_action(target)
            time.sleep(1 / 30)
        j6 = max(e[1] for e in log_since(sims["right"], "joint_6", t0))
        j4 = min(e[1] for e in log_since(sims["right"], "joint_4", t0))
        assert j6 <= 40.0 + 0.03 and j4 >= -0.03, (j6, j4)  # 0.03 deg = MIT position resolution
        assert sent["right_joint_6.pos"] == 40.0
        return f"commanded right J6 max {j6:.2f} (limit 40), J4 min {j4:.2f} (limit 0)"
    finally:
        teardown(robot, sims)


def _smoothness(entries, t_from, t_to):
    e = [x for x in entries if t_from <= x[0] <= t_to]
    q = np.array([x[1] for x in e])
    steps = np.abs(np.diff(q))
    return steps.max(), np.abs(np.diff(q, 2)).max(), np.mean([abs(x[2]) for x in e])


def test_command_interpolation_vs_hold():
    out = {}
    for mode in ("interpolate", "hold"):
        robot, sims = make_robot([f"--robot.command_interpolation={mode}"], initial={"joint_4": 40.0})
        try:
            start = {k: 0.0 for k in robot.action_features}
            start["left_joint_4.pos"] = 40.0
            drive_until_caught_up(robot, lambda: start, side="left", settle=True)
            t0 = time.monotonic()
            while time.monotonic() - t0 < 3.0:  # 30 Hz policy-style commands: 0.5 Hz, 30 deg sine on left J4
                t = time.monotonic() - t0
                a = {k: 0.0 for k in robot.action_features}
                a["left_joint_4.pos"] = 40.0 + 30.0 * math.sin(2 * math.pi * 0.5 * t)
                robot.send_action(a)
                time.sleep(1 / 30)
            out[mode] = _smoothness(sims["left"].motors["joint_4"].log, t0 + 1.5, t0 + 3.0)
        finally:
            teardown(robot, sims)
    (si, ci, vi), (sh, ch, vh) = out["interpolate"], out["hold"]
    # In hold mode the speed cap turns each 33 ms jump into a burst at full speed
    # followed by a stop; interpolation spreads it evenly. Compare target acceleration.
    assert ci < 0.3 * ch, (ci, ch)
    assert vi > 5.0, vi
    return (f"largest per-tick change in target speed {ci:.3f} deg (interpolate) vs {ch:.3f} deg (hold); "
            f"velocity feed-forward mean {vi:.0f} deg/s")


def test_error_propagates():
    robot, sims = make_robot()
    try:
        bus = robot.left_arm.bus
        orig = bus._mit_control_batch
        def boom(commands):
            raise OSError("simulated CAN failure")
        bus._mit_control_batch = boom
        time.sleep(0.1)
        try:
            robot.get_observation()
        except RuntimeError as e:
            bus._mit_control_batch = orig
            return f"loop failure surfaces in main thread: {e}"
        raise AssertionError("no error raised")
    finally:
        teardown(robot, sims)


def test_disconnect_stops_loops_and_disables_torque():
    robot, sims = make_robot()
    loops = list(robot._loops.values())
    robot.disconnect()
    n = sims["left"].mit_frames
    time.sleep(0.1)
    alive = [l.is_alive() for l in loops]
    disabled = all(not m.enabled for s in sims.values() for m in s.motors.values())
    for s in sims.values():
        s.stop()
    assert not any(alive) and sims["left"].mit_frames == n and disabled
    return "threads stopped, no MIT frames after disconnect, all 16 motors disabled"


def test_reply_window_covers_slow_replies():
    """Replies held back 1 ms (classic CAN): LeRobot's 1 ms wait misses them, the default window doesn't.

    The simulator adds roughly another 0.5 ms of its own, so they arrive 1.5-2 ms after the commands.
    """
    missed = {}
    for label, args in (("1 ms", ["--robot.reply_window_ms=1.0"]), ("default", [])):
        robot, sims = make_robot(args, reply_delay_s=0.001)
        try:
            time.sleep(1.5)
            stats = robot.streaming_stats()
            missed[label] = max(s["missed_replies"] / (8 * s["ticks"]) for s in stats.values())
            obs = robot.get_observation()
            assert all(abs(obs[k]) < 1.0 for k in obs if k.endswith(".pos")), "states went wrong"
        finally:
            teardown(robot, sims)
    assert missed["1 ms"] > 0.5 and missed["default"] < 0.1, missed
    return (f"replies held back 1 ms: {missed['1 ms']:.0%} missed with a 1 ms window, "
            f"{missed['default']:.1%} with the default 2.5 ms")


# ---------------------------------------------------------------- KER stream tests
class FakeKERStream:
    """Stands in for openarm_ker.KERStream: 500 Hz frames from a function of time."""

    def __init__(self, fn, **_):
        self.fn = fn
        self.metadata = {"fw": "sim"}
        self.is_link_up = True
        self._c = False
        self.frozen_at = None

    def connect(self):
        self._c = True

    @property
    def is_connected(self):
        return self._c

    def send_command(self, c):
        pass

    def close(self):
        self._c = False

    def latest(self):
        t = time.monotonic() if self.frozen_at is None else self.frozen_at
        seq = int(t * 500)
        return {"timestamp": seq, "seq": seq, "angles": self.fn(seq / 500.0), "errors": [False] * 16}


def make_ker(fn, **cfg):
    import lerobot_teleoperator_openarm_ker.openarm_ker as km
    from lerobot_teleoperator_openarm_ker import OpenArmKER, OpenArmKERConfig

    holder = {}

    def factory(**kw):
        holder["s"] = FakeKERStream(fn)
        return holder["s"]

    km.KERStream = factory
    ker = OpenArmKER(OpenArmKERConfig(stream_to_follower=True, **cfg))
    ker.connect()
    return ker, holder["s"]


def ker_angles_factory(amp=20.0, f=0.5, base=40.0):
    def fn(t):
        a = [0.0] * 16
        a[3] = base + amp * math.sin(2 * math.pi * f * t)  # right J4 channel
        return a
    return fn


def test_stream_mode_tracks_ker_at_loop_rate():
    ker, kstream = make_ker(ker_angles_factory())
    robot = sims = None
    try:
        robot, sims = make_robot(initial={"joint_4": 40.0})
        drive_until_caught_up(robot, ker.get_action, settle=True)
        t0 = time.monotonic()
        recorded = []
        while time.monotonic() - t0 < 3.0:  # lerobot-record style: 30 Hz get_action + send_action
            act = ker.get_action()
            robot.send_action(act)
            recorded.append((time.monotonic(), act["right_joint_4.pos"]))
            time.sleep(1 / 30)
        log = log_since(sims["right"], "joint_4", t0)
        log = [e for e in log if e[0] <= t0 + 3.0]
        q = np.array([e[1] for e in log])
        distinct_per_s = np.count_nonzero(np.abs(np.diff(q)) > 1e-6) / (log[-1][0] - log[0][0])
        vff = np.mean([abs(e[2]) for e in log])
        # recorded action vs what the loop was commanding at that moment
        tq = np.array([e[0] for e in log])
        errs = [abs(np.interp(t, tq, q) - v) for t, v in recorded if tq[0] <= t <= tq[-1]]
        assert distinct_per_s > 0.6 * HZ, distinct_per_s
        assert vff > 5.0, vff
        # timestamps come from two threads competing for the CPU, so judge the bulk
        assert np.median(errs) < 0.3 and np.percentile(errs, 95) < 1.0, (np.median(errs), np.percentile(errs, 95))
        return (f"follower target updates {distinct_per_s:.0f}x/s (vs 30 recorded frames/s), "
                f"vff mean {vff:.0f} deg/s, recorded action within {np.median(errs):.2f} deg (median) of the commanded target")
    finally:
        try:
            if robot is not None:
                teardown(robot, sims)
        finally:
            ker.disconnect()


def test_stream_inactive_without_polling_policy_keeps_control():
    ker, kstream = make_ker(ker_angles_factory(amp=0.0, base=80.0))  # KER held at right J4 = 80
    robot = sims = None
    try:
        robot, sims = make_robot(initial={"joint_4": 30.0})
        ker.get_action()  # a single call (like DAgger's handover) must not hand control to the KER
        t0 = time.monotonic()
        a = {k: 0.0 for k in robot.action_features}
        a["right_joint_4.pos"] = 30.0  # the "policy" holds right J4 at 30
        while time.monotonic() - t0 < 1.5:
            robot.send_action(a)
            time.sleep(1 / 30)
        held = log_since(sims["right"], "joint_4", t0)
        assert max(abs(e[1] - 30.0) for e in held) < 0.5, max(e[1] for e in held)
        assert robot.streaming_stats()["right"]["source"] == "commands"
        # human takes over: polling get_action activates the stream, approach first
        t1 = time.monotonic()
        while time.monotonic() - t1 < 4.0:
            robot.send_action(ker.get_action())
            time.sleep(1 / 30)
        took = log_since(sims["right"], "joint_4", t1)
        approach = [e for e in took if abs(e[1] - 80.0) > 2.0]
        assert abs(took[-1][1] - 80.0) < 0.5 and max_rate(approach) <= 22.0, (took[-1], max_rate(approach))
        assert robot.streaming_stats()["right"]["source"] == "stream"
        return "single get_action call ignored; polling hands control to the KER with a 20 deg/s approach"
    finally:
        try:
            if robot is not None:
                teardown(robot, sims)
        finally:
            ker.disconnect()


def test_stream_stall_holds_still():
    ker, kstream = make_ker(ker_angles_factory())
    robot = sims = None
    try:
        robot, sims = make_robot(initial={"joint_4": 40.0})
        drive_until_caught_up(robot, ker.get_action)
        kstream.frozen_at = time.monotonic()  # M5 stops streaming (jump detection / STOP)
        t1 = time.monotonic()
        while time.monotonic() - t1 < 1.0:
            robot.send_action(ker.get_action())
            time.sleep(1 / 30)
        late = log_since(sims["right"], "joint_4", t1 + 0.5)
        span = max(e[1] for e in late) - min(e[1] for e in late)
        vmax = max(abs(e[2]) for e in late)
        assert span < 1e-6 and vmax < 0.3, (span, vmax)  # 0.3 deg/s = MIT velocity resolution
        return "frozen KER: follower holds a constant target with zero velocity feed-forward"
    finally:
        try:
            if robot is not None:
                teardown(robot, sims)
        finally:
            ker.disconnect()


PARK_READY = {"left": [66, -3, -4, 120, -1, -2, -17], "right": [-72, 11, 8, 114, 4, 3, 17]}


def test_park_ready_takeover_and_rest():
    """KER park commands: hold rest, play the ready path, hand over on a pose match, return to rest."""
    import json
    import tempfile

    lift = {f"{s}_joint_{i + 1}": v for s, q in PARK_READY.items() for i, v in enumerate(q[:6])}
    mapping = {
        "wrist_remap": {"mode": "none"},  # stock channels, so KER angles map straight through
        "park": {"ready_path": [lift, {"left_joint_7": -17, "right_joint_7": 17}]},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(mapping, f)
    pose = {"a": [0.0] * 16}
    pose["a"][3] = 40.0  # KER starts with right J4 at 40, nowhere near rest or ready
    ker, _ = make_ker(lambda t: list(pose["a"]), park=True, park_gesture=True, park_hold=False,
                      mapping_path=f.name)
    robot = sims = None

    def run(seconds=None, until=None, timeout=20.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < (seconds or timeout):
            robot.send_action(ker.get_action())
            time.sleep(1 / 30)
            if until is not None and ker.park_state == until:
                return
        assert until is None, f"park never reached '{until}' (state {ker.park_state})"

    def squeeze(right=True, left=True, hold=0.15):
        pose["a"][7], pose["a"][15] = (-60.0 if right else 0.0), (60.0 if left else 0.0)
        run(seconds=hold)
        pose["a"][7] = pose["a"][15] = 0.0
        run(seconds=0.15)

    try:
        robot, sims = make_robot()
        run(seconds=1.0)
        rest = sims["right"].positions_deg()
        assert ker.park_state == "rest" and abs(rest["joint_4"]) < 1.0, (ker.park_state, rest)

        # not the gesture: one long squeeze, a double squeeze of one trigger, two squeezes far apart
        squeeze(hold=1.5)
        squeeze(left=False), squeeze(left=False)
        run(seconds=1.5)
        squeeze()
        run(seconds=1.5)
        assert ker.park_state == "rest", ker.park_state

        t0 = time.monotonic()
        squeeze(), squeeze()  # both triggers, twice, quickly: same as typing ready
        assert ker.park_state == "to_ready", ker.park_state
        run(until="ready")
        t_ready = time.monotonic() - t0
        j1, j4, j7 = (log_since(sims["right"], m, t0) for m in ("joint_1", "joint_4", "joint_7"))
        # joints 1 and 4 rise together (a straight line in joint space), the wrist turns afterwards
        off_line = max(abs(q4 - 114 / 72 * -np.interp(t, [e[0] for e in j1], [e[1] for e in j1])) for t, q4, *_ in j4)
        wrist_early = max(abs(q7) for (t, q7, *_), (_, q4, *_) in zip(j7, j4) if q4 < 110)
        assert off_line < 3.0 and wrist_early < 1.0, (off_line, wrist_early)
        assert max_rate(j4) <= 45.0, max_rate(j4)

        run(seconds=1.0)  # KER is still elsewhere: the followers must stay at ready
        assert ker.park_state == "ready"
        # shoulder and elbow roughly there (15 deg off), wrist nowhere near: still takes over
        for side, base in (("right", 0), ("left", 8)):
            q = PARK_READY[side]
            pose["a"][base:base + 7] = [q[0] + 15, q[1] - 15, q[2] + 15, q[3] - 15, 60.0, 30.0, -60.0]
        run(until="teleop")
        wrist = sims["left"].positions_deg()
        assert abs(wrist["joint_5"] - 60.0) < 3.0 and abs(wrist["joint_7"] + 60.0) < 3.0, wrist

        # back to ready, then take over with "go" while the KER is far from the ready pose
        ker.park_command("ready")
        run(until="ready")
        pose["a"][3] = 60.0
        run(seconds=1.0)
        assert ker.park_state == "ready"
        t_go = time.monotonic()
        ker.park_command("go")
        run(until="teleop")
        took = time.monotonic() - t_go
        assert took > 3.0 and max_rate(log_since(sims["right"], "joint_4", t_go)) <= 45.0, took
        pose["a"][3] = 100.0  # operator moves the right elbow
        run(seconds=1.0)
        assert abs(sims["right"].positions_deg()["joint_4"] - 100.0) < 3.0

        ker.park_command("rest")
        run(until="rest")
        run(seconds=0.5)
        final = {s: sims[s].positions_deg() for s in sims}
        worst = max(abs(final[s][f"joint_{i}"]) for s in final for i in range(1, 8))
        assert worst < 2.0, final
        return (f"rest held with the KER elsewhere; double squeeze starts the ready path, lookalikes don't; "
                f"path in {t_ready:.1f} s on a straight joint-space line "
                f"(max {off_line:.1f} deg off), wrist last; KER took over on a rough shoulder/elbow match "
                f"and again {took:.1f} s after 'go'; "
                f"'rest' from KER control ended {worst:.1f} deg from the calibration pose")
    finally:
        try:
            if robot is not None:
                teardown(robot, sims)
        finally:
            ker.disconnect()


def test_park_holds_connect_until_ker_in_control():
    """With park on, connect() returns only after the park move and takeover: nothing of it gets recorded."""
    import json
    import tempfile

    lift = {f"{s}_joint_{i + 1}": v for s, q in PARK_READY.items() for i, v in enumerate(q[:6])}
    mapping = {"wrist_remap": {"mode": "none"}, "park": {"ready_path": [lift]}}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(mapping, f)
    angles = [0.0] * 16  # KER held 12 deg off the ready pose on the elbows
    for side, base in (("right", 0), ("left", 8)):
        angles[base:base + 6] = PARK_READY[side][:6]
        angles[base + 3] -= 12.0
    ker, _ = make_ker(lambda t: list(angles), park=True, mapping_path=f.name)
    robot = sims = None
    try:
        threading.Timer(0.7, lambda: ker.park_command("ready")).start()
        t0 = time.monotonic()
        robot, sims = make_robot()  # blocks in connect()
        held = time.monotonic() - t0
        assert ker.park_state == "teleop" and held > 4.0, (ker.park_state, held)
        first = ker.get_action()
        obs = robot.get_observation()
        gap = max(abs(first[k] - obs[k]) for k in first if "gripper" not in k)
        assert abs(first["right_joint_4.pos"] - 102.0) < 0.5 and gap < 3.0, (first["right_joint_4.pos"], gap)

        # parking again mid-session: get_action() waits out the park move and the takeover,
        # then returns the action from before the park
        for _ in range(10):
            before = ker.get_action()
            time.sleep(1 / 30)
        threading.Timer(0.3, lambda: ker.park_command("ready")).start()
        t1 = time.monotonic()
        seen = []
        while time.monotonic() - t1 < 6.0:
            t_call = time.monotonic()
            a = ker.get_action()
            seen.append((time.monotonic() - t_call, ker.park_state, a))
            time.sleep(1 / 30)
        longest, state, returned = max(seen, key=lambda e: e[0])
        assert longest > 1.0 and state == "teleop" and returned == before, (longest, state)
        assert all(st == "teleop" for _, st, _ in seen), {st for _, st, _ in seen}
        return (f"connect() held {held:.1f} s for the park move and takeover; the first action LeRobot sees is the "
                f"KER pose, with the followers {gap:.1f} deg from it; a later park held get_action() for {longest:.1f} s")
    finally:
        try:
            if robot is not None:
                teardown(robot, sims)
        finally:
            ker.disconnect()


def test_guided_recording_session():
    """lerobot-ker-record end to end: auto ready, takeover, hands-free episode end, discard, auto rest."""
    import glob
    import json
    import tempfile

    import pandas as pd

    import lerobot.scripts.lerobot_record as rec
    import lerobot_teleoperator_openarm_ker.guide as guide_module
    import lerobot_teleoperator_openarm_ker.openarm_ker as km
    from lerobot_teleoperator_openarm_ker import record as launcher

    tmp = Path(tempfile.mkdtemp())
    motors = [f"joint_{i}" for i in range(1, 8)] + ["gripper"]
    for side in ("left", "right"):  # calibration files, so connect() does not prompt
        cal = {m: MotorCalibration(id=i, drive_mode=0, homing_offset=0, range_min=-90, range_max=90)
               for i, m in enumerate(motors)}
        with open(tmp / f"guided_{side}.json", "w") as f, draccus.config_type("json"):
            draccus.dump(cal, f, indent=4)
    ready = {"left_joint_1": -25, "left_joint_4": 90, "right_joint_1": 25, "right_joint_4": 90}
    (tmp / "map.json").write_text(json.dumps({"wrist_remap": {"mode": "none"}, "park": {"ready_path": [ready]}}))

    _counter[0] += 1
    channels = {"left": f"sim_left_{_counter[0]}", "right": f"sim_right_{_counter[0]}"}
    sims = {side: SimArm(ch) for side, ch in channels.items()}
    for sim in sims.values():
        sim.start()
    pose = [0.0] * 16  # the simulated KER; index 3 = right J4, 0 = right J1, 11 = left J4, 8 = left J1
    km.KERStream = lambda **kw: FakeKERStream(lambda t: list(pose))
    holder, failures = {}, []
    make_teleop = rec.make_teleoperator_from_config
    rec.make_teleoperator_from_config = lambda c: holder.setdefault("ker", make_teleop(c))

    def wait(cond, what, timeout=40.0):
        t0 = time.monotonic()
        while not cond():
            if time.monotonic() - t0 > timeout:
                raise AssertionError(f"operator script timed out waiting for {what}")
            time.sleep(0.01)

    def glide(index, target, seconds=1.0):
        start = pose[index]
        for i in range(1, 51):
            pose[index] = start + (target - start) * i / 50
            time.sleep(seconds / 50)

    def operator():
        try:
            wait(lambda: "ker" in holder, "the teleoperator")
            ker = holder["ker"]
            for take in ("discarded", "episode 0", "episode 1"):
                wait(lambda: ker.park_state == "ready", f"the ready pose before {take}")
                pose[0], pose[3], pose[8], pose[11] = 25.0, 90.0, -25.0, 90.0  # align the KER
                wait(lambda: ker.guide.phase == "record", f"recording of {take}")
                time.sleep(0.5)
                glide(3, 40.0)  # the task: right elbow down and back
                time.sleep(0.3)
                if take == "discarded":
                    for _ in range(2):  # double squeeze = bad take
                        pose[7], pose[15] = -60.0, 60.0
                        time.sleep(0.15)
                        pose[7], pose[15] = 0.0, 0.0
                        time.sleep(0.15)
                else:
                    glide(3, 90.0)  # back to ready, then hold still: the episode ends by itself
                wait(lambda: ker.guide.phase != "record", f"the end of {take}")
        except BaseException as e:  # reported by the main thread
            failures.append(e)

    cfg = draccus.parse(rec.RecordConfig, args=[
        "--robot.type=bi_openarm_streaming_follower", "--robot.id=guided", f"--robot.calibration_dir={tmp}",
        f"--robot.left_arm_config.port={channels['left']}", "--robot.left_arm_config.side=left",
        "--robot.left_arm_config.can_interface=virtual",
        f"--robot.right_arm_config.port={channels['right']}", "--robot.right_arm_config.side=right",
        "--robot.right_arm_config.can_interface=virtual",
        "--teleop.type=openarm_ker", "--teleop.stream_to_follower=true", f"--teleop.mapping_path={tmp / 'map.json'}",
        "--teleop.park=true", "--teleop.park_gesture=true", "--teleop.park_auto_ready_s=0.3",
        "--teleop.park_speed_deg_s=90", "--teleop.park_end_hold_s=0.7",
        "--dataset.repo_id=local/guided", f"--dataset.root={tmp / 'ds'}", "--dataset.single_task=test",
        "--dataset.num_episodes=2", "--dataset.episode_time_s=30", "--dataset.reset_time_s=1", "--dataset.fps=30",
        "--dataset.video=false", "--dataset.push_to_hub=false", "--play_sounds=false",
    ])
    thread = threading.Thread(target=operator, daemon=True)
    try:
        launcher.install()
        thread.start()
        with contextlib.redirect_stdout(io.StringIO()):
            rec.record(cfg)
        thread.join(timeout=5)
        assert not failures, failures[0]
        ker = holder["ker"]
        assert ker.park_state == "rest", ker.park_state
        rest = max(abs(q) for sim in sims.values() for name, q in sim.positions_deg().items() if name != "gripper")
        assert rest < 2.0, f"followers ended {rest:.1f} deg from the rest pose"

        info = json.loads((tmp / "ds" / "meta" / "info.json").read_text())
        df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(str(tmp / "ds" / "data" / "*" / "*.parquet"))))
        names = info["features"]["action"]["names"]
        arm = [i for i, n in enumerate(names) if "gripper" not in n]
        j4 = names.index("right_joint_4.pos")
        assert info["total_episodes"] == 2, info["total_episodes"]
        notes = []
        for ep in (0, 1):
            rows = df[df["episode_index"] == ep]
            a, o = np.stack(rows["action"].to_numpy()), np.stack(rows["observation.state"].to_numpy())
            seconds = len(a) / 30
            jump = np.abs(np.diff(a[:, arm], axis=0)).max()
            start_gap = np.abs(a[0, arm] - o[0, arm]).max()
            grippers_flick = np.abs(np.diff(a[:, [names.index("left_gripper.pos")]], axis=0)).max()
            assert 2.0 < seconds < 12.0, f"episode {ep} lasted {seconds:.1f} s (timer is 30 s)"
            assert abs(a[0, j4] - 90.0) < 2.0 and abs(o[0, j4] - 90.0) < 4.0, (a[0, j4], o[0, j4])  # starts at ready
            assert a[:, j4].min() < 45.0 and abs(a[-1, j4] - 90.0) < 2.0, (a[:, j4].min(), a[-1, j4])  # task, then ready
            assert jump < 8.0 and start_gap < 4.0 and grippers_flick < 1.0, (jump, start_gap, grippers_flick)
            notes.append(f"{seconds:.1f} s")
        return (f"followers rose to ready unprompted; a double-squeezed take was discarded; 2 episodes recorded "
                f"({', '.join(notes)}, each from takeover to the KER held still at ready, no park move or jump in "
                f"the data); followers returned to rest ({rest:.1f} deg) before disconnect")
    finally:
        rec.make_teleoperator_from_config = make_teleop
        guide_module.launcher_active = False
        for sim in sims.values():
            with contextlib.suppress(Exception):
                sim.stop()


@dataclass
class _CliBoth:
    robot: RobotConfig
    teleop: TeleoperatorConfig


def test_cli_parses_record_style_args():
    c = draccus.parse(_CliBoth, args=[
        "--robot.type=bi_openarm_streaming_follower",
        "--robot.left_arm_config.port=can0", "--robot.left_arm_config.side=left",
        "--robot.right_arm_config.port=can1", "--robot.right_arm_config.side=right",
        "--robot.control_hz=250", "--robot.max_speed_deg_s=150",
        "--teleop.type=openarm_ker", "--teleop.stream_to_follower=true", "--teleop.filter=one_euro",
    ])
    assert c.robot.control_hz == 250 and c.robot.max_speed_deg_s == 150 and c.teleop.stream_to_follower
    return "lerobot-record style CLI flags parse for both plugins"


def test_can_timeout_script():
    import importlib.util
    path = Path(__file__).resolve().parent.parent / "scripts" / "damiao_can_timeout.py"
    spec = importlib.util.spec_from_file_location("damiao_can_timeout", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _counter[0] += 1
    sim = SimArm(f"sim_param_{_counter[0]}")
    sim.start()
    try:
        common = ["--port", sim.channel, "--interface", "virtual"]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert mod.main(common) == 0
            assert mod.main(common + ["--set-ms", "100"]) == 0
        assert all(m.params[9] == 2000 for m in sim.motors.values()), {n: m.params for n, m in sim.motors.items()}
        assert not any(m.saved for m in sim.motors.values())
        with contextlib.redirect_stdout(out):
            assert mod.main(common + ["--set-ms", "100", "--save", "--yes", "--ids", "1,8"]) == 0
        saved = [n for n, m in sim.motors.items() if m.saved.get(9) == 2000]
        assert saved == ["joint_1", "gripper"], saved
        assert sim.disable_count == 2
        return "reads, writes (100 ms = 2000 counts of 50 us) and saves register 9 against the simulated motors"
    finally:
        sim.stop()


TESTS = [
    test_cli_parses_record_style_args,
    test_loop_rate_and_observation,
    test_main_thread_never_touches_bus,
    test_startup_approach_then_full_speed,
    test_joint_limits_enforced,
    test_command_interpolation_vs_hold,
    test_error_propagates,
    test_disconnect_stops_loops_and_disables_torque,
    test_reply_window_covers_slow_replies,
    test_stream_mode_tracks_ker_at_loop_rate,
    test_stream_inactive_without_polling_policy_keeps_control,
    test_stream_stall_holds_still,
    test_park_ready_takeover_and_rest,
    test_park_holds_connect_until_ker_in_control,
    test_guided_recording_session,
    test_can_timeout_script,
]

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.ERROR)
    failed = 0
    for t in TESTS:
        try:
            print(f"PASS {t.__name__}: {t()}")
        except Exception as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e!r}")
    sys.exit(1 if failed else 0)
