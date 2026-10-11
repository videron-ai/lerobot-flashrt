"""The recorder process: lerobot-ker-record plus a bridge to the GUI server.

    python -m record_gui.worker -- --config_path=... --teleop.type=openarm_ker ...
    python -m record_gui.worker --demo -- --dataset.num_episodes=3 ...   # no hardware

Started by server.py with RECORD_GUI_SOCKET pointing at its Unix socket. The bridge
hooks a few places in lerobot's record script to learn what the session is doing, and
a background thread sends that to the server ten times a second together with small
JPEG previews of the cameras. Commands coming back (next, re-record, stop, park moves)
set the same `events` flags lerobot's keyboard listener would.

Nothing here runs on the control path: the hooks store references, and the thread only
reads. If the server goes away the session is stopped the normal way, so the arms
return to the rest pose instead of running with no one able to control them.
"""

import argparse
import logging
import math
import os
import signal
import socket
import sys
import threading
import time

from . import ipc

logger = logging.getLogger("record_gui")

STATUS_HZ = 10.0
PREVIEW_MAX_WIDTH = 800  # wider camera frames are halved before JPEG encoding
JPEG_QUALITY = 70


class Link:
    """Blocking socket to the server. Sends are dropped once the connection is gone."""

    def __init__(self, path: str):
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(path)
        self._lock = threading.Lock()
        self.alive = True

    def send(self, data: bytes) -> None:
        if not self.alive:
            return
        try:
            with self._lock:
                self._sock.sendall(data)
        except OSError:
            self.alive = False

    def _read(self, n: int) -> bytes:
        chunks = []
        while n:
            chunk = self._sock.recv(n)
            if not chunk:
                raise EOFError
            chunks.append(chunk)
            n -= len(chunk)
        return b"".join(chunks)

    def listen(self, on_message, on_lost) -> None:
        def run() -> None:
            import json

            try:
                while True:
                    size, kind = ipc.HEADER.unpack(self._read(ipc.HEADER.size))
                    payload = self._read(size)
                    if kind == ipc.JSON:
                        on_message(json.loads(payload))
            except (OSError, EOFError, ValueError):
                pass
            self.alive = False
            on_lost()

        threading.Thread(target=run, name="record-gui-commands", daemon=True).start()


def _encode(frame, rgb: bool = True) -> bytes | None:
    import cv2

    if frame is None or frame.ndim != 3:
        return None
    if frame.shape[1] > PREVIEW_MAX_WIDTH:
        frame = frame[::2, ::2]
    if rgb:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return jpeg.tobytes() if ok else None


def _rounded(values) -> list[float]:
    return [round(float(v), 1) for v in values]


class Bridge:
    def __init__(self, link: Link, preview_fps: float):
        self.link = link
        self.preview_fps = preview_fps
        self.robot = None
        self.teleop = None
        self.dataset = None
        self.events: dict | None = None
        self.stage = "starting"
        self.loop: str | None = None  # "episode" or "reset" while inside lerobot's record_loop
        self.limit = 0.0
        self.saving = False
        self.saved = 0
        self.discarded = 0
        self._loop_t0 = 0.0
        self._rec_t0: float | None = None
        self._ker: dict | None = None  # latest mapped KER pose, before the park controller

    # ------------------------------------------------------------------ hooks
    def install(self) -> None:
        try:
            from lerobot_teleoperator_openarm_ker import record as guided

            guided.install()
        except ImportError as e:
            logger.warning(f"guided recording is unavailable ({e}); running plain lerobot-record")
        import lerobot.scripts.lerobot_record as rec

        make_robot, make_teleop, inner = rec.make_robot_from_config, rec.make_teleoperator_from_config, rec.record_loop

        def make_robot_hook(*args, **kwargs):
            self.robot = make_robot(*args, **kwargs)
            self.stage = "connecting"
            return self.robot

        def make_teleop_hook(*args, **kwargs):
            teleop = make_teleop(*args, **kwargs)
            self._adopt_teleop(teleop)
            return teleop

        def record_loop(*args, **kwargs):
            self._loop_started(kwargs)
            try:
                return inner(*args, **kwargs)
            finally:
                self._loop_ended()

        record_loop._ker_guided = getattr(inner, "_ker_guided", False)
        rec.make_robot_from_config = make_robot_hook
        rec.make_teleoperator_from_config = make_teleop_hook
        rec.record_loop = record_loop

    def _adopt_teleop(self, teleop) -> None:
        self.teleop = teleop
        apply = getattr(teleop, "_apply_park", None)
        if apply is None:
            return

        def tap(action, now):
            self._ker = action
            return apply(action, now)

        teleop._apply_park = tap

    def _adopt_dataset(self, dataset) -> None:
        self.dataset = dataset
        save, finalize = dataset.save_episode, dataset.finalize

        def save_episode(*args, **kwargs):
            self.saving = True
            try:
                out = save(*args, **kwargs)
                self.saved += 1
                return out
            finally:
                self.saving = False

        def finalize_hook(*args, **kwargs):
            self.stage = "finalizing"
            return finalize(*args, **kwargs)

        dataset.save_episode = save_episode
        dataset.finalize = finalize_hook

    def _loop_started(self, kwargs: dict) -> None:
        self.robot = kwargs.get("robot", self.robot)
        self.events = kwargs.get("events")
        dataset = kwargs.get("dataset")
        if dataset is not None and self.dataset is None:
            self._adopt_dataset(dataset)
        self.limit = float(kwargs.get("control_time_s") or 0.0)
        self._loop_t0 = time.monotonic()
        self._rec_t0 = None
        self.stage = "running"
        self.loop = "episode" if dataset is not None else "reset"

    def _loop_ended(self) -> None:
        if self.loop == "episode" and self._rec_t0 is not None and (self.events or {}).get("rerecord_episode"):
            self.discarded += 1
        self.loop = None

    # ----------------------------------------------------------------- status
    @property
    def _park(self):
        return getattr(self.teleop, "_park", None)

    @property
    def _guide(self):
        return getattr(self.teleop, "guide", None)

    def _recording(self) -> bool:
        if self.loop != "episode":
            return False
        guide = self._guide
        return guide is None or guide.phase == "record"

    def status(self) -> dict:
        now = time.monotonic()
        park, guide = self._park, self._guide
        recording = self._recording()
        if recording and self._rec_t0 is None:
            self._rec_t0 = now
        t0 = self._rec_t0 if recording else self._loop_t0
        out = {
            "stage": self.stage,
            "loop": self.loop,
            "park": park.state if park is not None else None,
            "guided": guide is not None,
            "recording": recording,
            "elapsed": round(now - t0, 2) if self.loop else 0.0,
            "limit": self.limit,
            "saving": self.saving,
            "saved": self.saved,
            "discarded": self.discarded,
            "controllable": self.events is not None,
        }
        for name, section in (
            ("total", self._total),
            ("countdown", self._countdown),
            ("align", self._align),
            ("end", self._episode_end),
            ("joints", self._joints),
            ("loops", self._loops),
            ("ker_age", self._ker_age),
        ):
            try:
                out[name] = section(now)
            except Exception:  # the session may be half built or half torn down
                out[name] = None
        return out

    def _total(self, now):
        return self.dataset.num_episodes if self.dataset is not None else None

    def _countdown(self, now):
        park = self._park
        return round(max(0.0, park._go_t - now), 1) if park is not None and park.state == "countdown" else None

    def _align(self, now):
        """How far the KER is from the ready pose, while the followers wait there for a takeover."""
        park, ker = self._park, self._ker
        if park is None or ker is None or park.state != "ready" or not park.engage_enabled:
            return None
        from lerobot_teleoperator_openarm_ker.park import alignment_hint

        errors = {k: ker[k] - park.ready[k] for k in park.match_keys}
        hint = dict(part.split(": ", 1) for part in alignment_hint(errors, park.engage_tolerance).split(" | "))
        return {
            "keys": list(errors),
            "errors": _rounded(errors.values()),
            "tol": park.engage_tolerance,
            "hint": {"left": hint.get("L", ""), "right": hint.get("R", "")},
        }

    def _episode_end(self, now):
        """Progress of the guided end-of-episode gesture: back at the ready pose, held still."""
        park, guide, ker = self._park, self._guide, self._ker
        if guide is None or ker is None or guide.phase != "record" or park.state != "teleop":
            return None
        cfg = self.teleop.config
        if cfg.park_end_hold_s <= 0:
            return None
        off = max(abs(ker[k] - park.ready[k]) for k in park.match_keys)
        near = guide._was_away and off <= cfg.park_end_tolerance_deg
        held = min(1.0, (now - guide._still_since) / cfg.park_end_hold_s) if near and guide._still_ref else 0.0
        return {"off": round(off, 1), "tol": cfg.park_end_tolerance_deg, "away": guide._was_away, "hold": round(held, 2)}

    def _joints(self, now):
        ker = self._ker
        keys = list(self.teleop.action_features) if self.teleop is not None else (list(ker) if ker else [])
        if not keys:
            return None
        park = self._park
        cmd = park._out if park is not None else ker
        actual = None
        loops = getattr(self.robot, "_loops", None)
        if loops:
            states = {side: loop.snapshot() for side, loop in loops.items()}
            actual = {
                f"{side}_{motor}.pos": state["position"] for side, arm in states.items() for motor, state in arm.items()
            }
        return {
            "keys": [k.removesuffix(".pos") for k in keys],
            "ker": _rounded(ker[k] for k in keys) if ker else None,
            "cmd": _rounded(cmd[k] for k in keys) if cmd else None,
            "actual": _rounded(actual[k] for k in keys) if actual else None,
        }

    def _loops(self, now):
        stats = getattr(self.robot, "streaming_stats", None)
        return (stats() or None) if stats is not None else None

    def _ker_age(self, now):
        teleop = self.teleop
        if teleop is None or getattr(teleop, "_stream", None) is None:
            return None
        source = getattr(teleop, "_source", None)
        last = source.last_advance_t if source is not None else teleop._last_advance_t
        return round(now - last, 3)

    def frames(self):
        cameras = getattr(self.robot, "cameras", None) or {}
        for name, cam in cameras.items():
            try:
                frame = cam.read_latest(max_age_ms=1000)
            except Exception:  # not connected yet, or the camera has stalled
                continue
            jpeg = _encode(frame, rgb="rgb" in str(getattr(cam, "color_mode", "rgb")).lower())
            if jpeg:
                yield name, jpeg

    # --------------------------------------------------------------- commands
    def command(self, msg: dict) -> None:
        cmd = msg.get("cmd")
        events = self.events
        if cmd == "stop":
            self.stop(discard=bool(msg.get("discard")))
        elif cmd == "next":
            if events is not None and self.loop:
                events["exit_early"] = True
        elif cmd == "rerecord":
            if events is not None and self._recording():
                events["rerecord_episode"] = True
                events["exit_early"] = True
        elif cmd == "park":
            park, move = self._park, msg.get("move")
            # Park moves are for the parked states only: mid-episode they would be recorded.
            if park is not None and move in ("ready", "rest", "go", "toggle") and park.state in ("rest", "ready", "countdown"):
                park.request(move)

    def stop(self, discard: bool = False) -> None:
        events = self.events
        if events is None:  # still connecting: same as Ctrl-C in the terminal
            os.kill(os.getpid(), signal.SIGINT)
            return
        if discard and self._recording():
            events["rerecord_episode"] = True
        events["stop_recording"] = True
        events["exit_early"] = True


def pump(link: Link, source, preview_fps: float, done: threading.Event) -> None:
    """Send status and camera previews until the session is over."""
    next_status = next_frames = time.monotonic()
    while not done.is_set() and link.alive:
        now = time.monotonic()
        if now >= next_status:
            next_status = now + 1.0 / STATUS_HZ
            try:
                link.send(ipc.pack_json(source.status()))
            except Exception:
                logger.exception("status report failed")
        if preview_fps > 0 and now >= next_frames:
            next_frames = now + 1.0 / preview_fps
            try:
                for name, jpeg in source.frames():
                    link.send(ipc.pack_frame(name, jpeg))
            except Exception:
                logger.exception("camera preview failed")
        time.sleep(0.02)


# ----------------------------------------------------------------------- demo
class Demo:
    """A scripted session with no hardware, for trying the GUI. Same status as Bridge."""

    KEYS = [f"{s}_{m}" for s in ("left", "right") for m in (*[f"joint_{i}" for i in range(1, 8)], "gripper")]
    READY = {"left_joint_1": -25.0, "left_joint_4": 90.0, "right_joint_1": 25.0, "right_joint_4": 90.0}
    MATCH = [f"{s}_joint_{j}" for s in ("left", "right") for j in (1, 2, 3, 4)]

    def __init__(self, num_episodes: int, episode_time_s: float, reset_time_s: float, cameras: list[str]):
        self.num_episodes, self.episode_time_s, self.reset_time_s = num_episodes, episode_time_s, reset_time_s
        self.cameras = cameras
        self.stage, self.loop, self.park = "connecting", None, "rest"
        self.saving = False
        self.saved = self.discarded = 0
        self.recording = False
        self._t0 = time.monotonic()
        self._stop = self._next = self._rerecord = self._discard = False
        self._go = False
        self._lift = 0.0  # 0 at the rest pose, 1 at the ready pose
        self._gap = 1.0  # how far the "operator" still is from the ready pose

    def _wait(self, seconds: float, until=lambda: False) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end and not until():
            time.sleep(0.05)

    def _move(self, state: str, target: float, seconds: float) -> None:
        self.park = state
        start, t0 = self._lift, time.monotonic()
        while (u := (time.monotonic() - t0) / seconds) < 1.0:
            self._lift = start + (target - start) * (0.5 - 0.5 * math.cos(math.pi * u))
            time.sleep(0.02)
        self._lift = target

    def run(self) -> None:
        self._wait(1.5, lambda: self._stop)
        self.stage = "running"
        while self.saved < self.num_episodes and not self._stop:
            self.loop, self._t0 = "episode", time.monotonic()
            if self.park == "rest":
                self._wait(2.0, lambda: self._stop)
                self._move("to_ready", 1.0, 3.0)
            self.park, self._gap, self._go = "ready", 1.0, False
            t_ready = time.monotonic()
            while not self._stop and not self._go and self._gap > 0.25:
                self._gap = max(0.0, 1.0 - (time.monotonic() - t_ready) / 6.0)
                time.sleep(0.05)
            if self._stop:
                break
            self.park = "engaging"
            self._wait(1.0)
            self.park, self.recording, self._t0 = "teleop", True, time.monotonic()
            self._next = self._rerecord = False
            self._wait(min(self.episode_time_s, 15.0), lambda: self._stop or self._next or self._rerecord)
            self.recording, self.loop = False, None
            discard = self._rerecord or (self._stop and self._discard)
            if not self._stop:
                self.loop, self._t0 = "reset", time.monotonic()
                self._move("to_ready", 1.0, 1.0)
                self.park = "ready"
                self._wait(self.reset_time_s, lambda: self._stop or self._next)
                self.loop = None
            if discard:
                self.discarded += 1
                continue
            self.saving = True
            self._wait(1.5)
            self.saving = False
            self.saved += 1
        self.loop = None
        if self.park != "rest":
            self._move("to_rest", 0.0, 3.0)
        self.park, self.stage = "rest", "finalizing"
        self._wait(1.5)

    def _pose(self, now: float) -> dict[str, float]:
        pose = dict.fromkeys(self.KEYS, 0.0)
        for key, value in self.READY.items():
            pose[key] = value * self._lift
        if self.park == "teleop":
            for i, key in enumerate(self.KEYS):
                pose[key] += 18.0 * math.sin(0.7 * now + i)
        return pose

    def status(self) -> dict:
        now = time.monotonic()
        cmd = self._pose(now)
        ker = dict(cmd)
        if self.park != "teleop":
            for i, key in enumerate(self.MATCH):
                ker[key] = self.READY.get(key, 0.0) + (55.0 - 9.0 * i) * self._gap * (1 if i % 2 else -1)
        waiting = self.park == "ready" and self.loop == "episode"
        errors = [ker[k] - self.READY.get(k, 0.0) for k in self.MATCH]
        phase = 0.5 + 0.5 * math.sin(0.4 * (now - self._t0))
        return {
            "stage": self.stage,
            "loop": self.loop,
            "park": self.park,
            "guided": True,
            "recording": self.recording,
            "elapsed": round(now - self._t0, 2) if self.loop else 0.0,
            "limit": self.episode_time_s if self.loop == "episode" else self.reset_time_s,
            "saving": self.saving,
            "saved": self.saved,
            "discarded": self.discarded,
            "controllable": self.stage == "running",
            "total": self.saved,
            "countdown": None,
            "align": {
                "keys": [f"{k}.pos" for k in self.MATCH],
                "errors": _rounded(errors),
                "tol": 20.0,
                "hint": {
                    "left": "ok" if abs(errors[0]) <= 20 else f"upper arm forward {abs(errors[0]):.0f}",
                    "right": "ok" if abs(errors[7]) <= 20 else f"bend elbow {abs(errors[7]):.0f}",
                },
            }
            if waiting
            else None,
            "end": {"off": round(60 * phase, 1), "tol": 20.0, "away": True, "hold": 0.0} if self.recording else None,
            "joints": {
                "keys": self.KEYS,
                "ker": _rounded(ker.values()),
                "cmd": _rounded(cmd.values()),
                "actual": _rounded(v - 1.5 for v in cmd.values()),
            },
            "loops": {
                side: {
                    "rate_hz": 250.0,
                    "ticks": int(250 * now) % 10**7,
                    "overruns": int(now) % 7,
                    "max_work_ms": 2.31,
                    "missed_replies": 0,
                    "approaching": self.park in ("to_ready", "to_rest"),
                    "source": "stream",
                }
                for side in ("left", "right")
            },
            "ker_age": 0.002,
        }

    def frames(self):
        import cv2
        import numpy as np

        now = time.time()
        for i, name in enumerate(self.cameras):
            h, w = (240, 320) if name == "base" else (360, 640)
            x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
            y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
            wave = 0.5 + 0.5 * np.sin(6.0 * (x + y * 0.6) + now * (0.8 + 0.3 * i))
            img = np.stack([40 + 50 * wave, 46 + 70 * wave * y, 60 + 90 * wave * x], axis=-1).astype(np.uint8)
            cv2.putText(img, f"DEMO {name}", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2, cv2.LINE_AA)
            cv2.putText(img, time.strftime("%H:%M:%S"), (16, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
            jpeg = _encode(img, rgb=False)
            if jpeg:
                yield name, jpeg

    def command(self, msg: dict) -> None:
        cmd = msg.get("cmd")
        if cmd == "stop":
            self.stop(discard=bool(msg.get("discard")))
        elif cmd == "next":
            self._next = True
        elif cmd == "rerecord":
            self._rerecord = self.recording
        elif cmd == "park" and msg.get("move") == "go":
            self._go = True

    def stop(self, discard: bool = False) -> None:
        self._discard = discard
        self._stop = True


def _flag(args: list[str], name: str, default: float) -> float:
    for arg in args:
        if arg.startswith(f"--{name}="):
            return float(arg.split("=", 1)[1])
    return default


def main() -> None:
    ap = argparse.ArgumentParser(description="Recorder process for the recording GUI (started by record_gui.server).")
    ap.add_argument("--demo", action="store_true", help="simulate a session without hardware")
    ap.add_argument("--preview-fps", type=float, default=10.0, help="camera preview rate sent to the GUI; 0 turns it off")
    ap.add_argument("--cameras", default="left_wrist,right_wrist,base", help="camera names for --demo")
    ap.add_argument("record_args", nargs=argparse.REMAINDER, help="-- followed by lerobot-ker-record flags")
    args = ap.parse_args()
    record_args = [a for a in args.record_args if a != "--"]

    link = Link(os.environ["RECORD_GUI_SOCKET"])
    if args.demo:
        source = Demo(
            int(_flag(record_args, "dataset.num_episodes", 3)),
            _flag(record_args, "dataset.episode_time_s", 120),
            _flag(record_args, "dataset.reset_time_s", 2),
            [c for c in args.cameras.split(",") if c],
        )
    else:
        source = Bridge(link, args.preview_fps)
        source.install()
    link.listen(source.command, on_lost=source.stop)
    done = threading.Event()
    reporter = threading.Thread(
        target=pump, args=(link, source, args.preview_fps, done), name="record-gui-status", daemon=True
    )
    reporter.start()

    try:
        if args.demo:
            print("record_gui demo session: no robot, no dataset", flush=True)
            source.run()
        else:
            import lerobot.scripts.lerobot_record as rec

            sys.argv = ["lerobot-ker-record", *record_args]
            rec.main()
    finally:
        done.set()
        reporter.join(timeout=1.0)


if __name__ == "__main__":
    main()
