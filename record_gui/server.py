"""The GUI server: serves the page, checks the rig and runs one recorder at a time.

    python -m record_gui [--port 8642] [--host 127.0.0.1] [--datasets-dir ./datasets] [--demo]

Run it from the repo root inside the container. The page talks to it over a WebSocket
(state and log, pushed) and a few POST endpoints (actions). Camera previews are MJPEG
streams. The recorder is a child process, see worker.py.
"""

import argparse
import asyncio
import contextlib
import importlib.metadata
import json
import os
import re
import shlex
import shutil
import signal
import socket
import stat
import sys
import tempfile
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from aiohttp import WSMsgType, web

from . import ipc
from .worker import _encode

STATIC = Path(__file__).parent / "static"
KER_USB_ID = ("303a", "4002")
# Installed in this order: openarm_videron needs the streaming and hard-stop plugins.
PLUGINS = {
    "lerobot_teleoperator_openarm_ker": "./lerobot_teleoperator_openarm_ker",
    "lerobot_robot_openarm_streaming": "./lerobot_robot_openarm_streaming",
    "lerobot_robot_openarm_hardstop": "./lerobot_robot_openarm_hardstop",
    "lerobot_robot_openarm_videron": "./lerobot_robot_openarm_videron",
}
# Robot types whose arms use hard-stop calibration: they need a calibration file per arm before a
# session, because the recorder has no terminal to run the calibration from.
HARDSTOP_ROBOT_TYPES = ("openarm_videron", "bi_openarm_follower_hardstop")
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
LOG_KEEP = 4000
LOG_BACKLOG = 600
STOP_WAIT_S = 90.0  # the return to the rest pose can take a while

DEFAULTS = {
    "task": "Recording test",
    "name": "ker_guided_test",
    "datasets_dir": "./datasets",
    "resume": "",
    "repo_id": "videron/ker_record_test",
    "num_episodes": 3,
    "episode_time_s": 120.0,
    "reset_time_s": 2.0,
    "fps": 30,
    "streaming_encoding": True,
    "encoder_threads": 1,
    "push_to_hub": False,
    "config_path": "./helpers/rollout_videron.yaml",
    "robot_type": "openarm_videron",
    "teleop_id": "ker",
    "mapping_path": "./lerobot_teleoperator_openarm_ker/ker_mapping.json",
    "filter": "one_euro",
    "stream_to_follower": True,
    "park": True,
    "park_gesture": True,
    "extra_args": "",
    "preview_fps": 10.0,
    "auto_prepare": True,
}
LIMITS = {
    "num_episodes": (1, 10000),
    "episode_time_s": (1.0, 3600.0),
    "reset_time_s": (0.0, 3600.0),
    "fps": (1, 240),
    "encoder_threads": (0, 64),
    "preview_fps": (0.0, 30.0),
}


class BadRequest(Exception):
    pass


def clean_settings(raw: dict) -> dict:
    """Coerce what the page sent onto the types and ranges of DEFAULTS."""
    out = dict(DEFAULTS)
    for key, default in DEFAULTS.items():
        if key not in raw:
            continue
        value = raw[key]
        try:
            if isinstance(default, bool):
                value = bool(value)
            elif isinstance(default, int):
                value = int(float(value))
            elif isinstance(default, float):
                value = float(value)
            else:
                value = str(value).strip()
        except (TypeError, ValueError):
            raise BadRequest(f"{key}: not a valid value ({raw[key]!r})") from None
        if key in LIMITS:
            lo, hi = LIMITS[key]
            if not lo <= value <= hi:
                raise BadRequest(f"{key} must be between {lo:g} and {hi:g}")
        out[key] = value
    if not re.fullmatch(r"[\w.\-]+", out["name"]) or out["name"].startswith("."):
        raise BadRequest("dataset name: use letters, digits, '_', '-' and '.' only")
    folder = os.path.expanduser(out["datasets_dir"])
    if not folder or "\n" in folder:
        raise BadRequest("datasets folder: enter a path")
    out["datasets_dir"] = folder.rstrip("/") or "/"
    if out["filter"] not in ("none", "one_euro"):
        raise BadRequest("filter must be none or one_euro")
    try:
        shlex.split(out["extra_args"])
    except ValueError as e:
        raise BadRequest(f"extra arguments: {e}") from None
    return out


def build_record_args(s: dict, stamp: str) -> tuple[list[str], str]:
    """The lerobot-ker-record flags for these settings, and the dataset directory they write."""

    def b(value: bool) -> str:
        return "true" if value else "false"

    folder = s["datasets_dir"].rstrip("/")
    root = f"{folder}/{s['resume']}" if s["resume"] else f"{folder}/{s['name']}_{stamp}"
    args = [
        f"--config_path={s['config_path']}",
        f"--robot.type={s['robot_type']}",
        "--teleop.type=openarm_ker",
        f"--teleop.id={s['teleop_id']}",
        f"--teleop.mapping_path={s['mapping_path']}",
        f"--teleop.filter={s['filter']}",
        f"--teleop.stream_to_follower={b(s['stream_to_follower'])}",
        f"--teleop.park={b(s['park'])}",
        f"--teleop.park_gesture={b(s['park'] and s['park_gesture'])}",
        f"--dataset.repo_id={s['repo_id']}",
        f"--dataset.root={root}",
        f"--dataset.single_task={s['task']}",
        f"--dataset.num_episodes={s['num_episodes']}",
        f"--dataset.episode_time_s={s['episode_time_s']:g}",
        f"--dataset.reset_time_s={s['reset_time_s']:g}",
        f"--dataset.fps={s['fps']}",
        f"--dataset.streaming_encoding={b(s['streaming_encoding'])}",
        f"--dataset.push_to_hub={b(s['push_to_hub'])}",
        "--play_sounds=false",
    ]
    if s["encoder_threads"] > 0:
        args.append(f"--dataset.encoder_threads={s['encoder_threads']}")
    if s["resume"]:
        args.append("--resume=true")
    return [*args, *shlex.split(s["extra_args"])], root


# ------------------------------------------------------------------ rig checks
def rig_config(path: str) -> dict:
    """Camera devices and CAN interfaces named by the robot config file."""
    import yaml

    with open(path, encoding="utf-8") as f:
        robot = (yaml.safe_load(f) or {}).get("robot") or {}
    cameras = dict(robot.get("cameras") or {})
    ports = {}
    for side in ("left", "right"):
        arm = robot.get(f"{side}_arm_config") or {}
        cameras.update(arm.get("cameras") or {})
        if arm.get("port"):
            ports[side] = str(arm["port"])
    return {
        "cameras": cameras,
        "ports": ports,
        "id": robot.get("id"),
        "calibration_dir": robot.get("calibration_dir"),
    }


def _item(key: str, label: str, state: str, detail: str, fix: str | None = None) -> dict:
    return {"id": key, "label": label, "state": state, "detail": detail, "fix": fix}


def _check_can(side: str, port: str) -> dict:
    label = f"CAN {side}"
    base = Path("/sys/class/net") / port
    if not base.exists():
        return _item(f"can_{side}", label, "fail", f"interface '{port}' not found. Adapter unplugged, or not named yet (helpers/setup_can.sh).")
    try:
        up = int((base / "flags").read_text().strip(), 16) & 1
    except (OSError, ValueError):
        up = 0
    if not up:
        return _item(f"can_{side}", label, "fail", f"'{port}' is down. On the host: sudo ip link set {port} up type can bitrate 1000000")
    return _item(f"can_{side}", label, "ok", f"'{port}' is up")


def _check_ker() -> dict:
    for dev in Path("/sys/bus/usb/devices").glob("*"):
        try:
            ids = ((dev / "idVendor").read_text().strip(), (dev / "idProduct").read_text().strip())
            if ids != KER_USB_ID:
                continue
            node = Path(f"/dev/bus/usb/{int((dev / 'busnum').read_text()):03d}/{int((dev / 'devnum').read_text()):03d}")
        except (OSError, ValueError):
            continue
        if node.exists():
            return _item("ker", "KER", "ok", f"USB {':'.join(ids)} at {node}")
        return _item(
            "ker", "KER", "fail",
            f"plugged in at {node}, but that device is not in this container (re-plugged since it started?). Restart the container.",
        )  # fmt: skip
    return _item("ker", "KER", "fail", f"no USB device {':'.join(KER_USB_ID)} found. Check the cable and that the M5 is on.")


def _check_plugins() -> dict:
    missing = []
    for name in (*PLUGINS, "openarm_ker"):
        try:
            importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
    if missing:
        return _item("plugins", "Plugins", "fail", f"not installed: {', '.join(missing)}", fix="install_plugins")
    return _item("plugins", "Plugins", "ok", "KER teleoperator and follower plugins installed")


def _check_calibration(robot_type: str, rig: dict) -> dict | None:
    """Hard-stop calibration files for both arms. None for robot types that do not use them."""
    if robot_type not in HARDSTOP_ROBOT_TYPES:
        return None
    if rig.get("calibration_dir"):
        folder = Path(rig["calibration_dir"])
    else:
        try:
            from lerobot.utils.constants import HF_LEROBOT_CALIBRATION

            folder = Path(HF_LEROBOT_CALIBRATION) / "robots" / "openarm_follower_hardstop"
        except Exception as e:
            return _item("calibration", "Calibration", "unknown", f"could not locate the calibration folder: {e}")
    missing = [
        str(folder / f"{rig.get('id')}_{side}.json")
        for side in ("left", "right")
        if not (folder / f"{rig.get('id')}_{side}.json").is_file()
    ]
    if missing:
        return _item(
            "calibration", "Calibration", "fail",
            f"no hard-stop calibration at {', '.join(missing)}. Run lerobot-calibrate --robot.type={robot_type} "
            "from a terminal first (lerobot_robot_openarm_videron/README.md); a session cannot run it.",
        )  # fmt: skip
    return _item("calibration", "Calibration", "ok", f"hard-stop calibration for both arms in {folder}")


def _check_disk(datasets_dir: Path) -> dict:
    """Free space where the datasets go. A folder that does not exist yet is created at start."""
    existing = datasets_dir.absolute()
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    note = "" if existing == datasets_dir.absolute() else " (the folder is created when a session starts)"
    if not existing.is_dir():
        return _item("disk", "Disk", "fail", f"{existing} is not a folder")
    if not os.access(existing, os.W_OK | os.X_OK):
        return _item("disk", "Disk", "fail", f"{existing} is not writable from this container")
    try:
        free_gb = shutil.disk_usage(existing).free / 1e9
    except OSError as e:
        return _item("disk", "Disk", "fail", str(e))
    state = "ok" if free_gb > 20 else "warn" if free_gb > 2 else "fail"
    return _item("disk", "Disk", state, f"{free_gb:.0f} GB free in {datasets_dir}{note}")


def list_folders(typed: str) -> list[str]:
    """Folders to suggest while a path is being typed: the children of its last complete part."""
    typed = os.path.expanduser(typed)
    parent, prefix = (typed, "") if typed.endswith("/") else os.path.split(typed)
    try:
        names = sorted(e.name for e in os.scandir(parent or ".") if e.is_dir() and not e.name.startswith("."))
    except OSError:
        return []
    return [os.path.join(parent, name) for name in names if name.startswith(prefix)][:60]


def run_preflight(settings: dict, datasets_dir: Path, mit: dict | None, demo: bool) -> dict:
    items = []
    cameras: dict = {}
    try:
        rig = rig_config(settings["config_path"])
        cameras = rig["cameras"]
        items.append(_item("config", "Config", "ok", settings["config_path"]))
    except Exception as e:
        rig = {"cameras": {}, "ports": {}, "id": None, "calibration_dir": None}
        items.append(_item("config", "Config", "fail", f"{settings['config_path']}: {e}"))
    if not Path(settings["mapping_path"]).is_file():
        items.append(_item("mapping", "KER mapping", "fail", f"{settings['mapping_path']} not found"))

    for side in ("left", "right"):
        if side in rig["ports"]:
            items.append(_check_can(side, rig["ports"][side]))
    for name, cam in cameras.items():
        path = str(cam.get("index_or_path", ""))
        try:
            ok = stat.S_ISCHR(os.stat(path).st_mode)
        except OSError:
            ok = False
        detail = path if ok else f"{path} is missing. The cameras are mapped by USB port when the container starts."
        items.append(_item(f"cam_{name}", f"Cam {name}", "ok" if ok else "fail", detail))
    items.append(_check_ker())
    items.append(_check_plugins())
    if rig["ports"] and (calibration := _check_calibration(settings["robot_type"], rig)) is not None:
        items.append(calibration)

    if mit is None:
        items.append(_item("mit", "MIT mode", "unknown", "Not checked since the GUI started. Motors fall back to their flashed mode after a power cycle.", fix="mit_set"))
    elif mit["ok"]:
        items.append(_item("mit", "MIT mode", "ok", f"all motors in MIT mode (checked {mit['at']})", fix="mit_check"))
    else:
        items.append(_item("mit", "MIT mode", "fail", f"{mit['detail']} (checked {mit['at']})", fix="mit_set"))

    items.append(_check_disk(datasets_dir))
    if demo:
        for item in items:
            item["detail"] += "  (demo mode: not required)"
    return {"items": items, "cameras": list(cameras), "at": time.time()}


def parse_mit_output(text: str, returncode: int) -> dict:
    """Result of helpers/set_mit_mode.py, from what it printed."""
    modes: dict[str, str] = {}
    iface = "?"
    for line in text.splitlines():
        if m := re.match(r"\s*\[(\S+)\]", line):
            iface = m.group(1)
        elif m := re.match(r"\s*(0x[0-9A-Fa-f]+): current = (\S+)", line):
            modes[f"{iface} {m.group(1)}"] = m.group(2)
        elif m := re.match(r"\s*(0x[0-9A-Fa-f]+): \S+ -> (\S+)\s+OK\s*$", line):
            modes[f"{iface} {m.group(1)}"] = m.group(2)
        elif m := re.match(r"\s*(0x[0-9A-Fa-f]+): ", line):
            modes[f"{iface} {m.group(1)}"] = "unconfirmed"
    wrong = [f"{motor} = {mode}" for motor, mode in modes.items() if mode != "MIT"]
    ok = returncode == 0 and bool(modes) and not wrong
    if ok:
        detail = ""
    elif wrong:
        detail = "not in MIT mode: " + ", ".join(wrong[:6]) + (" ..." if len(wrong) > 6 else "")
    else:
        detail = "the mode script failed; see the log"
    return {"ok": ok, "detail": detail, "at": datetime.now().strftime("%H:%M:%S")}


# -------------------------------------------------------------------- datasets
def list_datasets(datasets_dir: Path) -> list[dict]:
    out = []
    if not datasets_dir.is_dir():
        return out
    for path in datasets_dir.iterdir():
        info_path = path / "meta" / "info.json"
        if path.name.startswith(".") or not info_path.is_file():
            continue
        try:
            info = json.loads(info_path.read_text())
        except (OSError, ValueError):
            continue
        fps = info.get("fps") or 0
        frames = info.get("total_frames") or 0
        out.append(
            {
                "name": path.name,
                "episodes": info.get("total_episodes") or 0,
                "frames": frames,
                "fps": fps,
                "duration_s": frames / fps if fps else 0,
                "size": sum(f.stat().st_size for f in path.rglob("*") if f.is_file()),
                "modified": info_path.stat().st_mtime,
            }
        )
    return sorted(out, key=lambda d: d["modified"], reverse=True)


def list_episodes(datasets_dir: Path, name: str) -> dict:
    import pandas as pd

    root = datasets_dir / name
    info = json.loads((root / "meta" / "info.json").read_text())
    cameras = [key for key, feature in info.get("features", {}).items() if feature.get("dtype") == "video"]
    template = info.get("video_path") or "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    files = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    episodes = []
    if files:
        table = pd.concat([pd.read_parquet(f, columns=None) for f in files]).sort_values("episode_index")
        for row in table.to_dict("records"):
            videos = {}
            for key in cameras:
                if f"videos/{key}/chunk_index" not in row:
                    continue
                rel = template.format(
                    video_key=key,
                    chunk_index=int(row[f"videos/{key}/chunk_index"]),
                    file_index=int(row[f"videos/{key}/file_index"]),
                )
                videos[key.removeprefix("observation.images.")] = {
                    "src": f"/media/{name}/{rel}",
                    "from": float(row[f"videos/{key}/from_timestamp"]),
                    "to": float(row[f"videos/{key}/to_timestamp"]),
                }
            tasks = row.get("tasks")
            episodes.append(
                {
                    "index": int(row["episode_index"]),
                    "length": int(row["length"]),
                    "task": ", ".join(map(str, tasks)) if tasks is not None else "",
                    "videos": videos,
                }
            )
    return {"name": name, "fps": info.get("fps") or 30, "episodes": episodes}


# ----------------------------------------------------------------- live pieces
class FrameStore:
    """Latest JPEG per camera, with a wake-up for the MJPEG streams."""

    def __init__(self):
        self._frames: dict[str, tuple[bytes, int, float]] = {}
        self._cond = asyncio.Condition()

    async def put(self, name: str, jpeg: bytes) -> None:
        async with self._cond:
            seq = self._frames[name][1] + 1 if name in self._frames else 0
            self._frames[name] = (jpeg, seq, time.monotonic())
            self._cond.notify_all()

    async def wait(self, name: str, seen: int, timeout: float) -> tuple[bytes, int] | None:
        def fresh() -> bool:
            return name in self._frames and self._frames[name][1] != seen

        async with self._cond:
            try:
                await asyncio.wait_for(self._cond.wait_for(fresh), timeout)
            except asyncio.TimeoutError:
                return None
            jpeg, seq, _ = self._frames[name]
            return jpeg, seq

    def ages(self) -> dict[str, float]:
        now = time.monotonic()
        return {name: round(now - t, 1) for name, (_, _, t) in self._frames.items()}


class IdlePreview:
    """Reads the cameras directly while no recorder is running, to frame the scene."""

    def __init__(self, store: FrameStore):
        self._store = store
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self.errors: dict[str, str] = {}

    @property
    def active(self) -> bool:
        return bool(self._threads)

    def start(self, cameras: dict, loop: asyncio.AbstractEventLoop) -> None:
        if self.active:
            return
        self._stop = threading.Event()
        self.errors = {}
        for name, cfg in cameras.items():
            thread = threading.Thread(target=self._run, args=(name, cfg, loop, self._stop), name=f"preview-{name}", daemon=True)
            self._threads.append(thread)
            thread.start()

    def _run(self, name: str, cfg: dict, loop, stop: threading.Event) -> None:
        import cv2

        cap = cv2.VideoCapture(cfg.get("index_or_path"), cv2.CAP_V4L2)
        try:
            if not cap.isOpened():
                self.errors[name] = "could not open the camera"
                return
            if cfg.get("fourcc"):
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg["fourcc"]))
            for prop, key in ((cv2.CAP_PROP_FRAME_WIDTH, "width"), (cv2.CAP_PROP_FRAME_HEIGHT, "height"), (cv2.CAP_PROP_FPS, "fps")):
                if cfg.get(key):
                    cap.set(prop, cfg[key])
            last = 0.0
            while not stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    self.errors[name] = "the camera stopped sending frames"
                    return
                if time.monotonic() - last < 0.1:
                    continue
                last = time.monotonic()
                jpeg = _encode(frame, rgb=False)
                if jpeg:
                    asyncio.run_coroutine_threadsafe(self._store.put(name, jpeg), loop)
        finally:
            cap.release()

    async def stop(self) -> None:
        """Returns once every camera has been released, so the recorder can open them."""
        if not self.active:
            return
        self._stop.set()
        threads, self._threads = self._threads, []
        await asyncio.get_running_loop().run_in_executor(None, lambda: [t.join(timeout=5.0) for t in threads])


def describe_phase(session: dict, live: dict | None) -> dict:
    """What the big banner says. tone: idle, info, warn, go, rec, ok or bad."""

    def phase(key: str, title: str, sub: str, tone: str) -> dict:
        return {"key": key, "title": title, "sub": sub, "tone": tone}

    state = session["state"]
    if state == "idle":
        return phase("idle", "Ready", "Set up the session and press Start.", "idle")
    if state == "preparing":
        return phase("preparing", "Preparing", session.get("step") or "Checking the plugins and the motor mode.", "info")
    if state == "exited" and session.get("error"):
        return phase("failed", "Could not start", session["error"], "bad")
    if state == "exited":
        saved = (live or {}).get("saved", 0)
        where = f"{saved} episode{'' if saved == 1 else 's'} saved to {session['root']}"
        if session["exit_code"] == 0:
            return phase("done", "Session complete", where, "ok")
        if session.get("stop_mode") in ("interrupt", "kill"):
            return phase("done", "Session stopped", where, "warn")
        return phase("failed", "Recorder failed", f"Exit code {session['exit_code']}. The reason is in the log below.", "bad")
    if live is None or live["stage"] in ("starting", "connecting"):
        return phase("connecting", "Connecting", "Opening the cameras, the KER and both arms.", "info")
    if live["stage"] == "finalizing":
        return phase("finalizing", "Finalizing", "Writing the last video and the dataset metadata.", "info")

    park, guided = live["park"], live["guided"]
    episode = live["saved"] + 1
    if live["recording"]:
        sub = "Bring the KER back to the ready pose and hold still to end the episode." if guided else ""
        return phase("recording", "Recording", sub, "rec")
    if live["saving"]:
        return phase("saving", "Saving episode", "Encoding and writing the episode.", "info")
    if park == "to_rest":
        return phase("to_rest", "Returning to rest", "Keep clear of the arms.", "warn")
    if park == "to_ready":
        return phase("to_ready", "Moving to ready pose", "Keep clear of the arms.", "warn")
    if park == "countdown":
        left = live.get("countdown")
        title = f"Taking over in {left:.0f} s" if left is not None else "Taking over"
        return phase("countdown", title, "Hold the KER where the arms should go.", "warn")
    if park == "engaging":
        return phase("engaging", "Taking over", "The arms are easing onto the KER pose.", "warn")
    if park == "ready":
        if live["loop"] == "episode":
            return phase("takeover", "Take over", f"Bring the KER to the ready pose to start episode {episode}.", "go")
        return phase("reset", "Reset the scene", "The arms are holding the ready pose.", "info")
    if park == "rest":
        return phase("rest", "Arms at rest", "They rise to the ready pose shortly. A double squeeze of both triggers also does it.", "idle")
    if live["loop"] == "reset":
        return phase("reset", "Reset the scene", "", "info")
    return phase("running", "Running", "", "info")


class Console:
    def __init__(self, args):
        self.demo: bool = args.demo
        # --datasets-dir is the default folder for datasets, and where the GUI keeps its own
        # settings and logs whichever folder is chosen on the page.
        DEFAULTS["datasets_dir"] = args.datasets_dir.rstrip("/") or "/"
        self.state_dir = Path(args.datasets_dir) / ".record_gui"
        self.clients: set[web.WebSocketResponse] = set()
        self.log: deque[dict] = deque(maxlen=LOG_KEEP)
        self._log_seq = 0
        self._log_file = None
        self.settings = self._load_settings()
        self.session = {"state": "idle"}
        self.live: dict | None = None
        self.frames = FrameStore()
        self.preview = IdlePreview(self.frames)
        self.proc: asyncio.subprocess.Process | None = None
        self.worker: asyncio.StreamWriter | None = None
        self.task: str | None = None
        self.mit: dict | None = None
        self.preflight: dict | None = None
        self.closing = False
        self._sock_dir = tempfile.mkdtemp(prefix="record_gui_")
        self.sock_path = os.path.join(self._sock_dir, "worker.sock")
        self._background: set[asyncio.Task] = set()

    @property
    def datasets_dir(self) -> Path:
        return Path(self.settings["datasets_dir"])

    def apply_settings(self, raw: dict) -> bool:
        """Take new settings from the page. True if the datasets folder changed."""
        before = self.settings["datasets_dir"]
        self.settings = clean_settings(raw)
        if self.settings["resume"] and not (self.datasets_dir / self.settings["resume"] / "meta" / "info.json").is_file():
            self.settings["resume"] = ""  # belonged to the previous folder
        self._save_settings()
        return self.settings["datasets_dir"] != before

    # ------------------------------------------------------------- plumbing
    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    def _load_settings(self) -> dict:
        try:
            return clean_settings(json.loads((self.state_dir / "settings.json").read_text()))
        except (OSError, ValueError, BadRequest):
            return dict(DEFAULTS)

    def _save_settings(self) -> None:
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            (self.state_dir / "settings.json").write_text(json.dumps(self.settings, indent=2))
        except OSError as e:
            self.add_log(f"could not save the settings: {e}", "gui")

    async def send(self, msg: dict) -> None:
        data = json.dumps(msg)
        for ws in list(self.clients):
            try:
                await asyncio.wait_for(ws.send_str(data), 2.0)
            except (ConnectionError, asyncio.TimeoutError, RuntimeError):
                self.clients.discard(ws)

    def add_log(self, text: str, source: str = "rec") -> None:
        """Queue log lines for the page (and the session's log file). source: rec, gui or task."""
        lines = []
        for line in text.splitlines():
            line = ANSI.sub("", line).rstrip()
            if not line:
                continue
            level = "err" if re.search(r"\bERROR\b|Traceback|Error:|Exception", line) else "warn" if re.search(r"\bWARN(ING)?\b", line) else ""
            self._log_seq += 1
            entry = {"i": self._log_seq, "t": time.time(), "src": source, "lvl": level, "text": line}
            self.log.append(entry)
            lines.append(entry)
            if self._log_file is not None:
                self._log_file.write(f"{datetime.now():%H:%M:%S} {line}\n")
        if self._log_file is not None:
            self._log_file.flush()
        if lines:
            self._spawn(self.send({"type": "log", "lines": lines}))

    def state_message(self) -> dict:
        return {
            "type": "state",
            "session": self.session,
            "live": self.live if self.session["state"] != "idle" else None,
            "phase": describe_phase(self.session, self.live),
            "cams": self.frames.ages(),
            "task": self.task,
            "preview": self.preview.active,
            "preview_errors": self.preview.errors,
        }

    async def push_state(self) -> None:
        await self.send(self.state_message())

    @property
    def busy(self) -> bool:
        return self.session["state"] in ("preparing", "starting", "running", "stopping")

    async def refresh_preflight(self) -> dict:
        loop = asyncio.get_running_loop()
        self.preflight = await loop.run_in_executor(None, run_preflight, self.settings, self.datasets_dir, self.mit, self.demo)
        await self.send({"type": "preflight", "preflight": self.preflight})
        return self.preflight

    async def heartbeat(self) -> None:
        """While no recorder is reporting, keep the page's camera ages and state fresh."""
        while not self.closing:
            await asyncio.sleep(1.0)
            if self.clients and self.session["state"] != "running":
                await self.push_state()

    # ------------------------------------------------------------- recorder
    async def start_session(self, raw: dict) -> None:
        if self.busy:
            raise BadRequest("a session is already running")
        if self.task:
            raise BadRequest(f"wait for '{self.task}' to finish")
        wanted = clean_settings(raw)
        if wanted["resume"] and not (Path(wanted["datasets_dir"]) / wanted["resume"] / "meta" / "info.json").is_file():
            raise BadRequest(f"dataset to add to not found: {wanted['datasets_dir']}/{wanted['resume']}")
        try:
            Path(wanted["datasets_dir"]).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise BadRequest(f"cannot create the datasets folder: {e}") from None
        if self.apply_settings(wanted):
            await self.send({"type": "datasets_changed"})
        await self.preview.stop()

        s = self.settings
        args, root = build_record_args(s, datetime.now().strftime("%Y%m%d_%H%M%S"))
        cmd = [sys.executable, "-u", "-m", "record_gui.worker", f"--preview-fps={s['preview_fps']:g}"]
        if self.demo:
            cameras = (self.preflight or {}).get("cameras") or ["left_wrist", "right_wrist", "base"]
            cmd += ["--demo", f"--cameras={','.join(cameras)}"]
        prepare = s["auto_prepare"] and not self.demo
        self.session = {
            "state": "preparing" if prepare else "starting",
            "started_at": time.time(),
            "root": root,
            "task": s["task"],
            "num_episodes": s["num_episodes"],
            "episode_time_s": s["episode_time_s"],
            "resume": bool(s["resume"]),
            "command": "lerobot-ker-record " + shlex.join(args),
            "exit_code": None,
            "stop_mode": None,
            "demo": self.demo,
        }
        self.live = None
        self.frames = FrameStore()
        self.preview = IdlePreview(self.frames)
        try:
            (self.state_dir / "logs").mkdir(parents=True, exist_ok=True)
            self._log_file = open(self.state_dir / "logs" / f"{Path(root).name}_{datetime.now():%H%M%S}.log", "a", encoding="utf-8")
        except OSError:
            self._log_file = None
        await self.push_state()
        self._spawn(self._launch([*cmd, "--", *args], prepare))

    async def _launch(self, cmd: list[str], prepare: bool) -> None:
        session = self.session
        if prepare:
            try:
                error = await self._prepare(session)
            except Exception as e:  # never leave the session stuck in "preparing"
                error = f"Preparing the rig failed: {type(e).__name__}: {e}"
            if error is None and session["state"] == "stopping":
                error = "Stopped before the recorder started."
            if error is not None:
                self.add_log(error, "gui")
                session.update(state="exited", error=error, ended_at=time.time())
                self._close_log_file()
                await self.push_state()
                return
            session["state"] = "starting"
        self.add_log(f"$ {session['command']}", "gui")
        env = {**os.environ, "RECORD_GUI_SOCKET": self.sock_path, "PYTHONUNBUFFERED": "1"}
        self.proc = await asyncio.create_subprocess_exec(
            *cmd, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env
        )
        self._spawn(self._watch(self.proc))
        await self.push_state()

    async def _prepare(self, session: dict) -> str | None:
        """What used to be typed by hand before a session. Returns why it failed, or None.

        Installs the plugins when they are missing. Then reads the motors' control
        mode and switches them to MIT only if some are not in it: after a power cycle
        that is the first session, later ones just pass the check.
        """

        async def step(text: str) -> None:
            session["step"] = text
            await self.push_state()

        if _check_plugins()["state"] != "ok":
            await step("Installing the KER teleoperator and follower plugins.")
            code, _ = await self._run_commands(self._task_commands("install_plugins"))
            await self.refresh_preflight()
            if code:
                return "Installing the plugins failed. The reason is in the log below."
        if session["state"] == "stopping":
            return None
        await step("Checking the motors' control mode.")
        code, output = await self._run_commands(self._task_commands("mit_check"))
        self.mit = parse_mit_output(output, code)
        if not self.mit["ok"] and session["state"] != "stopping":
            await step("Switching the motors to MIT mode.")
            code, output = await self._run_commands(self._task_commands("mit_set"))
            self.mit = parse_mit_output(output, code)
        await self.refresh_preflight()
        if not self.mit["ok"] and session["state"] != "stopping":
            detail = self.mit["detail"].removeprefix("not in MIT mode: ") or "see the log below"
            return f"Could not switch the motors to MIT mode ({detail}). Check that they are powered and the CAN interfaces are up."
        return None

    def _close_log_file(self) -> None:
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    async def _pipe_to_log(self, stream: asyncio.StreamReader, source: str, sink: list[str] | None = None) -> None:
        pending = b""
        while chunk := await stream.read(8192):
            # Progress bars redraw with \r; show each redraw as its own line.
            *lines, pending = re.split(rb"[\r\n]", pending + chunk)
            text = "\n".join(line.decode(errors="replace") for line in lines)
            if sink is not None:
                sink.append(text)
            self.add_log(text, source)
        if pending:
            if sink is not None:
                sink.append(pending.decode(errors="replace"))
            self.add_log(pending.decode(errors="replace"), source)

    async def _watch(self, proc: asyncio.subprocess.Process) -> None:
        await self._pipe_to_log(proc.stdout, "rec")
        code = await proc.wait()
        self.session.update(state="exited", exit_code=code, ended_at=time.time())
        self.proc = None
        self.add_log(f"recorder exited with code {code}", "gui")
        self._close_log_file()
        await self.push_state()
        await self.send({"type": "datasets_changed"})

    async def on_worker(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.worker = writer
        if self.session["state"] == "starting":
            self.session["state"] = "running"
        try:
            while True:
                size, kind = ipc.HEADER.unpack(await reader.readexactly(ipc.HEADER.size))
                if size > ipc.MAX_PAYLOAD:
                    break
                payload = await reader.readexactly(size)
                if kind == ipc.FRAME:
                    await self.frames.put(*ipc.unpack_frame(payload))
                elif kind == ipc.JSON:
                    self.live = json.loads(payload)
                    await self.push_state()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            if self.worker is writer:
                self.worker = None
            writer.close()

    def command(self, msg: dict) -> None:
        if self.worker is None:
            raise BadRequest("the recorder is not connected")
        self.worker.write(ipc.pack_json(msg))

    async def stop_session(self, mode: str, discard: bool = False) -> None:
        proc = self.proc
        if proc is None and self.session["state"] == "preparing":
            self.add_log("stop requested; the recorder will not be started", "gui")
            self.session["state"] = "stopping"
            await self.push_state()
            return
        if proc is None and self.session["state"] == "stopping":
            raise BadRequest("stopping as soon as the current preparation step finishes")
        if proc is None:
            raise BadRequest("no session is running")
        self.session["stop_mode"] = mode
        if mode == "kill":
            self.add_log("killing the recorder", "gui")
            proc.kill()
        elif mode == "interrupt" or self.worker is None:
            self.add_log("interrupting the recorder (Ctrl-C)", "gui")
            proc.send_signal(signal.SIGINT)
        else:
            self.add_log("stop requested" + (", discarding the current take" if discard else ""), "gui")
            self.command({"cmd": "stop", "discard": discard})
        self.session["state"] = "stopping"
        await self.push_state()

    # ---------------------------------------------------------------- tasks
    async def run_task(self, name: str) -> None:
        if self.busy:
            raise BadRequest("not while a session is running")
        if self.task:
            raise BadRequest(f"'{self.task}' is still running")
        if self.demo and name.startswith("mit_"):
            raise BadRequest("demo mode does not touch the motors")
        commands = self._task_commands(name)
        self.task = name
        self._spawn(self._run_task(name, commands))
        await self.push_state()

    def _task_commands(self, name: str) -> list[list[str]]:
        if name == "install_plugins":
            return [[sys.executable, "-m", "pip", "install", "-e", path] for path in PLUGINS.values()]
        if name in ("mit_check", "mit_set"):
            ports = rig_config(self.settings["config_path"])["ports"].values()
            mit = [sys.executable, "helpers/set_mit_mode.py", "--interfaces", *ports]
            return [[*mit, "--read-only"]] if name == "mit_check" else [mit]
        raise BadRequest(f"unknown task {name}")

    async def _run_commands(self, commands: list[list[str]]) -> tuple[int, str]:
        """Run helper commands in turn, output to the log. (exit code of the last one run, its output)"""
        output: list[str] = []
        code = 0
        try:
            for cmd in commands:
                self.add_log(f"$ {shlex.join(cmd)}", "gui")
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
                )
                await self._pipe_to_log(proc.stdout, "task", output)
                code = await proc.wait()
                if code:
                    self.add_log(f"'{shlex.join(cmd)}' failed with code {code}", "gui")
                    break
        except OSError as e:
            code = 1
            self.add_log(f"{commands[0][0]}: {e}", "gui")
        return code, "\n".join(output)

    async def _run_task(self, name: str, commands: list[list[str]]) -> None:
        try:
            code, output = await self._run_commands(commands)
            if name.startswith("mit_"):
                self.mit = parse_mit_output(output, code)
        finally:
            self.task = None
            await self.refresh_preflight()
            await self.push_state()

    async def shutdown(self) -> None:
        """Server going down: end the session the normal way so the arms return to rest."""
        self.closing = True
        await self.preview.stop()
        proc = self.proc
        if proc is not None:
            print("record_gui: stopping the recorder, the arms return to the rest pose ...", flush=True)
            with contextlib.suppress(ProcessLookupError):
                proc.send_signal(signal.SIGINT)
            try:
                await asyncio.wait_for(proc.wait(), STOP_WAIT_S)
            except asyncio.TimeoutError:
                proc.kill()
        for ws in list(self.clients):
            await ws.close()
        shutil.rmtree(self._sock_dir, ignore_errors=True)


# ---------------------------------------------------------------------- routes
def routes(console: Console) -> list:
    async def body(request: web.Request) -> dict:
        try:
            data = await request.json()
        except ValueError:
            raise BadRequest("expected a JSON body") from None
        if not isinstance(data, dict):
            raise BadRequest("expected a JSON object")
        return data

    def dataset_name(request: web.Request) -> str:
        name = request.match_info["name"]
        if not re.fullmatch(r"[\w.\-]+", name) or name.startswith("."):
            raise web.HTTPNotFound()
        return name

    async def index(request):
        return web.FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    async def ws_handler(request):
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        preflight = console.preflight or await console.refresh_preflight()
        await ws.send_str(
            json.dumps(
                {
                    "type": "hello",
                    "settings": console.settings,
                    "defaults": DEFAULTS,
                    "preflight": preflight,
                    "log": list(console.log)[-LOG_BACKLOG:],
                    "demo": console.demo,
                    "host": socket.gethostname(),
                }
            )
        )
        await ws.send_str(json.dumps(console.state_message()))
        console.clients.add(ws)
        try:
            async for msg in ws:
                if msg.type == WSMsgType.ERROR:
                    break
        finally:
            console.clients.discard(ws)
        return ws

    async def camera(request):
        name = request.match_info["name"]
        response = web.StreamResponse(
            headers={"Content-Type": "multipart/x-mixed-replace; boundary=frame", "Cache-Control": "no-store"}
        )
        await response.prepare(request)
        seen = -1
        try:
            await response.write(b"--frame\r\n")
            while not console.closing:
                if request.transport is None or request.transport.is_closing():
                    break
                got = await console.frames.wait(name, seen, 2.0)
                if got is None:
                    continue
                jpeg, seen = got
                # The next boundary goes out with the frame: Chrome only paints a part once it has seen it.
                head = b"Content-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg)
                await response.write(head + jpeg + b"\r\n--frame\r\n")
        except (ConnectionError, asyncio.CancelledError):
            pass
        return response

    async def post_preflight(request):
        return web.json_response(await console.refresh_preflight())

    async def post_settings(request):
        if console.busy:
            raise BadRequest("not while a session is running")
        if console.apply_settings(await body(request)):
            await console.send({"type": "datasets_changed"})
        await console.refresh_preflight()
        return web.json_response({"settings": console.settings})

    async def get_folders(request):
        loop = asyncio.get_running_loop()
        return web.json_response({"folders": await loop.run_in_executor(None, list_folders, request.query.get("path", ""))})

    async def post_start(request):
        await console.start_session(await body(request))
        return web.json_response({"ok": True})

    async def post_stop(request):
        data = await body(request)
        mode = data.get("mode", "graceful")
        if mode not in ("graceful", "interrupt", "kill"):
            raise BadRequest("mode must be graceful, interrupt or kill")
        await console.stop_session(mode, bool(data.get("discard")))
        return web.json_response({"ok": True})

    async def post_command(request):
        data = await body(request)
        if data.get("cmd") not in ("next", "rerecord", "park"):
            raise BadRequest("unknown command")
        console.command({"cmd": data["cmd"], "move": data.get("move")})
        return web.json_response({"ok": True})

    async def post_task(request):
        await console.run_task(str((await body(request)).get("name")))
        return web.json_response({"ok": True})

    async def post_preview(request):
        on = bool((await body(request)).get("on"))
        if on:
            if console.busy:
                raise BadRequest("the recorder has the cameras")
            cameras = rig_config(console.settings["config_path"])["cameras"]
            console.preview.start(cameras, asyncio.get_running_loop())
        else:
            await console.preview.stop()
        await console.push_state()
        return web.json_response({"ok": True})

    async def get_datasets(request):
        loop = asyncio.get_running_loop()
        return web.json_response({"datasets": await loop.run_in_executor(None, list_datasets, console.datasets_dir)})

    async def get_episodes(request):
        name = dataset_name(request)
        loop = asyncio.get_running_loop()
        try:
            return web.json_response(await loop.run_in_executor(None, list_episodes, console.datasets_dir, name))
        except FileNotFoundError:
            raise web.HTTPNotFound() from None

    async def media(request):
        root = (console.datasets_dir / dataset_name(request)).resolve()
        path = (root / request.match_info["tail"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path)

    return [
        web.get("/", index),
        web.get("/ws", ws_handler),
        web.get("/cam/{name}.mjpg", camera),
        web.post("/api/preflight", post_preflight),
        web.post("/api/settings", post_settings),
        web.get("/api/folders", get_folders),
        web.post("/api/start", post_start),
        web.post("/api/stop", post_stop),
        web.post("/api/command", post_command),
        web.post("/api/task", post_task),
        web.post("/api/preview", post_preview),
        web.get("/api/datasets", get_datasets),
        web.get("/api/datasets/{name}/episodes", get_episodes),
        web.get("/media/{name}/{tail:.+}", media),
    ]


@web.middleware
async def errors(request, handler):
    try:
        return await handler(request)
    except BadRequest as e:
        return web.json_response({"error": str(e)}, status=400)
    except (OSError, ValueError, KeyError) as e:
        if request.path.startswith("/api/"):
            return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=500)
        raise


def make_app(args) -> web.Application:
    console = Console(args)
    app = web.Application(middlewares=[errors])
    app["console"] = console
    app.add_routes(routes(console))

    async def on_startup(app):
        app["worker_server"] = await asyncio.start_unix_server(console.on_worker, path=console.sock_path)
        console._spawn(console.heartbeat())
        # A fresh container has none of the plugins: install them now rather than at the first session.
        if not console.demo and _check_plugins()["state"] != "ok":
            await console.run_task("install_plugins")

    async def on_shutdown(app):
        await console.shutdown()
        app["worker_server"].close()

    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    return app


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m record_gui", description="Browser GUI for guided KER recording sessions.")
    ap.add_argument("--host", default="127.0.0.1", help="address to listen on; 0.0.0.0 lets other machines on the network control the rig")
    ap.add_argument("--port", type=int, default=8642)
    ap.add_argument("--datasets-dir", default="./datasets", help="default datasets folder (changeable on the page); GUI settings and logs are kept in it")
    ap.add_argument("--demo", action="store_true", help="simulated sessions, no robot needed")
    args = ap.parse_args()
    if not Path("record_gui").is_dir():
        ap.error("run this from the repo root (the recorder resolves ./helpers, ./datasets ... from there)")
    print(f"record_gui: open http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}", flush=True)
    web.run_app(make_app(args), host=args.host, port=args.port, print=None, shutdown_timeout=1.0)


if __name__ == "__main__":
    main()
