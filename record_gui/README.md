# record_gui: browser GUI for guided KER recording

Runs `lerobot-ker-record` sessions on the bimanual OpenArm rig from a web page: rig
check, session setup, live phase / timer / cameras, takeover guidance, episode
controls, log and a dataset browser. It runs inside the `lerobot_flashrt` container
and needs nothing the image does not already have.

## Start

From the host (finds a running container of the image, or starts one):

    bash scripts/run_record_gui.sh           # then http://localhost:8642 opens
    bash scripts/run_record_gui.sh --demo    # simulated sessions, no robot needed

Or inside a container, from `/lerobot-flashrt`:

    python -m record_gui [--port 8642] [--host 127.0.0.1] [--datasets-dir ./datasets] [--demo]

The page is only reachable from the rig PC unless `--host 0.0.0.0` is given. Anyone
who can open the page can move the arms, so only do that on a trusted network.

The launcher's environment overrides (`IMAGE`, `CONTAINER`, `PORT`, camera paths ...)
are listed at the top of `scripts/run_record_gui.sh`. When it has to start a container
it passes the KER by its current USB address, found by USB id (`303a:4002`).

After changing anything in `record_gui/`, restart the server (Ctrl-C, then start it
again) and reload the page. The repo is mounted into the container, so no rebuild or
reinstall is needed.

## A session

1. **Rig check** (right column, and the pills in the top bar): CAN interfaces, the
   three cameras, the KER, the plugins, the followers' calibration, MIT mode and free disk space. The plugins
   and MIT mode are taken care of when a session starts (see "Automatic preparation"
   below); the "Install" and "Set MIT mode" buttons do the same by hand.
2. **Session**: task, number of episodes, episode limit, the datasets folder, and
   either a new dataset (`<folder>/<name>_<date>_<time>`) or "Add to" an existing one
   (`--resume=true`). The folder is a path inside the container (the repo is
   `/lerobot-flashrt`, the NAS is `/nas`); it is created when a session starts, and
   the Datasets tab lists whatever it holds. Everything else from the usual command
   line is under Advanced. Settings are kept in `<datasets-dir>/.record_gui/settings.json`,
   where `<datasets-dir>` is the `--datasets-dir` the server was started with.
3. **Start session**. After a short *Preparing* step the banner follows the guided
   workflow: arms at rest, moving
   to ready, *Take over* (with per-joint guidance for matching the ready pose),
   *Recording* (timer, and progress of the hold-still gesture that ends the episode),
   saving, and so on until the arms are back at rest.

| Control | Key | What it does |
| --- | --- | --- |
| Save & next | `N` / `→` | end the episode now and save it |
| Re-record | `R` / `←` | discard this take and record it again |
| Stop | `Esc` / `Q` | end the session; while recording, choose to keep or discard the take |
| Take over now | | at the ready pose: hand over to the KER after the countdown (`go`) |
| Raise to ready / Lower to rest | | park moves while the arms are parked |

Stop is the normal end: the arms return to the rest pose. If that hangs, the button
becomes "Interrupt" (Ctrl-C) and "Kill" (no return to rest: support the arms first).
Closing the server (Ctrl-C in its terminal) also stops a running session the normal way.

"Preview cameras" opens the cameras while no session runs, to frame the scene. The
Datasets tab lists what is in the datasets folder and plays back saved episodes.
Each session's log is also written to `<datasets-dir>/.record_gui/logs/`.

## Notes

**Where data goes**

- The datasets folder is a path inside the container. With the usual mounts,
  `/lerobot-flashrt/datasets` is the repo's `datasets/` and `/nas/...` is `/mnt/nas/...`
  on the host. A path that is not on a mount is lost when a `--rm` container exits.
- The container runs as root, so datasets and the `.record_gui` folder are owned by
  root on the host.
- With streaming encoding the videos are written while the episode is recorded. On a
  slow or busy network share that can add control-loop overruns; compare the overrun
  percentage in the Control loops card against a session on local disk.
- The GUI's own settings and session logs always stay under the `--datasets-dir` the
  server was started with (`./datasets/.record_gui/` by default), whichever folder is
  chosen on the page.
- "Add to" records `Episodes` more episodes into the chosen dataset. It lists only
  datasets in the current folder.

**Automatic preparation**

Every Start first does what used to be typed by hand in a new container:

- If any plugin (or `openarm_ker`) is missing, it runs the `pip install -e`
  commands for the KER teleoperator and the three follower plugins. That only happens in a fresh container, and needs network access there
  because `openarm_ker` comes from PyPI.
- The GUI also installs missing plugins as soon as it starts, so in a fresh container the rig check
  shows them as installed before the first session.
- The hard-stop followers (`openarm_videron`, `bi_openarm_follower_hardstop`) switch their own motors
  to MIT mode whenever they connect, from the GUI or a terminal. The step below does the same before the
  recorder starts, for every robot type.
- It reads every motor's control mode (`helpers/set_mit_mode.py --read-only`). Only if
  some motor is not in MIT mode does it run the script for real. So the first session
  after the motors are powered on switches them, and later sessions just pass the
  check, which takes about a second.
- Switching disables the motors first, so the arms go limp for a moment. Start
  sessions with the arms hanging at the rest pose, which the guided workflow expects
  anyway. Nothing else may be driving the CAN buses at that moment.
- If a step fails the recorder is not started; the banner says which step and the log
  has the output. Stop during *Preparing* cancels the start after the current step.
- "Install plugins and set MIT mode automatically" under Advanced turns all of this
  off. Demo mode never does it.

**Rig check**

- The default robot is `openarm_videron` with `helpers/rollout_videron.yaml`: the
  250 Hz streaming follower on hard-stop calibrated arms. Its *Calibration* check
  fails until both arms have a calibration file. Calibrate once from a terminal
  (`lerobot_robot_openarm_videron/README.md`); a session has no terminal, so it
  cannot run the calibration itself. Robot type and config file are under Advanced;
  settings saved by an earlier version keep their old values until changed there.
- The pills in the top bar repeat the checks; hover one for the detail. A failed check
  does not block Start, it asks for confirmation (missing plugins and the motor mode
  are not asked about while automatic preparation is on).
- MIT mode shows amber until the first session or check: the GUI cannot tell whether
  the motors were power-cycled before it started.
- "KER ... not in this container" means the KER was re-plugged after the container
  started and got a new USB address. Start a new container.
- A CAN interface that is down has to be brought up on the host; the check shows the
  command.

**During a session**

- Stop while recording asks whether to keep the current take. Stop at any other time
  loses nothing: there is no take in progress.
- "Take over now" hands control to the KER wherever it is held, after the countdown.
  The arms move from the ready pose to the KER pose, so hold it somewhere sensible.
- Park moves (Raise to ready, Lower to rest, Take over now) are only accepted while
  the arms are parked, never in the middle of an episode.
- Keyboard shortcuts are ignored while typing in a field or while a dialog is open.
- The camera tiles show a half-resolution JPEG preview at `Preview FPS` (Advanced,
  default 10; 0 turns it off). What is recorded is unaffected. Click a tile to enlarge it.
- Sound (top right) beeps when recording starts and ends. Browsers only allow it after
  a click on the page, which pressing Start provides.
- If the recorder fails, the banner says so and the reason is at the end of the Log
  tab. "Copy command" gives the equivalent `lerobot-ker-record` command line for
  running the same session in a terminal.

**Demo mode**

`--demo` runs a scripted session with made-up joint angles and camera images. It never
opens the CAN buses or the KER and refuses the MIT mode buttons, but the rig check and
"Preview cameras" still look at the real devices. Give it its own `--datasets-dir`
(for example `/tmp/demo`) so it does not overwrite the saved settings of the real rig.

**Limits**

- One session at a time, and one server per port. The launcher only opens the page if
  a server is already answering.
- Episode playback uses the browser's own video decoder. If it cannot play the
  dataset's codec the player says so; the dataset itself is fine.
- The page has no login. Keep the default `--host 127.0.0.1` unless the network is trusted.

## How it is put together

- `server.py`: aiohttp server. Serves `static/index.html`, pushes state and log over
  a WebSocket, streams camera previews as MJPEG, runs the checks and helper commands.
- `worker.py`: the recorder, a child process. It is `lerobot-ker-record` with hooks
  on lerobot's `record_loop`, robot/teleoperator factories and the dataset's
  `save_episode` / `finalize`, plus a thread that reports status (10 Hz) and JPEG
  previews to the server and applies its commands to lerobot's `events` flags.
  `--demo` replaces all of that with a scripted session.
- `ipc.py`: the framing used on the Unix socket between the two.

The recorder is a separate process so the web server never competes with the 250 Hz
arm loops for the interpreter. If the server disappears, the recorder stops the
session and returns the arms to rest.
