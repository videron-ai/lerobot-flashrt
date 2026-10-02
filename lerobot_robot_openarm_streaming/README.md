# lerobot_robot_openarm_streaming

A LeRobot robot plugin that adds `--robot.type=bi_openarm_streaming_follower`.
It is the stock `bi_openarm_follower` plus a background MIT control loop for each
arm, running at 250 Hz by default.

Stock LeRobot sends each arm one MIT command per recorded frame, so at 30 fps the
arm moves in 33 ms steps. With this plugin, LeRobot still records at 30 fps, but
the arms are commanded 250 times a second:

- **Teleop with the KER** (`--teleop.stream_to_follower=true` on
  `lerobot_teleoperator_openarm_ker` 0.3.0+): the follower reads the KER's
  500 Hz stream directly, filtered with One Euro, at the loop rate. The
  recorded action is the same target the follower is being sent at that moment.
- **Policies, or any other teleop:** the 30 Hz actions from `send_action()` are
  interpolated into a 250 Hz trajectory. The ramp's velocity is sent as MIT
  velocity feed-forward.

Datasets are unchanged. The robot reports its name as `bi_openarm_follower`, so
the dataset `robot_type`, features, keys and calibration files all match the
stock follower. New episodes merge with your existing OpenArm 1.0 data, and
policies trained on that data run on this follower without changes.

**Status:** KER teleoperation has run on the rig, including `reply_window_ms`
and the KER plugin's park moves and gesture (see Hardware runs under First time
on hardware). Recording, policy rollouts, interventions and the CAN timeout
script have only run against simulated motors. The tests run LeRobot 0.6.2's
real config parsing, plugin discovery and `DamiaoMotorsBus` code on a virtual
CAN bus with simulated Damiao motors, and all 14 pass.

## Install

```bash
pip install -e path/to/lerobot_robot_openarm_streaming
pip install -e path/to/lerobot_teleoperator_openarm_ker   # 0.3.0+, for KER streaming
```

LeRobot auto-imports installed packages whose names start with `lerobot_robot_`
and `lerobot_teleoperator_`. Calibrate with the stock robot type as before. The
calibration files are shared.

The commands below are written for the rig described by `helpers/rollout.yaml`
in the lerobot-flashrt repo (id `my_bimanual_follower`, classic CAN on
`left_arm` / `right_arm`, custom joint limits with `side: null`, gripper 0 to
-160 deg, `left_wrist` / `right_wrist` / `base` cameras). Run them with:

```bash
export FLASHRT=~/Desktop/Experiments/lerobot-flashrt   # /lerobot-flashrt inside the container
```

## Teleoperate with the KER

```bash
lerobot-teleoperate \
  --config_path=$FLASHRT/helpers/rollout.yaml \
  --robot.type=bi_openarm_streaming_follower \
  --teleop.type=openarm_ker --teleop.id=ker \
  --teleop.mapping_path=path/to/lerobot_teleoperator_openarm_ker/ker_mapping.json \
  --teleop.filter=one_euro \
  --teleop.stream_to_follower=true
```

The followers approach the KER pose at 20 deg/s, then track it at 250 Hz. Use
this to check tracking and tune the filter before recording.

To start with the arms held at the calibration pose and move them with the KER
plugin's `ready` / `go` / `rest` commands instead (see Park in that plugin's README),
add `--teleop.park=true`. Park moves run through the same 250 Hz loops.
`--teleop.park_gesture=true` adds a hands-free toggle: squeeze both KER triggers
together twice.

```bash
lerobot-teleoperate \
  --config_path=$FLASHRT/helpers/rollout.yaml \
  --robot.type=bi_openarm_streaming_follower \
  --teleop.type=openarm_ker --teleop.id=ker \
  --teleop.mapping_path=path/to/lerobot_teleoperator_openarm_ker/ker_mapping.json \
  --teleop.filter=one_euro \
  --teleop.stream_to_follower=true \
  --teleop.park=true \
  --teleop.park_gesture=true
```
 `--fps` only sets
how often LeRobot polls the KER and reads the cameras (default 60); the arms are
driven by the control loops either way.

## Record with the KER

```bash
lerobot-record \
  --config_path=$FLASHRT/helpers/rollout.yaml \
  --robot.type=bi_openarm_streaming_follower \
  --teleop.type=openarm_ker --teleop.id=ker \
  --teleop.mapping_path=path/to/lerobot_teleoperator_openarm_ker/ker_mapping.json \
  --teleop.filter=one_euro \
  --teleop.stream_to_follower=true \
  --dataset.repo_id=${HF_USER}/openarm_ker_demo \
  --dataset.single_task="Put the cup on the plate" \
  --dataset.fps=30
```

`rollout.yaml` says `type: bi_openarm_follower`; the `--robot.type` flag
overrides it and everything else in the file (ports, classic CAN, joint limits,
cameras) still applies. `ker_mapping.json` is the rig mapping from the KER
plugin's dry run (software wrist remap, joint 7 inverted on both sides).

`rollout.yaml` also sets `max_relative_target`; it is ignored by both commands
(with a warning per arm) and replaced by the speed caps below.

In stream mode the One Euro filter runs on every KER frame (500 Hz) rather than
every recorded frame. The filter accounts for the time step, so the same settings
behave much the same, but retune by feel in `lerobot-teleoperate` before
recording a dataset.

One thing to keep in mind when mixing data: follower states in new episodes lag
the action less than in your 30 Hz episodes, and the motion between frames is
smoother. The action space and its meaning are unchanged, but this is a small
shift in `observation.state` dynamics. Check it on held-out evaluation if you
train on both.

## Run a policy

Use your usual `lerobot-record --policy.path=...` or `lerobot-rollout` command
and change only the robot type: with `--config_path=$FLASHRT/helpers/rollout.yaml`,
add `--robot.type=bi_openarm_streaming_follower`.
`command_interpolation=interpolate` (the default) ramps from the previous action
to the latest one over the measured interval between them.
- **Cost:** the target lags by up to one action interval (33 ms at 30 Hz).
- **Gain:** velocity feed-forward removes roughly `kd/kp` of tracking lag
  (12–21 ms with the default gains).
- **Net, in simulation:** about the same total lag as stock, with much less
  shake.

`command_interpolation=hold` jumps to each action instead (still under the speed
cap).

**DAgger / interventions** (`lerobot-rollout` with the KER attached and
`--teleop.stream_to_follower=true`): with `target_source=auto` the policy drives
until a human takes over. The KER stream counts as active only while
`get_action()` is being called regularly, so a single call (such as LeRobot's
handover read) doesn't take control.
- **Takeover:** when correction starts, the follower slowly approaches the KER
  pose, then tracks it at the loop rate.
- **Handback:** when correction ends, the follower approaches the policy's next
  action, then follows the policy again.
- **Before taking over,** match the KER to the follower's pose. Otherwise the
  approach takes a while.

## Safety behavior

Each arm's loop applies all of this on every tick:

- **Joint limits:** every target is clipped to the arm config's `joint_limits`,
  the same limits as the stock follower: the custom ones in `rollout.yaml`, or
  the side defaults when `side` is set.
- **Speed caps:** the commanded target never moves faster than
  `max_speed_deg_s` (180) or `max_gripper_speed_deg_s` (720). Stock
  `max_relative_target=3` at 30 fps is about 90 deg/s, so set
  `--robot.max_speed_deg_s=90` to match it. The 10 deg/step in `rollout.yaml`
  allows about 300 deg/s at 30 fps, so the default 180 is already the tighter
  cap on this rig.
- **Slow approach:** the follower moves at `approach_speed_deg_s` (20; gripper
  60) until every motor is within `approach_tolerance_deg` (2) of the target,
  then raises the cap to full speed over `approach_ramp_s` (0.5 s). This
  replaces the stock startup snap, so the KER no longer has to be matched to
  the followers' pose before starting. It applies:
  - after connecting;
  - whenever the target source changes (policy ↔ KER, KER stalled ↔ live);
  - after `send_action()` pauses for longer than `command_gap_s` (0.25 s);
  - when an action jumps faster than 1.5× the speed cap, which means a new
    target rather than the next point on a trajectory.
- **Stalls:** if the KER stops streaming, the teleop holds its last action and
  the follower holds still with zero velocity feed-forward. If the policy or
  main loop stops calling `send_action()`, the follower finishes the current ramp
  and holds.
- **Loop failures:** a CAN error in a loop thread is raised in the main thread on
  the next `get_observation()` or `send_action()`. LeRobot then disconnects,
  which disables torque by default, so the arms go limp. A motor that stops
  replying for more than 0.2 s logs a warning.
- **One owner per bus:** after `connect()`, only the arm's loop thread talks to
  its CAN bus. `get_observation()` returns the states decoded from the replies to
  the loop's own MIT commands, so it adds no CAN traffic and the states are at
  most one tick old.
- **Reply window:** each tick waits up to `reply_window_ms` (2.5) for the eight
  replies and stops as soon as they are all in. LeRobot's own batch call waits
  only 1 ms, which classic CAN at 1 Mbps does not meet: eight commands plus
  eight replies take about 2 ms of bus time. A reply that still arrives late is
  processed at the start of the next tick, and counts as one missed reply.
- **Gains:** `position_kp`/`position_kd` come from the arm configs as usual.
  `custom_kp`/`custom_kd` arguments to `send_action()` are ignored.

### Optional: the motors' CAN timeout

Damiao motors have their own watchdog. If no command arrives within the CAN
timeout (register 9, in 50 µs steps, 0 = off), the motor disables itself.
The loop sends every 4 ms, so 100 ms never trips in normal use. The trade-off is
what happens if the process dies or hangs:
- **Timeout off:** the motors hold the last command stiffly.
- **Timeout on:** they go limp after the timeout, and the arms drop under
  gravity.

Pick whichever your setup and lab rules prefer. To check or set it, run this with
nothing else connected to that CAN interface:

```bash
python scripts/damiao_can_timeout.py --port left_arm --no-fd                         # read motors 1-8
python scripts/damiao_can_timeout.py --port left_arm --no-fd --set-ms 100            # until power-off
python scripts/damiao_can_timeout.py --port left_arm --no-fd --set-ms 100 --save     # save to flash; disables the motors first
```

`--no-fd` matches `use_can_fd: False` in `rollout.yaml`. Repeat with
`--port right_arm` for the other arm.

The frame format follows Damiao's reference library. The script has been
checked against the simulator, not real motors, so read the value back after
writing it.

## Options (`--robot.*`, on top of all `bi_openarm_follower` options)

| Flag | Default | Meaning |
|---|---|---|
| `control_hz` | `250` | loop rate per arm (20–1000) |
| `target_source` | `auto` | `auto`: KER stream while it is being polled, otherwise `send_action()`; `commands`: only `send_action()`; `stream`: only the stream (holds when it is idle) |
| `command_interpolation` | `interpolate` | `interpolate` or `hold` for `send_action()` targets |
| `velocity_feedforward` | `true` | send target velocity in the MIT velocity field (joints) |
| `gripper_velocity_feedforward` | `false` | same, for the grippers |
| `max_speed_deg_s` / `max_gripper_speed_deg_s` | `180` / `720` | speed caps on the commanded target |
| `approach_speed_deg_s` / `approach_gripper_speed_deg_s` | `20` / `60` | speed during approach |
| `approach_tolerance_deg` | `2` | approach ends when every motor is this close |
| `approach_ramp_s` | `0.5` | time to raise the cap from approach speed to full speed |
| `command_gap_s` | `0.25` | `send_action()` gap that counts as a pause |
| `reply_window_ms` | `2.5` | how long a tick waits for motor replies (minimum 1, capped at 70% of the control period: 2.8 ms at 250 Hz) |

When the robot disconnects, each loop logs its stats at INFO: rate, ticks,
overruns, worst cycle time, missed replies and the current source. To read them
mid-session, call `robot.streaming_stats()` from your own script.

## First time on hardware

1. Run the tests in your LeRobot environment (below).
2. Bring up `left_arm`/`right_arm` as usual (`helpers/setup_can.sh` in
   lerobot-flashrt). Support the arms and clear the workspace.
3. Start `lerobot-teleoperate` with the streaming flags and conservative caps:
   `--robot.max_speed_deg_s=60 --robot.approach_speed_deg_s=10`. For this first
   run, hold the KER near the followers' pose anyway. The log should show `control loops running at 250 Hz`,
   then `caught up with target` per arm.
4. Move one joint at a time, then everything. Stop with Ctrl-C and check the
   stats logged at disconnect:
   - `rate_hz` should be close to 250.
   - `overruns` should be a small fraction of `ticks`.
   - `missed_replies` should be near 0. It counts motor replies, so compare it
     with 8 x `ticks`.

   If overruns climb, the PC is busy, for example with camera decoding or policy
   inference sharing Python's GIL. Try `--robot.control_hz=200`.

   If replies are still missed on classic CAN, raise `--robot.reply_window_ms`
   (up to 2.8 at 250 Hz), or run at `--robot.control_hz=200`, which allows up to
   3.5.
5. Raise the caps back to the defaults and record a test episode.
6. For policies, try `--robot.command_interpolation=hold` first, then the default
   `interpolate`, and compare.

### Hardware runs (2026-10-02)

All with `lerobot-teleoperate`, the KER, `rollout.yaml` (classic CAN, three
cameras) and the default caps.

| | Run 1: before `reply_window_ms`, 7.8 min | Run 2: `reply_window_ms=2.5`, 43 s |
|---|---|---|
| `rate_hz` (left / right) | 250.0 / 249.9 | 248.4 / 247.9 |
| `overruns` / `ticks` | 85 / 117,616 and 66 / 117,626 (0.07%, 0.06%) | 40 / 10,680 and 28 / 10,688 (0.37%, 0.26%) |
| `max_work_ms` | 7.35 / 7.71 | 14.05 / 7.31 |
| `missed_replies` | 17,233 / 17,595 (1.8%, 1.9% of replies) | 20 / 39 (0.02%, 0.05%) |

- In both runs the arms approached the KER pose slowly, then tracked it, and no
  motor went silent for more than 0.2 s.
- The reply window removed almost all missed replies on classic CAN.
- Run 2 was short and includes startup, so its overrun rate and the one 14 ms
  tick are not yet a fair comparison with run 1. Check them on a longer run.
- Later sessions ran the KER plugin's park moves (rest to ready, takeover, back
  to rest) and its double-squeeze gesture through these loops. No stats were
  kept from those.

## Tests

```bash
python tests/test_streaming_follower.py      # prints a summary line per test
pytest tests/                                # same tests
```

No hardware is needed. `tests/fake_damiao.py` simulates each arm's eight motors
on a python-can virtual bus, including MIT frames, enable/disable, state refresh,
parameter registers, PD physics and Damiao feedback frames. The tests check:
- loop rate and observation keys;
- that the main thread never touches the bus;
- startup approach, speed cap and jump handling;
- joint limits;
- interpolate vs hold smoothness;
- error propagation, and that disconnect stops the loops and disables torque;
- the reply window, with simulated motors that reply late;
- KER streaming at the loop rate;
- policy-vs-KER priority;
- stall hold;
- KER park: hold at rest, the double-squeeze gesture, the ready path, takeover on a rough pose match and on `go`, return to rest;
- the CAN timeout script.

The KER tests need `lerobot_teleoperator_openarm_ker` installed.

## Limitations

- On real hardware, only KER teleoperation (with and without park) has run so
  far. Recording, policy rollouts, interventions and the CAN timeout script have
  not.
- `reply_window_ms=2.5` has been confirmed on this rig's classic CAN bus in one
  short run. Other buses and longer runs may differ, so keep checking
  `missed_replies`.
- The simulator also drops some replies under CPU contention, which the loop
  tolerates by keeping the last state.
- The gripper range comes from the arm config's `joint_limits`, as with the stock
  follower. `rollout.yaml` sets it to -160 to 0 deg, and the KER plugin's
  `gripper_open_deg` now defaults to -160 to match. With LeRobot's `side`
  presets instead, the follower clips the gripper to -65, so a released KER
  trigger would only open it that far.
