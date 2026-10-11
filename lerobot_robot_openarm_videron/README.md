# lerobot_robot_openarm_videron

`openarm_videron` is a bimanual OpenArm follower for [LeRobot](https://github.com/huggingface/lerobot)
that combines two plugins from this repo:

| From | What it brings |
| --- | --- |
| `lerobot_robot_openarm_streaming` | A 250 Hz MIT control loop per arm, with speed caps, a slow approach phase and velocity feed-forward. A streaming teleoperator (the OpenArm KER with `--teleop.stream_to_follower=true`) feeds the loops directly. |
| `lerobot_robot_openarm_hardstop` | Zero calibration against each joint's mechanical stop, saved as per-joint software offsets. The motors' own zero is never rewritten and nothing is re-zeroed on connect. |

The robot is the streaming follower with its two arms replaced by the hard-stop type. The control
loops read and command the arms through the hard-stop motor bus, so observations, `send_action()`
targets, KER targets, joint limits and park poses are all in the calibrated joint frame.

It takes every `bi_openarm_streaming_follower` option, with these differences:

- **`side` is optional.** `left_arm_config.side` defaults to `left` and `right_arm_config.side` to
  `right`. Setting the opposite side is an error.
- **Custom joint limits are kept.** Stock LeRobot replaces `joint_limits` with the side's presets
  whenever `side` is set. Here the presets apply only when `joint_limits` is left at its default.
- **`check_positions_on_connect`** (default `true`): refuse to connect when a joint reads clearly
  outside its range, which means the saved calibration no longer matches the motors.
- **Datasets record `openarm_videron` as their robot type**, not `bi_openarm_follower`. The joint
  frame differs from stock-calibrated data by each motor's zero offset (up to 5.6 degrees on the rig),
  so the two should not be merged as if they were the same robot.
- **MIT mode is set on connect.** Each arm reads its motors' control mode and switches any that are
  not in MIT (in RAM only). `helpers/set_mit_mode.py` is not needed after a power cycle. A switched
  motor is disabled for a moment, so connect with the arms hanging.

**Status:** on the rig, both arms have been calibrated with this robot type (stops repeat within
0.02 degrees), their zeros corrected with the gravity sweep, and the park moves to the rest and ready
poses checked (arms hang straight, grippers level). KER teleoperation, recording, a record GUI session
and the MIT switch on connect have only run against simulated arms (see Tests).

## Install

Into the LeRobot environment, from the repo root. The order matters: this package depends on the
other two follower plugins, which are not on PyPI.

```bash
pip install -e ./lerobot_robot_openarm_streaming
pip install -e ./lerobot_robot_openarm_hardstop
pip install -e ./lerobot_robot_openarm_videron
pip install -e ./lerobot_teleoperator_openarm_ker    # for the KER
```

The record GUI (`record_gui/`) installs all four by itself when it starts.

## Calibrate (once per arm)

Start with both arms hanging straight down, elbows straight, grippers closed. **The arms move on
their own**, one after the other; keep the emergency stop in reach. Run from the repo root:

```bash
lerobot-calibrate --robot.type=openarm_videron --robot.id=my_bimanual_follower \
  --robot.calibration_dir=calibration/openarm_follower_hardstop \
  --robot.left_arm_config.port=left_arm --robot.left_arm_config.use_can_fd=false \
  --robot.right_arm_config.port=right_arm --robot.right_arm_config.use_can_fd=false
```

The flags are spelled out because `lerobot-calibrate` rejects the rig's config file (it has no
`display_data` field). The files land in `calibration/openarm_follower_hardstop/` as
`my_bimanual_follower_left.json` and `..._right.json`, which is where
`helpers/rollout_videron.yaml` looks for them. They are the same files
`bi_openarm_follower_hardstop` uses.

To check how repeatable an arm's stops are first, without saving anything:

```bash
openarm-stop-repeatability --port left_arm --side left --runs 5 --no-can-fd
```

### Correct the zeros (once per arm)

The stops repeat very well, but their nominal angles can be a few degrees off on a given arm. On the
rig that left the arms hanging slightly off vertical and one gripper higher than the other at the
ready pose. The gravity sweep measures where each arm really hangs and corrects joints 1, 2 and 4.
**The arm moves on its own.** Run it for each arm, from the repo root:

```bash
openarm-zero-trim --port left_arm --side left --id my_bimanual_follower_left \
  --calibration-dir calibration/openarm_follower_hardstop --no-can-fd --gravity
openarm-zero-trim --port right_arm --side right --id my_bimanual_follower_right \
  --calibration-dir calibration/openarm_follower_hardstop --no-can-fd --gravity
```

Each correction is written into the arm's calibration file, which is the only file loaded in normal
use. It is also recorded in `<id>.stop_trim.json` beside it, which a later hard-stop calibration reads
so the correction carries over. The hard-stop plugin's README explains the method and the other
options (`--joints`, `--shift`).

### When to recalibrate

Nothing drifts with time or power cycles, so there is no schedule.

- **Hard-stop calibration again:** a motor's stored zero is rewritten (LeRobot's stock calibration,
  Enactic's zero tool, Damiao's debugging tool), a motor is replaced, a joint is reassembled, screws
  are found loose, the arm takes a hard knock, or the connect-time check refuses to start.
- **Gravity sweep again:** the arm's weight or mounting changes (gripper, wrist camera, cable
  routing, link extensions, the tower moved or re-levelled).
- **To check without changing anything:** look at the rest and ready poses each session, and run
  `openarm-stop-repeatability` after moving the rig. A stop that has moved by more than about
  0.2 degrees means something mechanical shifted.

Do not run the stock `bi_openarm_follower` calibration on a rig calibrated this way: it rewrites the
motors' zero, which the saved offsets are relative to.

## Use

`helpers/rollout_videron.yaml` describes the rig for this robot type (ports, classic CAN, custom
joint limits, cameras, calibration folder). Teleoperate with the KER:

```bash
lerobot-teleoperate \
  --config_path=helpers/rollout_videron.yaml \
  --teleop.type=openarm_ker --teleop.id=ker \
  --teleop.mapping_path=lerobot_teleoperator_openarm_ker/ker_mapping.json \
  --teleop.filter=one_euro --teleop.stream_to_follower=true
```

Guided recording, policy rollouts and every other command in the streaming and KER READMEs work
the same way: use `helpers/rollout_videron.yaml` as the config file and drop
`--robot.type=bi_openarm_streaming_follower`.

After the first hard-stop calibration, repeat the KER dry run
(`lerobot_teleoperator_openarm_ker/scripts/compare_ker_follower.py --hardstop
--calibration-dir calibration/openarm_follower_hardstop ...`). The followers' zero has moved from
the hand-held pose of the stock calibration to the true hanging pose, so the mapping file's
`joint_offsets` may need adjusting.

A session started without a terminal (the record GUI) cannot run the calibration. If the
calibration files are missing, the GUI's rig check says so before the session starts. The GUI
defaults to this robot type and `helpers/rollout_videron.yaml`.

## Tests

The tests run LeRobot's real Damiao bus, the streaming follower's real control loops and the real
hard-stop calibration against the hard-stop plugin's simulated arms (hidden motor zeros, mechanical
stops, gravity, friction). They cover the first connect (calibrate, then stream), reuse of a saved
calibration after a power cycle, a teleop stream and the KER plugin landing on true joint angles,
`lerobot-calibrate` on a connected robot, and refusing a stale calibration. Run them inside the
`lerobot_flashrt` container; the host Python has no LeRobot.

```bash
pip install -e ".[test]"
pytest
```
