# lerobot_robot_openarm_hardstop

Repeatable zero calibration for OpenArm arms in [LeRobot](https://github.com/huggingface/lerobot), as a plugin.

LeRobot's stock OpenArm calibration has you hold the arm hanging straight down and writes that pose to the
motors as zero. The stock leader also re-zeros its motors every time it connects, and up to LeRobot 0.6.0 so
does the follower, wherever the arm happens to rest.

This plugin adds OpenArm follower and leader types that calibrate against the arm's mechanical stops
instead. Each joint is driven slowly into its stop, its position is averaged while pressed against it, and
the difference from the stop's known angle is saved as a per-joint offset in the calibration file. The
motors' stored zero is never changed, and nothing is re-zeroed on connect. The joint order and stop angles
follow Enactic's `openarm-can-zero-position-calibration` tool.

| Plugin type | Replaces |
| --- | --- |
| robot `openarm_follower_hardstop` | `openarm_follower` |
| teleop `openarm_leader_hardstop` | `openarm_leader` |
| robot `bi_openarm_follower_hardstop` | `bi_openarm_follower` |
| teleop `bi_openarm_leader_hardstop` | `bi_openarm_leader` |

They take the same options as the types they replace. The single-arm types also need `side`.

`lerobot_robot_openarm_videron` (`openarm_videron`) puts these arms under the 250 Hz streaming follower of
`lerobot_robot_openarm_streaming`.

Two steps differ from Enactic's sequence, both found on the rig: joint 2's stall threshold is 5 Nm instead
of 2 Nm (cable drag alone reached 2 Nm before the stop), and the right arm seeks joint 7's opposite stop
(the +90° direction turns the right wrist up into the wrist camera's cable).

## Install

Into the same Python environment as LeRobot 0.4.4 or newer:

```bash
pip install lerobot_robot_openarm_hardstop-0.1.0-py3-none-any.whl
# or, from this folder, to edit it:  pip install -e .
```

LeRobot's commands pick the plugin up automatically. If you applied the earlier
`openarm-hard-stop-calibration.patch` to LeRobot, revert it first; the plugin detects it and refuses to run.

## Check your arm's stops (optional, recommended)

Moves the arm through the calibration several times and reports how repeatable each joint's stop is.
Saves nothing.

```bash
openarm-stop-repeatability --port can0 --side right --runs 5
```

A spread of a few tenths of a degree or less is good.

## Calibrate (once per arm)

Start with the arm hanging straight down, elbow straight, gripper closed. **The arm moves on its own**;
keep the emergency stop in reach. Ctrl+C makes it hold its position; support it, then press ENTER to
release torque.

```bash
lerobot-calibrate --robot.type=openarm_follower_hardstop --robot.port=can0 --robot.side=right --robot.id=my_follower
lerobot-calibrate --teleop.type=openarm_leader_hardstop --teleop.port=can1 --teleop.side=right --teleop.id=my_leader
```

Bimanual:

```bash
lerobot-calibrate --robot.type=bi_openarm_follower_hardstop --robot.id=my_bi \
  --robot.left_arm_config.port=can0 --robot.left_arm_config.side=left \
  --robot.right_arm_config.port=can1 --robot.right_arm_config.side=right
lerobot-calibrate --teleop.type=bi_openarm_leader_hardstop --teleop.id=my_bi_leader \
  --teleop.left_arm_config.port=can2 --teleop.right_arm_config.port=can3
```

Each arm needs `side`. Unlike the stock types, joint limits you set yourself are kept when `side` is set;
the side's preset limits apply only when `joint_limits` is left at its default.

Calibration files go in their own folders (`.../calibration/robots/openarm_follower_hardstop/` and
`.../calibration/teleoperators/openarm_leader_hardstop/`), separate from the stock types' files.

## Correct the zero at a reference pose (optional)

The stops are very repeatable, but their angles are nominal design values. On a given arm a stop can sit a
few degrees from its nominal angle, which shifts that joint's zero by the same amount every time. If the
calibrated arm does not hang at zero, or the two arms of a pair sit differently in the same pose, correct it
once per arm:

```bash
openarm-zero-trim --port can0 --side right --id my_follower
# bimanual: once per arm, with --id <robot id>_left / <robot id>_right and the robot's --calibration-dir
```

The arm goes limp and nothing moves. Let it hang straight down, jiggle it so it settles, and confirm. The
joints given by `--joints` (default `joint_1 joint_2`, which hang vertical under gravity) are made to read
zero there. The correction is saved next to the calibration file as `<id>.stop_trim.json` and reused by
every later hard-stop calibration of that arm, so this is not repeated after recalibrating. For other
joints, hold them at their zero by eye and name them with `--joints`.

A hanging arm settles within a degree or two of vertical, because of friction in the motors. `--gravity`
measures the zero instead of relying on that:

```bash
openarm-zero-trim --port can0 --side right --id my_follower --gravity
```

**The arm moves on its own.** Joints 1 and 2 swing slowly a few degrees either side of hanging. The elbow
bends to 90° and back, then is swept again between 10° and 60° with the upper arm tilted 30° back, so the
forearm passes through vertical inside the sweep. Everything runs forward and backward, twice. The torque needed to hold the arm against its own
weight is zero exactly where it would hang without friction; averaging the two directions cancels the
friction, and a sine fitted to the rest gives that angle. The result is shown with its uncertainty and how
far the second pass moved it, and is saved to the same trim file once you confirm. A joint whose fit is too
uncertain is reported and left alone.

What it finds is where the arm's weight hangs under each joint, which is the same on both arms of a mirrored
pair. A constant error in a shoulder motor's torque reading is not separated from the zero (about 0.4° per
0.1 Nm). The elbow's two sweeps together cover enough of the sine to fit and remove one; the tool also
prints the zero from the tilted sweeps alone with no constant fitted, which scatters less but would include
such an error, as a cross-check.

To move a joint's zero by hand instead: `--shift joint_1=1.0` makes every commanded pose sit 1.0° further in that joint's positive
direction. It only edits the two files; the arm is not touched.

## Use

Use the plugin types and the same ids in every command, e.g.:

```bash
lerobot-teleoperate \
  --robot.type=openarm_follower_hardstop --robot.port=can0 --robot.side=right --robot.id=my_follower \
  --teleop.type=openarm_leader_hardstop --teleop.port=can1 --teleop.side=right --teleop.id=my_leader
```

Datasets recorded this way list `openarm_follower_hardstop` (or `bi_openarm_follower_hardstop`) as their
robot type.

## When to recalibrate

Only when something changes how a motor sits relative to its joint: the motor's stored zero is rewritten
(by LeRobot's stock calibration, Enactic's zero tool or Damiao's debugging tool), a motor is replaced, a
joint is reassembled, or the arm takes a hard knock.

## Safety checks

- **Offsets applied exactly once.** The plugin wraps a few internal methods of LeRobot's Damiao motor bus.
  Every time a bus is created, it self-tests those paths without touching hardware, and refuses to run if a
  LeRobot update changed them.
- **Calibration matches the arm.** On connect, if any joint reads clearly outside its mechanical range, the
  calibration no longer matches the motors and the plugin refuses to connect. `lerobot-calibrate` then goes
  straight to recalibrating.
- **Offsets survive power cycles.** Damiao motors report a single-turn absolute position after power-up,
  so a joint's raw reading has to stay within ±180° of the motor's zero. Calibration is rejected if any
  offset would break that in the joint's working range. If that happens, re-zero the motors once near the
  hanging pose with Enactic's tool or LeRobot's stock calibration, then run this calibration.
- **MIT mode.** LeRobot commands Damiao motors in MIT mode only. Motors flashed with another control mode
  come back in it after every power cycle and ignore those commands. On connect, every device and tool
  in this plugin reads each motor's mode and switches the ones that are not in MIT (in RAM only, never
  saved to the motor's flash). A switched motor is disabled for a moment, so start with the arm hanging.
- **Bounded motion.** Every joint gets a travel limit, a push-torque limit, a timeout and a debounced
  stall test.

## Tests

The tests run LeRobot's real motor bus and device classes against a simulated arm on a fake CAN bus, with
hidden motor zeros, gravity, friction, noise and compliant stops:

```bash
pip install -e ".[test]"
pytest
```
