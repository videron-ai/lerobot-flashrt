# lerobot_teleoperator_openarm_ker

A LeRobot teleoperator plugin that makes the OpenArm KER a `--teleop.type` for
`lerobot-teleoperate` and `lerobot-record`, paired with LeRobot's built-in
`bi_openarm_follower`.

The KER is one USB device for both arms, so this is a single bimanual
teleoperator. Its action keys (`left_joint_1.pos` ... `right_gripper.pos`, in
degrees) match `bi_openarm_follower` exactly.

With the companion plugin `lerobot_robot_openarm_streaming` and
`--teleop.stream_to_follower=true`, the followers track the KER at 250 Hz
instead of once per recorded frame (see Streaming below).

Checked against LeRobot 0.6.2 and openarm_ker 0.3.0.

**Tested on the rig** (2026-10-02, KER firmware 2.0.0, `lerobot-teleoperate`,
`rollout.yaml`, `bi_openarm_streaming_follower`):
- the rig mapping in `ker_mapping.json`: software wrist remap, joint 7 inverted
  on both sides, gripper range 0 to -160 deg;
- `--teleop.filter=one_euro` with `--teleop.stream_to_follower=true`;
- park: the rest and ready poses, the `ready_path` in `ker_mapping.json`, and
  the return to rest;
- the double-squeeze park gesture.

**Only tested against a simulated KER stream and simulated motors:**
- `lerobot-record`;
- the filter without streaming, and anything with the stock `bi_openarm_follower`
  since the filter, taper and park were added.

Do the dry run below before letting it drive different arms or a changed mapping.

The commands below are written for the rig described by
`helpers/rollout.yaml` in the lerobot-flashrt repo (`bi_openarm_follower`, id
`my_bimanual_follower`, classic CAN on `left_arm` / `right_arm`, custom joint
limits, gripper 0 to -160 deg, `left_wrist` / `right_wrist` / `base` cameras).
Run them from this directory with:

```bash
export FLASHRT=~/Desktop/Experiments/lerobot-flashrt   # /lerobot-flashrt inside the container
```

## 1. Install

```bash
# KER host prerequisites (once)
sudo apt install libusb-1.0-0-dev
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="303a", MODE="0666"' | sudo tee /etc/udev/rules.d/99-m5stack.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

# In your LeRobot environment (Python >= 3.12, lerobot installed with [damiao])
pip install -e path/to/lerobot_teleoperator_openarm_ker
openarm-ker-cli ping      # should print firmware/hardware and the field schema
```

LeRobot auto-imports any installed package whose name starts with
`lerobot_teleoperator_`, so no LeRobot changes are needed.

## 2. Zero the KER and calibrate the followers

- **KER:** mount it in the calibration jig and tap *Zero Reset (All)* on the M5
  touchscreen. The offsets persist in the M5's flash.
- **Followers:** calibrate them as a pair so the calibration ids match what
  `lerobot-record` loads (`<id>_left` / `<id>_right`). Zero is arm hanging
  down, gripper closed.

  ```bash
  lerobot-calibrate --robot.type=bi_openarm_follower --robot.id=my_bimanual_follower \
    --robot.left_arm_config.port=left_arm --robot.left_arm_config.use_can_fd=false \
    --robot.right_arm_config.port=right_arm --robot.right_arm_config.use_can_fd=false
  ```

  The flags are spelled out here because `lerobot-calibrate` rejects
  `rollout.yaml` (it has no `display_data` field). If a calibration file already
  exists, type `c` at the prompt; pressing ENTER reuses the file without
  re-zeroing. Since LeRobot 0.6.1 the follower no longer re-zeroes itself on
  every connect, so the zero set here is the one every later session uses.

## 3. Dry run: check signs and offsets (nothing moves)

The KER firmware's zero offsets and joint directions were tuned for Enactic's own
driver, and LeRobot's follower zeroes itself separately, so check every joint.

```bash
python scripts/compare_ker_follower.py \
    --left-port left_arm --right-port right_arm \
    --left-id my_bimanual_follower_left --right-id my_bimanual_follower_right \
    --no-can-fd --mapping ker_mapping.json
```

No position command is sent, but the CAN handshake does send each motor the
Damiao enable frame, so support the arms or let them hang.

1. Put the KER and both followers in the same pose (arms hanging, grippers
   closed). The DELTA column is the offset for that joint.
2. Move one KER joint and the same follower joint by hand in the same physical
   direction. If the numbers move in opposite directions, that joint needs sign -1.

Put what you find in `ker_mapping.json`, this rig's mapping file (any joint not
listed defaults to sign +1, offset 0), and rerun until every DELTA is near zero
and every joint tracks. `ker_mapping.example.json` is a blank template; leave
`--mapping` off to see the plugin's bare defaults.

Dry-run results recorded in `ker_mapping.json` so far:

| Joint | Sign | Offset |
|---|---|---|
| `right_joint_7` | -1 | 0 |
| `left_joint_7` | -1 | 0 |
| all others | +1 | 0 |

The wrist remap described in the next section is on by default, in the dry run
too. Single-axis moves pass straight through it, so move the wrist joints one at
a time here.

Use the same ids your follower calibration files were saved under.

## OpenArm 1.0 followers

The KER replicates OpenArm 2.0. Per Enactic's URDFs, 1.0 and 2.0 share link
lengths and joint axis lines except the wrist: in 2.0, J6 turns about the axis
parallel to the elbow and J7 about the perpendicular one; in 1.0 it is the other
way round.

### Default: software remap on an unmodified KER

With no mapping file, or one that does not say otherwise, the plugin drives a
1.0 follower from a stock KER. Each KER wrist channel is paired with the
follower axis that points the same way (follower J6 reads CH7/CH15, J7 reads
CH6/CH14), and `wrist_remap` then re-solves J5-J7 so the gripper orientation
matches exactly wherever the KER can reach it. There are no M5 firmware changes,
because the KER hardware matches its stock config.
`ker_mapping.v1_software_only.example.json` spells these defaults out.

Limits and checks:
- **Reach.** The KER's wrist joint next to the forearm stops at +-45 deg. When a
  follower pose needs it slightly further, the follower ends up a few degrees
  short. In a sampled policy-rollout dataset this affected about 12% of
  right-arm and 2% of left-arm frames, by at most about 4 deg.
- **Follower J6 cap.** The remap saturates the follower's J6 at `j6_limit_deg`
  (default 40, matching LeRobot's `side` presets). Set it to just inside the
  follower's limit (44 for followers that reach +-45, as in `ker_mapping.json`;
  must be below 45). Near the KER's J6 stop the cap tapers off
  (`taper_slope`, default 1.0), so the follower's J6 reaches about 41 deg there
  instead of 44. Without the taper, small KER motions in that corner rotated
  the follower hand up to ~5x as much; with it, at most ~1.7x, and exactly 1x
  wherever the follower can follow.
- **Forearm roll differs.** The follower's forearm roll (J5) differs slightly from
  the operator's: a few degrees in typical use, up to about 25 deg at large
  wrist angles.
- **Handedness must be verified.** Single-axis moves pass straight through the
  remap, so the dry run calibrates signs and offsets as usual. The
  `handedness` values (URDF defaults: right +1, left -1) only affect combined
  wrist poses. Check them by holding both KER wrist joints at about 30 deg: if
  the follower gripper is visibly twisted relative to the KER, flip that
  side's value. A wrong value causes a 15-30 deg error at moderate angles.
  Repeat this check after changing a wrist joint's sign: `ker_mapping.json`
  inverts joint 7 on both sides and still carries the URDF handedness defaults.

### Not recommended: swapping the KER's J6/J7 modules

Swapping the J6 and J7 encoder modules restores the 1.0 rotation order, but each
module's hard stop travels with it. The +-45 deg module then lands on the axis
the follower's J7 uses, which has +-90 deg on a 1.0 arm, and the 1:1 mapping
leaves no other joint to compensate. In the same sampled dataset, follower J7
went past 45 deg in about 38% of right-arm and 26% of left-arm frames, by up to
about 16 deg. The swap only makes sense if the stop on that module can be
changed to at least +-60 deg.

If you do swap, turn the software remap off with
`"wrist_remap": {"mode": "none"}` in the mapping file, which also restores the
stock channel layout. Then either reflash the four moved modules' IDs so the IDs
follow their new positions (IDs 6/7 and 14/15, via UPDI), or keep the IDs and
start from `ker_mapping.v1_swap_keep_ids.example.json`. Either way, update the four
affected rows (ch6/7, ch14/15) of `ENCODER_CONFIG` in the M5 firmware's
`include/Common.h` and reflash the M5, otherwise the M5 wraps angles by 360 deg
near the wrist limits. After changing an `invert` flag, re-zero in the jig.

`"wrist_remap": {"mode": "none"}` on its own is also what an OpenArm 2.0
follower needs.

## Smoothing (`--teleop.filter=one_euro`)

The KER has almost no friction, so hand tremor and small unintended motions
reach the follower unchanged. `--teleop.filter=one_euro` smooths the raw KER
angles with a speed-adaptive low-pass filter: heavy smoothing when the hand is
nearly still, little lag when it moves. In simulation at 30 Hz with the
defaults (`filter_min_cutoff_hz=1.0`, `filter_beta=0.05`), it removed about
85-90% of tremor at rest and lagged 20-40 ms during deliberate moves, roughly
half the lag of a plain low-pass filter with less smoothing.

Tuning, in `lerobot-teleoperate`:
1. Set `filter_beta=0` and lower `filter_min_cutoff_hz` until the follower is
   steady while you hold the KER still (try 1.0, then 0.5).
2. Raise `filter_beta` until quick moves no longer feel laggy (try 0.05, 0.1).
   If jitter returns while moving, back it off.

The filter shapes the recorded actions too, so keep the same settings for a
whole dataset. Enactic's dora node has a different filter (`--hampel`, off by
default and not enabled in their KER dataflow). It only rejects isolated spikes
over 5 deg, so it does nothing for tremor, and at LeRobot's 30 Hz loop it can
freeze a joint after a brisk move from rest. Don't port it.

## Streaming to the follower (`--teleop.stream_to_follower=true`)

By default LeRobot reads the KER once per recorded frame and sends that to the
followers, so at 30 fps they move in 33 ms steps. With
`--teleop.stream_to_follower=true` and
`--robot.type=bi_openarm_streaming_follower` (from
`lerobot_robot_openarm_streaming`), the follower's 250 Hz control loops read the
KER directly:
- Every KER frame (500 Hz) is filtered and mapped once, wrist remap included.
- The loops send the result with velocity feed-forward.
- `get_action()` returns the same latest target, so the recorded action is what
  the follower was being sent at that moment.
- Recording stays at `--dataset.fps`.

Behavior worth knowing:
- **Polling:** the stream only drives the followers while `get_action()` is
  being called regularly (twice within 0.5 s). In a `lerobot-rollout` with the
  KER attached for interventions, the policy keeps control until the human
  takes over.
- **Filter rate:** the One Euro filter runs at the KER's rate instead of 30 Hz.
  The same settings feel similar; fine-tune them in `lerobot-teleoperate`.
- **Stalls:** behave as described below. The followers hold still with zero
  velocity.
- **Install:** without the robot plugin installed, `stream_to_follower=true`
  fails at connect with an ImportError.

See that plugin's README for the speed caps and slow approach, which replace
`max_relative_target`.

## Park: rest and ready poses (`--teleop.park=true`)

With `--teleop.park=true` the session starts with the followers held at their
rest pose (the calibration pose: arms dangling, grippers closed) and the KER
ignored. Typed commands, each followed by ENTER in the terminal running
LeRobot, move the arms without the KER:

| Command | From | What happens |
|---|---|---|
| `ready` | rest | rise, swing out over the work table, hold at the ready pose |
| `ready` | KER control | release the KER, go straight to the ready pose and hold |
| `go` | ready | hand control to the KER after a 3 s countdown |
| `rest` | ready | play the path backwards to the calibration pose and hold |
| `rest` | KER control | release the KER, go to the ready pose, then down to the calibration pose |
| bare ENTER | any | `ready`, or `rest` when already at ready |

Commands typed while a move is running are ignored.

**Hands-free toggle** (`--teleop.park_gesture=true`): squeeze both gripper
triggers together twice within `park_gesture_window_s` (1.2 s). It does the
same as bare ENTER: rest to ready, ready to rest, KER control to ready. So a
whole session needs no typing: double squeeze to rise to the ready pose, take
over by the rough pose match, double squeeze to hand the arms back to the ready
pose, double squeeze again to send them to rest.
- A press counts when both triggers pass 70% squeezed after both were below
  30%. One long squeeze does nothing, and neither does double-tapping one
  trigger while the other is held closed.
- In KER control the follower grippers follow the two squeezes before the arms
  leave for the ready pose, so don't do it while holding something.
- Closing both grippers twice within 1.2 s during a task triggers it too.
  Shorten `park_gesture_window_s` if that happens.
- The log prints `park: double squeeze` each time it fires.

Full command with the streaming follower (every line but the last ends in `\`;
without it the shell runs the next flag as a separate command and park stays
off, which shows as `'park': False` in the config printed at startup):

```bash
lerobot-teleoperate \
  --config_path=$FLASHRT/helpers/rollout.yaml \
  --robot.type=bi_openarm_streaming_follower \
  --teleop.type=openarm_ker --teleop.id=ker \
  --teleop.mapping_path=ker_mapping.json \
  --teleop.filter=one_euro \
  --teleop.stream_to_follower=true \
  --teleop.park=true \
  --teleop.park_gesture=true
```

When park is on, the log prints `park: holding the rest pose` once the control
loops are running.

### The ready pose

Upper arms 25 deg forward, elbows at 90 deg, forearms straight ahead, wrists
neutral, grippers closed:

| | J1 | J2 | J3 | J4 | J5 | J6 | J7 |
|---|---|---|---|---|---|---|---|
| left | -25 | 0 | 0 | 90 | 0 | 0 | 0 |
| right | 25 | 0 | 0 | 90 | 0 | 0 | 0 |

On this rig forward shoulder pitch is negative J1 on the left arm and positive
on the right. In the rollouts
(`videron/rollout_gen3_level2_multitask_20260913_195707`, 131 grasps over 10
episodes) both arms grasp on the table with the forearm about level, so a level
forearm puts the gripper at the table surface. The ready pose tilts the forearm
25 deg above level, which should hold the grippers very roughly 15-20 cm above
the table and 45-50 cm in front of the shoulders. Those figures assume about
35 cm from shoulder to elbow (with the 5 cm biceps extension) and 36 cm from
elbow to gripper; they are not measured.

### The path

`ready_path` in the `park` block of `ker_mapping.json`, in follower joint space.
The first three waypoints are the opening move of the rollouts (medians of 10
episodes); the fourth is the ready pose.

1. **Lift:** joints 1 and 4 together on a straight line in joint space. The
   upper arms go back about 70 deg while the elbows fold to about 120 deg, so
   the grippers rise close to the body.
2. **Turn the wrists** (joint 7).
3. **Swing forward** with the elbows still folded, to where the rollouts' arms
   are just before their first grasp.
4. **Open the elbows to 90 deg** while the upper arms continue forward. In the
   rough side view the grippers only rise during this segment.

Each waypoint lists only the joints it changes. All joints of a segment start
and stop together, eased in and out, with the fastest joint peaking at
`park_speed_deg_s` (40). The whole path then takes about 10 s; the policy did
the lift alone in about 1.2 s. `rest` plays the same waypoints in reverse.
`rest` defaults to all zeros and can be overridden with
`"park": {"rest": {...}}`.

### Taking over

At the ready pose the followers wait. Either:

- **Match roughly.** Hold the KER with the upper arms slightly forward, elbows
  at 90 deg and forearms straight ahead. Once joints 1-4 of both arms have been
  within `park_engage_tolerance_deg` (20) of the ready pose for
  `park_engage_hold_s` (0.5 s), the KER takes over. The wrists and the gripper
  triggers are not part of the match. While waiting, the log shows the three
  joints furthest off every 2 s.
- **Type `go`.** After `park_go_delay_s` (3 s) the KER takes over wherever it
  is. Use the countdown to get both hands on the KER and hold it over the table.

Either way the followers close the remaining gap at the park speed, then track
the KER.

### Things to know

- **Start at the calibration pose.** The plugin cannot see the followers. It
  commands the rest pose and assumes they are there. If they are not,
  `bi_openarm_streaming_follower` approaches it slowly; the stock follower
  moves there at `max_relative_target` per step. If LeRobot asks to calibrate at
  startup, the pose the arms are in when you press ENTER becomes the rest pose.
- **After changing the path or the rig.** The `ready_path` in `ker_mapping.json`
  has run on this rig with this table. The clearance figures above are still
  estimates, not measurements. After editing a waypoint, moving the table or
  changing the arms, run it once with `--teleop.park_speed_deg_s=15` and watch
  the table edge and the wrist cameras.
- **`go` with the KER far from the ready pose** moves each joint straight to the
  KER's angle at the park speed. Nothing checks that path against the table.
- **`ready` from KER control** moves in a straight joint-space line from
  wherever the arms are to the ready pose.
- **Shutting down.** Type `rest` and wait for `at the rest pose` before Ctrl-C.
  The arms are then already dangling when LeRobot cuts torque.
- **Needs a live KER stream and an interactive terminal** (`docker run -it`).
  Without a terminal, call `teleop.park_command("ready")` from your own script.
- **Recording.** Park moves are returned by `get_action()`, so `lerobot-record`
  stores them as actions like any other.

## 4. Record

The whole `--robot.*` side comes from `rollout.yaml`. `ker_mapping.json` is the
rig mapping from the dry run; without `--teleop.mapping_path` the joint 7
inversion is not applied.

Try teleoperation first:

```bash
lerobot-teleoperate \
  --config_path=$FLASHRT/helpers/rollout.yaml \
  --teleop.type=openarm_ker \
  --teleop.id=ker \
  --teleop.mapping_path=ker_mapping.json \
  --teleop.filter=one_euro
```

Then record:

```bash
lerobot-record \
  --config_path=$FLASHRT/helpers/rollout.yaml \
  --teleop.type=openarm_ker \
  --teleop.id=ker \
  --teleop.mapping_path=ker_mapping.json \
  --teleop.filter=one_euro \
  --dataset.repo_id=${HF_USER}/openarm_ker_demo \
  --dataset.single_task="Put the cup on the plate" \
  --dataset.num_episodes=20 \
  --dataset.fps=30
```

`rollout.yaml` sets `display_data: False`; add `--display_data=true` to override it.
The three cameras are declared at the top level of the robot config, so their
keys stay unprefixed in observations (`left_wrist`, `right_wrist`, `base`).

Add `--teleop.park=true` to either command to start from the rest pose and use
the `ready` / `rest` commands (see Park above).

For 250 Hz streaming, add these two flags to either command:

```bash
  --robot.type=bi_openarm_streaming_follower \
  --teleop.stream_to_follower=true
```

The rest of `rollout.yaml` (ports, classic CAN, joint limits, cameras) still
applies. Its `max_relative_target` is ignored by the streaming follower, with a
warning; the speed caps and slow approach replace it.

## Options (`--teleop.*`)

| Flag | Default | Meaning |
|---|---|---|
| `transport` | `usb` | `usb` or `serial` (match the firmware build) |
| `port`, `baud` | `/dev/m5_ker_485`, `2000000` | serial transport only |
| `mapping_path` | none | JSON with `channels`, `joint_signs`, `joint_offsets`, `wrist_remap`, `gripper_direction`; without it the software-only 1.0 wrist remap applies |
| `gripper_travel_deg` | `60` | trigger travel from released to fully squeezed |
| `gripper_open_deg` / `gripper_closed_deg` | `-160` / `0` | follower gripper range; matches the gripper `joint_limits` in `rollout.yaml` (LeRobot's `side` presets clip it to `-65`) |
| `filter` | `none` | `none` or `one_euro` (see Smoothing) |
| `filter_min_cutoff_hz` / `filter_beta` / `filter_d_cutoff_hz` | `1.0` / `0.05` / `1.0` | One Euro settings: smoothing at rest / lag on fast moves / speed estimate |
| `stream_to_follower` | `false` | feed `bi_openarm_streaming_follower`'s control loops at the KER's rate (see Streaming) |
| `park` | `false` | start at the rest pose and accept `ready` / `go` / `rest` commands (see Park) |
| `park_speed_deg_s` | `40` | peak joint speed of park moves and of the takeover (grippers 4x) |
| `park_engage_joints` | `[1, 2, 3, 4]` | joints of each arm the KER must match at the ready pose to take over |
| `park_engage_tolerance_deg` / `park_engage_hold_s` | `20` / `0.5` | how closely and how long those joints must match |
| `park_go_delay_s` | `3` | countdown after typing `go` before the KER takes over without a match |
| `park_gesture` | `false` | double squeeze of both triggers toggles park, like bare ENTER (needs `park=true`) |
| `park_gesture_window_s` | `1.2` | longest time between the two squeezes |
| `stale_timeout_s` | `0.25` | no new KER frame for this long = stalled |
| `on_stall` | `hold` | `hold` keeps the last action and warns; `raise` aborts the session |

## Safety notes

- **Startup snap.** On the first frame the follower is commanded to wherever the
  KER is. With the stock follower, hold the KER roughly in the followers' pose
  when you start. `max_relative_target` (degrees per control step) only softens
  the snap if it is small: the 10 deg/step in `rollout.yaml` allows 300 deg/s at
  30 fps and 600 deg/s at `lerobot-teleoperate`'s default 60 fps, while 2-3
  deg/step gives a 60-90 deg/s glide at 30 fps but caps teleop speed for the
  whole session. `bi_openarm_streaming_follower` removes the need to match
  poses: it approaches the KER pose at 20 deg/s, then ramps to full speed.
  `--teleop.park=true` avoids the snap with either follower: the arms stay
  parked until the KER is near the ready pose or you type `go`, and then
  close the gap at the park speed.
- **Stalls.** If the M5 stops streaming (its jump detection trips, someone taps
  STOP, the USB link drops), `hold` freezes the followers in place. Fix the cause,
  press START on the M5, and re-record the episode (left arrow key).
  `raise` ends the session instead; LeRobot then disconnects the followers,
  which by default disables torque, so the arms go limp.
- **Frozen channels.** On the current KER firmware, a joint module that stops
  replying mid-session keeps reporting its last angle without setting an error
  flag. Neither this plugin nor the M5 can detect that; watch for a joint that
  stops tracking.
- Keep `use_velocity_and_torque` off on the follower arms (the default).
  Otherwise the follower expects `.vel`/`.torque` actions the KER cannot provide.
