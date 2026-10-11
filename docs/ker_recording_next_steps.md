# KER recording: next steps

Possible improvements to the OpenArm KER teleoperation and recording setup
(`lerobot_teleoperator_openarm_ker`, `lerobot_robot_openarm_streaming`), written
down on 2026-10-02 after the first guided recording sessions on the rig. Listed
in the order they are worth doing.

## Worth doing

### 1. Scripted ready move for policy rollouts (required)

Guided recording (`lerobot-ker-record`) starts every episode at the ready pose,
and the lift from the calibration pose is never recorded. A policy trained on
that data has to be brought to the ready pose the same way before it starts. The
rollout script does not do that yet.

- Reuse the park path in `ker_mapping.json` (`park.ready_path`, with `blend`).
- Also return to the rest pose when the rollout ends.
- Older rollout data includes the lift as part of the policy's behaviour, so
  old and new data have different episode starts.

### 2. A zero that can be trusted

The follower zero is set by hand ("arms hanging, press ENTER") and lives in the
motors, not in the calibration files, which hold placeholder values only. It was
re-set several times on 2026-10-02, so that day's datasets may not share a zero.
At the ready pose the two grippers end up at slightly different heights, which
is most likely a zero difference between the arms (about 0.8 cm per degree at
the shoulder, 0.6 cm per degree at the elbow). Tracking error is not the cause:
both arms sit about 2 deg low at the shoulder and 1 deg at the elbow.

- Add a script that reads the Damiao registers `M_OFF` (54), `P_M` (80) and
  `XOUT` (81) for all eight motors of an arm. Run it, re-zero, run it again, to
  confirm `M_OFF` is the stored zero. This is not yet verified.
- If it is: store the values with each session and warn at startup when they
  have changed. Writing them back to restore a zero may not be possible.
- Make the hanging pose repeatable with a jig or reference marks.
- Until then, check the zero with the dry-run script: with the arms hanging and
  settled, every joint in the "follower" column should read close to 0.

### 3. Dataset check after every session

Each recording has been checked by hand so far. Turn that into a command that
prints a pass/fail report when recording stops:

- every video has exactly one frame per data row;
- no frame-to-frame jump in the action (the guided session peaked at 5.5 deg;
  parking mid-episode under plain `lerobot-record` left jumps of up to 42 deg);
- the follower is on the KER at frame 0 (within about 3 deg);
- each episode ends near the ready pose;
- median gap between commanded and actual angles (about 2 deg so far).

### 4. Consistent gripper state at episode start

In the first guided session two episodes started with the grippers closed and
one with them open, depending on whether the triggers were squeezed at takeover.
The ready pose holds the grippers closed, while a released trigger means open.
Making the ready pose hold them open would line the two up.

### 5. Encoding choice for guided sessions

Streaming video encoding (`--dataset.streaming_encoding=true
--dataset.encoder_threads=1`) cost 0.5% to 1.5% control-loop overruns on the
rig, with worst ticks of 60-75 ms. Encoding after each episode showed 0.1%.
With the guided workflow the save happens while the arms are parked and the
scene is being reset, so the wait (roughly half the episode length) may cost
nothing in practice. Record one session with
`--dataset.streaming_encoding=false` and compare.

Not options, as measured:
- `--dataset.video_encoding_batch_size` crashes in this LeRobot checkout
  (`_batch_save_episode_video` reads episode metadata that is not loaded yet).
- A different codec does not help: libsvtav1, h264 and hevc all took about 11 s
  per 20 s of 720p footage, and GPU encoding still gave 5% overruns.

### 6. Bake the plugins into the container image

Every new container needs both plugins and `openarm_ker` installed by hand,
because `scripts/run_rollout_container.sh` starts containers with `--rm`. Add
them to the image. `lerobot_flashrt:1.1` already has working torchcodec; it was
built by applying the Dockerfile's tokenizer, lerobot and torchcodec steps on
top of 1.0 and is not itself described by a Dockerfile in the repo.

## Hold until there is evidence they are needed

### Gravity compensation

The arms sit about 2 deg low at the shoulder and 1 deg at the elbow under load,
roughly 2 cm at the gripper. Feeding gravity torque forward would close that
gap, but it changes how commanded and actual angles relate in the data.

### Control loops in their own process

Moving the two 250 Hz loops into child processes would stop the video encoder,
policy inference and camera decoding from delaying them. It is a rewrite of
about half the streaming follower plus its test simulator, and all rig testing
would have to be repeated. Worth it only if rollouts show overruns from policy
inference, or if streaming encoding has to be used and its overruns turn out to
matter.

## Smaller open items

- **Takeover ergonomics.** Matching the KER to the ready pose is hard. Live
  guidance and a wider zone (`--teleop.park_engage_tolerance_deg=35
  --teleop.park_engage_joints=[1,4]`) exist; if that is not enough, move the
  ready pose closer to where the KER is naturally held, or add mapping offsets.
- **Not yet run on the rig:** the higher rest-to-ready path with corner
  blending, the live alignment guidance (check that its directions read the
  right way), the discard gesture, `q` and Ctrl-C under `lerobot-ker-record`,
  and guided recording with the stock follower.
- **Unpushed:** branch `feat/openarm-ker-streaming-plugins`.
