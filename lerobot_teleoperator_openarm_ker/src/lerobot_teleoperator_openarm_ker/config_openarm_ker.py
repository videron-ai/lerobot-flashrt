"""Configuration for the OpenArm KER teleoperator."""

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("openarm_ker")
@dataclass
class OpenArmKERConfig(TeleoperatorConfig):
    # "usb" (default firmware build, vendor-class bulk) or "serial" (CDC build).
    transport: str = "usb"
    # Only used when transport == "serial".
    port: str = "/dev/m5_ker_485"
    baud: int = 2_000_000

    # Optional JSON file with per-joint signs/offsets, wrist remap and gripper
    # settings. See ker_mapping.example.json. Anything missing falls back to the
    # defaults: sign +1, offset 0, and the software-only wrist remap for an
    # OpenArm 1.0 follower ("wrist_remap": {"mode": "none"} turns that off).
    mapping_path: Path | None = None

    # Gripper trigger travel on the KER, in degrees from released to fully squeezed.
    gripper_travel_deg: float = 60.0
    # Follower gripper range on this rig: 0 deg = closed, -160 deg = fully open
    # (matches the gripper joint_limits in lerobot-flashrt's helpers/rollout.yaml).
    # LeRobot's side="left"/"right" presets clip the gripper to -65 instead.
    gripper_open_deg: float = -160.0
    gripper_closed_deg: float = 0.0

    # Smoothing applied to the raw KER angles (all 16 channels) before mapping.
    # "none" passes them through; "one_euro" is a speed-adaptive low-pass filter
    # (Casiez et al. 2012): heavy smoothing when the hand is nearly still, so
    # tremor and jitter are suppressed, and little lag when it moves quickly.
    # min_cutoff_hz: smoothing at rest; lower = smoother but more lag on slow moves.
    # beta (Hz per deg/s): how fast the cutoff rises with speed; higher = less lag
    # on fast moves but more jitter gets through while moving.
    filter: str = "none"
    filter_min_cutoff_hz: float = 1.0
    filter_beta: float = 0.05
    filter_d_cutoff_hz: float = 1.0

    # Feed targets straight into lerobot_robot_openarm_streaming's control loop at the
    # KER's own rate (500 Hz) instead of once per recorded frame. Requires that
    # plugin and --robot.type=bi_openarm_streaming_follower. The filter then runs on
    # every KER frame; get_action() still returns (and records) the same targets.
    stream_to_follower: bool = False

    # Park: start by holding the followers' rest pose (their calibration pose, arms
    # dangling) and ignore the KER. Typing "ready" + ENTER in the terminal moves them
    # along the mapping file's "park" ready_path to the ready pose above the work
    # surface, at no more than park_speed_deg_s per joint. The KER takes over once
    # the joints in park_engage_joints (shoulder and elbow by default; the wrist is
    # hard to judge by eye) have been within park_engage_tolerance_deg of the ready
    # pose for park_engage_hold_s, or park_go_delay_s after typing "go". The
    # followers then close the remaining gap at the park speed. "rest" + ENTER
    # plays the path backwards to the rest pose, from the ready pose or from KER
    # control (via the ready pose). Bare ENTER does the usual next move: ready, or
    # rest when already at ready.
    park: bool = False
    park_speed_deg_s: float = 40.0
    park_engage_joints: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    park_engage_tolerance_deg: float = 20.0
    park_engage_hold_s: float = 0.5
    park_go_delay_s: float = 3.0
    # Hands-free park toggle: squeeze both gripper triggers together twice within
    # park_gesture_window_s. Same as bare ENTER: rest -> ready, ready -> rest,
    # KER control -> ready. One long squeeze, or a double squeeze of one trigger
    # while the other is held, does nothing.
    park_gesture: bool = False
    park_gesture_window_s: float = 1.2

    # If the KER frame counter stops advancing for this long, the stream is
    # considered stalled (M5 jump detection tripped, STOP pressed, link lost...).
    stale_timeout_s: float = 0.25
    # What to do on a stall: "hold" keeps sending the last good action and logs a
    # warning (the follower stays still; re-record the episode), "raise" aborts
    # the session. Note that aborting disconnects the follower, which by default
    # disables torque, so the arms will go limp.
    on_stall: str = "hold"
