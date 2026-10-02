"""Configuration for the streaming bimanual OpenArm follower."""

from dataclasses import dataclass

from lerobot.robots.bi_openarm_follower import BiOpenArmFollowerConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("bi_openarm_streaming_follower")
@dataclass(kw_only=True)
class BiOpenArmStreamingFollowerConfig(BiOpenArmFollowerConfig):
    """bi_openarm_follower plus a background MIT control loop per arm.

    All bi_openarm_follower options (ports, sides, gains, cameras, calibration ids)
    work unchanged. `max_relative_target` is not used; the speed caps below
    replace it.
    """

    # Rate of each arm's control loop. 250 Hz matches Enactic's own driver.
    control_hz: float = 250.0

    # Where the loop gets its targets:
    #   "auto"     - a live teleop stream (KER with --teleop.stream_to_follower=true)
    #                while someone is teleoperating, otherwise send_action() commands
    #   "commands" - only send_action() (policies, any other teleop)
    #   "stream"   - only the live stream; holds position when it is idle
    target_source: str = "auto"

    # How 30 Hz send_action() commands are turned into a 250 Hz trajectory:
    #   "interpolate" - ramp from the previous command to the latest over one
    #                   command interval (smoothest; the default)
    #   "hold"        - jump to each command (like stock LeRobot, but still at 250 Hz)
    command_interpolation: str = "interpolate"

    # Send the target velocity in the MIT velocity field instead of 0. Removes the
    # lag the damping term otherwise adds (about kd/kp seconds).
    velocity_feedforward: bool = True
    gripper_velocity_feedforward: bool = False

    # Speed caps on the commanded target (deg/s). These replace max_relative_target.
    max_speed_deg_s: float = 180.0
    max_gripper_speed_deg_s: float = 720.0

    # After connecting, after a stall, or when the target source changes, the
    # follower moves at this speed until it is within approach_tolerance_deg of
    # the target on every motor.
    approach_speed_deg_s: float = 20.0
    approach_gripper_speed_deg_s: float = 60.0
    approach_tolerance_deg: float = 2.0
    # Once caught up, the speed cap rises from the approach speed to the max over this time.
    approach_ramp_s: float = 0.5

    # send_action() commands further apart than this are not ramped between, and
    # the next one is approached slowly.
    command_gap_s: float = 0.25

    # How long each tick waits for the motors' replies to its MIT commands. LeRobot's
    # own batch call waits 1 ms, which is enough on CAN FD but not on classic CAN at
    # 1 Mbps, where 8 commands plus 8 replies take about 2 ms of bus time. The loop
    # stops waiting as soon as every motor has replied, so a longer window costs
    # nothing when replies are on time. Capped at 70% of the control period.
    reply_window_ms: float = 2.5
