"""Configuration for the Videron bimanual OpenArm follower."""

from dataclasses import dataclass

from lerobot.robots.config import RobotConfig
from lerobot_robot_openarm_streaming import BiOpenArmStreamingFollowerConfig


@RobotConfig.register_subclass("openarm_videron")
@dataclass(kw_only=True)
class OpenArmVideronConfig(BiOpenArmStreamingFollowerConfig):
    """bi_openarm_streaming_follower with hard-stop calibrated arms.

    All bi_openarm_streaming_follower options work unchanged. Differences:

    * Each arm's `side` may be left unset; it then follows from the arm's position
      (left_arm_config -> left).
    * `joint_limits` given for an arm are kept even though `side` is set. The side's
      preset limits apply only when `joint_limits` is left at its default.
    * Calibration is the hard-stop calibration of lerobot_robot_openarm_hardstop
      (`lerobot-calibrate --robot.type=openarm_videron ...`), followed by its gravity
      zero sweep. The motors are never re-zeroed.
    """

    # Refuse to connect if a joint reads clearly outside its range, which means the
    # saved calibration no longer matches the motors.
    check_positions_on_connect: bool = True

    # After each arm's hard-stop calibration, sweep joints 1, 2 and 4 to find their zero
    # from gravity torque. The arm moves; the result is confirmed before it is saved.
    gravity_zero: bool = True
