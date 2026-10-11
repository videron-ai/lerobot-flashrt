"""Bimanual OpenArm follower: the streaming follower's control loops on hard-stop calibrated arms.

lerobot_robot_openarm_streaming runs a 250 Hz MIT loop per arm and can be fed directly
by a streaming teleoperator (the OpenArm KER). lerobot_robot_openarm_hardstop finds each
joint's zero from its mechanical stop and applies it as a software offset on the motor
bus. Here the streaming follower's two arms are the hard-stop type, so everything the
loops read and command (observations, send_action() targets, KER targets, joint limits,
park poses) is in the calibrated joint frame.
"""

from __future__ import annotations

import logging

from lerobot_robot_openarm_hardstop import hardstop_follower_arms
from lerobot_robot_openarm_streaming import BiOpenArmStreamingFollower

from .config_openarm_videron import OpenArmVideronConfig

logger = logging.getLogger(__name__)

SIDES = ("left", "right")


class OpenArmVideron(BiOpenArmStreamingFollower):
    """bi_openarm_streaming_follower with hard-stop calibrated arms."""

    config_class = OpenArmVideronConfig
    # Recorded as the dataset's robot_type. Not "bi_openarm_follower": joint angles are in
    # the hard-stop calibrated frame, which differs from the stock follower's by each
    # motor's zero offset.
    name = "openarm_videron"

    def __init__(self, config: OpenArmVideronConfig):
        for side in SIDES:
            arm_config = getattr(config, f"{side}_arm_config")
            if arm_config.side is None:
                arm_config.side = side
        super().__init__(config)
        self.left_arm, self.right_arm = hardstop_follower_arms(
            config, self, check_positions_on_connect=config.check_positions_on_connect
        )
        self.cameras = {**self.left_arm.cameras, **self.right_arm.cameras}

    def calibrate(self) -> None:
        """Hard-stop calibration of both arms.

        lerobot-calibrate calls this on a connected robot, whose control loops own the CAN
        buses. They are stopped for the calibration (the motors keep holding their last
        command meanwhile) and restarted from the newly calibrated positions.
        """
        running = bool(self._loops)
        self._stop_loops()
        super().calibrate()
        if running:
            self._start_loops()
