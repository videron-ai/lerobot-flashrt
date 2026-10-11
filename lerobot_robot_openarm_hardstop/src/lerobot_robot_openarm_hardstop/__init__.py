# SPDX-License-Identifier: Apache-2.0
"""Hard-stop zero calibration for OpenArm arms in LeRobot.

Importing this package registers four device types with LeRobot. LeRobot's command-line tools import it
automatically once it is installed (its name starts with `lerobot_robot_`):

- robot `openarm_follower_hardstop`, teleoperator `openarm_leader_hardstop`
- robot `bi_openarm_follower_hardstop`, teleoperator `bi_openarm_leader_hardstop`
"""

from .bimanual import (
    BiOpenArmFollowerHardstop,
    BiOpenArmFollowerHardstopConfig,
    BiOpenArmLeaderHardstop,
    BiOpenArmLeaderHardstopConfig,
    hardstop_follower_arms,
)
from .calibration import CalibrationAbortedError
from .follower import OpenArmFollowerHardstop, OpenArmFollowerHardstopConfig
from .hardstop_arm import CalibrationMismatchError
from .leader import OpenArmLeaderHardstop, OpenArmLeaderHardstopConfig
from .offset_bus import IncompatibleLeRobotError, OffsetDamiaoMotorsBus

__version__ = "0.1.0"

__all__ = [
    "BiOpenArmFollowerHardstop",
    "BiOpenArmFollowerHardstopConfig",
    "BiOpenArmLeaderHardstop",
    "BiOpenArmLeaderHardstopConfig",
    "CalibrationAbortedError",
    "CalibrationMismatchError",
    "IncompatibleLeRobotError",
    "OffsetDamiaoMotorsBus",
    "OpenArmFollowerHardstop",
    "OpenArmFollowerHardstopConfig",
    "OpenArmLeaderHardstop",
    "OpenArmLeaderHardstopConfig",
    "hardstop_follower_arms",
]
