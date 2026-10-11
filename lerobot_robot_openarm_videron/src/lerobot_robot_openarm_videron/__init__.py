"""LeRobot robot plugin: bimanual OpenArm follower with a 250 Hz MIT control loop and hard-stop calibration."""

from .config_openarm_videron import OpenArmVideronConfig
from .openarm_videron import OpenArmVideron

__all__ = ["OpenArmVideron", "OpenArmVideronConfig"]
