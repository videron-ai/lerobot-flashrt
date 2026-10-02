"""LeRobot teleoperator plugin for the OpenArm KER leader arm."""

# Importing the config registers "openarm_ker" as a --teleop.type choice.
from .config_openarm_ker import OpenArmKERConfig
from .openarm_ker import OpenArmKER

__all__ = ["OpenArmKER", "OpenArmKERConfig"]
