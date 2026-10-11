# SPDX-License-Identifier: Apache-2.0
"""`openarm_follower_hardstop`: LeRobot's OpenArm follower with hard-stop calibration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields

from lerobot.robots.config import RobotConfig
from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig

from .hardstop_arm import HardStopArmMixin

logger = logging.getLogger(__name__)


@RobotConfig.register_subclass("openarm_follower_hardstop")
@dataclass
class OpenArmFollowerHardstopConfig(OpenArmFollowerConfig):
    """Same options as `openarm_follower`; `side` is required.

    Unlike the stock follower, `joint_limits` given in the config are kept when `side` is set; the side's
    preset limits are used only when `joint_limits` is left at its default.

    Calibration files are kept apart from the stock follower's, under
    `.../calibration/robots/openarm_follower_hardstop/`.
    """

    # Refuse to connect if a joint reads clearly outside its mechanical range, which means the saved
    # calibration no longer matches the motors.
    check_positions_on_connect: bool = True

    # After every hard-stop calibration, sweep joints 1, 2 and 4 to find their zero from gravity torque and
    # correct the stops' nominal angles for this arm. The arm moves; the result is confirmed before saving.
    gravity_zero: bool = True


def custom_joint_limits(config: OpenArmFollowerConfig) -> dict[str, tuple[float, float]] | None:
    """The config's joint limits if they were set by the user, else None."""
    limits = {motor: tuple(bounds) for motor, bounds in config.joint_limits.items()}
    default = next(f for f in fields(OpenArmFollowerConfig) if f.name == "joint_limits").default_factory()
    return None if limits == default else limits


class OpenArmFollowerHardstop(HardStopArmMixin, OpenArmFollower):
    config_class = OpenArmFollowerHardstopConfig
    name = "openarm_follower_hardstop"

    def __init__(self, config: OpenArmFollowerHardstopConfig):
        # The stock follower replaces the limits with the side's presets.
        custom_limits = custom_joint_limits(config)
        super().__init__(config)
        if custom_limits is not None:
            config.joint_limits = custom_limits
            logger.info(f"Keeping the configured joint limits: {custom_limits}.")
        self._setup_hard_stop(
            config.side, config.check_positions_on_connect, custom_limits, gravity_zero=config.gravity_zero
        )

    def connect(self, calibrate: bool = True) -> None:
        """Like the stock follower's connect(), except the motors are never re-zeroed.

        LeRobot up to 0.6.0 re-zeros the follower's motors on every connect, wherever the arm happens to rest.
        """
        if self.is_connected:
            raise RuntimeError(f"{self} is already connected.")
        logger.info(f"Connecting arm on {self.config.port}...")
        self._connect_bus_and_calibrate(calibrate)
        for cam in self.cameras.values():
            cam.connect()
        self.configure()
        self.bus.enable_torque()
        logger.info(f"{self} connected.")
