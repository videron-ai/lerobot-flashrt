# SPDX-License-Identifier: Apache-2.0
"""`openarm_leader_hardstop`: LeRobot's OpenArm leader with hard-stop calibration."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.openarm_leader import OpenArmLeader, OpenArmLeaderConfig

from .hardstop_arm import HardStopArmMixin

logger = logging.getLogger(__name__)


@TeleoperatorConfig.register_subclass("openarm_leader_hardstop")
@dataclass
class OpenArmLeaderHardstopConfig(OpenArmLeaderConfig):
    """Same options as `openarm_leader`, plus `side` (required).

    Calibration files are kept apart from the stock leader's, under
    `.../calibration/teleoperators/openarm_leader_hardstop/`.
    """

    # Side of the arm: "left" or "right". The calibration sequence is mirrored between the two.
    side: str | None = None

    # Refuse to connect if a joint reads clearly outside its mechanical range, which means the saved
    # calibration no longer matches the motors.
    check_positions_on_connect: bool = True


class OpenArmLeaderHardstop(HardStopArmMixin, OpenArmLeader):
    config_class = OpenArmLeaderHardstopConfig
    name = "openarm_leader_hardstop"

    def __init__(self, config: OpenArmLeaderHardstopConfig):
        super().__init__(config)
        self._setup_hard_stop(config.side, config.check_positions_on_connect)

    def connect(self, calibrate: bool = True) -> None:
        """Like the stock leader's connect(), except the motors are never re-zeroed.

        The stock leader re-zeros its motors on every connect, wherever the arm happens to rest.
        """
        if self.is_connected:
            raise RuntimeError(f"{self} is already connected.")
        logger.info(f"Connecting arm on {self.config.port}...")
        self._connect_bus_and_calibrate(calibrate)
        self.configure()
        logger.info(f"{self} connected.")
