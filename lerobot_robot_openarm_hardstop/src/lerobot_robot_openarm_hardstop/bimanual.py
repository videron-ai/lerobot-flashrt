# SPDX-License-Identifier: Apache-2.0
"""`bi_openarm_follower_hardstop` and `bi_openarm_leader_hardstop`: the bimanual versions.

They take exactly the same options as LeRobot's `bi_openarm_follower` / `bi_openarm_leader`. The stock
classes build their two arms internally; these let them do that, then rebuild each arm as the hard-stop
type from the arm's resolved config, so any per-arm handling LeRobot adds (camera naming, ids, ...) carries
over unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from lerobot.robots.bi_openarm_follower import BiOpenArmFollower, BiOpenArmFollowerConfig
from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.bi_openarm_leader import BiOpenArmLeader, BiOpenArmLeaderConfig
from lerobot.teleoperators.config import TeleoperatorConfig

from .follower import OpenArmFollowerHardstop, OpenArmFollowerHardstopConfig
from .leader import OpenArmLeaderHardstop, OpenArmLeaderHardstopConfig


def _config_fields(config) -> dict:
    return {f.name: getattr(config, f.name) for f in fields(config) if f.init}


@RobotConfig.register_subclass("bi_openarm_follower_hardstop")
@dataclass(kw_only=True)
class BiOpenArmFollowerHardstopConfig(BiOpenArmFollowerConfig):
    """Same options as `bi_openarm_follower`. Each arm needs `side` set (left_arm_config.side=left, ...).

    `joint_limits` given for an arm are kept; the side's presets apply only when they are left at default.
    """

    # After each arm's hard-stop calibration, find the zero of joints 1, 2 and 4 from gravity torque.
    gravity_zero: bool = True


def hardstop_follower_arms(
    config: BiOpenArmFollowerConfig, robot: BiOpenArmFollower, **arm_options
) -> tuple[OpenArmFollowerHardstop, OpenArmFollowerHardstop]:
    """Hard-stop (left, right) arms to replace the stock arms `robot` built from `config`.

    `arm_options` are extra `OpenArmFollowerHardstopConfig` fields, applied to both arms. Other bimanual
    followers built on `BiOpenArmFollower` can use this to get hard-stop calibrated arms.
    """
    arms = {}
    for side, arm in (("left", robot.left_arm), ("right", robot.right_arm)):
        if arm.config.side != side:
            raise ValueError(
                f"Set {side}_arm_config.side={side} (got {arm.config.side!r}). Hard-stop calibration needs it, "
                "and it also selects the arm's joint limits."
            )
        arm_fields = _config_fields(arm.config)
        # The stock arm has already swapped the limits for the side's presets.
        arm_fields["joint_limits"] = getattr(config, f"{side}_arm_config").joint_limits
        arms[side] = OpenArmFollowerHardstop(OpenArmFollowerHardstopConfig(**arm_fields, **arm_options))
    return arms["left"], arms["right"]


class BiOpenArmFollowerHardstop(BiOpenArmFollower):
    config_class = BiOpenArmFollowerHardstopConfig
    name = "bi_openarm_follower_hardstop"

    def __init__(self, config: BiOpenArmFollowerHardstopConfig):
        super().__init__(config)
        self.left_arm, self.right_arm = hardstop_follower_arms(config, self, gravity_zero=config.gravity_zero)
        self.cameras = {**self.left_arm.cameras, **self.right_arm.cameras}


@TeleoperatorConfig.register_subclass("bi_openarm_leader_hardstop")
@dataclass
class BiOpenArmLeaderHardstopConfig(BiOpenArmLeaderConfig):
    """Same options as `bi_openarm_leader`. Each arm's side follows from its position."""


class BiOpenArmLeaderHardstop(BiOpenArmLeader):
    config_class = BiOpenArmLeaderHardstopConfig
    name = "bi_openarm_leader_hardstop"

    def __init__(self, config: BiOpenArmLeaderHardstopConfig):
        super().__init__(config)
        arms = {}
        for side, arm in (("left", self.left_arm), ("right", self.right_arm)):
            arm_fields = _config_fields(arm.config)
            arm_fields["side"] = side
            arms[side] = OpenArmLeaderHardstop(OpenArmLeaderHardstopConfig(**arm_fields))
        self.left_arm, self.right_arm = arms["left"], arms["right"]
