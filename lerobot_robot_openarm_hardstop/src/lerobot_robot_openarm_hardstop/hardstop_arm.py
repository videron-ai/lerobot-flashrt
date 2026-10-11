# SPDX-License-Identifier: Apache-2.0
"""Behaviour shared by the hard-stop follower and leader."""

from __future__ import annotations

import logging

from .calibration import position_problems, run_openarm_hard_stop_calibration
from .offset_bus import OffsetDamiaoMotorsBus
from .stop_trim import load_stop_trim, stop_trim_path

logger = logging.getLogger(__name__)


class CalibrationMismatchError(RuntimeError):
    """The saved calibration does not match the arm (e.g. its motors were re-zeroed since)."""


class HardStopArmMixin:
    """Mixed into OpenArmFollower / OpenArmLeader subclasses.

    - Swaps the stock Damiao bus for one that applies the calibration's offsets.
    - Replaces `calibrate()` with hard-stop calibration.
    - Provides the connect-time check that the saved calibration still matches the arm.
    """

    def _setup_hard_stop(
        self,
        side: str | None,
        check_positions_on_connect: bool,
        joint_limits: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        if side not in ("left", "right"):
            raise ValueError(
                f"{type(self).__name__} needs `side` set to 'left' or 'right' (got {side!r}); "
                "the calibration sequence is mirrored between the two arms."
            )
        self._hard_stop_side = side
        # User-set joint limits, if any: positions inside them are not a sign of a stale calibration.
        self._hard_stop_joint_limits = joint_limits
        self._check_positions_on_connect = check_positions_on_connect
        self._calibration_mismatch: list[str] = []
        self.bus = OffsetDamiaoMotorsBus.from_bus(self.bus)

    def _connect_bus_and_calibrate(self, calibrate: bool) -> None:
        """First half of connect(): open the bus, then calibrate or check the saved calibration."""
        self.bus.connect()
        if not self.is_calibrated:
            if calibrate:
                self.calibrate()
            return

        self._calibration_mismatch = self._check_calibration()
        if not self._calibration_mismatch:
            return
        details = "\n  - ".join(self._calibration_mismatch)
        if not calibrate:
            # lerobot-calibrate connects with calibrate=False and calls calibrate() next, which will insist on
            # recalibrating.
            logger.warning(f"The saved calibration does not match {self}:\n  - {details}")
            return
        self.bus.disconnect(disable_torque=True)
        raise CalibrationMismatchError(
            f"The saved calibration ({self.calibration_fpath}) does not match {self}:\n  - {details}\n"
            "Were the motors re-zeroed, or swapped, since it was made? Recalibrate with lerobot-calibrate."
        )

    def _check_calibration(self) -> list[str]:
        if not self._check_positions_on_connect:
            return []
        positions = self.bus.sync_read("Present_Position")
        return position_problems(positions, self._hard_stop_side, self._hard_stop_joint_limits)

    def calibrate(self) -> None:
        """Hard-stop calibration: drive each joint into its stop and save per-joint software offsets."""
        if self.calibration and not self._calibration_mismatch:
            user_input = input(
                f"Press ENTER to use the calibration file associated with the id {self.id}, or type 'c' and "
                "press ENTER to run hard-stop calibration: "
            )
            if user_input.strip().lower() != "c":
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"Running hard-stop calibration for {self}")
        stop_trim = load_stop_trim(self.calibration_fpath)
        if stop_trim:
            logger.info(f"Using this arm's stop trim from {stop_trim_path(self.calibration_fpath)}: {stop_trim}")
        self.calibration = run_openarm_hard_stop_calibration(
            self.bus, self._hard_stop_side, stop_trim_deg=stop_trim
        )
        self._calibration_mismatch = []
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")
