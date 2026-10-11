# SPDX-License-Identifier: Apache-2.0
"""Find a joint's zero from gravity instead of from a reference pose judged by eye.

A joint whose axis is horizontal has to push against the arm's weight with a torque proportional to
sin(angle from where the arm would hang), which is zero exactly where the arm hangs. Each such joint is swept
slowly across a window, forward and back, while the others are held. Friction opposes the motion, so it has
opposite signs in the two directions and averaging them leaves the gravity torque. A sine fitted to that gives
the angle of zero torque: where the joint would hang if it had no friction.

    joint_1, joint_2   swept a few degrees either side of hanging. Fit: a*sin(q - q0).
    joint_4            its stop is at zero, so with the upper arm vertical it can only be swept one way.
                       It is swept twice: from nearly straight to bent 90 degrees, and again with the upper
                       arm tilted back by a known angle, where the forearm passes through vertical inside
                       the sweep. What matters for the fit is the forearm's angle from vertical, x = q -
                       tilt, and the two sweeps together cover it from behind vertical to horizontal. That
                       is enough of the sine to also fit a constant torque (a sensor offset, a cable
                       pulling), which a narrow range cannot tell apart from a shifted zero.
                       Fit: a*sin(x - q0) + c.

What this measures is where the arm's weight hangs under the joint. That differs from the joint's kinematic
zero by however far the links' centres of mass sit off the joint axes, which is small and the same on both
arms of a mirrored pair. A constant torque error on the shoulder joints is not separated from the zero; it
shifts the result by (error / fitted amplitude) radians, about 0.4 degrees per 0.1 Nm on a 13 Nm shoulder.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace

import numpy as np

from .calibration import (
    DEFAULT_HOLD_KD,
    DEFAULT_HOLD_KP,
    CalibrationAbortedError,
    HardStopCalibrator,
    RelativeMove,
)

logger = logging.getLogger(__name__)

BIN_DEG = 0.5
MIN_BINS = 12
# A fit less certain than this is reported but not used.
MAX_STDERR_DEG = 0.3


@dataclass(frozen=True)
class GravitySweep:
    motor: str
    # Sweep window in degrees. Relative to the current estimate of the joint's zero when `centered`,
    # otherwise in the joint's calibrated frame as it is.
    lo_deg: float
    hi_deg: float
    centered: bool = True
    # Also fit a constant torque. Only well determined when the window spans a large part of the sine.
    fit_constant: bool = False
    speed_deg_s: float = 4.0
    kp: float = 150.0
    kd: float = 2.5
    # Abort above this torque, or if the joint falls this far behind its target (something is in the way).
    max_torque_nm: float = 8.0
    max_lag_deg: float = 8.0
    # Reject a fit whose gravity amplitude is below this: the joint is not carrying the arm as expected.
    min_amplitude_nm: float = 3.0
    # Samples this close to either end of the window are left out (the joint is still speeding up there).
    margin_deg: float = 1.5
    # Extra sweeps with another joint moved away from its zero first: (that joint's offset in degrees,
    # window lo, window hi). The offset tilts the link this joint hangs from, so the swept link's angle from
    # vertical is `position + tilt_gain * offset actually reached`.
    tilt_motor: str | None = None
    tilt_gain: float = 0.0
    tilted: tuple[tuple[float, float, float], ...] = ()


@dataclass(frozen=True)
class GravityFit:
    motor: str
    # Where gravity torque is zero, in the frame the sweep was measured in.
    zero_deg: float
    # One standard error of the zero, from the scatter of the fit. See `trusted`.
    stderr_deg: float
    amplitude_nm: float
    friction_nm: float
    constant_nm: float
    rms_nm: float
    bins: int
    # The zero from the tilted sweeps alone with no constant torque fitted: much less scatter, but any
    # constant torque error is in it. None when the joint has no tilted sweep.
    two_sided_zero_deg: float | None = None

    @property
    def trusted(self) -> bool:
        """Whether the fit pins the zero down well enough to act on."""
        return self.stderr_deg <= MAX_STDERR_DEG


def gravity_sweeps(side: str, joints: tuple[str, ...] | list[str] | None = None) -> list[GravitySweep]:
    """Sweeps for one OpenArm arm hanging straight down. Joint 2's window is mirrored between the arms."""
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    s = 1 if side == "right" else -1
    # Joint 2's inward stop is about 10 degrees from hanging, so its window leans outward.
    j2 = (-4.0, 16.0) if s == 1 else (-16.0, 4.0)
    sweeps = {
        "joint_1": GravitySweep("joint_1", -12.0, 12.0),
        "joint_2": GravitySweep("joint_2", *j2),
        # Tilted sweep: upper arm 30 degrees back (negative joint 1 on the right arm, positive on the left),
        # elbow between 10 and 60. The forearm then hangs vertical with the elbow bent 30.
        "joint_4": GravitySweep(
            "joint_4", 3.0, 90.0, centered=False, fit_constant=True, speed_deg_s=8.0, kp=80.0, kd=2.0,
            min_amplitude_nm=0.5, tilt_motor="joint_1", tilt_gain=float(s), tilted=((-30.0 * s, 10.0, 60.0),),
        ),
    }  # fmt: skip
    joints = list(sweeps) if joints is None else list(joints)
    unknown = [j for j in joints if j not in sweeps]
    if unknown:
        raise ValueError(f"No gravity sweep is defined for {unknown}; choose from {list(sweeps)}.")
    return [sweeps[j] for j in sweeps if j in joints]


# One sweep's samples: (forward, backward) lists of (position deg, torque Nm), and how far the swept link's
# hanging point was moved by tilting the joint above it (0 for a plain sweep).
Segment = tuple[list[tuple[float, float]], list[tuple[float, float]], float]


def fit_gravity_zero(
    motor: str,
    forward: list[tuple[float, float]],
    backward: list[tuple[float, float]],
    *,
    fit_constant: bool = False,
) -> GravityFit:
    """Fit the zero-gravity angle to (position deg, torque Nm) samples from a sweep in each direction."""
    return fit_gravity_segments(motor, [(forward, backward, 0.0)], fit_constant=fit_constant)


def fit_gravity_segments(motor: str, segments: list[Segment], *, fit_constant: bool = False) -> GravityFit:
    """Fit one zero to several sweeps of a joint, each with its own tilt.

    A segment's tilt is added to its positions to give the swept link's angle from vertical when the joint
    is at zero, so all segments share one sine.
    """

    def binned(samples: list[tuple[float, float]]) -> dict[int, float]:
        sums: dict[int, list[float]] = {}
        for q, tau in samples:
            sums.setdefault(math.floor(q / BIN_DEG), []).append(tau)
        return {b: float(np.mean(v)) for b, v in sums.items()}

    angles: list[float] = []
    gravity: list[float] = []
    friction: list[float] = []
    for forward, backward, tilt in segments:
        fwd, bwd = binned(forward), binned(backward)
        for b in sorted(set(fwd) & set(bwd)):
            angles.append((b + 0.5) * BIN_DEG + tilt)
            # Friction opposes the motion, so it cancels in the mean of the two directions.
            gravity.append((fwd[b] + bwd[b]) / 2)
            friction.append(abs(fwd[b] - bwd[b]) / 2)
    if len(angles) < MIN_BINS:
        raise CalibrationAbortedError(
            f"{motor}: the two sweep directions overlap over only {len(angles) * BIN_DEG:.1f}°; not enough "
            "to fit."
        )
    q = np.radians(angles)
    tau = np.array(gravity)

    columns = [np.sin(q), np.cos(q)] + ([np.ones_like(q)] if fit_constant else [])
    x = np.column_stack(columns)
    coef, *_ = np.linalg.lstsq(x, tau, rcond=None)
    residual = tau - x @ coef
    dof = max(1, len(q) - x.shape[1])
    cov = float(residual @ residual) / dof * np.linalg.inv(x.T @ x)

    # a*sin(q - q0) = (a cos q0) sin q - (a sin q0) cos q. A flipped torque sign convention flips both
    # coefficients and leaves q0 where it is.
    alpha, beta = float(coef[0]), float(coef[1])
    sign = 1.0 if alpha >= 0 else -1.0
    zero = math.atan2(-sign * beta, sign * alpha)
    amplitude = math.hypot(alpha, beta)
    variance = (alpha**2 * cov[1, 1] + beta**2 * cov[0, 0] - 2 * alpha * beta * cov[0, 1]) / amplitude**4
    return GravityFit(
        motor=motor,
        zero_deg=math.degrees(zero),
        stderr_deg=math.degrees(math.sqrt(max(variance, 0.0))),
        amplitude_nm=amplitude,
        friction_nm=float(np.mean(friction)),
        constant_nm=float(coef[2]) if fit_constant else 0.0,
        rms_nm=float(np.sqrt(np.mean(residual**2))),
        bins=len(angles),
    )


class GravityZeroFinder(HardStopCalibrator):
    """Sweeps joints on a calibrated bus and returns where gravity puts their zero.

    Positions are in the bus's calibrated frame; the calibration is not changed here.
    """

    # A zero further than this from the current one is a bad fit or a stale calibration, not a correction.
    MAX_ZERO_SHIFT_DEG = 10.0
    GOTO_SPEED_DEG_S = 10.0

    def find(self, sweeps: list[GravitySweep], passes: int = 2) -> dict[str, list[GravityFit]]:
        """Each joint's fit after every pass. Later passes hold the other joints at the zero found so far,
        which matters because one joint being off moves the arm's weight for the others. A joint with
        tilted sweeps is fitted to the sweeps of all passes so far together.

        Leaves the arm held with the swept joints at their final zero.
        """
        motors = list(self.bus.motors)
        self.bus.enable_torque(motors)
        self._hold_current(motors)
        zero = {sweep.motor: 0.0 for sweep in sweeps}
        fits: dict[str, list[GravityFit]] = {sweep.motor: [] for sweep in sweeps}
        # Per joint: (forward, backward, where the tilt joint actually was) for every sweep so far.
        raw: dict[str, list[tuple[list, list, float | None]]] = {sweep.motor: [] for sweep in sweeps}
        try:
            for n in range(passes):
                for sweep in sweeps:
                    for other in sweeps:
                        if other.motor != sweep.motor:
                            self._goto(other.motor, self._rest_target(other, zero[other.motor]))
                    if not sweep.tilted:
                        raw[sweep.motor] = []  # a plain sweep is refitted from scratch around its new centre
                    center = zero[sweep.motor] if sweep.centered else 0.0
                    tilt_zero = zero.get(sweep.tilt_motor, 0.0)
                    samples = raw[sweep.motor]
                    samples.append((*self._sweep(sweep, center + sweep.lo_deg, center + sweep.hi_deg), None))
                    for offset, lo, hi in sweep.tilted:
                        samples.append(self._tilted_sweep(sweep, tilt_zero, offset, lo, hi))

                    fit = self._fit(sweep, samples, tilt_zero)
                    self._check(sweep, fit)
                    logger.info(
                        f"pass {n + 1} {sweep.motor}: zero at {fit.zero_deg:+.2f}° (±{fit.stderr_deg:.2f}°), "
                        f"gravity {fit.amplitude_nm:.2f} Nm, friction {fit.friction_nm:.2f} Nm, "
                        f"constant {fit.constant_nm:+.3f} Nm, fit rms {fit.rms_nm:.3f} Nm"
                    )
                    if fit.trusted:
                        zero[sweep.motor] = fit.zero_deg
                    else:
                        logger.warning(f"{sweep.motor}: that fit is too uncertain to use; its zero is kept")
                    fits[sweep.motor].append(fit)
            for sweep in sweeps:
                self._goto(sweep.motor, self._rest_target(sweep, zero[sweep.motor]))
            self._wait(self.params.pause_s)
        except BaseException:
            try:
                self._hold_current(motors)
            except Exception:
                logger.exception("Could not command the arm to hold its position after an error.")
            raise
        return fits

    @staticmethod
    def _fit(
        sweep: GravitySweep, raw: list[tuple[list, list, float | None]], tilt_zero_deg: float
    ) -> GravityFit:
        def segment(forward, backward, tilt_position) -> Segment:
            # Measured from the tilt joint's zero as it is known now, which may be better than when the
            # sweep was made. A plain sweep was made with the tilt joint held at its zero of that moment.
            tilt = 0.0 if tilt_position is None else sweep.tilt_gain * (tilt_position - tilt_zero_deg)
            return forward, backward, tilt

        segments = [segment(*item) for item in raw]
        fit = fit_gravity_segments(sweep.motor, segments, fit_constant=sweep.fit_constant)
        tilted = [seg for seg, item in zip(segments, raw, strict=True) if item[2] is not None]
        if tilted:
            fit = replace(fit, two_sided_zero_deg=fit_gravity_segments(sweep.motor, tilted).zero_deg)
        return fit

    def _tilted_sweep(
        self, sweep: GravitySweep, tilt_zero_deg: float, offset_deg: float, lo: float, hi: float
    ) -> tuple[list, list, float]:
        """Sweep with the tilt joint moved `offset_deg` from its zero. Returns where that joint really was."""
        # Bend the swept joint to the window's start before tilting, and straighten the tilt before leaving
        # the window, so the arm only ever tilts with the joint inside the window.
        self._goto(sweep.motor, lo)
        self._goto(sweep.tilt_motor, tilt_zero_deg + offset_deg)
        self._wait(0.5)
        before = self._read(sweep.tilt_motor)["position"]
        forward, backward = self._sweep(sweep, lo, hi)
        after = self._read(sweep.tilt_motor)["position"]
        self._goto(sweep.tilt_motor, tilt_zero_deg)
        return forward, backward, (before + after) / 2

    @staticmethod
    def _rest_target(sweep: GravitySweep, zero_deg: float) -> float:
        # A joint swept from its stop (not centred) has the stop at the bottom of its range: never hold it
        # at a zero that lies beyond the stop.
        return zero_deg if sweep.centered else max(zero_deg, 0.0)

    def _check(self, sweep: GravitySweep, fit: GravityFit) -> None:
        problems = []
        if fit.amplitude_nm < sweep.min_amplitude_nm:
            problems.append(
                f"gravity amplitude {fit.amplitude_nm:.2f} Nm is below the {sweep.min_amplitude_nm:.1f} Nm "
                "expected (is the arm hanging freely?)"
            )
        if abs(fit.zero_deg) > self.MAX_ZERO_SHIFT_DEG:
            problems.append(f"zero at {fit.zero_deg:+.1f}° is more than {self.MAX_ZERO_SHIFT_DEG:.0f}° away")
        if problems:
            raise CalibrationAbortedError(f"{sweep.motor}: " + "; ".join(problems) + ". Nothing was changed.")

    def _goto(self, motor: str, target_deg: float) -> None:
        delta = target_deg - self._read(motor)["position"]
        kp = self.params.hold_kp.get(motor, DEFAULT_HOLD_KP[motor])
        kd = self.params.hold_kd.get(motor, DEFAULT_HOLD_KD[motor])
        self._move(RelativeMove(motor, delta, max(0.4, abs(delta) / self.GOTO_SPEED_DEG_S), kp, kd))
        # _move ends on the measured start plus delta; make the held target exact.
        self._command(motor, kp, kd, target_deg)

    def _sweep(self, sweep: GravitySweep, lo: float, hi: float) -> tuple[list, list]:
        """Sweep lo -> hi -> lo. Returns the (position, torque) samples of each direction."""
        p = self.params
        samples: dict[int, list[tuple[float, float]]] = {1: [], -1: []}

        self._goto(sweep.motor, lo)
        for direction, start, end in ((1, lo, hi), (-1, hi, lo)):
            self._command(sweep.motor, sweep.kp, sweep.kd, start)
            self._wait(0.5)
            target = start
            last = self._clock()
            while direction * (end - target) > 0:
                now = self._clock()
                target += direction * sweep.speed_deg_s * (now - last)
                last = now
                self._command(sweep.motor, sweep.kp, sweep.kd, target)
                self._sleep(p.control_period_s)
                state = self._read(sweep.motor)
                if abs(state["torque"]) > sweep.max_torque_nm:
                    raise CalibrationAbortedError(
                        f"{sweep.motor} pushed {abs(state['torque']):.1f} Nm at {state['position']:.1f}° "
                        f"(limit {sweep.max_torque_nm:.1f} Nm). Is something in its way?"
                    )
                if abs(target - state["position"]) > sweep.max_lag_deg:
                    raise CalibrationAbortedError(
                        f"{sweep.motor} is {abs(target - state['position']):.1f}° behind its target at "
                        f"{state['position']:.1f}°. Is something in its way?"
                    )
                if lo + sweep.margin_deg <= state["position"] <= hi - sweep.margin_deg:
                    samples[direction].append((state["position"], state["torque"]))
        return samples[1], samples[-1]
