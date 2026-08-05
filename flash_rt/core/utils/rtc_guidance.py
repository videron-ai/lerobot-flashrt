"""Real-Time Chunking prefix-guidance schedules for flow-matching VLAs.

Host-side math only — the per-timestep prefix weights and the per-denoise-step
guidance ceiling.  The device-side correction lives in the model pipeline.

Ported from ``lerobot/policies/rtc/modeling_rtc.py`` (``RTCProcessor``), which
follows Physical Intelligence's Kinetix implementation:
https://www.physicalintelligence.company/download/real_time_chunking.pdf
"""

from __future__ import annotations

import math

import numpy as np

# Schedule names accepted by :func:`get_prefix_weights`, matching
# ``lerobot.configs.RTCAttentionSchedule``.
PREFIX_ATTENTION_SCHEDULES = ("zeros", "ones", "linear", "exp")


def get_prefix_weights(start: int, end: int, total: int,
                       schedule: str = "exp") -> np.ndarray:
    """Prefix attention weights over a chunk of ``total`` timesteps.

    ``start`` is the inference delay (timesteps already consumed by the robot
    while inference ran — these are pinned to weight 1.0), ``end`` is the
    execution horizon (weight reaches 0.0 here and stays there).

    Mirrors ``RTCProcessor.get_prefix_weights`` exactly, including the
    ``start = min(start, end)`` clamp.

    Returns:
        float32 array of shape ``(total,)``.
    """
    sched = str(schedule).lower()
    if sched not in PREFIX_ATTENTION_SCHEDULES:
        raise ValueError(
            f"unknown prefix_attention_schedule {schedule!r}; "
            f"expected one of {PREFIX_ATTENTION_SCHEDULES}")

    start = min(int(start), int(end))
    end = int(end)
    total = int(total)

    if sched == "zeros":
        weights = np.zeros(total, dtype=np.float32)
        weights[:start] = 1.0
        return weights
    if sched == "ones":
        weights = np.ones(total, dtype=np.float32)
        weights[end:] = 0.0
        return weights

    # linear / exp: ramp down between `start` and `end`
    skip_at_end = max(total - end, 0)
    n = total - skip_at_end - start
    if end <= start or n <= 0:
        ramp = np.zeros(0, dtype=np.float32)
    else:
        ramp = np.linspace(1.0, 0.0, n + 2, dtype=np.float32)[1:-1]
        if sched == "exp":
            ramp = ramp * np.expm1(ramp) / (math.e - 1.0)

    weights = ramp
    if skip_at_end > 0:
        weights = np.concatenate([weights, np.zeros(skip_at_end, dtype=np.float32)])
    ones_len = min(start, total)
    if ones_len > 0:
        weights = np.concatenate([np.ones(ones_len, dtype=np.float32), weights])
    return weights.astype(np.float32, copy=False)


def guidance_ceiling(num_steps: int) -> np.ndarray:
    """Per-denoise-step guidance factor before clamping by ``max_guidance_weight``.

    For denoise step ``k`` the flow-matching time is ``time_k = 1 - k/num_steps``
    and ``tau_k = 1 - time_k = k/num_steps``.  ``RTCProcessor.denoise_step``
    computes::

        inv_r2 = ((1 - tau)**2 + tau**2) / (1 - tau)**2
        c      = (1 - tau) / tau                      # +inf at tau = 0
        g      = min(nan_to_num(c * inv_r2, posinf=max_w), max_w)

    Returning ``c * inv_r2`` unclamped (``+inf`` at ``k = 0``) lets the device
    side recover ``g`` with a single ``minimum`` against the runtime
    ``max_guidance_weight`` — ``min(+inf, max_w) == max_w`` reproduces the
    reference ``nan_to_num`` behaviour exactly.

    Returns:
        float32 array of shape ``(num_steps,)``.
    """
    num_steps = int(num_steps)
    tau = np.arange(num_steps, dtype=np.float64) / num_steps
    one_minus = 1.0 - tau
    sq = one_minus ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_r2 = (sq + tau ** 2) / sq
        c = one_minus / tau                      # tau == 0 -> +inf
    ceiling = c * inv_r2
    ceiling[0] = np.inf                          # tau == 0 exactly
    return ceiling.astype(np.float32)


__all__ = ["get_prefix_weights", "guidance_ceiling", "PREFIX_ATTENTION_SCHEDULES"]
