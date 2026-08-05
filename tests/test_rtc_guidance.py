"""Unit tests for RTC prefix guidance.

The schedule tests are CPU-only and check FlashRT's host-side math against
lerobot's ``RTCProcessor`` reference where lerobot is importable.

The device test exercises the pipeline's ``_apply_rtc_guidance`` correction
against a numpy reference of the documented update; it needs a GPU but not a
checkpoint.
"""

import math

import numpy as np
import pytest

from flash_rt.core.utils.rtc_guidance import (
    PREFIX_ATTENTION_SCHEDULES,
    get_prefix_weights,
    guidance_ceiling,
)


# ── Host-side schedules ───────────────────────────────────────────────────


def test_schedule_shapes_and_bounds():
    for sched in PREFIX_ATTENTION_SCHEDULES:
        w = get_prefix_weights(3, 12, 30, sched)
        assert w.shape == (30,)
        assert w.dtype == np.float32
        assert np.all(w >= 0.0) and np.all(w <= 1.0)
        # Beyond the execution horizon the previous chunk has no say.
        assert np.all(w[12:] == 0.0)


def test_leading_delay_is_pinned():
    """The timesteps the robot already consumed stay locked to the old chunk."""
    for sched in ("linear", "exp"):
        w = get_prefix_weights(4, 12, 30, sched)
        assert np.all(w[:4] == 1.0)
        # and the ramp is monotonically non-increasing
        assert np.all(np.diff(w[4:12]) <= 1e-7)


def test_zero_horizon_gives_no_guidance():
    for sched in ("linear", "exp"):
        assert np.all(get_prefix_weights(0, 0, 30, sched) == 0.0)


def test_start_clamped_to_end():
    # start > end must behave as start == end (matches the reference clamp).
    assert np.array_equal(
        get_prefix_weights(20, 12, 30, "exp"),
        get_prefix_weights(12, 12, 30, "exp"),
    )


def test_exp_decays_faster_than_linear():
    lin = get_prefix_weights(0, 12, 30, "linear")
    exp = get_prefix_weights(0, 12, 30, "exp")
    assert np.all(exp[:12] <= lin[:12] + 1e-7)
    assert exp[:12].sum() < lin[:12].sum()


def test_unknown_schedule_rejected():
    with pytest.raises(ValueError, match="unknown prefix_attention_schedule"):
        get_prefix_weights(0, 12, 30, "cosine")


@pytest.mark.parametrize("max_w", [1.0, 5.0, 10.0, 100.0])
def test_guidance_ceiling_clamps_to_max(max_w):
    g = np.minimum(guidance_ceiling(10), np.float32(max_w))
    assert g.shape == (10,)
    assert np.all(g <= max_w + 1e-6)
    # tau == 0 is the +inf case that must saturate at max_w exactly
    assert g[0] == pytest.approx(max_w)


def test_guidance_ceiling_matches_closed_form():
    n = 10
    ceil = guidance_ceiling(n)
    for k in range(1, n):
        tau = k / n
        sq = (1 - tau) ** 2
        expected = ((1 - tau) / tau) * ((sq + tau ** 2) / sq)
        assert ceil[k] == pytest.approx(expected, rel=1e-5)
    assert math.isinf(ceil[0])


# ── Cross-check against lerobot's reference implementation ────────────────


@pytest.mark.parametrize("sched", ["ZEROS", "ONES", "LINEAR", "EXP"])
def test_matches_lerobot_reference(sched):
    pytest.importorskip("lerobot", reason="lerobot not installed")
    from lerobot.configs import RTCAttentionSchedule
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.policies.rtc.modeling_rtc import RTCProcessor

    proc = RTCProcessor(
        RTCConfig(prefix_attention_schedule=RTCAttentionSchedule[sched]))
    for total in (10, 30, 50):
        for end in (0, 1, 5, 12, total, total + 5):
            for start in (0, 1, 3, 12, end + 2):
                ref = proc.get_prefix_weights(start, end, total).numpy()
                got = get_prefix_weights(start, end, total, sched.lower())
                np.testing.assert_allclose(
                    got, ref, atol=1e-6,
                    err_msg=f"{sched} total={total} start={start} end={end}")


# ── Device-side correction ────────────────────────────────────────────────


@pytest.mark.parametrize("delay,horizon", [(0, 12), (3, 12), (5, 30)])
def test_device_correction_matches_reference(delay, horizon):
    """`a' = a + (g_k/N)(prev - x - (N-k)a)*w`, evaluated on the GPU."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    from flash_rt.core.cuda_buffer import CudaBuffer
    from flash_rt.models.pi05.pipeline_rtx import ACTION_DIM, Pi05Pipeline

    chunk, n_steps, adim, max_w = 30, 10, 16, 10.0

    class _Stub:
        chunk_size, num_steps = chunk, n_steps

        def __init__(self):
            self.bufs = {
                "diffusion_noise": CudaBuffer.device_zeros(chunk * ACTION_DIM, np.float16),
                "decoder_action_buf": CudaBuffer.device_zeros(chunk * ACTION_DIM, np.float16),
                "rtc_prev_action_chunk": CudaBuffer.device_zeros(chunk * ACTION_DIM, np.float16),
                "rtc_prefix_weights": CudaBuffer.device_zeros(chunk, np.float32),
                "rtc_guidance_weight": CudaBuffer.device_zeros(1, np.float32),
            }
            self._rtc = None
            self._rtc_streams = {}

    for name in ("enable_rtc_guidance", "_rtc_torch_stream", "rtc_write_inputs",
                 "_apply_rtc_guidance"):
        setattr(_Stub, name, getattr(Pi05Pipeline, name))
    _Stub.rtc_guidance_enabled = Pi05Pipeline.rtc_guidance_enabled

    pipe = _Stub()
    pipe.enable_rtc_guidance()

    rng = np.random.default_rng(7)
    prev = rng.normal(0, 0.5, (horizon, adim)).astype(np.float32)
    weights = get_prefix_weights(delay, horizon, chunk, "exp")
    pipe.rtc_write_inputs(prev, weights, max_w)

    def as_bf16(v):
        return torch.from_numpy(v).to(torch.bfloat16).float().numpy()

    x = as_bf16(rng.normal(0, 1.0, (chunk, ACTION_DIM)).astype(np.float32))
    a = as_bf16(rng.normal(0, 0.1, (chunk, ACTION_DIM)).astype(np.float32))

    prev_padded = np.zeros((chunk, ACTION_DIM), dtype=np.float32)
    prev_padded[:horizon, :adim] = as_bf16(prev)
    g = np.minimum(guidance_ceiling(n_steps), np.float32(max_w))

    for k in range(n_steps):
        pipe._rtc["x"].copy_(torch.from_numpy(x))
        pipe._rtc["a"].copy_(torch.from_numpy(a))
        torch.cuda.synchronize()
        pipe._apply_rtc_guidance(k, 0)
        torch.cuda.synchronize()
        got = pipe._rtc["a"].float().cpu().numpy()

        x1 = x + (n_steps - k) * a                     # time_k * N == N - k
        expected = a + (g[k] / n_steps) * (prev_padded - x1) * weights[:, None]

        scale = max(float(np.abs(expected).max()), 1e-6)
        assert np.abs(got - expected).max() / scale < 0.02, f"step {k}"


def test_zero_weights_are_an_exact_noop():
    """No previous chunk must leave the denoiser bit-identical."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    from flash_rt.core.cuda_buffer import CudaBuffer
    from flash_rt.models.pi05.pipeline_rtx import ACTION_DIM, Pi05Pipeline

    chunk, n_steps = 30, 10

    class _Stub:
        chunk_size, num_steps = chunk, n_steps

        def __init__(self):
            self.bufs = {
                "diffusion_noise": CudaBuffer.device_zeros(chunk * ACTION_DIM, np.float16),
                "decoder_action_buf": CudaBuffer.device_zeros(chunk * ACTION_DIM, np.float16),
                "rtc_prev_action_chunk": CudaBuffer.device_zeros(chunk * ACTION_DIM, np.float16),
                "rtc_prefix_weights": CudaBuffer.device_zeros(chunk, np.float32),
                "rtc_guidance_weight": CudaBuffer.device_zeros(1, np.float32),
            }
            self._rtc = None
            self._rtc_streams = {}

    for name in ("enable_rtc_guidance", "_rtc_torch_stream", "rtc_write_inputs",
                 "_apply_rtc_guidance"):
        setattr(_Stub, name, getattr(Pi05Pipeline, name))
    _Stub.rtc_guidance_enabled = Pi05Pipeline.rtc_guidance_enabled

    pipe = _Stub()
    pipe.enable_rtc_guidance()
    pipe.rtc_write_inputs(None, None, 10.0)

    rng = np.random.default_rng(1)
    a = torch.from_numpy(rng.normal(0, 0.1, (chunk, ACTION_DIM)).astype(np.float32))
    pipe._rtc["x"].copy_(torch.from_numpy(
        rng.normal(0, 1.0, (chunk, ACTION_DIM)).astype(np.float32)))
    pipe._rtc["a"].copy_(a)
    torch.cuda.synchronize()
    before = pipe._rtc["a"].clone()

    for k in range(n_steps):
        pipe._apply_rtc_guidance(k, 0)
    torch.cuda.synchronize()

    assert torch.equal(before, pipe._rtc["a"])
