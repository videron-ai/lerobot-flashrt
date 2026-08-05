# RTC Prefix Attention — FlashRT Integration

Status: **implemented** for Pi0.5 torch RTX (`flash_rt/frontends/torch/pi05_rtx.py`)
via the identity-Jacobian approximation, using elementwise corrections fused into the
existing captured denoise graph.

Opt in at load time:

```python
model = flash_rt.load_model(
    checkpoint=..., framework="torch", num_views=3, action_horizon=30,
    rtc_guidance=True,                     # arms the correction before capture
    rtc_execution_horizon=12,
    rtc_prefix_attention_schedule="exp",   # zeros | ones | linear | exp
    rtc_max_guidance_weight=10.0,
)

actions = model.predict(images=imgs, prompt=task, state=state,
                        prev_actions=prev_chunk_tail,   # (T, action_dim), normalized
                        inference_delay=delay)          # frames consumed during inference
```

`rtc_guidance` defaults to **False**. Arming it adds correction kernels to the captured
graph, so leaving it off keeps the default graph and its numerics byte-identical.

---

## What the prefix attention schedule does

RTC guidance is applied **inside the flow-matching denoising loop** — at each of the 10
denoising steps, not at the outer `predict()` call boundary.

At every step `t`:

```
v_t   = denoiser(x_t)                              # forward pass through transformer
x1_t  = x_t − time × v_t                           # denoised action prediction

weights = get_prefix_weights(inference_delay, execution_horizon, T)
# weights shape (T,): 1.0 for frames [0 : inference_delay],
# decaying to 0.0 by execution_horizon using the chosen schedule (EXP, LINEAR, …)

err        = (prev_chunk_left_over − x1_t) × weights
correction = autograd.grad(x1_t, x_t, grad_outputs=err)  # backward through one denoiser step

guidance_weight = time-dependent scalar, clamped at max_guidance_weight
v_guided  = v_t − guidance_weight × correction
x_t_next  = x_t + dt × v_guided
```

The effect: the generated chunk is steered to agree with the previous chunk's unexecuted
tail for the first `execution_horizon` timesteps, producing smooth continuity across
re-prediction boundaries.

Source: `lerobot/policies/rtc/modeling_rtc.py` — `RTCProcessor.denoise_step()` and
`RTCProcessor.get_prefix_weights()`.

---

## Why the exact form does not fit a static graph

`autograd.grad` is a dynamic backward pass through the transformer. FlashRT's Pi0.5
pipeline is custom kernels plus CUTLASS GEMM launches with no autograd surface, and the
whole 10-step denoise loop is captured as one static CUDA graph. Options considered:

**Option A — per-step subgraphs (exact).** Break the loop into 10 replayable subgraphs and
compute the VJP in eager torch between replays. Needs a differentiable reference decoder
that does not exist, and costs 10 graph launches plus 10 sync points (~50–80 ms vs ~20 ms).

**Option B — identity-Jacobian approximation (implemented).** The Jacobian is
`J = ∂x1_t/∂x_t = I − time·(∂v/∂x)`. Taking `J ≈ I` gives `correction ≈ err`, and the
update collapses to pure elementwise arithmetic that captures into the existing graph.

> **Sign.** With `J ≈ I` the update is `v' = v_t − g·err`, *not* `+`. Check it against the
> geometry: `x1 = x − time·v`, so `∂x1/∂v = −time < 0` — to move `x1` **toward** `prev`
> when `err = prev − x1 > 0`, `v` must **decrease**. An earlier draft of this document had
> `v_t + g·err`, which pushes each chunk *away* from its predecessor.

---

## How it maps onto the pipeline

`decoder_action_out_proj_w` is pre-scaled by `−1/N`, so `decoder_action_buf` holds
`a = −v/N` and the step is `x += a` (`Pi05Pipeline.transformer_decoder`). Rewriting
Option B in those terms:

```
x1  = x + (time_k · N) · a            # time_k · N == N − k, exactly
err = (prev − x1) ⊙ w
a' = a + (g_k / N) · err              # since a' = −v'/N = a + (g/N)·err
```

`_apply_rtc_guidance` runs this as four elementwise ops per step, inserted between the
output bias add and the `residual_add` (it reads `diffusion_noise` as `x_k`, so it has to
sit before the accumulate):

```python
torch.sub(prev, x, out=s)          # prev − x
s.add_(a, alpha=-(N - k))          # prev − x1
s.mul_(w)                          # err
a.addcmul_(s, g_slices[k], value=1.0 / N)
```

Plus three ops hoisted to step 0 that derive `g_k = min(ceiling_k, max_guidance_weight)`
for all steps at once and cast the weights to bf16. `ceiling_k` is `+inf` at `k = 0`, so
`min(+inf, max_w)` reproduces the reference's `nan_to_num(posinf=max_w)` exactly.

### Ports

The three device buffers were already allocated by the pipeline; guidance is what finally
reads the latter two:

| buffer | shape | written by |
|---|---|---|
| `rtc_prev_action_chunk` | (chunk, 32) bf16 | `rtc_write_inputs` |
| `rtc_prefix_weights` | (chunk,) fp32 | `rtc_write_inputs` |
| `rtc_guidance_weight` | (1,) fp32 | `rtc_write_inputs` |

Because every runtime-varying input is a fixed-address device buffer read *inside* the
graph, `inference_delay`, the prefix weights and `max_guidance_weight` are all tunable
per inference with **no recapture** — unlike `subgraphs/pi05/rtc_prefix.py`, whose
`prefix_len` is baked at capture time. Host staging is one 4·chunk-byte upload.

Zero weights make the correction exactly zero, so the first chunk (no previous chunk)
needs no separate graph. `enable_rtc_guidance` zeroes the three ports at arm time, because
they are `device_empty` and a NaN reaching FP8 calibration would poison the activation
scales.

---

## Measured

Pi0.5 folding checkpoint (chunk 30, 3 views, `state_prompt_mode="fixed"`), GB10 / SM121:

| | median | p99 |
|---|---|---|
| `rtc_guidance=False` | 76.37 ms | 79.38 ms |
| armed, `prev_actions=None` | 76.67 ms | 81.42 ms |
| armed, full staging | 76.62 ms | 82.09 ms |

Guidance costs **~0.25 ms**; host staging is 0.04 ms.

Chunk-splice discontinuity over 6 consecutive re-predictions
(`execution_horizon=12`, `inference_delay=2`, EXP):

| | mean splice jump | vs natural within-chunk step |
|---|---|---|
| unguided | 0.0349 | 5.53× |
| guided | 0.0058 | 1.34× |

An 83% reduction — transitions drop to roughly the size of ordinary motion between
adjacent timesteps.

---

## Accuracy caveat

The identity-Jacobian approximation is weakest exactly where the guidance is strongest:
`guidance_weight` saturates at `max_guidance_weight` at `tau = 0` (the first denoise step)
and decays afterward. Nothing here has been validated against the exact VJP on task
success — only against the algebraic update and against chunk continuity. Treat
`rtc_max_guidance_weight` as the knob to back off if guided rollouts degrade.

For the exact path, `flash_rt/subgraphs/pi05/rtc_vjp_guided.py` still defines the
producer-supplied `DenoiserVjpProvider` contract; it remains unimplemented.

---

## Tests

`tests/test_rtc_guidance.py`

- host schedules cross-checked against `RTCProcessor.get_prefix_weights` for all four
  schedules across a grid of `(start, end, total)`
- `guidance_ceiling` against the closed form, and its `+inf`/clamp behaviour
- the device correction against a numpy reference of `a' = a + (g_k/N)(prev − x1)⊙w`,
  every denoise step, within bf16 tolerance
- zero weights are a bit-exact no-op
