# RTC Prefix Attention — FlashRT Integration Notes

Status: **not implemented** — `prev_chunk_left_over` is currently dropped in the FlashRT
`predict_action_chunk` patch. Chunk transitions work but are generated independently; the
prefix schedule flags (`--inference.rtc.prefix_attention_schedule`, `--inference.rtc.max_guidance_weight`,
`--inference.rtc.execution_horizon`) have no effect on FlashRT inference.

---

## What the prefix attention schedule does

RTC guidance is applied **inside the flow-matching denoising loop** — at each of the 10
denoising steps, not at the outer `predict()` call boundary.

At every step `t`:

```
v_t   = denoiser(x_t)                              # forward pass through transformer
x1_t  = x_t − time × v_t                           # denoised action prediction

weights = get_prefix_weights(inference_delay, execution_horizon, T)
# weights shape (T,): ~1.0 for frames [0 : inference_delay],
# decaying to 0.0 by execution_horizon using the chosen schedule (EXP, LINEAR, etc.)

err        = (prev_chunk_left_over − x1_t) × weights
correction = autograd.grad(x1_t, x_t, grad_outputs=err)  # backward through one denoiser step

guidance_weight = time-dependent scalar, clamped at max_guidance_weight
v_guided  = v_t − guidance_weight × correction
x_t_next  = x_t + dt × v_guided
```

The effect: the generated chunk is steered to agree with the previous chunk's unexecuted
tail for the first `execution_horizon` timesteps, producing smooth continuity across
re-prediction boundaries. The EXP schedule makes the weight decay exponential rather than
linear, concentrating guidance on the immediately-executing frames.

Source: `lerobot/policies/rtc/modeling_rtc.py` — `RTCProcessor.denoise_step()` and
`RTCProcessor.get_prefix_weights()`.

---

## Why it can't be added at the `predict()` boundary

`flash_rt.VLAModel.predict()` returns a completed chunk after all 10 denoising steps.
There is no hook to inject per-step corrections from outside. The guidance must be applied
**between** each denoiser call.

---

## The CUDA graph compatibility problem

FlashRT captures the entire 10-step denoising loop as a single static CUDA graph.
The `autograd.grad` call in the correction step involves a dynamic backward pass through
the transformer — this **cannot** be captured in a static graph.

### Option A — Per-step subgraphs (exact)

Break the single 10-step graph into 10 per-step subgraphs. Between each replay:

1. Transfer `x_t` to host (or keep on device in a pre-allocated buffer)
2. Compute `x1_t`, `err`, `correction` on the host (or in eager mode)
3. Write corrected `x_t` back into the graph's input buffer

Correct, but adds 10× graph launch overhead and 10 synchronization points.
Estimated latency: ~50–80 ms vs. the current ~20 ms.

### Option B — Identity-Jacobian approximation (stays in one graph)

Replace `correction = autograd.grad(x1_t, x_t, err)` with `correction ≈ −err`,
assuming the Jacobian of `x1_t` w.r.t. `x_t` is approximately identity. The update becomes:

```
v_guided = v_t + guidance_weight × (prev_chunk_left_over − x1_t) × weights
```

This is pure tensor arithmetic — it can be fused into the captured CUDA graph as an
additive correction applied after each denoiser call. The approximation is reasonable late
in denoising (when `x_t` is close to clean) and less accurate early on.

**This is the recommended path** — maintains ~20 ms latency while adding meaningful
guidance. Expected to produce smoother chunk transitions than Option A at the cost of
slight guidance accuracy.

---

## Required changes

### 1. flash_rt internals (the bulk of the work)

**`flash_rt/api.py` / `VLAModel.predict()`** — add two new parameters:

```python
def predict(self, images, prompt=None, state=None,
            prev_actions=None,    # np.ndarray (T, action_dim) float32, normalized; or None
            inference_delay=0):   # int — frames consumed during inference
```

**`frontends/torch/pi05_rtx.py`** — inject RTC correction inside the denoising loop after
each forward call. Port `get_prefix_weights()` from `lerobot/policies/rtc/modeling_rtc.py`
(pure Python/torch — no framework dependency).

With Option B, the correction is:

```python
# precompute once before the loop
if prev_actions is not None:
    weights = get_prefix_weights(inference_delay, execution_horizon, T)  # (T,)
    weights = torch.from_numpy(weights).to(device).unsqueeze(-1)          # (T, 1)
    prev_t  = torch.from_numpy(prev_actions).to(device, bf16)             # (T, A)

# inside the loop at each step
v_t  = denoiser(x_t)
x1_t = x_t - time * v_t
if prev_actions is not None:
    err       = (prev_t - x1_t) * weights
    guidance  = compute_guidance_weight(tau, max_guidance_weight)
    v_t       = v_t + guidance * err          # additive correction, no autograd
x_t = x_t + dt * v_t
```

`get_prefix_weights` with `RTCAttentionSchedule.EXP`:

```python
def get_prefix_weights(start, end, total):
    import math, torch
    start = min(start, end)
    skip  = max(total - end, 0)
    n     = total - skip - start
    if n > 0:
        w = torch.linspace(1, 0, n + 2)[1:-1]
        w = w * torch.expm1(w) / (math.e - 1)   # EXP schedule
    else:
        w = torch.tensor([])
    # prepend ones for [0:start], append zeros for [end:total]
    return torch.cat([torch.ones(start), w, torch.zeros(skip)])
```

`compute_guidance_weight` (from `modeling_rtc.py:221–227`):

```python
def compute_guidance_weight(tau, max_w):
    tau   = torch.as_tensor(tau)
    sq    = (1 - tau) ** 2
    inv_r = (sq + tau ** 2) / sq
    c     = torch.nan_to_num((1 - tau) / tau, posinf=max_w)
    return torch.minimum(torch.nan_to_num(c * inv_r, posinf=max_w), torch.as_tensor(max_w))
```

### 2. `examples/online_rollout.py` (trivial once FlashRT exposes the API)

Stop dropping `prev_chunk_left_over` and `inference_delay` — forward them to `predict()`:

```python
def predict_action_chunk(self, batch: dict, **kwargs) -> torch.Tensor:
    prev_chunk = kwargs.get("prev_chunk_left_over")   # (1, T, A) or (T, A) normalized
    delay      = kwargs.get("inference_delay") or 0

    imgs     = [...]  # as before
    state_np = batch[OBS_STATE].squeeze(0).float().cpu().numpy()

    prev_np = None
    if prev_chunk is not None:
        p = prev_chunk.squeeze(0) if prev_chunk.dim() == 3 else prev_chunk
        prev_np = p[:, :_state_dim].float().cpu().numpy()   # (T, action_dim) normalized

    with torch.no_grad():
        chunk_np = _model.predict(
            images=imgs,
            prompt=_task,
            state=state_np[:_state_dim],
            prev_actions=prev_np,
            inference_delay=delay,
        )

    return torch.from_numpy(chunk_np[:, :_act_dim]).float().unsqueeze(0).to(_device)
```

`prev_chunk_left_over` is already in normalized space (it's the `original` tensor from
`ActionQueue.merge()`) and already action-dim-sized — no conversion needed before passing
to FlashRT.

---

## Summary

| Component | Change | Effort |
|---|---|---|
| `flash_rt/api.py` | Add `prev_actions`, `inference_delay` to `predict()` | Small |
| `frontends/torch/pi05_rtx.py` | Inject correction inside denoising loop | Medium |
| CUDA graph compat | Use Option B approximation (no autograd) | Medium |
| `examples/online_rollout.py` | Forward kwargs → `predict()` | ~10 lines |

All the real work is inside FlashRT. The lerobot side is trivial once the API is exposed.
