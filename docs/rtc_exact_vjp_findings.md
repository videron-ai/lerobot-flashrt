# Exact-VJP RTC Guidance — Investigation and Measurement

Session notes: implementing Kinetix's exact vector-Jacobian-product form of RTC
prefix guidance for Pi0.5, measuring what the shipped identity-Jacobian
approximation actually costs in accuracy, and deciding whether the exact form
is worth its latency.

**Verdict up front: not worth deploying as implemented.** The approximation
induces an error of ~38% of the guidance effect (~0.5× the natural
within-chunk step size at `max_guidance_weight=10`), but the exact path costs
2.8× latency, and in RTC that latency penalty is self-defeating — it forces a
longer `execution_horizon`, which `rollout_rtc_findings.md` §4.1 measured as
the worse operating point by a wider margin than the correction gains back.

**Status: the implementation was reverted.** Nothing in `flash_rt/` changed.
See §8 for what was removed and where it is preserved.

**Setup.** GB10 / SM121, `Pi05PipelineFP16` (`use_fp8=False`), chunk 30,
3 views, `state_prompt_mode="fixed"` (`encoder_seq_len=846`),
`num_inference_steps=10`, action dim 16.
Checkpoint `openarm_folding_high_quality_60k/checkpoints/060000/pretrained_model`.
Guidance: `inference_delay=3`, `execution_horizon=6`, EXP schedule.

---

## 1. lerobot's "exact" reference is itself the approximation

Before building anything, the reference implementation was checked.
`RTCProcessor.denoise_step` (`lerobot/policies/rtc/modeling_rtc.py`) reads:

```python
with torch.enable_grad():
    v_t = original_denoise_step_partial(x_t)   # x_t.requires_grad is False here
    x_t.requires_grad_(True)                   # ← set AFTER the forward
    x1_t = x_t - time * v_t
    err = (prev_chunk_left_over - x1_t) * weights
    correction = torch.autograd.grad(x1_t, x_t, err.clone().detach())[0]
```

`x_t` is `clone().detach()`-ed, then `v_t` is computed while it does **not**
require grad, so no autograd graph connects them. `requires_grad_(True)`
afterwards only makes `x_t` a leaf for the *subsequent* ops. The only path from
`x1_t` back to `x_t` is the direct `x_t` term, so `∂x1/∂x = I` exactly and
`correction` is returned **bit-identical to `err`**.

Confirmed with a non-identity toy denoiser:

```
err            [0.011358, 0.577740, -0.040569, 0.014405]
corr(lerobot)  [0.011358, 0.577740, -0.040569, 0.014405]   <- equal to err
corr(true VJP) [0.095010, 0.525205,  0.052357, -0.042175]
lerobot corr == err exactly : True
```

**Consequence.** FlashRT's identity-Jacobian path is not an approximation *of
lerobot* — it already reproduces lerobot bit-for-bit. Implementing the true VJP
means matching **Kinetix** (the JAX original, where `jax.vjp` does capture the
denoiser dependency) and deliberately diverging from lerobot. Any future
"validate against the reference" plan needs to know this; the reference cannot
distinguish the two.

---

## 2. What the exact term requires

With `x1 = x − t·v_θ(x)`:

```
correction = (∂x1/∂x)^T err = err − t·(∂v/∂x)^T err
```

The Jacobian is only 960×960 (chunk 30 × action dim 32), but `∂v/∂x` runs
through the whole action expert: `action_in_proj` → 18 Gemma-300M layers
(d=1024, GQA 8q/1kv, head_dim 256, non-causal attention over the 30 action
tokens plus the frozen 968-token encoder KV) → `action_out_proj`.

Two structural facts make it cheaper than a training backward, and both were
confirmed in the code:

- **Only input-gradients are needed** — no weight gradients.
- **The encoder K/V and the style modulations are constant w.r.t. `x`.** Styles
  depend only on the denoise step (`_precompute_decoder_styles`), and the
  prefix KV is written once before the loop. Cross-attention backward therefore
  needs only `dQ`.

The decoder's *own* K/V do depend on `x` (attention is `causal=False`, so the
30 action tokens all see each other), which is why the Jacobian is dense across
the chunk. This matters in §6.

---

## 3. Approach: a differentiable torch mirror

The fused FP8/FP16 kernel decoder has no autograd surface, so the exact VJP
cannot reuse it. A differentiable torch mirror of the same computation was
built instead, binding the pipeline's own weights, style buffers, RoPE table
and encoder KV cache zero-copy so it cannot drift from what the kernels run.

GEMMs run in the pipeline's 16-bit storage dtype (tensor cores accumulate in
fp32, as CUTLASS does); elementwise stages run in fp32, which is what the
kernels do internally before storing. For the Jacobian this is correct rather
than a compromise: 16-bit rounding is piecewise-constant with zero derivative
almost everywhere, so the useful Jacobian is that of the underlying
real-valued function.

### 3.1 Kernel semantics the mirror had to match

Recovered by reading the kernels and confirmed numerically:

| stage | semantics |
|---|---|
| `ada_rms_norm_style` | `rsqrt(mean(x²)+1e-6)`, weight is all-ones; style row is `[scale\|shift\|gate]`, output `n·(1+scale)+shift`, gate passed through |
| `gate_mul_residual` | `residual += x · gate` |
| `qkv_split_rope` | **interleaved-pair** RoPE on adjacent channels `(2p, 2p+1)`, cos/sin interleaved along head_dim; V unrotated |
| attention | non-causal, GQA 1 KV head broadcast to 8, `softmax_scale = 1/√256` |
| FFN | **GeGLU** — `gelu(gate) · up` |
| `decoder_action_out_proj_w/b` | pre-scaled by `−1/N`, so the buffer holds `a = −v/N` and the step is `x += a` |
| `decoder_time_emb` | allocated and uploaded but **never read** — time conditioning is entirely in the styles |

### 3.2 The FFN is GeGLU, not SwiGLU

This was the one genuine surprise and it cost the most time. The kernel is
named `gate_geglu`, but `csrc/gemm/cutlass_sm80_int8_silu_gated.cu` documents
"SiLU-gated", which sent the first implementation down the wrong path. Testing
layer 0's residual stream against the kernel settles it:

| activation | gated arg | mirror \|h\| | rel err |
|---|---|---|---|
| silu | gate | 73.96 | 0.17802 |
| **gelu (tanh)** | **gate** | **62.780** | **0.00048** |
| gelu (erf) | gate | 62.780 | 0.00048 |
| silu | up (swapped) | 105.16 | 0.67497 |

Kernel reference: 62.781. `gelu_tanh` and `gelu_erf` are indistinguishable at
fp16; Gemma uses the tanh approximation.

> **Worth checking separately:** if the INT8 decoder path routes this MLP
> through `cutlass_sm80_int8_silu_gated.cu`, it is applying SiLU where the
> model wants GELU — a 0.178 relative error per layer. The INT8 path was not
> exercised in this session, so this is a flag, not a finding.

---

## 4. Verification

| check | result |
|---|---|
| mirror vs kernel decoder, per-step delta, all 10 steps | **0.33%** worst-case relative |
| VJP is the transpose of the JVP, `<u,Jd>` vs `<Jᵀu,d>` | **2.6e-5** … 5.4e-3 |
| mirror JVP vs kernel finite-difference JVP | cosine **0.992** at ε=0.4 |
| zero prefix weights on the exact path | **0.000000** — exact no-op |

The finite-difference agreement *improves* monotonically with larger ε
(cosine 0.992 / 0.966 / 0.893 / 0.758 at ε = 0.4 / 0.2 / 0.1 / 0.05), which is
fp16 noise dominating at small ε, not a Jacobian mismatch — truncation error
would trend the other way.

> A first attempt at this check compared the scalar `u·(Jd)` for random `u`,
> which lands at ~1e-3 and drowns in the fp16 noise floor. Comparing the full
> 960-vector JVP instead is what makes it meaningful. Worth remembering.

---

## 5. How wrong is the approximation?

### 5.1 In the correction term

Feeding the same `x_k` to both, with a realistic previous chunk:

| step | `g_k` | \|err\| | \|corr\| | \|corr−err\|/\|err\| | cos(corr,err) |
|---|---|---|---|---|---|
| 0 | 10.000 | 2.8329 | 1.1317 | 0.6133 | 0.9806 |
| 1 | 9.111 | 2.7004 | 1.0868 | 0.6086 | 0.9834 |
| 2 | 4.250 | 2.5703 | 1.0445 | 0.6046 | 0.9839 |
| 4 | 2.167 | 2.2996 | 0.9516 | 0.6004 | 0.9796 |
| 5 | 2.000 | 2.1608 | 0.9177 | 0.5945 | 0.9735 |
| 7 | 2.762 | 1.8530 | 0.9071 | 0.5686 | 0.9359 |
| 9 | 9.111 | 1.5492 | 0.9272 | 0.4659 | 0.9533 |

The exact correction is **~60% different in norm and consistently smaller** —
about **2.5× smaller at step 0**, precisely where `g_k` saturates at
`max_guidance_weight`. Direction is largely preserved (cosine 0.92–0.98).

This quantifies the caveat already stated in `rtc_prefix_attention.md`
("weakest exactly where the guidance is strongest") and explains why backing
off `rtc_max_guidance_weight` helps: it partly compensates for a correction
the approximation inflates.

### 5.2 In the final action chunk — the decision-relevant number

Both corrections run through the *same* mirror from identical initial noise, so
the only variable is `corr = err` vs `corr = Jᵀerr`. This also sidesteps the
kernel-path nondeterminism in §7.2. Normalized action units, averaged over the
prefix-weighted rows and the checkpoint's 16 real action dims:

| `max_w` | \|exact − identity\| | guidance effect | error / effect | error / natural step |
|---|---|---|---|---|
| 1 | 0.0016 | 0.0033 | 48% | 0.16× |
| 2 | 0.0027 | 0.0064 | 42% | 0.26× |
| 5 | 0.0039 | 0.0107 | 37% | 0.39× |
| **10** | **0.0052** | **0.0138** | **38%** | **0.51×** |
| 20 | 0.0052 | 0.0136 | 38% | 0.51× |

Natural within-chunk step size is 0.0101. At the `g=10` operating point the
approximation displaces the action by **about half of one control tick's normal
motion**, which is **~38% of everything guidance is doing**. It saturates above
`max_w≈10` because `g_k` is already clamped there.

---

## 6. The error is *not* absorbable into `max_guidance_weight`

Since the discrepancy is mostly gain (cosine 0.92–0.98), the obvious cheap fix
is to retune `max_w` on the identity path. It does not work — the minimum is
flat and shallow:

| identity `max_w` | 1 | 2 | 3 | 4 | 5 | 7 | 10 |
|---|---|---|---|---|---|---|---|
| residual vs exact(10) | 78.9% | 58.2% | 47.0% | 40.9% | 37.9% | **34.6%** | 37.7% |

Best case `max_w=7` leaves 34.6%, barely better than 37.7%.

The reason is structural, not numerical: **the identity correction is exactly
zero wherever the prefix weight is zero, and the exact one is not.** At
`max_w=10`, timesteps t=6–9 carry `w=0` yet still differ by 0.002–0.006,
because the dense Jacobian (§2) redistributes the correction across the entire
chunk:

```
  t= 0  w=1.000  |diff|=0.00448      t= 5  w=0.041  |diff|=0.00597
  t= 2  w=1.000  |diff|=0.00747      t= 6  w=0.000  |diff|=0.00555
  t= 3  w=0.488  |diff|=0.00421      t= 9  w=0.000  |diff|=0.00167
```

No scalar retune can reproduce a redistribution the approximation cannot
express.

---

## 7. Two bugs found along the way

### 7.1 FP16 guidance reads its buffers under the wrong dtype — **still present**

`Pi05PipelineFP16.enable_rtc_guidance` binds `bf16_t = torch.bfloat16` and
takes `torch_view`s of `diffusion_noise`, `decoder_action_buf` and
`rtc_prev_action_chunk` with it. Those buffers hold **fp16**: this module's
`BF16` is `np.float16` (a 2-byte sizing placeholder, per its own comment) and
the torch frontend binds its `bf16` alias to `torch.float16`. Reading the
converged action chunk both ways:

```
  as   float16: absmax=0.9575  mean=+0.0354   <- normalized action range
  as  bfloat16: absmax=0.0052  mean=+9.3e-05  <- reinterpreted bits
```

The guidance correction therefore reads garbage ~150× too small and writes
bf16 bit patterns back into a buffer the kernels read as fp16. This plausibly
explains the shelved FP16 guidance port. With it corrected during this session,
identity guidance measurably worked: weighted `|chunk − prev|` went
0.0721 → 0.0277 against 0.0624 unguided.

**This fix was reverted along with everything else.** It is a one-line change
(route the views through the pipeline's real storage dtype) and is worth
applying on its own, independently of any VJP work.

### 7.2 The kernel path does not reproduce under a fixed seed — **not diagnosed**

Two `predict` calls with the same `torch.manual_seed` differ by 0.66 in the
output chunk, while the eager mirror path reproduces exactly (0.000000). The
frontend creates a fresh `torch.cuda.Stream()` per inference and copies the
noise on it, while the graph replays on `self._graph_stream`; a missing
sync between those two is a plausible cause, but it was **not confirmed**.

This matters beyond reproducibility: it would confound any A/B measurement
taken on the kernel path, including the offline rollout comparison that was
the original plan. Understand this before running that sweep.

---

## 8. What was built, and what was reverted

Implemented, verified, then removed at the end of the session:

| file | change |
|---|---|
| `flash_rt/models/pi05/decoder_vjp.py` *(new)* | `TorchActionExpert` — differentiable mirror of the denoise step; `guided_action_delta` applying the exact correction; `attach(frontend)` |
| `flash_rt/models/pi05/pipeline_rtx_fp16.py` | `torch_storage_dtype` property (fixes §7.1); `enable_rtc_vjp_guidance()`; `_transformer_decoder_vjp()`; branch in `transformer_decoder`; capture guard in `record_infer_graph` |

Preserved outside the repo for reconstruction:
`<scratchpad>/decoder_vjp.py.keep` and `<scratchpad>/pipeline_vjp_wiring.patch`.

Design notes worth keeping if this is revisited:

- Arming the exact path must drop `self._graph` and `self._decoder_only_graph`.
  Otherwise `forward()` replays the captured kernel loop and silently bypasses
  the whole thing.
- Autograd is host-side control flow, so the exact decode cannot be captured;
  the pipeline falls back to eager `run_pipeline`, which is part of why the
  measured cost is 2.8× rather than the ~2× the extra FLOPs alone imply.
- The pipeline's existing `_save_rtc_x` / `_apply_rtc_guidance` call sites are
  a convenient zero-modification hook for snapshotting `x_k` and `x_{k+1}` per
  denoise step — useful for any future parity work.

---

## 9. Recommendations

1. **Do not adopt the exact VJP as a torch mirror.** 2.8× latency
   (102 → 286 ms). In RTC that penalty compounds: higher latency → larger
   `inference_delay` → longer `execution_horizon` required, and §4.1 of
   `rollout_rtc_findings.md` measured h=12 at MAE 5.69 versus h=6 at 4.18.
   That regression is larger than the 0.005-unit correction gain.
2. **Fix §7.1 regardless.** One line, and it makes FP16 guidance function at
   all.
3. **Diagnose §7.2 before any offline A/B**, or the comparison is unsound.
4. **If the exact path is ever wanted, build it as backward kernels.** Roughly
   latency-neutral (~95 ms estimated), at which point a 38% fidelity gain is
   essentially free. The cheap-first ordering matters: `cache_frames=2` is
   ~26 ms for far less work (§5.3 there).
5. **Nothing here settles task success.** All of it is open-loop, single-seed,
   single-observation, one checkpoint. The ratios were stable across `max_w`,
   but the third digit should not be quoted.
