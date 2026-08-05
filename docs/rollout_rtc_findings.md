# FlashRT × lerobot Rollout — Review, Fixes, and RTC Guidance

Session notes: a review of `examples/online_rollout.py`, the fixes applied, the
implementation of RTC prefix guidance in the Pi0.5 torch RTX frontend, and the
offline evaluation that came out of it.

**Environment.** NVIDIA GB10 (SM121) → `hardware="auto"` resolves to
`rtx_sm120` → `flash_rt/frontends/torch/pi05_rtx.py`. lerobot from
`/usr/local/lib/python3.12/dist-packages/lerobot` inside the `flashrt:dev`
container; repo mounted at `/flashrt`.

**Checkpoint under test.** `openarm_folding_high_quality_60k/checkpoints/060000`
— `chunk_size=30`, `n_action_steps=30`, action dim 16, state dim 16,
`num_inference_steps=10`, `max_state_dim=32`, **`use_relative_actions=true`**
(gripper joints excluded). Cameras: two 1280×720 wrists + one 640×480 base.

---

## 1. Review of `examples/online_rollout.py`

Seven findings, ordered by severity. Four were fixed this session.

### 1.1 Image resize did not match training preprocessing — **fixed**

`F.interpolate(..., mode="bilinear")` stretched every camera to 224×224. Native
PI0.5 uses `resize_with_pad_torch` (`lerobot/policies/common/vla_utils.py:142`)
— aspect-preserving resize with centered black padding — inside
`_preprocess_images`, which FlashRT bypasses.

The in-code claim that "lerobot preprocessor already resizes to 224×224" was
false: `make_pi05_pre_post_processors` has **no** resize step (rename → batch →
relative → normalize → state-tokenize → tokenize → device). The resize happens
inside the model. So the "safety net" branch was the only branch, always taken.

Verified against native PI0.5's SigLIP input on synthetic 1280×720 / 640×480 frames:

| path | max abs diff | mean |
|---|---|---|
| old (stretch + truncate) | **1.9843** | 0.6835 |
| new (`resize_with_pad_torch` + round) | 0.0039 | 0.0011 |

The residual 0.0039 is half a uint8 LSB — the floor imposed by FlashRT's uint8
image interface. Padding is correct: 98 black rows for the 16:9 wrists
(224×126 + 49 px bars), 56 for the 4:3 base (224×168 + 28 px bars). The old path
had zero padded rows.

### 1.2 FP8 calibration ran on all-black dummy frames — **fixed**

`_warmup_flashrt` issued the first `predict()` with `np.zeros((224,224,3))`.
`VLAModel.predict` lazily fires `calibrate_with_real_data([obs])`
(`flash_rt/api.py:139-146`), and `_calibrate_single_frame`
(`pi05_rtx.py:1315`) runs a full forward on that observation and **freezes
per-GEMM FP8 activation scales from it**. The RTX path does not persist
calibration to disk (`load_calibration` is only wired on the Thor path), so this
recurred on every process start.

`docs/calibration.md` §"Practical guidance" specifies calibrating at "an
operating-point frame". Fix: `_capture_warmup_inputs()` pulls one real
observation (`robot_wrapper.get_observation()` → `build_dataset_frame` →
`prepare_observation_for_inference` → preprocessor) and warms up on it. Done via
the warmup frame rather than a separate `calibrate()` call because `calibrate`
requires `set_prompt` to have run first; going through `predict()` orders both
correctly and matches what `offline_rollout.py` already did.

### 1.3 `online_rollout_sync.py` cannot run with this checkpoint — **not fixed**

`use_relative_actions: true` ⇒ `build_rollout_context` raises
`NotImplementedError: SyncInferenceEngine does not support policies with
relative actions for now` (`rollout/context.py:477-484`).

The raise happens at step 6, *after* `robot.connect()` at step 3, and
`build_rollout_context` is called **outside** the script's try/finally
(`online_rollout_sync.py:253`, same at `online_rollout.py:276`) — so the arms are
left connected and torqued with no disconnect path. The comment "Everything
after robot connect is wrapped so the robot is always disconnected" is
inaccurate; it covers everything after `build_rollout_context` *returns*.

### 1.4 State truncated to `action_dim` — **fixed**

`state_np[:_state_dim]` with `_state_dim = action_dim`. Harmless here (both 16)
but wrong in general; on a robot with a wider state (velocity channels,
LeKiwi-style) it silently drops state dims from the prompt. `state_dim` now
comes from `policy.config.input_features[OBS_STATE].shape[0]` and the full state
is passed, with a one-shot warning on mismatch. This also aligned the online and
offline scripts, which previously disagreed.

### 1.5 uint8 conversion truncated instead of rounding — **fixed**

`(hwc * 255).to(torch.uint8)`. Inputs come from `uint8/255`, so `x*255` lands at
e.g. 199.99997 and truncates to 199. Measured: **28.1% of pixels** differed from
the rounded value, mean bias −0.281 LSB. `resize_with_pad_torch` uses
`torch.round`.

### 1.6 Missing camera raises instead of padding — **not fixed**

`batch[key]` is indexed directly. Native `_preprocess_images` tolerates a
dropped camera by substituting an all-`−1` image with mask 0. Under RTC a
mid-rollout camera dropout is swallowed and retried 10× before shutdown.

### 1.7 Warmup runs with the robot connected and torqued — **not fixed**

`build_rollout_context` connects the robot before `_install_flashrt_backend`;
graph capture + autotune is 1–60 s with the arms holding position.

### Verified-correct (checked, no change needed)

- View-key order from `input_features` is `[left_wrist, right_wrist, base]`,
  matching FlashRT's `images[0..2] → image/wrist_image/wrist_image_right`.
- Prompt format matches exactly: `format_pi05_prompt`
  (`flash_rt/core/utils/pi05_prompt.py:35`) produces the same
  `f"Task: {t}, State: {s};\nAction: "` as
  `Pi05PrepareStateTokenizerProcessorStep`, and state discretization
  (`np.digitize(linspace(-1,1,257)[:-1]) - 1`) is identical. `openpi` is not
  installed, so the sentencepiece fallback — the matching path — is the one that runs.
- Flow-matching steps agree: checkpoint `num_inference_steps=10`,
  `NUM_STEPS_DEFAULT=10`.
- FlashRT returns raw normalized actions (no unnormalize), so feeding them to
  lerobot's postprocessor is correct.
- The `types.MethodType` instance patch is picked up by both
  `RTCInferenceEngine._policy` and `select_action`; `torch.compile` wrapping is
  skipped for `pi05` (`context.py:275`) so it cannot clobber the patch.
- `cfg.device` is always resolved in `RolloutConfig.__post_init__`.

---

## 2. RTC prefix guidance (the "documented caveat")

Previously `prev_chunk_left_over` and `inference_delay` were dropped, so
`--inference.rtc.*` had no effect on FlashRT inference.

### 2.1 What already existed

`docs/rtc_prefix_attention.md` said "not implemented", but the pipeline already
allocated all three RTC ports (`pipeline_rtx.py:384-390`, accessors 2072-2084):

| buffer | shape | prior status |
|---|---|---|
| `rtc_prev_action_chunk` | (chunk, 32) bf16 | used by `_copy_rtc_prefix` |
| `rtc_prefix_weights` | (chunk,) fp32 | allocated, **never read** |
| `rtc_guidance_weight` | (1,) fp32 | allocated, **never read** |

`transformer_decoder(rtc_prefix_len=N)` implemented a **hard prefix lock**
(equivalent to `ONES` with no decay). `subgraphs/pi05/rtc_prefix.py` captures it,
but only reachable via the export/Nexus stage-plan path —
`Pi05Pipeline.forward()` only replays `self._graph`.
`subgraphs/pi05/rtc_vjp_guided.py` is a contract with no implementation.

### 2.2 Sign error in the old design doc

The doc recommended `v_t = v_t + guidance * err`. That is backwards. With
`J = ∂x1/∂x ≈ I` the update is `v' = v − g·err`; geometrically `∂x1/∂v = −time < 0`,
so moving `x1` **toward** `prev` requires `v` to **decrease**. As written it
would push each chunk *away* from its predecessor. Corrected in the rewritten doc.

### 2.3 Implementation (Option B, identity-Jacobian)

Out-projection weights are pre-scaled by `−1/N`, so `decoder_action_buf` holds
`a = −v/N` and the step is `x += a`:

```
x1  = x + (time_k · N) · a        # time_k · N == N − k, exactly
err = (prev − x1) ⊙ w
a' = a + (g_k / N) · err
```

Four elementwise ops per denoise step, inserted between the output bias add and
`residual_add` (it reads `diffusion_noise` as `x_k`, so it must precede the
accumulate), plus three hoisted to step 0 deriving
`g_k = min(ceiling_k, max_guidance_weight)` for all steps at once.
`ceiling_k = +inf` at `k=0`, so `min(+inf, max_w)` reproduces the reference's
`nan_to_num(posinf=max_w)` exactly.

Feasibility was probed first: torch elementwise ops **do** capture into
FlashRT's raw `cudaStreamBeginCapture` graph and replay correctly. This is
consistent with existing design — the attention backend already runs
`flash_attn_func` inside the captured region, which is why
`record_infer_graph` takes `external_stream_int`.

**Files changed**

| file | change |
|---|---|
| `flash_rt/core/cuda_buffer.py` | `CudaBuffer.torch_view()` — aliases device memory as a torch tensor, no copy (bf16 via uint16 alias; torch imported lazily so JAX is unaffected) |
| `flash_rt/core/utils/rtc_guidance.py` *(new)* | `get_prefix_weights()`, `guidance_ceiling()` ported from `RTCProcessor` |
| `flash_rt/models/pi05/pipeline_rtx.py` | `enable_rtc_guidance()`, `rtc_write_inputs()`, `_apply_rtc_guidance()`; injection into `transformer_decoder` |
| `flash_rt/frontends/torch/pi05_rtx.py` | `rtc_guidance` / `rtc_execution_horizon` / `rtc_prefix_attention_schedule` / `rtc_max_guidance_weight` ctor args; `infer(..., prev_actions=, inference_delay=)`; per-call staging |
| `flash_rt/api.py` | `predict(..., prev_actions=, inference_delay=)`; matching `load_model` options |
| `examples/online_rollout.py` | arms FlashRT from `cfg.inference.rtc`; forwards `prev_chunk_left_over` + `inference_delay` |
| `docs/rtc_prefix_attention.md` | rewritten |
| `tests/test_rtc_guidance.py` *(new)* | 19 tests |

**Design properties**

- **Opt-in.** `rtc_guidance=False` by default; the default graph and its
  numerics are untouched.
- **Runtime tunable, no recapture.** Weights, `inference_delay` and
  `max_guidance_weight` are read from fixed-address device buffers *inside* the
  graph. Host staging is 0.04 ms. (Contrast `subgraphs/pi05/rtc_prefix.py`,
  whose `prefix_len` is baked at capture time.)
- **Zero weights are an exact no-op**, so the first chunk needs no second graph.
- `enable_rtc_guidance()` zeroes the three ports at arm time — they are
  `device_empty`, and a NaN reaching FP8 calibration would poison the scales.

### 2.4 Verification

- **Schedules** match `RTCProcessor.get_prefix_weights` exactly for all four
  schedules across a grid of `(start, end, total)` — 0 mismatches.
- **Device correction** matches a numpy reference of
  `a' = a + (g_k/N)(prev − x1)⊙w` at every denoise step; worst relative error
  0.0072 (bf16 eps ≈ 0.008).
- **`max_guidance_weight=0` is bit-identical** to unguided in-process (max diff 0.0).
- **19/19 tests pass.**

**Latency** (chunk 30, 3 views, `state_prompt_mode="fixed"`, GB10):

| | median | p99 |
|---|---|---|
| `rtc_guidance=False` | 76.37 ms | 79.38 ms |
| armed, `prev_actions=None` | 76.67 ms | 81.42 ms |
| armed, full staging | 76.62 ms | 82.09 ms |

Guidance costs **~0.25 ms**.

> The ~20 ms figure quoted in the example docstrings is stale. The real baseline
> for this config is **76 ms** — `state_prompt_mode="fixed"` runs every
> inference at the padded 200-token prompt length, with 3 views. This matters for
> `online_rollout_sync.py`, whose docstring argues sync mode is free because
> inference fits in one 20 ms tick. At 76 ms it does not. (The offline harness
> shows ~116 ms because it also runs lerobot pre/postprocessing per prediction.)

---

## 3. `examples/offline_rollout.py` parity update

Same preprocessing fixes as §1.1 / §1.5, plus:

- **View keys from the checkpoint's `input_features`** in training order,
  replacing the hardcoded `_VIEW_KEYS`; `num_views` derived from it.
- **RTC guidance wired through** — the harness already computed
  `prev_chunk_left_over` and `inference_delay` and discarded both.
- **Re-prediction trigger changed** from `queue.empty()` to
  `qsize() <= requeue_threshold` (default = `execution_horizon`). With the old
  trigger `get_left_over()` always returned a **zero-length** tensor, which
  `_normalize_prev_actions_length` zero-pads — guidance toward an all-zero
  prefix. The RTC path was untestable before this.
- `--rtc_guidance`, `--requeue_threshold`, splice/step-size metrics, and
  `_guided` / `_unguided` output tags.

### 3.1 Relative-action re-anchoring — a real bug this exposed

The first guided run was **worse on every metric**, including a *higher* max
splice than unguided.

Cause: with `use_relative_actions=true`, `ActionQueue`'s leftover is stored in
**absolute** joint space while the policy consumes actions **relative to the
current state**. `RTCInferenceEngine._rtc_loop` calls
`reanchor_relative_rtc_prefix()` before passing the prefix
(`rollout/inference/rtc.py:305-320`); the offline harness did not. Guidance was
steering toward a target expressed in the *previous* chunk's coordinate frame.

`online_rollout.py` was never affected — the RTC engine re-anchors before
calling `predict_action_chunk`. But **any caller that builds a prefix itself
must re-anchor** for relative-action checkpoints.

---

## 4. Offline evaluation

Episode 0, 300 frames, EXP schedule, `requeue_threshold=12` throughout.
`h` = execution_horizon, `g` = max_guidance_weight.

| config | MAE | Cos mean | Cos min | Step size | Splice mean | Splice max |
|---|---|---|---|---|---|---|
| unguided | **3.4327** | 0.9672 | 0.4218 | 0.5909 | 3.6583 | 8.6297 |
| g=10, h=12, *no re-anchor* | 7.0654 | 0.9445 | 0.1079 | 0.7910 | 3.3937 | 11.5707 |
| g=10, h=12 | 5.6896 | 0.9458 | 0.1089 | 0.6198 | 1.4500 | 3.1798 |
| g=2, h=12 | 4.5071 | 0.9544 | 0.1639 | 0.4869 | 1.7932 | 3.7691 |
| g=2, h=6 | 3.6647 | **0.9709** | 0.3798 | **0.4752** | 1.9948 | 3.6305 |
| **g=10, h=6** | 4.1812 | 0.9704 | **0.4410** | 0.5869 | **1.4036** | 3.2081 |

### 4.1 The two knobs are largely orthogonal

- **`execution_horizon` drives GT error.** Dropping 12 → 6 improves MAE at both
  guidance levels (5.69 → 4.18 at g=10; 4.51 → 3.66 at g=2) and restores cosine
  min (0.109 → 0.441; 0.164 → 0.380).
- **`max_guidance_weight` drives splice.** Raising 2 → 10 lowers splice at both
  horizons (1.99 → 1.40 at h=6; 1.79 → 1.45 at h=12).

### 4.1a Re-measured after the §5 latency work

The sweep above ran at ~116 ms inference (`delay=4`). After the tokenizer cache
and thread pinning, inference is ~73 ms and `delay` drops to **3**, so `merge()`
discards one fewer action per chunk and 21 predictions cover the episode instead
of 23:

| | unguided | guided g=10, h=6 |
|---|---|---|
| MAE | **3.0968** | 3.9376 |
| Cosine mean | 0.9653 | **0.9751** |
| Cosine min | 0.3749 | **0.4633** |
| Step size | 0.5438 | 0.5514 |
| Splice mean | 3.3272 | **1.0839** |
| Splice max | 9.2791 | **2.5509** |
| Latency median | 73.2 ms | 74.1 ms |

At the faster operating point guidance now wins on **both** cosine metrics as
well as splice (−67% mean, −73% max) — previously cosine min was the holdout.
Only MAE-vs-GT remains worse, which §4.3 argues is largely harness artifact.

**Why the horizon matters so much:** it must cover roughly the actions consumed
during inference. Native PI0.5 at several hundred ms needs ~12; FlashRT at
~116 ms computes `delay=4`, so 6 sits just above the real requirement. Pinning
12 timesteps when only ~4 are stale over-constrains the chunk against an
observation that is no longer current — which showed up as both GT error and the
bad cosine-min outliers.

**`g=10, h=6` is the best overall point measured**: splice mean −62% and splice
max −63% vs unguided, cosine min **better than unguided** (0.4410 vs 0.4218),
cosine mean better (0.9704 vs 0.9672), at 22% higher MAE.

Suggested sizing rather than a fixed constant:

```
execution_horizon ≈ ceil(latency × fps / interpolation_multiplier) + margin
```

≈ 4–6 here, versus lerobot's default of 12.

### 4.2 The GT is a policy rollout, not a demonstration

`rollout_test_20260718_143829` — `bi_openarm_follower`, 1 episode, 2704 frames
@ 30 fps. lerobot only permits the `rollout_` prefix for policy-deployment
datasets. So MAE measures "do we reproduce *that* rollout's commanded actions",
not "are these good actions".

### 4.3 `--interpolation_multiplier` is a dead argument

Autocorrelation of the GT action's per-frame jerk:

```
peaks (lag, corr): [(3, 0.517), (27, 0.216), (30, 0.241), (42, 0.267), (45, 0.348)]
```

The **lag-3 peak at 0.517** is `--interpolation_multiplier=3` from the
documented CLI. `ActionInterpolator(multiplier=3)`
(`lerobot/utils/action_interpolator.py:24-46`, driven from
`strategies/core.py:63`) pops one policy action every 3 control ticks and emits
2 linear blends between. So the recording consumed policy actions at **10 Hz**
while storing at 30 Hz, and 2 of every 3 GT rows are interpolated.

`offline_rollout.py` parses `--interpolation_multiplier` (line 84) and mentions
it in the docstring (line 30) but **never uses it**. Three consequences, all
inflating absolute MAE for *both* arms roughly equally:

1. **Action consumption 3× too fast** — one policy action per dataset frame, so
   a 30-action chunk burns in 30 frames instead of 90.
2. **Delay computed against the wrong denominator** — `ceil(latency/(1/fps))`
   gives 4; with actions consumed every 3 frames the true figure is ~2, so
   `merge()` discards twice as many actions as it should.
3. **Raw vs blended comparison** — two-thirds of GT rows are interpolations.

### 4.4 Standing limitation

This is **open-loop replay**: observations are the consequence of the *recorded*
policy's actions, so once predictions diverge the observations do not follow.
GT-error metrics are a weak proxy. `splice` is measured at the merge boundary
*within* a single run rather than against GT, which is why it is the most
trustworthy number here.

---

## 5. Latency breakdown and recovery

Measured per prediction on GB10, chunk 30, 3 views, `state_prompt_mode="fixed"`.

### 5.1 Tokenizer reload on the hot path — **fixed**

`_embed_prompt` called `load_paligemma_sentencepiece()` on every invocation, and
that function re-loaded the ~4 MB SentencePiece model from disk each time:
**40.01 ms per call**. Because Pi0.5 renders robot state into the prompt,
`set_prompt` runs on *every* inference, so this landed once per control tick.

Fix: `functools.lru_cache` on the resolved path
(`flash_rt/utils/paligemma_tokenizer.py`). `SentencePieceProcessor` is read-only
after load, so sharing one instance is safe.

| | before | after |
|---|---|---|
| `load_paligemma_sentencepiece()` | 40.01 ms | 0.02 ms (first call 42.8 ms) |
| `set_prompt` | 38.06 ms | **0.23 ms** |
| `model.predict` | 116.06 ms | **76.70 ms** |

**−34% end-to-end for a one-line cache.**

### 5.2 Where the remaining time goes

| stage | ms |
|---|---|
| `prepare_observation_for_inference` (uint8→f32 CHW + 26 MB H2D) | 5.45 |
| lerobot preprocessor | 0.23 |
| image resize + uint8 + D2H | 0.14 |
| `set_prompt` | 0.23 |
| **GPU graph replay** | **74.5** |
| lerobot postprocessor | 0.04 |
| **total** | **~81** |

Graph internals, from `forward()` vs `forward_decode_only()`:

| | ms | share |
|---|---|---|
| full graph | 74.50 | — |
| decoder only | 21.48 | 29% |
| **vision + encoder** | **53.02** | **71%** |

### 5.3 Remaining levers, ranked

1. **`cache_frames=2`** — alternates full and decoder-only inference, skipping
   vision+encoder every other call. Measured basis: average would drop to
   ~48 ms (74.5 / 21.5 alternating), **saving ~26 ms**. Largest remaining lever
   by a wide margin. Costs one frame of visual staleness on alternate ticks.
2. **`vision_pool_factor=2`** and/or **`vision_num_layers < 27`** — vision
   dominates the 53 ms and 3 views × 256 = 768 of the 968 encoder tokens are
   visual. Both are documented RTX knobs; accuracy impact unmeasured here.
3. **`num_steps` 10 → 5** — halves the 21.5 ms decoder, ~10 ms saved. Direct
   flow-matching accuracy tradeoff.
4. ~~Skip the redundant image round trip~~ — **measured and mostly not worth
   it.** The FlashRT-side GPU→CPU→GPU hop (`_extract_flashrt_inputs` D2H +
   `_fill_img_buf` CPU float convert + H2D) is only **0.68 ms**, of which
   ~0.55 ms is recoverable. The 5.45 ms figure quoted here originally was
   `prepare_observation_for_inference`, a *different* function — see §5.4.
5. **`state_prompt_fixed_max_len` on RTX** — the real prompt is **70 tokens**,
   padded to 200 in fixed mode; encoder seq 968 → 838 (**13% fewer tokens**).
   The RTX frontend hardcodes `PI05_STATE_PROMPT_MAX_LEN = 200` and does not
   accept the kwarg (`api.py` forwards it only "if accepted"; only Thor has it).
   Small code change, modest gain.

Note that under RTC the background thread already overlaps inference with
execution, so latency is not directly a control-rate limit. It still matters:
lower latency shrinks `inference_delay`, so `merge()` discards fewer actions per
chunk and a shorter `execution_horizon` suffices — see §4.1.

### 5.4 `prepare_observation_for_inference` — **fixed (online)**

lerobot's host-side observation prep expands uint8 → float32 *before* the H2D,
so it moves 24.6 MiB instead of 6.2 MiB and runs a cache-hostile HWC→CHW
transpose on the CPU:

```
CPU uint8→float32 /255            7.52 ms
CPU permute HWC→CHW .contiguous() 1.40 ms
H2D float32 (pageable)            0.47 ms
```

It is also badly thread-pool sensitive — the same workload measured 5 ms to
21 ms purely as a function of `torch.set_num_threads`.

`_fast_prepare_observation_for_inference` in `examples/online_rollout.py`
uploads uint8 and converts on device. Both engines bind the name at import, so
`_install_fast_observation_prep()` rebinds it in `policies.utils`,
`rollout.inference.rtc` and `rollout.inference.sync`
(`FLASHRT_FAST_OBS_PREP=0` disables).

| | min | median | p90 |
|---|---|---|---|
| stock | 13.4 | 17.7 | 18.7 ms |
| fast | 0.52 | **0.53** | 0.54 ms |

Output matches to within **1 ULP** of float32 (max abs diff 6e-8, from CPU vs
GPU division rounding) — ~65,000× below the uint8 quantization step these
images pass through afterwards. Pinned staging was tried and is *slower*
(~1.2 ms): the extra host memcpy costs more than the 6 MiB transfer saves.

End-to-end the saving is **~5.6 ms**, not 17 — the tick only synchronizes at the
end, so some of the stock CPU work overlaps GPU execution. The p90 tightening
(89.7 → 83.8 ms) is the more valuable part under RTC, where jitter feeds
straight into `inference_delay`.

**Not ported to `offline_rollout.py`**: `LeRobotDataset` already returns float32
CHW tensors, so that harness's `preprocess_frame` is a straight H2D measuring
**0.69 ms**. There is no CPU conversion or transpose to move.

### 5.5 Thread pinning — **done (both scripts)**

`torch.set_num_threads` measurably dominates host-side jitter. Full online tick:

| threads | stock prep | patched prep |
|---|---|---|
| 20 (default) | 87.6 | 82.0 ms |
| 8 | 83.1 | 78.6 |
| 4 | 82.9 | 78.1 |
| 2 | 85.4 | 77.5 |
| **1** | 81.5 | **76.8** |

Online: `FLASHRT_TORCH_THREADS` (opt-in, since `set_num_threads` is
process-global). Offline: `--torch_threads`, default **1**, applied *after* the
dataset is decoded so loading keeps the full pool.

### 5.6 Net result

| | median |
|---|---|
| session start (inferred) | ~126 ms |
| after tokenizer cache | 87.6 ms |
| + GPU observation prep | 82.0 ms |
| + threads=1 | **76.8 ms** |
| GPU graph floor | 74.5 ms |

**−39%**, with the graph now 97% of the tick. `cache_frames=2` (~26 ms) is the
only remaining lever of size.

---

## 6. Recommendations

1. **Do not enable `rtc_guidance` by default on hardware yet.** It is opt-in at
   every layer; the unguided path is unchanged and verified bit-identical.
2. **If enabling: `execution_horizon` 4–6, not lerobot's 12**, sized from
   measured latency. `max_guidance_weight` 10 gives the best smoothness;
   2 trades some of that for lower GT error. No defaults were changed in code —
   `rtc_max_guidance_weight` still matches `RTCConfig`'s 10.0 — because silently
   diverging from lerobot would be worse than documenting this.
3. **Implement `--interpolation_multiplier` in the offline harness** before
   reading further into MAE, then re-run the sweep and re-pick the horizon.
4. **Closed-loop is the real acceptance test.** No offline metric here settles
   whether guidance helps task success.

### 6.1 Known-unfixed

- `online_rollout_sync.py` raises for this checkpoint, leaking a torqued robot
  (§1.3); `build_rollout_context` sits outside try/finally in both scripts.
- Missing-camera `KeyError` instead of PI0.5's mask-padding (§1.6).
- Warmup with the robot connected and torqued (§1.7).
- `Indexes diff is not equal to real delay` fires every prediction offline — the
  synchronous harness captures `idx_before` and merges without consuming `delay`
  actions in between. Log noise (`_check_and_resolve_delays` returns
  `real_delay` regardless), but the offline delay bookkeeping is not a faithful
  simulation of the threaded engine.
- The identity-Jacobian approximation is weakest where guidance is strongest
  (`guidance_weight` saturates at `max_guidance_weight` on the first denoise
  step). `flash_rt/subgraphs/pi05/rtc_vjp_guided.py` still defines the exact-VJP
  contract; unimplemented.
