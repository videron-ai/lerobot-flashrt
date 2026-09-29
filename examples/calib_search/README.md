# FP8 calibration search — `openarm_folding_gen3_level2_multitask_50k`

Goal: pick the FP8 activation-calibration frames that minimise action error across
the model's training dataset
(`openarm_bimanual_shirt_folding_gen3_345_episodes_fold_unfold_home_rightfirst`:
345 episodes, 455k frames. Tasks: fold 244, return-home 53, unfold 48).

**Result:** `/nas/models_fast/openarm_folding_gen3_level2_multitask_50k/flashrt_calib_32.npz`
(host: `/mnt/nas/models_fast/...`). It has 32 frames from 29 episodes, all three tasks
(15 fold / 12 unfold / 5 home), spread from 4% to 96% of episode progress.
It is in the exact `online_rollout.py` cache format. Use it with:

```bash
FLASHRT_CALIB_FRAMES=32    # picks up <ckpt>/flashrt_calib_32.npz automatically
# or: FLASHRT_CALIB_CACHE=/nas/models_fast/openarm_folding_gen3_level2_multitask_50k/flashrt_calib_32.npz
```

`FLASHRT_CALIB_PERCENTILE` stays at its default of 99.9. Calibrating with the fold task
prompt, which is what `_calibrate_flashrt` does, is fine (see the prompt section below).

## Final numbers (deployed path, like for like)

Each file below was loaded into a fresh FP8 model and calibrated exactly like
`online_rollout._calibrate_flashrt` does it: `set_prompt(fold, state=obs[0])`, then
`calibrate(obs, 99.9)`. Each was then scored on 640 held-out frames.

| cache | GT MAE | GT MAE (first 10 steps) | FP8 vs FP16 | fold | home | unfold |
|---|---|---|---|---|---|---|
| old `/openarm/gen3_level2/flashrt_calib_16.npz` | 0.9661 | 0.7465 | 0.3201 | 0.8597 | 1.1486 | 1.1383 |
| **new `flashrt_calib_32.npz`** | **0.9429** | **0.7336** | **0.2772** | **0.8389** | 1.1476 | **1.0846** |
| Δ, with 95% CI from an episode bootstrap | −0.023 [−0.034, −0.014] | | −0.043 [−0.055, −0.033] | | | |

For context, a single-frame (lazy) calibration scores about 1.12 GT MAE and 0.55 FP8-vs-FP16.

## Method

Scripts are in this directory. Outputs go to `/flashrt/rollout_outputs_calib_search/`.

1. **`build_pool.py`**: sample frames, stratified by task and split by *episode*.
   - 1,200 candidate frames: 100 episodes (60 fold, 20 home, 20 unfold) × 12 frames.
   - 640 eval frames: 64 other episodes (40 fold, 12 home, 12 unfold) × 10 frames.
   - Frame positions are uniform with jitter. Each frame is stored exactly as
     `predict()` receives it (same lerobot preprocessor, `resize_with_pad`, uint8
     rounding), along with the raw state and the 30-step ground-truth action chunk.
2. **`collect.py --mode amax`**: the per-GEMM dynamic amax for every frame, 970 FP8
   scales per frame (vision, encoder, and decoder per denoise step). Frames from
   the old caches are included as baselines.
3. **`collect.py --mode ref`**: an FP16 (unquantized) reference chunk for each eval
   frame, using a fixed noise seed per frame.
4. **`search.py`**: all FP8 scales are read through device pointers when the graph
   replays, so a candidate calibration can be swapped into one captured model by
   uploading new values. That costs no reload and no recapture. Each candidate is
   scored on the 640 eval frames with the same seeds.
   - Metrics are joint-space MAE vs GT (the objective) and MAE vs FP16 (pure
     quantization error, much lower variance).
   - Two runs of the same candidate were bit-identical.
5. **`analyze.py`**: paired 95% CI against a reference candidate, resampling whole
   episodes.
6. **`prompt_effect.py`**: the same frames' amax under each of the three task prompts.
7. **`validate.py`**: writes the chosen frames as a cache file, then re-scores that
   file through a fresh model and the real `calibrate()` path.

Everything is logged in `search_results.jsonl`, with per-frame errors in
`search_per_frame.npz`.

## Findings

- **Coverage beats count.** Any calibration drawn across the whole dataset beat
  the old caches, which came from a single fold episode. The plateau is flat:
  - Pool-wide per-GEMM quantiles q98–q100, random 16/64-frame draws, and greedy
    sets all land at GT MAE 0.934–0.942 and FP8-vs-FP16 0.266–0.273.
  - Those differences are within noise.
  - Lower quantiles clip too much: q75 scores 0.960 / 0.311 and q50 scores 0.997 / 0.360.
  - Taking a (near) max over diverse frames is the right reduction.
- **Most of the per-frame spread is in the decoder** (log-std about 0.13) and is
  driven by the denoising noise. Vision (about 0.05) and encoder (about 0.01) barely
  move. Task and episode phase shift the median scales by under 1%.
- **The final set.** A greedy selection picks K frames whose deployed p99.9
  reduction best matches the pool-wide q100 vector, with at least one frame per task.
  - With K=32 it had the best GT MAE in the swap search (0.9345, Δ −0.011
    [−0.018, −0.003] vs the old cache).
  - K=16 was statistically equivalent (0.9389 / 0.2685).
- **Prompt: calibrating per task is not needed.** For the same 120 frames, scale
  vectors built with each frame's own prompt, the fold prompt only, or all three
  prompts scored 0.9431 / 0.9388 / 0.9395 GT and 0.2716 / 0.2697 / 0.2686 FP8-vs-FP16.
  All pairwise CIs include 0.
  - A few individual scales do move up to about 2× with the prompt.
  - After the max-reduction the average change (0.03 in log terms) is smaller than
    the frame-sampling spread (0.085).
  - The scales are shared by all prompts at runtime anyway: `calibrate()` runs once
    per process.
- **Gripper joints benefit most.** The GT MAE for j7 / j15 is 2.94 / 3.90 with a
  single frame, 2.30 / 2.65 with the old cache, and 2.26 / 2.60 with the pool q100 vector.
- **The remaining FP8-vs-FP16 gap (about 0.27) is a quantization floor.** No choice of
  calibration dataset closes it.
  - `calibrate()` warns about an outlier-dominated scale at
    `encoder_ffn_down_w_16`, about 1000× the median.
  - Per-tensor FP8 on the few outlier layers is the likely limiter. A mixed-precision
    fallback for those layers is the next lever, not more data.

## Caveats

- **Cross-process variation.** Scales computed in different processes differ by up
  to about 9% on some decoder down-projection GEMMs. The likely cause is cuBLASLt
  autotune picking different algorithms per process; this wasn't isolated.
  - Within one process, results are bit-exact.
  - All swap-search candidates were compared inside one process.
  - Re-scoring one candidate in a fresh process moved GT MAE by only 0.0002.
- **Deployed numbers sit a little above the swap numbers** (0.943 vs 0.935). Deployment
  calibrates every frame with the fold prompt and frame 0's *state* tokens, and
  the scales vary slightly between processes. The like-for-like deployed
  comparison in the table above is the one that counts.
- **Offline, open-loop metric.** These are single-chunk errors on training-distribution
  frames, not closed-loop success. The old caches' states were normalized with an
  older checkpoint's stats, but states only feed the prompt tokens here.

## Reproduce (inside the container, from `/flashrt`)

```bash
C=/nas/models_fast/openarm_folding_gen3_level2_multitask_50k
D=/nas/datasets_archive/openarm_bimanual_shirt_folding_gen3_345_episodes_fold_unfold_home_rightfirst
P=rollout_outputs_calib_search/pool.npz
python examples/calib_search/build_pool.py --ckpt $C --dataset_root $D --out $P          # ~3 min
python examples/calib_search/collect.py --mode amax --ckpt $C --pool $P \
    --baseline_npz /openarm/gen3_level2/flashrt_calib_16.npz /openarm/gen3/flashrt_calib_16.npz  # ~11 min
python examples/calib_search/collect.py --mode ref  --ckpt $C --pool $P                  # ~6 min
python examples/calib_search/search.py --stage sweep  --ckpt $C --pool $P                # ~3 min / candidate
python examples/calib_search/search.py --stage select --ckpt $C --pool $P --target_q 100 --ks 16,32
python examples/calib_search/analyze.py
python examples/calib_search/validate.py --ckpt $C --pool $P --write <out.npz> --idx_from_result greedy_q100_N32
python examples/calib_search/validate.py --ckpt $C --pool $P --calib_npz <out.npz>
```
