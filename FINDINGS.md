# FINDINGS.md — Phase 0 Inventory

Generated from reading installed source, not from memory.

---

## 0.1 LeRobot side

**Installation path:** `/home/videron/miniconda3/lib/python3.13/site-packages/lerobot`

### PreTrainedPolicy abstract interface (`lerobot/policies/pretrained.py`)

Abstract methods (all must be implemented by subclasses):
- `get_optim_params(self) -> dict`
- `reset(self)` — clears action queue and other internal state
- `forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]` — training loss
- `predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor`
- `select_action(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor`

`ActionSelectKwargs` in PI05 is `{inference_delay, prev_chunk_left_over, execution_horizon}` (different from the base class's `{noise}` — PI05 overrides with its own TypedDict).

### PI05Policy (`lerobot/policies/pi05/modeling_pi05.py`)

**`reset()`**: Resets `_action_queue = deque(maxlen=config.n_action_steps)` and `_queues = {ACTION: deque(maxlen=n_action_steps)}`.

**`select_action(batch)`**: If queue is empty, calls `predict_action_chunk(batch)[:, :n_action_steps]`, transposes to `(n_action_steps, B, dim)`, extends queue. Pops and returns one step.

**`predict_action_chunk(batch)`**:
```python
images, img_masks = self._preprocess_images(batch)
tokens, masks = batch["observation.language_tokens"], batch["observation.language_attention_mask"]
actions = self.model.sample_actions(images, img_masks, tokens, masks, **kwargs)
original_action_dim = self.config.output_features["action"].shape[0]   # 16
actions = actions[:, :, :original_action_dim]   # slice (30,32) → (30,16)
return actions  # shape (B, 30, 16)
```

State is NOT passed to `predict_action_chunk` directly. It is embedded in the tokenized prompt by the preprocessor (`Pi05PrepareStateTokenizerProcessorStep`) before `batch` reaches the policy.

**`_preprocess_images(batch)`**:
- Iterates `self.config.image_features` in insertion order (= JSON config order).
- Input: `float32 [B, C, H, W]` in `[0, 1]` range from LeRobot dataloader.
- Converts to `[B, H, W, C]`, applies `resize_with_pad_torch(img, 224, 224)`, then rescales to `[-1, 1]`, converts back to `[B, C, H, W]`.
- Missing cameras filled with `-1` padding + zero mask.
- Returns `(images: list[Tensor], img_masks: list[BoolTensor])`.

### Normalization architecture

Normalization lives **outside** the policy in `make_pi05_pre_post_processors` (`lerobot/policies/pi05/processor_pi05.py`). The policy itself does not contain `Normalize`/`Unnormalize` submodules.

**Preprocessor pipeline** (in order):
1. `RenameObservationsProcessorStep({})` — no-op rename, maintains shape
2. `AddBatchDimensionProcessorStep()` — adds B=1 dim
3. `RelativeActionsProcessorStep(enabled=True, exclude_joints=["left_gripper","right_gripper"])`
4. `NormalizerProcessorStep(features={inputs+outputs}, norm_map={"VISUAL":"IDENTITY","STATE":"QUANTILES","ACTION":"QUANTILES"}, stats=dataset_stats)`
5. `Pi05PrepareStateTokenizerProcessorStep(max_state_dim=32)` — digitizes normalized state into 256 bins, builds full prompt string `"Task: {task}, State: {bins};\nAction: "`
6. `TokenizerProcessorStep("google/paligemma-3b-pt-224", max_length=200, padding="max_length")`
7. `DeviceProcessorStep(device="cuda")`

**Postprocessor pipeline** (in order):
1. `UnnormalizerProcessorStep(features=output_features, norm_map={"ACTION":"QUANTILES"}, stats=dataset_stats)`
2. `AbsoluteActionsProcessorStep(enabled=True, relative_step=<the RelativeActionsStep above>)` — converts relative → absolute using the state captured at step 3
3. `DeviceProcessorStep("cpu")`

**Key consequence for our integration:** By the time `predict_action_chunk` is called, state is already baked into `tokens`/`masks`. The FlashRT wrapper bypasses the tokenizer step and instead passes the raw normalized state array to FlashRT's own `set_prompt(prompt_text, state=state_array)` — which calls `format_pi05_prompt` internally with the same digitize logic.

### Camera key order

Config `input_features` JSON insertion order (= `image_features` dict iteration order = model sees them in this order):

```
CANONICAL_VIEW_ORDER = [
    "observation.images.left_wrist",   # index 0 → FlashRT images[0]
    "observation.images.right_wrist",  # index 1 → FlashRT images[1]
    "observation.images.base",         # index 2 → FlashRT images[2]
]
```

`_preprocess_images` iterates `self.config.image_features` which is a dict comprehension over `input_features` preserving JSON insertion order. This is the authoritative order — the model learned whichever order the training loop produced, and that order is the JSON insertion order of the config.

**Native resolutions (before resize_with_pad to 224×224):**
- `left_wrist`: 720×1280 (landscape, 16:9)
- `right_wrist`: 720×1280 (landscape, 16:9)
- `base`: 480×640 (landscape, 4:3)

All three resize to 224×224 via `resize_with_pad_torch` (aspect-ratio-preserving, black-padded). The padding for 16:9 → 224×224 differs from 4:3 → 224×224; this is fine because the same resize was applied during training. The critical thing is that view N from live robot goes to index N.

---

## 0.2 Trained checkpoint

**Path:** `/home/videron/Desktop/openarm/outputs/train/openarm_folding_high_quality_60k/checkpoints/060000/pretrained_model/`

| Field | Value |
|---|---|
| `chunk_size` | **30** ✓ |
| `n_action_steps` | 30 |
| `max_state_dim` | 32 |
| `max_action_dim` | **32** ✓ |
| `normalization_mapping` | `{"VISUAL":"IDENTITY","STATE":"QUANTILES","ACTION":"QUANTILES"}` |
| `use_relative_actions` | `true` |
| `relative_exclude_joints` | `["left_gripper","right_gripper"]` |
| `output action shape` | `(16,)` — real DOF, padded to 32 for model |
| `paligemma_variant` | `gemma_2b` |
| `action_expert_variant` | `gemma_300m` |

**Action projection weight shapes:**
```
model.action_in_proj.weight   [1024, 32]   — 32-dim padded action → 1024-dim decoder hidden
model.action_in_proj.bias     [1024]
model.action_out_proj.weight  [32, 1024]   — 1024-dim → 32-dim output
model.action_out_proj.bias    [32]
```

**Key layout:** All weights have the `model.` prefix (lerobot HF release format). FlashRT's `_autodetect_strip_prefix` detects the `model.` prefix automatically and strips it before spec lookups — no conversion script needed.

**Norm stats:** The checkpoint directory contains:
- `policy_preprocessor_step_3_normalizer_processor.safetensors`
- `policy_postprocessor_step_0_unnormalizer_processor.safetensors`

FlashRT's `_load_norm_stats` calls `load_norm_stats(candidates, checkpoint_dir=checkpoint_dir)` which invokes `_find_lerobot_policy_stats(ckpt)` as a fallback — it will find these safetensors automatically. Norm stats load will succeed.

Since our integration returns raw **normalized** actions (no FlashRT-side unnormalize), the loaded norm stats are dead code at runtime but their presence satisfies the strict loader.

---

## 0.3 FlashRT side

### Hardware path

Will be determined at first `load_model()` call. Expected to be `thor` on Jetson Thor or `rtx_sm120`/`rtx_sm89` on RTX. Both paths are patched for `chunk_size=30` and full 32-dim output.

**Frontend class by hardware:**
- Thor SM110 → `Pi05TorchFrontendThor` (pi05_thor.py)
- RTX SM120/SM89 → `Pi05TorchFrontendRtxRtx` (pi05_rtx.py)
- Orin SM87 → `Pi05TorchFrontendRtxRtx` (pi05_rtx.py, experimental)

### Weight loading

The LeRobot checkpoint is in the lerobot HF release format (`model.` prefix). `_autodetect_strip_prefix` handles this transparently by detecting `model.` and stripping before spec lookups. No separate openpi-conversion script needed.

### `predict()` return shape

After Phase 0 patches: `(30, 32)` float32 normalized. Confirmed by weight shapes: `action_out_proj` outputs 32 dims, buffers are `(Sa, 32)`.

### Sequence-length headroom for 3 views (Thor path)

```
nv = 3
S_sig = nv * 256 = 768         # SigLIP vision tokens
Se_max = nv * 256 + 256 = 1024  # encoder budget: 768 image + 256 prompt+state
Sa = 30                          # action chunk (patched)
total_keys_max = Se_max + Sa = 1054
RoPE table = torch.arange(1200)  # margin = 1200 - 1054 = 146
```

Three views fit with 146-position margin. A fourth view would require 256 more positions (1310 > 1200) and would silently index past the RoPE table. **Do not add a fourth camera without rebuilding the RoPE table.**

### Prompt length headroom

`PI05_STATE_PROMPT_MAX_LEN = 200` tokens (the max the tokenizer+prompt occupies in the 256-slot prompt budget).

Full prompt format: `"Task: {task}, State: {bins};\nAction: "` where `{bins}` is 32 space-separated integers. Character length ≈182 chars for a 32-token state + medium task description. Estimated token count: ~120 tokens (rough estimate without transformers in this env). Headroom against 200-token cap: ~80 tokens for the task description. The `tokenizer_max_length: 200` config field confirms this budget.

Watch for: a very long task description crowding the 200-token cap. Keep task strings under ~60 words.

### Norm stats loading

Thor's `_load_norm_stats` uses `lerobot_candidates` + `_find_lerobot_policy_stats` fallback. The checkpoint directory contains `policy_*normalizer_processor.safetensors`, so loading succeeds.

RTX's `_load_norm_stats` (in pi05_rtx.py) uses `pi05_candidates` (openpi paths) first. If those fail and the RTX path is used, the RTX loader will reach its `FileNotFoundError` before the lerobot fallback. **Action if RTX is the resolved hardware:** add `lerobot_candidates(checkpoint_dir)` to the RTX candidates list in `pi05_rtx.py`, or point `OPENPI_ASSETS_DIR` to the checkpoint parent.

### VRAM notes (fp16)

`_enc_logits` shape: `(Se_max * NHe, total_keys_max)` = `(1024 * 8, 1054)` = ~17 MB. Total encoder buffer increase nv=2→3: ~8 MB. Not a concern on discrete GPU; measure on Orin/Thor before assuming.

### Cache invalidation

Calibration cache key includes `Se` (sequence length); changing nv from 2 to 3 changes Se_max and will cache-miss cleanly. JAX weight cache includes `num_views` and also misses cleanly. Clear `~/.flash_rt/` anyway before first nv=3 run.

---

## Cross-boundary contract (canonical reference)

| Direction | Field | Type | Space |
|---|---|---|---|
| → FlashRT | images (3×) | list of `np.ndarray (224,224,3)` uint8 | raw pixels |
| → FlashRT | state | `np.ndarray (32,)` float32 | **normalized** [-1,1], padded to 32 |
| → FlashRT | prompt | str | raw task text (no state; FlashRT formats internally) |
| ← LeRobot | actions | `np.ndarray (30,32)` float32 | **normalized** (LeRobot postprocessor unnormalizes) |

```python
CANONICAL_VIEW_ORDER = [
    "observation.images.left_wrist",   # index 0
    "observation.images.right_wrist",  # index 1
    "observation.images.base",         # index 2
]
```

This list is the single source of truth. Both the server (validates incoming list order) and the client (serializes named dict → ordered list) reference it from one location.

---

## View-permutation fixture table (Phase 4.5 placeholder)

| Permutation | Mean cosine | Notes |
|---|---|---|
| 0,1,2 (canonical) | — | to be filled |
| 0,2,1 | — | |
| 1,0,2 | — | |
| 1,2,0 | — | |
| 2,0,1 | — | |
| 2,1,0 | — | |
