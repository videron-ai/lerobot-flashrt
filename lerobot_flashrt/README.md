# lerobot_flashrt — FlashRT backend for LeRobot π₀.₅

High-performance inference for LeRobot π₀.₅ policies using FlashRT as the model backend. Supports two deployment topologies and two rollout modes.

---

## Deployment topologies

### HTTP variant — FlashRT as a network server

```
┌─────────────────────────────────────┐     HTTP      ┌──────────────────────────────┐
│          LeRobot host               │  ──────────►  │      FlashRT server          │
│                                     │               │                              │
│  FlashRTPI05Policy                  │  /predict     │  serving/lerobot_host/       │
│    └─ predict_action_chunk()        │  ◄──────────  │    server.py (FastAPI)       │
│         └─ FlashRTClient            │               │      └─ flash_rt.load_model  │
│                                     │               │                              │
│  Preprocessor  (LeRobot, local GPU) │               │  CUDA graph replay  ≤ 20 ms  │
│  Postprocessor (LeRobot, local GPU) │               │  FP8 quantized               │
└─────────────────────────────────────┘               └──────────────────────────────┘
```

Use when FlashRT and LeRobot run in separate processes or containers.

### Local (in-process) variant — FlashRT in the same process

```
┌──────────────────────────────────────────────────────────────┐
│                     Single process                           │
│                                                              │
│  LocalFlashRTPI05Policy                                      │
│    └─ predict_action_chunk()                                 │
│         └─ flash_rt.VLAModel.predict()  (direct call)        │
│                                                              │
│  Preprocessor / Postprocessor  (lerobot)                     │
│  ActionQueue / RTCInferenceEngine  (lerobot)                 │
│  Robot / cameras  (lerobot)                                  │
└──────────────────────────────────────────────────────────────┘
```

Use when LeRobot and FlashRT coexist in the same Docker container — no HTTP overhead, simpler deployment.

---

## Components

### `FlashRTClient` — `client.py`

HTTP client for the inference server. Validates camera view order on connect and enforces `CANONICAL_VIEW_ORDER` throughout.

```python
from lerobot_flashrt.client import FlashRTClient, CANONICAL_VIEW_ORDER

client = FlashRTClient("http://localhost:8000", timeout_s=10.0)
actions = client.predict(
    images={k: img_hwc_uint8 for k in CANONICAL_VIEW_ORDER},
    prompt="fold the fabric",
    state=normalized_state,   # (32,) float32
)
# actions: np.ndarray (30, 32) normalized
```

### `FlashRTPI05Policy` — `policy.py`

PI05Policy subclass for the HTTP topology. Overrides only `predict_action_chunk()`; all other methods (`reset`, `select_action`, training forward pass) are inherited unchanged.

```python
from lerobot_flashrt import make_flashrt_policy

policy, preprocessor, postprocessor = make_flashrt_policy(
    checkpoint="/path/to/pretrained_model",
    server_endpoint="http://localhost:8000",
)
batch  = preprocessor(observation)
chunk  = policy.predict_action_chunk(batch)   # (1, 30, 32) normalized
action = postprocessor(chunk)                  # unnormalized robot actions
```

### `LocalFlashRTPI05Policy` — `local_policy.py`

PI05Policy subclass for the in-process topology. Calls `flash_rt.VLAModel.predict()` directly — no HTTP, no server.

```python
from lerobot_flashrt import make_local_flashrt_policy

policy, preprocessor, postprocessor = make_local_flashrt_policy(
    checkpoint="/path/to/pretrained_model",
    device="cuda",
    action_horizon=30,
    num_views=3,
    state_prompt_mode="fixed",   # one CUDA graph; no re-capture during rollout
)
batch  = preprocessor(observation)
chunk  = policy.predict_action_chunk(batch)   # (1, 30, 32) normalized
action = postprocessor(chunk)
```

### `make_flashrt_policy` / `make_local_flashrt_policy` — `factory.py`

Factory functions that wire up the checkpoint config, FlashRT model, and lerobot pre/post-processors in one call. Both read `output_features.action.shape[0]` from `config.json` to compute `state_in_prompt_dim` — ensuring the discretized state bins in the FlashRT prompt match exactly what LeRobot's `Pi05PrepareStateTokenizerProcessorStep` produces.

### Inference server — `serving/lerobot_host/server.py`

FastAPI server for the HTTP topology. Reads the checkpoint's action dim at startup and truncates state to that many dims before building the prompt.

```bash
python serving/lerobot_host/server.py \
    --checkpoint /path/to/pretrained_model \
    --action-horizon 30 \
    --num-views 3 \
    --state-prompt-mode fixed \
    --port 8000
```

| Route | Description |
|---|---|
| `GET /health` | Status, view order, chunk size |
| `POST /predict` | One inference step → `(Sa, 32)` normalized actions |
| `POST /warmup` | FP8 calibration + prompt-bucket warming |

---

## Rollout scripts

### Offline rollout — `examples/offline_rollout.py`

Runs FlashRT inference over a recorded LeRobot dataset instead of a live robot. Produces per-joint plots, an error heatmap, and an inference latency histogram.

Uses lerobot's own `ActionQueue`, `LatencyTracker`, and `RTCConfig` directly — the RTC sliding window, inference-delay compensation, and `prev_chunk_left_over` bookkeeping all come from lerobot, not a reimplementation.

```bash
python examples/offline_rollout.py \
    --ckpt /path/to/pretrained_model \
    --dataset videron/rollout_test_20260718_143829 \
    --task "Fold the T-shirt properly" \
    --execution_horizon 12 \
    --max_guidance_weight 10.0 \
    --prefix_attention_schedule EXP \
    --out_dir ./rollout_outputs
```

**Output files** (in `--out_dir`):
| File | Contents |
|---|---|
| `ep000_per_joint.png` | GT vs FlashRT per joint over time, re-prediction events marked |
| `ep000_error_latency.png` | `\|predicted − GT\|` heatmap + inference latency histogram |
| `ep000_rtc_timeline.png` | Mean action with RTC re-prediction event markers |
| `ep000_results.npz` | Raw arrays: `predicted_actions`, `gt_actions`, `chunk_at`, `latencies_ms` |

**RTC parameters** (mirror `lerobot-rollout` CLI):

| Flag | Default | lerobot-rollout equivalent |
|---|---|---|
| `--execution_horizon` | 12 | `--inference.rtc.execution_horizon` |
| `--max_guidance_weight` | 10.0 | `--inference.rtc.max_guidance_weight` |
| `--prefix_attention_schedule` | EXP | `--inference.rtc.prefix_attention_schedule` |
| `--action_horizon` | 30 | chunk size for `flash_rt.load_model()` |

### Online rollout — `examples/online_rollout.py`

Drop-in replacement for `lerobot-rollout`. Calls `build_rollout_context()` exactly as lerobot does — same robot connection, preprocessors, postprocessors, `ActionQueue`, `RTCInferenceEngine`, and rollout strategy — then monkey-patches `predict_action_chunk` on the live policy object to route inference through FlashRT instead of the PI05 VLA forward pass.

A 20-iteration dummy warmup runs before the robot loop so CUDA graph capture latency is paid upfront rather than on the first live control tick.

```bash
python examples/online_rollout.py \
    --config_path=/openarm/rollout.yaml \
    --strategy.type=base \
    --policy.path=/path/to/pretrained_model \
    --task="Fold the T-shirt properly" \
    --interpolation_multiplier=3 \
    --inference.type=rtc \
    --inference.rtc.execution_horizon=12 \
    --inference.rtc.max_guidance_weight=10.0 \
    --inference.rtc.prefix_attention_schedule=EXP \
    --use_torch_compile=False \
    --duration=0
```

All flags are identical to `lerobot-rollout`. The only behavioral difference is that model inference goes through FlashRT's FP8 CUDA-graph replay (~20 ms) instead of the native PI05 VLA forward pass.

---

## Test suite

### Client unit tests
```bash
pytest tests/test_client.py -v
```
Mock-server tests: view-order enforcement, state dtype/shape validation, warmup schema, error paths. No GPU required.

### Parity gate (HTTP variant)
```bash
FLASHRT_SERVER=http://localhost:8000 \
LEROBOT_CKPT=/path/to/pretrained_model \
pytest tests/test_parity.py -v -s
```
Compares raw normalized `(30, 16)` chunks between PyTorch and FlashRT on the same observation with identical ODE seed. Gate: **mean cosine ≥ 0.99, no single frame below 0.98**. Prints per-frame summary and per-timestep detail for the first 10 frames.

### Parity gate (local variant)
```bash
LEROBOT_CKPT=/path/to/pretrained_model \
pytest tests/test_local_parity.py -v -s
```
Same gate as above, no server required. Optionally cross-checks local vs HTTP if `FLASHRT_SERVER` is set.

### Latency gate (HTTP variant)
```bash
FLASHRT_SERVER=http://localhost:8000 \
pytest tests/test_latency.py -v -s
```
200 predict calls after a settle window must all fall within **3× the warm median**. Reports median, mean, p95, p99, and a latency histogram.

### Latency gate (local variant)
```bash
LEROBOT_CKPT=/path/to/pretrained_model \
pytest tests/test_local_latency.py -v -s
```
Same gate as above, in-process. Pre-generates all observation batches before timing to exclude batch construction from measurements. Settle window discards CUDA graph capture calls.

| Test | Requires | Gate |
|---|---|---|
| `test_client.py` | nothing | unit |
| `test_parity.py` | server + GPU | cosine ≥ 0.99 |
| `test_local_parity.py` | GPU | cosine ≥ 0.99 |
| `test_latency.py` | server + GPU | all calls ≤ 3× median |
| `test_local_latency.py` | GPU | all calls ≤ 3× median |

---

## Key design decisions

**Raw normalized actions.** Both the HTTP server and `LocalFlashRTPI05Policy` return `(Sa, 32)` actions in LeRobot's normalized space. `unnormalize_actions` has been removed from the FlashRT frontends (`pi05_rtx`, `pi05_thor`). The LeRobot postprocessor handles unnormalization and relative→absolute conversion, exactly as it does for native inference.

**`state_in_prompt_dim` alignment.** π₀.₅ encodes state as discretized language tokens: `"Task: …, State: 127 128 …;\nAction: "`. Both the HTTP server and the local policy read `output_features.action.shape[0]` from `config.json` and pass only that many dims to the tokenizer. For a 16-DOF checkpoint this produces 16 bins — matching what `Pi05PrepareStateTokenizerProcessorStep` produces. Passing zero-padded 32-dim state would generate spurious extra bins and produce near-zero cosine similarity versus the PyTorch reference.

**Fixed state-prompt mode.** `state_prompt_mode="fixed"` captures one CUDA graph at the padded maximum prompt length. This eliminates re-capture during rollouts when state values change tick to tick. Use `exact` only when you need minimum peak latency and are willing to pay a re-capture on state-length changes.

**Online rollout via monkey-patch.** `online_rollout.py` patches `predict_action_chunk` as a bound instance method after `build_rollout_context()` returns. This means lerobot's robot driver, preprocessors, postprocessors, `ActionQueue`, `LatencyTracker`, `RTCInferenceEngine`, and `BaseStrategy` run completely unmodified — FlashRT replaces only the model forward pass.

**Offline RTC fidelity.** `offline_rollout.py` uses lerobot's `ActionQueue`, `LatencyTracker`, and `_normalize_prev_actions_length` directly. Inference latency is measured and fed into `ActionQueue.merge()` as `new_delay`, so the offline simulation applies the same inference-delay skip that the live RTC thread applies — predicting what the real robot would have experienced.

**20-iteration warmup.** The online rollout runs 20 dummy `model.predict()` calls with zero images and state before connecting to the robot. FlashRT's CUDA graph capture can take 1–60 s on the first call; doing this upfront ensures the first live control tick returns in ≤ 20 ms.
