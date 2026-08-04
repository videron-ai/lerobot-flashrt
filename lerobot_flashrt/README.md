# lerobot_flashrt — FlashRT backend for LeRobot π₀.₅

Drop-in replacement for LeRobot's local π₀.₅ inference. Routes
`predict_action_chunk()` through a FlashRT GPU server over HTTP, leaving
all action-queue management, preprocessing, and postprocessing in the
LeRobot environment.

---

## Architecture

```
┌─────────────────────────────────────┐     HTTP      ┌──────────────────────────────┐
│          LeRobot host               │  ──────────►  │      FlashRT server          │
│                                     │               │                              │
│  FlashRTPI05Policy                  │  /predict     │  serving/lerobot_host/       │
│    └─ predict_action_chunk()        │  ◄──────────  │    server.py (FastAPI)       │
│         └─ FlashRTClient            │               │      └─ flash_rt.load_model  │
│              └─ /predict, /warmup   │               │           └─ Pi05Frontend    │
│                                     │               │                              │
│  Preprocessor  (LeRobot, local GPU) │               │  CUDA graph replay  ≤ 20 ms  │
│  Postprocessor (LeRobot, local GPU) │               │  FP8 quantized               │
└─────────────────────────────────────┘               └──────────────────────────────┘
```

The server holds the model and owns the CUDA graph. The LeRobot host owns
preprocessing (image resize, state normalization, tokenization) and
postprocessing (unnormalization, relative → absolute actions). The server
returns raw **normalized** `(Sa, 32)` actions; the postprocessor converts
them before they reach the robot.

---

## Components

### `FlashRTClient` — `client.py`

HTTP client for the inference server.

```python
from lerobot_flashrt.client import FlashRTClient, CANONICAL_VIEW_ORDER

client = FlashRTClient("http://localhost:8000", timeout_s=10.0)

actions = client.predict(
    images={
        "observation.images.left_wrist":  left_img,   # (224,224,3) uint8
        "observation.images.right_wrist": right_img,
        "observation.images.base":        base_img,
    },
    prompt="fold the fabric",
    state=normalized_state,   # (32,) float32, LeRobot preprocessor output
)
# actions: np.ndarray (30, 32) raw normalized
```

`CANONICAL_VIEW_ORDER` is the single source of truth for camera ordering.
The client validates view order against the server on connect and raises if
they disagree — a missing or swapped camera raises rather than silently
producing wrong actions.

### `FlashRTPI05Policy` — `policy.py`

Subclass of `PI05Policy` that overrides only `predict_action_chunk()`.
Everything else — `reset()`, `select_action()`, action queue, `forward()`
training pass, `save_pretrained()` — is inherited unchanged.

```python
from lerobot_flashrt import make_flashrt_policy

policy, preprocessor, postprocessor = make_flashrt_policy(
    checkpoint="/path/to/checkpoint",
    server_endpoint="http://localhost:8000",
    device="cpu",
)

# Drop-in for PI05Policy in a rollout loop:
batch = preprocessor(observation)
action = policy.select_action(batch)
action = postprocessor(action)
```

### `make_flashrt_policy` — `factory.py`

Loads the checkpoint config, builds the preprocessor and postprocessor
pipelines from `make_pi05_pre_post_processors`, constructs the
`FlashRTPI05Policy`, and validates the server connection.

### Inference server — `../serving/lerobot_host/server.py`

FastAPI server that loads the model at startup and serves it over HTTP.

```bash
python serving/lerobot_host/server.py \
    --checkpoint /path/to/lerobot_pi05_checkpoint \
    --action-horizon 30 \
    --num-views 3 \
    --view-order observation.images.left_wrist \
                 observation.images.right_wrist \
                 observation.images.base \
    --state-prompt-mode fixed \
    --port 8000
```

**Endpoints:**

| Route | Description |
|---|---|
| `GET /health` | Server status, view order, frontend class, chunk size |
| `POST /predict` | Run one inference step; returns `(Sa, 32)` normalized actions |
| `POST /warmup` | FP8 calibration + state-prompt bucket warming; call once before rollout |

**State-in-prompt alignment.** π₀.₅ encodes proprioceptive state as
discretized language tokens in the prompt
(`"Task: …, State: 127 128 …;\nAction: "`). At startup the server reads
`output_features.action.shape[0]` from the checkpoint's `config.json` and
passes only that many dims to the tokenizer — matching the exact bin count
that LeRobot's `Pi05PrepareStateTokenizerProcessorStep` produces. Without
this, zero-padded dims would produce spurious extra bins and the server's
prompt would diverge from the PyTorch reference.

---

## Quick start

**1. Start the server** (in the FlashRT container / environment):

```bash
python serving/lerobot_host/server.py \
    --checkpoint /path/to/pretrained_model \
    --action-horizon 30 --num-views 3 --state-prompt-mode fixed
```

**2. Warmup** (once per session, before the rollout loop):

```python
client.warmup(
    images=representative_images,
    prompt="fold the fabric",
    states=[reset_state, mid_state, goal_state],
)
```

**3. Rollout loop** (in the LeRobot environment):

```python
policy, preprocessor, postprocessor = make_flashrt_policy(
    checkpoint=CKPT, server_endpoint="http://localhost:8000"
)
policy.reset()

while not done:
    obs = get_observation()           # dict of images + state + task
    batch = preprocessor(obs)
    action = policy.select_action(batch)
    action = postprocessor(action)
    robot.send(action)
```

---

## Test gates

### Phase 2 — Client unit tests
```bash
pytest tests/test_client.py -v
```
Runs against a local mock server. Validates view-order enforcement, state
dtype/shape checks, image validation, warmup schema, and error paths. No GPU
required.

### Phase 4 — Parity gate
```bash
FLASHRT_SERVER=http://localhost:8000 \
LEROBOT_CKPT=/path/to/pretrained_model \
pytest tests/test_parity.py::test_parity_raw_chunk -v -s
```
Compares raw normalized `(30, 16)` action chunks between PyTorch and FlashRT
on the same observation with identical ODE noise (seeded). Gate: **mean
cosine ≥ 0.99, no single frame below 0.98**.

Output includes a per-frame summary table and, for the first 10 frames, a
per-timestep table with `cos_t`, `max|Δ|`, and first-4-dim previews for both
sides.

### Phase 5 — Latency gate
```bash
FLASHRT_SERVER=http://localhost:8000 \
pytest tests/test_latency.py::test_latency_stability -v -s
```
200 predict calls after a settle window must all fall within **3× the warm
median**. Reports median, mean, p95, p99, and a histogram. A CUDA graph
capture event adds 100–500 ms; the 3× threshold catches any capture without
false positives from normal jitter.

---

## Key design decisions

**Raw normalized actions.** The server returns `(Sa, 32)` actions in
LeRobot's normalized space. `unnormalize_actions` has been removed from the
FlashRT frontends (`pi05_rtx`, `pi05_thor`). The LeRobot postprocessor
(`UnnormalizerProcessorStep` + `AbsoluteActionsProcessorStep`) handles
unnormalization and relative→absolute conversion, exactly as it does for
local inference.

**`chunk_size` forwarding.** `flash_rt.load_model(action_horizon=30)` now
forwards `action_horizon` as `chunk_size` to the π₀.₅ Thor frontend, which
previously hardcoded `Sa=10`. The server passes `--action-horizon 30` to get
30-step chunks.

**Fixed state-prompt mode for rollouts.** `--state-prompt-mode fixed` uses
one CUDA graph at the padded max prompt length regardless of state value
variation, eliminating re-capture during rollouts. Use `exact` + warmup only
when peak latency matters more than capture-free operation.

**Seed support.** `client.predict(..., seed=N)` seeds the server's CUDA RNG
before inference so both sides start the ODE from identical Gaussian noise.
Used by `test_parity_raw_chunk`; not needed in production rollouts.
