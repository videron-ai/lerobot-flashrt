"""Local parity test — LocalFlashRTPI05Policy (in-process FlashRT) vs PyTorch.

Same gate criteria as test_parity.py:
    - Mean cosine similarity ≥ 0.99
    - No single frame below 0.98

No HTTP server required — FlashRT runs in the same process.  Run in an
environment where both lerobot and flash_rt are installed.

Usage:
    LEROBOT_CKPT=/path/to/checkpoint pytest tests/test_local_parity.py -v -s

Environment variables:
    LEROBOT_CKPT      Path to the LeRobot pretrained model directory.
    PARITY_N_FRAMES   Number of frames to test (default 50).
    PARITY_SEED       RNG seed (default 42).
    DATASET_REPO      HuggingFace dataset repo ID (optional; synthetic if absent).
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pytest
import torch

from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, OBS_STATE
from lerobot_flashrt.client import CANONICAL_VIEW_ORDER

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

CKPT = os.environ.get(
    "LEROBOT_CKPT",
    "/home/videron/Desktop/openarm/outputs/train/openarm_folding_high_quality_60k"
    "/checkpoints/060000/pretrained_model",
)
N_FRAMES = int(os.environ.get("PARITY_N_FRAMES", "50"))
SEED     = int(os.environ.get("PARITY_SEED", "42"))
DATASET_REPO = os.environ.get("DATASET_REPO", None)

COSINE_MEAN_MIN  = 0.99
COSINE_FRAME_MIN = 0.98

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pytorch_policy():
    import torch._dynamo
    torch._dynamo.config.disable = True

    from lerobot.policies.pi05 import PI05Policy
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    import json
    with open(os.path.join(CKPT, "config.json")) as f:
        raw = json.load(f)
    config = PI05Config(**{k: v for k, v in raw.items()
                           if k in PI05Config.__dataclass_fields__})
    config.device = "cuda"
    config.validate_features()
    policy = PI05Policy.from_pretrained(CKPT, strict=False)
    policy.eval()
    return policy


@pytest.fixture(scope="module")
def local_flashrt_policy():
    from lerobot_flashrt import make_local_flashrt_policy
    policy, _, _ = make_local_flashrt_policy(
        CKPT,
        device="cuda",
        action_horizon=30,
        num_views=3,
        state_prompt_mode="fixed",
    )
    return policy


@pytest.fixture(scope="module")
def preprocessor():
    from lerobot.policies.pi05 import make_pi05_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot_flashrt.factory import _load_stats_from_checkpoint
    from pathlib import Path
    import json
    with open(os.path.join(CKPT, "config.json")) as f:
        raw = json.load(f)
    config = PI05Config(**{k: v for k, v in raw.items()
                           if k in PI05Config.__dataclass_fields__})
    config.device = "cuda"
    config.validate_features()
    stats = _load_stats_from_checkpoint(Path(CKPT))
    pre, _ = make_pi05_pre_post_processors(config, dataset_stats=stats)
    return pre


@pytest.fixture(scope="module")
def test_frames(pytorch_policy):
    return _load_frames(N_FRAMES)


# ── Helpers (shared with test_parity.py) ─────────────────────────────────────

def _load_frames(n: int) -> list[dict]:
    if DATASET_REPO:
        return _load_from_dataset(n)
    logger.warning("DATASET_REPO not set; using synthetic frames.")
    return _synthetic_frames(n)


def _synthetic_frames(n: int) -> list[dict]:
    rng = np.random.default_rng(SEED)
    frames = []
    for _ in range(n):
        images = {k: rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
                  for k in CANONICAL_VIEW_ORDER}
        state = rng.uniform(-1, 1, (32,)).astype(np.float32)
        frames.append({"images": images, "state": state, "prompt": "fold the fabric neatly"})
    return frames


def _load_from_dataset(n: int) -> list[dict]:
    import random
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        pytest.skip("lerobot.datasets not available")

    from lerobot.policies.pi05 import make_pi05_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot_flashrt.factory import _load_stats_from_checkpoint
    from pathlib import Path
    import json
    with open(os.path.join(CKPT, "config.json")) as f:
        raw = json.load(f)
    config = PI05Config(**{k: v for k, v in raw.items()
                           if k in PI05Config.__dataclass_fields__})
    config.device = "cpu"
    config.validate_features()
    stats = _load_stats_from_checkpoint(Path(CKPT))

    ds = LeRobotDataset(DATASET_REPO)
    episodes = list(range(ds.num_episodes))
    random.seed(SEED)
    random.shuffle(episodes)

    frames = []
    for ep_idx in episodes:
        if len(frames) >= n:
            break
        ep_frames = [i for i in range(len(ds)) if ds[i]["episode_index"] == ep_idx]
        if not ep_frames:
            continue
        for pos in [0, len(ep_frames) // 2, len(ep_frames) - 1]:
            if len(frames) >= n:
                break
            raw = ds[ep_frames[pos]]
            imgs = {k: _tensor_to_uint8_hwc(raw[k]) for k in CANONICAL_VIEW_ORDER if k in raw}
            raw_state = raw.get("observation.state", torch.zeros(16)).numpy().astype(np.float32)
            state_pad = np.zeros(32, dtype=np.float32)
            state_pad[:len(raw_state)] = raw_state
            frames.append({"images": imgs, "state": state_pad, "prompt": raw.get("task", "")})
    return frames[:n]


def _tensor_to_uint8_hwc(t: torch.Tensor) -> np.ndarray:
    if t.dim() == 3 and t.shape[0] in (1, 3):
        t = t.permute(1, 2, 0)
    return (t.clamp(0, 1) * 255).to(torch.uint8).numpy()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_flat, b_flat = a.reshape(-1), b.reshape(-1)
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    return float(np.dot(a_flat, b_flat) / denom) if denom > 1e-12 else 1.0


def _build_batch(frame: dict, config, preprocessor=None) -> dict[str, torch.Tensor]:
    images, state, prompt = frame["images"], frame["state"], frame["prompt"]
    device = getattr(config, "device", "cpu")

    if preprocessor is not None:
        flat: dict = {k: torch.from_numpy(v).permute(2, 0, 1).float() / 255.0
                      for k, v in images.items()}
        flat["observation.state"] = torch.from_numpy(state[:16])
        flat["task"] = [prompt]
        return preprocessor(flat)

    batch: dict[str, torch.Tensor] = {}
    for key, img in images.items():
        batch[key] = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    batch["observation.state"] = torch.from_numpy(state).unsqueeze(0).to(device)
    batch["task"] = [prompt]
    batch[OBS_LANGUAGE_TOKENS] = torch.zeros(1, 200, dtype=torch.long, device=device)
    batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones(1, 200, dtype=torch.bool, device=device)
    return batch


def _print_chunk_detail(frame_idx: int, chunk_pt: np.ndarray, chunk_frt: np.ndarray) -> None:
    T, D = chunk_pt.shape
    print(f"\n  ── Frame {frame_idx} ({T}t × {D}d) ──────────────────────────────────────────────")
    print(f"  {'t':>3}  {'cos_t':>7}  {'max|Δ|':>7}  {'pt[:4]':^36}  {'frt[:4]':^36}")
    print(f"  {'─'*3}  {'─'*7}  {'─'*7}  {'─'*36}  {'─'*36}")
    for t in range(T):
        a, b = chunk_pt[t], chunk_frt[t]
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        cos_t = float(np.dot(a, b) / denom) if denom > 1e-12 else 1.0
        max_delta = float(np.max(np.abs(a - b)))
        pt_str  = ", ".join(f"{v:7.4f}" for v in a[:4])
        frt_str = ", ".join(f"{v:7.4f}" for v in b[:4])
        flag = " ←" if cos_t < 0.98 else ""
        print(f"  {t:>3}  {cos_t:>7.4f}  {max_delta:>7.4f}  {pt_str}  {frt_str}{flag}")
    print()


# ── Parity test ───────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not os.path.exists(CKPT),
    reason=f"Checkpoint not found at {CKPT}",
)
def test_local_parity_raw_chunk(pytorch_policy, local_flashrt_policy, test_frames, preprocessor):
    """Compare raw normalized (30, 16) chunk between PyTorch and LocalFlashRTPI05Policy.

    Both sides are seeded identically so they start from the same ODE noise.
    No HTTP server involved — FlashRT runs in-process.
    """
    np.random.seed(SEED)

    cosines  = []
    failures = []

    W = 40
    print(f"\n{'─'*W}")
    print(f"  Local parity  frames={N_FRAMES}  seed={SEED}  mean≥{COSINE_MEAN_MIN}  frame≥{COSINE_FRAME_MIN}")
    print(f"{'─'*W}")
    print(f"  {'Frame':>5}  {'Cosine':>8}  Status")

    for i, frame in enumerate(test_frames):
        seed = SEED + i

        batch_pt = _build_batch(frame, pytorch_policy.config, preprocessor=preprocessor)

        normalized_state_np = batch_pt[OBS_STATE].squeeze(0).cpu().float().numpy()
        state_padded = np.zeros(32, dtype=np.float32)
        state_padded[:len(normalized_state_np)] = normalized_state_np

        batch_frt = _build_batch(frame, local_flashrt_policy.config)
        batch_frt[OBS_STATE] = torch.from_numpy(state_padded).unsqueeze(0)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        with torch.no_grad():
            chunk_pt = pytorch_policy.predict_action_chunk(batch_pt).squeeze(0).cpu().numpy()

        chunk_frt = local_flashrt_policy.predict_action_chunk(batch_frt, seed=seed).squeeze(0).cpu().numpy()

        act_dim = chunk_pt.shape[1]
        chunk_frt_cmp = chunk_frt[:, :act_dim]

        cos = _cosine(chunk_pt, chunk_frt_cmp)
        cosines.append(cos)

        status = "PASS" if cos >= COSINE_FRAME_MIN else "FAIL ←"
        print(f"  {i:>5}  {cos:>8.4f}  {status}")

        if i < 10:
            _print_chunk_detail(i, chunk_pt, chunk_frt_cmp)

        if cos < COSINE_FRAME_MIN:
            failures.append((i, cos, chunk_pt[0, :5], chunk_frt_cmp[0, :5]))

    mean_cos = float(np.mean(cosines))
    min_cos  = float(np.min(cosines))
    print(f"{'─'*W}")
    print(f"  mean={mean_cos:.4f}  min={min_cos:.4f}  passed={len(cosines)-len(failures)}/{len(cosines)}")
    print(f"{'─'*W}\n")

    assert not failures, (
        f"{len(failures)}/{len(cosines)} frames below {COSINE_FRAME_MIN}: "
        + ", ".join(f"frame {i}: {c:.4f}" for i, c, _, _ in failures)
        + "\n\nTriage:\n"
        "  all frames bad        → weight mapping / view ordering\n"
        "  t=0 ok, later drift   → action mask or decoder RoPE\n"
        "  state-dependent only  → state normalization or digitize path\n"
        "  local ok, HTTP fails  → serialization or seeding difference\n"
    )
    assert mean_cos >= COSINE_MEAN_MIN, (
        f"Mean cosine {mean_cos:.4f} < {COSINE_MEAN_MIN}."
    )


# ── Cross-variant consistency ─────────────────────────────────────────────────

@pytest.mark.skipif(
    not os.path.exists(CKPT),
    reason=f"Checkpoint not found at {CKPT}",
)
def test_local_vs_http_consistency():
    """Skip unless FLASHRT_SERVER is set — confirms local and HTTP variants agree."""
    server = os.environ.get("FLASHRT_SERVER")
    if not server:
        pytest.skip("FLASHRT_SERVER not set; skipping local-vs-HTTP cross-check.")

    from lerobot_flashrt import make_flashrt_policy, make_local_flashrt_policy

    _, pre_http, _ = make_flashrt_policy(CKPT, server_endpoint=server)
    policy_local, _, _ = make_local_flashrt_policy(CKPT)
    policy_http, _, _  = make_flashrt_policy(CKPT, server_endpoint=server)

    rng = np.random.default_rng(SEED)
    frames = _synthetic_frames(5)
    cosines = []

    for i, frame in enumerate(frames):
        seed = SEED + i
        batch_local = _build_batch(frame, policy_local.config)
        batch_http  = _build_batch(frame, policy_http.config)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        chunk_local = policy_local.predict_action_chunk(batch_local, seed=seed).squeeze(0).cpu().numpy()
        chunk_http  = policy_http.predict_action_chunk(batch_http,  seed=seed).squeeze(0).cpu().numpy()

        cosines.append(_cosine(chunk_local, chunk_http))

    mean_cos = float(np.mean(cosines))
    print(f"\nLocal vs HTTP: mean cosine={mean_cos:.4f}  min={min(cosines):.4f}")
    assert mean_cos >= COSINE_MEAN_MIN, (
        f"Local and HTTP variants diverge: mean cosine {mean_cos:.4f}. "
        "Check state_in_prompt_dim and seed handling on both paths."
    )
