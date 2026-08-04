"""Phase 4 gate — parity test between PI05Policy (PyTorch) and FlashRTPI05Policy.

Requirements:
    - Mean cosine similarity on the raw normalized (30, 32) chunk >= 0.99
    - No single frame below 0.98

Usage:
    # Default (needs both GPU environments — run on the LeRobot machine with
    # the FlashRT server already running):
    FLASHRT_SERVER=http://flashrt-host:8000 \\
    LEROBOT_CKPT=/path/to/checkpoint \\
    pytest tests/test_parity.py -v -s

    # View-permutation ablation only:
    pytest tests/test_parity.py::test_view_permutation_ablation -v -s

Environment variables:
    LEROBOT_CKPT      Path to the LeRobot pretrained model directory.
    FLASHRT_SERVER    FlashRT server endpoint (default http://localhost:8000).
    PARITY_N_FRAMES   Number of frames to test (default 50).
    PARITY_SEED       RNG seed for noise (default 42).
    DATASET_REPO      LeRobot dataset repo ID for sampling (optional;
                      if absent, synthetic data is used for smoke-test).
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import pytest
import torch

logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────

CKPT = os.environ.get(
    "LEROBOT_CKPT",
    "/home/videron/Desktop/openarm/outputs/train/openarm_folding_high_quality_60k"
    "/checkpoints/060000/pretrained_model",
)
SERVER = os.environ.get("FLASHRT_SERVER", "http://localhost:8000")
N_FRAMES = int(os.environ.get("PARITY_N_FRAMES", "50"))
SEED = int(os.environ.get("PARITY_SEED", "42"))
DATASET_REPO = os.environ.get("DATASET_REPO", None)

COSINE_MEAN_MIN = 0.99
COSINE_FRAME_MIN = 0.98

from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, OBS_STATE
from lerobot_flashrt.client import CANONICAL_VIEW_ORDER

# ── Fixtures ─────────────────────────────────────────────────────────────────

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
def flashrt_policy(pytorch_policy):
    from lerobot_flashrt import make_flashrt_policy
    policy, preprocessor, postprocessor = make_flashrt_policy(
        CKPT, server_endpoint=SERVER, device="cpu"
    )
    return policy


@pytest.fixture(scope="module")
def preprocessor():
    """Full LeRobot preprocessor pipeline for building properly-conditioned PyTorch batches."""
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
    """Load N frames stratified across episodes / positions within episode."""
    return _load_frames(N_FRAMES)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_frames(n: int) -> list[dict]:
    """Return list of dicts with keys:
        images: {cam: np.ndarray uint8 HWC 224×224×3}
        state:  np.ndarray float32 (32,) normalized
        prompt: str
        raw_state: np.ndarray float32 (actual DOF, unnormalized) for postprocessor
    """
    if DATASET_REPO:
        return _load_from_dataset(n)
    else:
        logger.warning(
            "DATASET_REPO not set; using synthetic frames for smoke test. "
            "Set DATASET_REPO=<hf_repo_id> for a real parity test."
        )
        return _synthetic_frames(n)


def _synthetic_frames(n: int) -> list[dict]:
    rng = np.random.default_rng(SEED)
    frames = []
    for _ in range(n):
        images = {k: rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
                  for k in CANONICAL_VIEW_ORDER}
        state = rng.uniform(-1, 1, (32,)).astype(np.float32)
        frames.append({
            "images": images,
            "state": state,
            "prompt": "fold the fabric neatly",
        })
    return frames


def _load_from_dataset(n: int) -> list[dict]:
    """Sample n frames stratified across episodes, from early/mid/late in each."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        pytest.skip("lerobot.datasets not available; use DATASET_REPO=none or install [dataset]")

    # Lazy import of normalizer to build normalized state
    from lerobot.policies.pi05 import make_pi05_pre_post_processors
    import json
    with open(os.path.join(CKPT, "config.json")) as f:
        raw = json.load(f)
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    config = PI05Config(**{k: v for k, v in raw.items()
                           if k in PI05Config.__dataclass_fields__})
    config.device = "cpu"
    config.validate_features()

    from lerobot_flashrt.factory import _load_stats_from_checkpoint
    from pathlib import Path
    stats = _load_stats_from_checkpoint(Path(CKPT))
    preprocessor, _ = make_pi05_pre_post_processors(config, dataset_stats=stats)

    ds = LeRobotDataset(DATASET_REPO)
    episodes = list(range(ds.num_episodes))
    random.seed(SEED)
    random.shuffle(episodes)

    frames = []
    per_ep = max(1, n // len(episodes))
    for ep_idx in episodes:
        if len(frames) >= n:
            break
        ep_frames = [i for i in range(len(ds)) if ds[i]["episode_index"] == ep_idx]
        if not ep_frames:
            continue
        # Sample from early / mid / late
        positions = [0, len(ep_frames) // 2, len(ep_frames) - 1]
        for pos in positions:
            if len(frames) >= n:
                break
            raw = ds[ep_frames[pos]]
            # Build normalized batch via preprocessor
            # Extract images (uint8 HWC) and normalized state
            imgs = {k: _tensor_to_uint8_hwc(raw[k]) for k in CANONICAL_VIEW_ORDER
                    if k in raw}
            raw_state = raw.get("observation.state", torch.zeros(16)).numpy().astype(np.float32)
            # Pad state to 32
            state_pad = np.zeros(32, dtype=np.float32)
            state_pad[:len(raw_state)] = raw_state
            frames.append({
                "images": imgs,
                "state": state_pad,  # NOTE: not normalized yet; see below
                "prompt": raw.get("task", ""),
            })
    return frames[:n]


def _tensor_to_uint8_hwc(t: torch.Tensor) -> np.ndarray:
    """Convert [C, H, W] float32 [0,1] → (H, W, C) uint8."""
    if t.dim() == 3 and t.shape[0] in (1, 3):
        t = t.permute(1, 2, 0)
    return (t.clamp(0, 1) * 255).to(torch.uint8).numpy()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Per-chunk cosine similarity: mean over (chunk, dim)."""
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom < 1e-12:
        return 1.0
    return float(np.dot(a_flat, b_flat) / denom)


def _build_batch(frame: dict, config, preprocessor=None) -> dict[str, torch.Tensor]:
    """Build a preprocessed batch suitable for predict_action_chunk."""
    images = frame["images"]
    state = frame["state"]
    prompt = frame["prompt"]

    if preprocessor is not None:
        # Flat-dict input: pipeline's to_transition() extracts keys starting
        # with "observation." as the obs dict and "task" as complementary data.
        flat: dict = {k: torch.from_numpy(v).permute(2, 0, 1).float() / 255.0
                      for k, v in images.items()}
        flat["observation.state"] = torch.from_numpy(state[:16])  # real DOF
        flat["task"] = [prompt]
        return preprocessor(flat)

    # Lightweight fallback for smoke test: manually normalize + resize.
    # Place tensors on config.device so the model's embedding lookup doesn't
    # get a CPU tensor against a CUDA weight (DeviceProcessorStep normally does this).
    device = getattr(config, "device", "cpu")

    batch: dict[str, torch.Tensor] = {}
    for key, img in images.items():
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # [C, H, W]
        batch[key] = t.unsqueeze(0).to(device)  # [1, C, H, W]

    # Pad state to 32 (already normalized in synthetic frames)
    state_t = torch.from_numpy(state).unsqueeze(0).to(device)  # [1, 32]
    batch["observation.state"] = state_t
    batch["task"] = [prompt]

    # Dummy tokens/masks (PI05Policy will use them, FlashRT ignores them)
    batch[OBS_LANGUAGE_TOKENS] = torch.zeros(1, 200, dtype=torch.long, device=device)
    batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones(1, 200, dtype=torch.bool, device=device)
    return batch


def _print_chunk_detail(frame_idx: int, chunk_pt: np.ndarray, chunk_frt: np.ndarray) -> None:
    """Print per-timestep cosines and first-4-dim previews for one frame."""
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


# ── Main parity test ─────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not os.path.exists(CKPT),
    reason=f"Checkpoint not found at {CKPT}",
)
def test_parity_raw_chunk(pytorch_policy, flashrt_policy, test_frames, preprocessor):
    """Compare raw normalized (30, 32) chunk between PyTorch and FlashRT.

    Both sides are seeded identically before each call so they start from the
    same ODE noise.  PyTorch uses the full preprocessor to build properly
    tokenized prompt+state conditioning; FlashRT builds its own prompt on the
    server from the raw task text and normalized state.
    """
    np.random.seed(SEED)

    cosines = []
    failures = []

    W = 40
    print(f"\n{'─'*W}")
    print(f"  Parity  frames={N_FRAMES}  seed={SEED}  mean≥{COSINE_MEAN_MIN}  frame≥{COSINE_FRAME_MIN}")
    print(f"{'─'*W}")
    print(f"  {'Frame':>5}  {'Cosine':>8}  Status")

    for i, frame in enumerate(test_frames):
        seed = SEED + i

        # PyTorch: use full preprocessor for correct token conditioning.
        batch_pt = _build_batch(frame, pytorch_policy.config, preprocessor=preprocessor)

        # Extract the preprocessor-normalized state so FlashRT digitizes the same
        # bins and produces the same prompt string as PyTorch's tokenizer step.
        # The preprocessor outputs the real DOF dims only (e.g. 16), not padded to 32.
        normalized_state_np = batch_pt[OBS_STATE].squeeze(0).cpu().float().numpy()

        # Zero-pad to 32 for _extract_state validation, but the server will only
        # use the first _state_in_prompt_dim dims (=original_action_dim) when
        # calling format_pi05_prompt(), matching the 16-bin PyTorch prompt exactly.
        state_padded = np.zeros(32, dtype=np.float32)
        state_padded[:len(normalized_state_np)] = normalized_state_np

        # FlashRT: lightweight batch, but override state with preprocessor-normalized values.
        batch_frt = _build_batch(frame, flashrt_policy.config)
        batch_frt[OBS_STATE] = torch.from_numpy(state_padded).unsqueeze(0)

        # Seed both sides to the same value so the ODE starts from identical noise.
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        with torch.no_grad():
            chunk_pt = pytorch_policy.predict_action_chunk(batch_pt).squeeze(0).cpu().numpy()

        chunk_frt = flashrt_policy.predict_action_chunk(batch_frt, seed=seed).squeeze(0).cpu().numpy()

        # PI05Policy.predict_action_chunk slices output to original_action_dim (16)
        # before returning.  FlashRT returns the full 32-wide buffer.  Align on
        # the real DOF count so we compare the same space.
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
    print(f"  mean={mean_cos:.4f}  min={min_cos:.4f}  "
          f"passed={len(cosines)-len(failures)}/{len(cosines)}")
    print(f"{'─'*W}\n")

    assert not failures, (
        f"{len(failures)}/{len(cosines)} frames below {COSINE_FRAME_MIN}: "
        + ", ".join(f"frame {i}: {c:.4f}" for i, c, _, _ in failures)
        + "\n\nTriage:\n"
        "  all frames bad        → weight mapping / view ordering\n"
        "  t=0 ok, later drift   → action mask or decoder RoPE for Sa=30\n"
        "  state-dependent only  → state normalization or digitize path\n"
        "  raw ok, final wrong   → postprocessor / relative-action handling\n"
        "Run test_view_permutation_ablation first if cosine is ~0.97 uniformly."
    )
    assert mean_cos >= COSINE_MEAN_MIN, (
        f"Mean cosine {mean_cos:.4f} < {COSINE_MEAN_MIN}. "
        "See failure triage above."
    )


# ── 4.5 View-permutation ablation ────────────────────────────────────────────

@pytest.mark.skipif(
    not os.path.exists(CKPT),
    reason=f"Checkpoint not found at {CKPT}",
)
def test_view_permutation_ablation(pytorch_policy, test_frames):
    """Run parity under all 6 view orderings, report cosines.

    The canonical ordering should score clearly higher than the others.
    If two orderings score similarly, run an occlusion test (blank one view
    and confirm which output dims change).

    Results are printed for manual inspection and should be recorded in
    FINDINGS.md §View-permutation fixture table.
    """
    from itertools import permutations

    views = list(CANONICAL_VIEW_ORDER)
    perms = list(permutations(range(len(views))))

    fixture_frames = test_frames[:10]

    print("\n=== View-permutation ablation ===")
    results = []

    for perm in perms:
        perm_order = [views[i] for i in perm]
        cosines = []
        for frame in fixture_frames:
            # Re-order images according to this permutation
            reordered = {perm_order[j]: frame["images"][views[j]]
                         for j in range(len(views))}
            frame_permuted = {**frame, "images": reordered}

            batch_canon = _build_batch(frame, pytorch_policy.config)
            batch_perm = _build_batch(frame_permuted, pytorch_policy.config)

            with torch.no_grad():
                chunk_canon = pytorch_policy.predict_action_chunk(batch_canon).squeeze(0).cpu().numpy()
                chunk_perm = pytorch_policy.predict_action_chunk(batch_perm).squeeze(0).cpu().numpy()

            cosines.append(_cosine(chunk_canon, chunk_perm))

        mean_cos = float(np.mean(cosines))
        marker = " ← CANONICAL" if list(perm) == list(range(len(views))) else ""
        print(f"  perm {list(perm)} ({perm_order}): mean_cosine={mean_cos:.4f}{marker}")
        results.append((list(perm), mean_cos))

    # Canonical perm should be 1.0 by definition (same → same)
    canonical_cos = next(c for p, c in results if p == list(range(len(views))))
    print(f"\nCanonical self-cosine: {canonical_cos:.6f}")

    # Record note (update FINDINGS.md manually with the printed table)
    print("\nRecord the table above in FINDINGS.md §View-permutation fixture table.")
