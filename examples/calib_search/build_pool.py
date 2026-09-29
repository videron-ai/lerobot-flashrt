#!/usr/bin/env python3
"""build_pool.py — sample a stratified frame pool for FP8 calibration search.

Draws frames from a LeRobot dataset, split by *episode* into two disjoint sets:

  * ``cand`` — frames calibration datasets may be built from
  * ``eval`` — held-out frames every candidate calibration is scored on

Episodes are stratified by task so the multitask mix (fold / unfold / home) is
represented in both sets, and frames within an episode are jittered-uniform so
every phase of the motion is covered.

Each frame is stored exactly as FlashRT's ``predict()`` sees it — the same
lerobot preprocessor + ``resize_with_pad`` + uint8 rounding as
``offline_rollout.py`` / ``online_rollout.py`` — together with the raw state
(for relative→absolute postprocessing) and the ground-truth action chunk.

Usage (inside the FlashRT container):
    python examples/calib_search/build_pool.py \\
        --ckpt /nas/models_fast/openarm_folding_gen3_level2_multitask_50k \\
        --dataset_root /nas/datasets_archive/openarm_bimanual_shirt_folding_gen3_345_episodes_fold_unfold_home_rightfirst \\
        --out rollout_outputs_calib_search/pool.npz
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from offline_rollout import extract_flashrt_inputs, preprocess_frame  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--out", default="rollout_outputs_calib_search/pool.npz")
    p.add_argument("--chunk", type=int, default=30)
    # episodes per task: (candidate, eval)
    p.add_argument("--cand_eps", default="60,20,20",
                   help="candidate episodes per task: fold,home,unfold")
    p.add_argument("--eval_eps", default="40,12,12",
                   help="eval episodes per task: fold,home,unfold")
    p.add_argument("--cand_frames", type=int, default=12, help="frames per candidate episode")
    p.add_argument("--eval_frames", type=int, default=10, help="frames per eval episode")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


TASK_ORDER = ["Fold the T-shirt properly", "Return to home position", "Unfold the T-shirt"]


def jittered_positions(length: int, n: int, rng: np.random.Generator) -> list[int]:
    edges = np.linspace(0, length, n + 1)
    return sorted({int(rng.uniform(edges[i], edges[i + 1])) for i in range(n)})


def main():
    args = parse_args()
    import json
    import pandas as pd
    import glob
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    rng = np.random.default_rng(args.seed)
    ckpt = Path(args.ckpt)
    raw = json.load(open(ckpt / "config.json"))
    config = PI05Config(**{k: v for k, v in raw.items() if k in PI05Config.__dataclass_fields__})
    view_keys = [k for k, v in raw["input_features"].items() if v.get("type") == "VISUAL"]
    preprocessor, _ = make_pre_post_processors(config, pretrained_path=str(ckpt))

    eps = pd.concat([pd.read_parquet(f, columns=["episode_index", "tasks", "length",
                                                 "dataset_from_index", "dataset_to_index"])
                     for f in sorted(glob.glob(f"{args.dataset_root}/meta/episodes/*/*.parquet"))])
    eps["task"] = eps["tasks"].map(lambda t: str(t[0]))

    n_cand = [int(x) for x in args.cand_eps.split(",")]
    n_eval = [int(x) for x in args.eval_eps.split(",")]
    plan = []  # (split, episode_row)
    for task, nc, ne in zip(TASK_ORDER, n_cand, n_eval):
        rows = eps[eps.task == task].sample(frac=1.0, random_state=args.seed)
        if nc + ne > len(rows):
            raise ValueError(f"{task!r}: need {nc + ne} episodes, have {len(rows)}")
        plan += [("cand", r) for _, r in rows.iloc[:nc].iterrows()]
        plan += [("eval", r) for _, r in rows.iloc[nc:nc + ne].iterrows()]

    ds = LeRobotDataset("local/calib_pool", root=args.dataset_root, video_backend="pyav")
    all_actions = np.asarray(ds.hf_dataset.with_format("numpy")["action"], dtype=np.float32)

    recs = {k: [] for k in ["images", "state_norm", "state_raw", "gt_chunk", "gt_mask",
                            "episode", "frame", "length", "task_id", "split"]}
    t0 = time.time()
    total = sum(args.cand_frames if s == "cand" else args.eval_frames for s, _ in plan)
    done = 0
    for split, r in plan:
        n = args.cand_frames if split == "cand" else args.eval_frames
        L = int(r.length)
        start = int(r.dataset_from_index)
        for fi in jittered_positions(L, n, rng):
            frame = ds[start + fi]
            assert int(frame["episode_index"]) == int(r.episode_index)
            task = str(frame["task"])
            with torch.no_grad():
                pre = preprocess_frame(frame, preprocessor, task, args.device, view_keys)
                imgs, state_np = extract_flashrt_inputs(pre, view_keys)

            end = min(fi + args.chunk, L)
            chunk = np.zeros((args.chunk, all_actions.shape[1]), np.float32)
            mask = np.zeros(args.chunk, bool)
            chunk[: end - fi] = all_actions[start + fi: start + end]
            mask[: end - fi] = True

            recs["images"].append(np.stack(imgs))
            recs["state_norm"].append(state_np.astype(np.float32))
            recs["state_raw"].append(frame["observation.state"].numpy().astype(np.float32))
            recs["gt_chunk"].append(chunk)
            recs["gt_mask"].append(mask)
            recs["episode"].append(int(r.episode_index))
            recs["frame"].append(fi)
            recs["length"].append(L)
            recs["task_id"].append(TASK_ORDER.index(task))
            recs["split"].append(0 if split == "cand" else 1)
            done += 1
            if done % 100 == 0:
                el = time.time() - t0
                print(f"  {done}/{total}  {el:.0f}s  eta {el / done * (total - done):.0f}s",
                      flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **{k: np.stack(v) if k in ("images", "state_norm", "state_raw", "gt_chunk",
                                               "gt_mask") else np.asarray(v)
                     for k, v in recs.items()},
             view_keys=np.array(view_keys).astype("U"),
             tasks=np.array(TASK_ORDER).astype("U"))
    split = np.asarray(recs["split"])
    print(f"Saved {out}: {len(split)} frames (cand={int((split == 0).sum())}, "
          f"eval={int((split == 1).sum())}) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
