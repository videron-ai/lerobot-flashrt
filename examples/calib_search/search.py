#!/usr/bin/env python3
"""search.py — score candidate FP8 calibrations on held-out training frames.

Each candidate is an FP8 activation-scale vector, hot-swapped into one captured
model (see ``harness.py``). A candidate is scored by predicting an action chunk
for every eval frame with a fixed per-frame noise seed and measuring:

  gt_mae     joint-space MAE vs the dataset's ground-truth action chunk
             (the objective: error across the training distribution)
  gt_mae10   same, first 10 steps only (what RTC actually executes)
  f16_mae    joint-space MAE vs the FP16 reference on the same noise —
             pure quantization error, far lower variance than gt_mae

Candidate families (``--stage``):
  sweep   baselines (stock single frame, existing 16-frame caches), pool-wide
          per-GEMM quantiles, and random K-frame subsets at the deployed
          percentile reduction
  select  greedy K-frame calibration sets whose deployed reduction matches the
          best quantile target found by ``sweep``

Results append to ``<pool dir>/search_results.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from harness import Harness  # noqa: E402

TASKS_SHORT = ["fold", "home", "unfold"]


def reduce(amax: np.ndarray, percentile: float) -> np.ndarray:
    """Exactly flash_rt.core.calibration.accumulate_amax."""
    return np.percentile(amax.astype(np.float64), percentile, axis=0).astype(np.float32)


class Evaluator:
    def __init__(self, h: Harness, pool, ref, tasks):
        self.h, self.pool, self.tasks = h, pool, tasks
        self.ev = ref["eval_idx"]
        self.mask = pool["gt_mask"][self.ev]                      # (E, T)
        self.gt = pool["gt_chunk"][self.ev]                        # (E, T, D)
        self.task_id = pool["task_id"][self.ev]
        self.ref_joint = np.stack([
            h.to_joint(ref["chunks"][k], pool["state_raw"][i])
            for k, i in enumerate(self.ev)])

    def run(self, vec: np.ndarray) -> dict:
        h, pool = self.h, self.pool
        h.set_scales(vec)
        pred = np.zeros_like(self.gt)
        for k, i in enumerate(self.ev):
            raw = h.predict(self.tasks[pool["task_id"][i]], pool["images"][i],
                            pool["state_norm"][i], seed=int(i))
            pred[k] = h.to_joint(raw, pool["state_raw"][i])
        return self.metrics(pred)

    def metrics(self, pred: np.ndarray) -> dict:
        m = self.mask[..., None]
        err_gt = np.abs(pred - self.gt) * m                       # (E, T, D)
        per_frame_gt = err_gt.sum((1, 2)) / (m.sum((1, 2)) * pred.shape[2])
        m10 = m[:, :10]
        per_frame_gt10 = (err_gt[:, :10].sum((1, 2))
                          / (m10.sum((1, 2)) * pred.shape[2]))
        per_frame_f16 = np.abs(pred - self.ref_joint).mean((1, 2))
        out = {
            "gt_mae": float(per_frame_gt.mean()),
            "gt_mae10": float(per_frame_gt10.mean()),
            "f16_mae": float(per_frame_f16.mean()),
            "gt_joint": (err_gt.sum((0, 1)) / m.sum((0, 1))).round(4).tolist(),
            "f16_joint": np.abs(pred - self.ref_joint).mean((0, 1)).round(4).tolist(),
        }
        for t, name in enumerate(TASKS_SHORT):
            sel = self.task_id == t
            out[f"gt_{name}"] = float(per_frame_gt[sel].mean())
            out[f"f16_{name}"] = float(per_frame_f16[sel].mean())
        out["_per_frame_gt"] = per_frame_gt
        out["_per_frame_f16"] = per_frame_f16
        return out


def greedy_select(amax: np.ndarray, cand_idx: np.ndarray, target: np.ndarray, k: int,
                  percentile: float, task_id: np.ndarray, min_per_task: int = 1,
                  ) -> list[int]:
    """Pick k frames whose percentile-reduced amax best matches ``target``.

    Distance is mean |log(reduced) - log(target)| over GEMMs, so every layer
    counts equally regardless of its absolute scale. Forced to include at least
    ``min_per_task`` frames of each task so the set stays multitask.
    """
    la = np.log(np.maximum(amax[cand_idx], 1e-12))
    lt = np.log(np.maximum(target, 1e-12))
    chosen: list[int] = []

    def dist(sel):
        r = reduce(np.exp(la[sel]), percentile)
        return float(np.abs(np.log(np.maximum(r, 1e-12)) - lt).mean())

    for _ in range(k):
        need = [t for t in range(3)
                if sum(task_id[cand_idx[c]] == t for c in chosen) < min_per_task]
        remaining = k - len(chosen)
        best, best_d = None, np.inf
        for c in range(len(cand_idx)):
            if c in chosen:
                continue
            if len(need) >= remaining and task_id[cand_idx[c]] not in need:
                continue
            d = dist(chosen + [c])
            if d < best_d:
                best, best_d = c, d
        chosen.append(best)
    # One swap pass to escape the greedy order.
    improved = True
    while improved:
        improved = False
        cur = dist(chosen)
        for pos in range(k):
            for c in range(len(cand_idx)):
                if c in chosen:
                    continue
                trial = chosen.copy()
                trial[pos] = c
                if any(sum(task_id[cand_idx[x]] == t for x in trial) < min_per_task
                       for t in range(3)):
                    continue
                d = dist(trial)
                if d < cur - 1e-6:
                    chosen, cur, improved = trial, d, True
    return [int(cand_idx[c]) for c in chosen]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--pool", required=True)
    p.add_argument("--stage", choices=["sweep", "select", "custom"], required=True)
    p.add_argument("--target_q", type=float, default=None,
                   help="select: pool quantile the K-frame set should reproduce")
    p.add_argument("--ks", default="16,32", help="select: calibration set sizes")
    p.add_argument("--percentile", type=float, default=99.9,
                   help="deployed calibrate() percentile reduction")
    p.add_argument("--quantiles", default="50,75,90,95,98,99,99.5,99.9,100")
    p.add_argument("--random_draws", type=int, default=3)
    p.add_argument("--custom_json", default=None,
                   help="custom: JSON list of {name, idx:[pool idx], percentile}")
    p.add_argument("--vecs", nargs="*", default=[],
                   help="custom: name=path.npy scale vectors to score as-is")
    args = p.parse_args()

    out_dir = Path(args.pool).parent
    pool = np.load(args.pool)
    tasks = [str(t) for t in pool["tasks"]]
    A = np.load(out_dir / "amax.npz")
    amax = A["amax"]
    ref = np.load(out_dir / "ref_fp16.npz")
    cand = np.where(pool["split"] == 0)[0]

    h = Harness(args.ckpt, precision="fp8")
    i0 = int(ref["eval_idx"][0])
    h.bootstrap(tasks[pool["task_id"][i0]], pool["images"][i0], pool["state_norm"][i0])
    assert list(h.scale_names()) == [str(n) for n in A["names"]], "scale name mismatch"
    ev = Evaluator(h, pool, ref, tasks)

    candidates: list[tuple[str, np.ndarray, dict]] = []
    rng = np.random.default_rng(1)
    if args.stage == "sweep":
        for key in sorted(k for k in A.files if k.startswith("baseline") and
                          not k.endswith("_path")):
            ba = A[key]
            src = str(A[f"{key}_path"])
            candidates.append((f"{key}_single_frame0", ba[0], {"src": src, "n": 1}))
            candidates.append((f"{key}_N{len(ba)}_p{args.percentile}",
                               reduce(ba, args.percentile), {"src": src, "n": len(ba)}))
        for q in [float(x) for x in args.quantiles.split(",")]:
            candidates.append((f"pool_q{q:g}", reduce(amax[cand], q),
                               {"n": len(cand), "q": q}))
        for k in (16, 32, 64):
            for d in range(args.random_draws):
                sel = rng.choice(cand, size=k, replace=False)
                candidates.append((f"random_N{k}_d{d}", reduce(amax[sel], args.percentile),
                                   {"idx": sel.tolist()}))
        # Repeat the first candidate to measure run-to-run determinism.
        candidates.append((candidates[0][0] + "_repeat", candidates[0][1], {}))
    elif args.stage == "select":
        target = reduce(amax[cand], args.target_q)
        for k in [int(x) for x in args.ks.split(",")]:
            sel = greedy_select(amax, cand, target, k, args.percentile, pool["task_id"])
            candidates.append((f"greedy_q{args.target_q:g}_N{k}",
                               reduce(amax[sel], args.percentile),
                               {"idx": sel, "target_q": args.target_q}))
        candidates.append((f"pool_q{args.target_q:g}_recheck", target, {}))
    else:
        for spec in args.vecs:
            name, path = spec.split("=", 1)
            candidates.append((name, np.load(path).astype(np.float32), {"src": path}))
        for c in (json.load(open(args.custom_json)) if args.custom_json else []):
            idx = np.asarray(c["idx"])
            candidates.append((c["name"], reduce(amax[idx], c.get("percentile",
                                                                   args.percentile)),
                               {"idx": idx.tolist()}))

    res_path = out_dir / "search_results.jsonl"
    pf_path = out_dir / "search_per_frame.npz"
    per_frame = dict(np.load(pf_path)) if pf_path.exists() else {}
    for name, vec, meta in candidates:
        t0 = time.time()
        m = ev.run(vec)
        per_frame[f"{name}__gt"] = m.pop("_per_frame_gt")
        per_frame[f"{name}__f16"] = m.pop("_per_frame_f16")
        rec = {"name": name, "stage": args.stage, **m, **meta,
               "vec_logmean": float(np.log(np.maximum(vec, 1e-12)).mean())}
        with open(res_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        np.savez(pf_path, **per_frame)
        np.save(out_dir / f"vec_{name}.npy", vec)
        print(f"  {name:36s} gt={m['gt_mae']:.4f} gt10={m['gt_mae10']:.4f} "
              f"f16={m['f16_mae']:.4f}  [fold {m['gt_fold']:.4f} home {m['gt_home']:.4f} "
              f"unfold {m['gt_unfold']:.4f}]  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
