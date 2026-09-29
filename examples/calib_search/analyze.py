#!/usr/bin/env python3
"""analyze.py — rank search results with paired bootstrap CIs vs a reference.

Every candidate is scored on the same eval frames with the same noise seeds, so
per-frame differences are paired; the bootstrap resamples *episodes* (frames of
one episode are correlated) to get a 95% CI on the mean difference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="rollout_outputs_calib_search")
    p.add_argument("--ref", default="baseline0_N16_p99.9")
    p.add_argument("--n_boot", type=int, default=2000)
    args = p.parse_args()

    d = Path(args.dir)
    pool = np.load(d / "pool.npz")
    ev = np.load(d / "ref_fp16.npz")["eval_idx"]
    ep = pool["episode"][ev]
    uniq = np.unique(ep)
    members = [np.where(ep == u)[0] for u in uniq]
    rng = np.random.default_rng(0)
    boots = [np.concatenate([members[j] for j in rng.integers(0, len(uniq), len(uniq))])
             for _ in range(args.n_boot)]

    pf = dict(np.load(d / "search_per_frame.npz"))
    for f in d.glob("validate_*.npz"):
        v = np.load(f)
        pf[f"{f.stem}__gt"], pf[f"{f.stem}__f16"] = v["gt"], v["f16"]
    names = sorted({k.rsplit("__", 1)[0] for k in pf})

    def ci(metric, name):
        diff = pf[f"{name}__{metric}"] - pf[f"{args.ref}__{metric}"]
        bs = np.array([diff[b].mean() for b in boots])
        return diff.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    rows = []
    for n in names:
        g = ci("gt", n)
        f = ci("f16", n)
        rows.append((pf[f"{n}__f16"].mean(), n, pf[f"{n}__gt"].mean(), g, f))
    print(f"reference = {args.ref}   (Δ = candidate − reference; negative is better)")
    print(f"{'candidate':42s} {'gt_mae':>7s} {'Δgt [95% CI]':>26s} {'f16':>7s} "
          f"{'Δf16 [95% CI]':>26s}")
    for f16, n, gt, g, f in sorted(rows):
        print(f"{n:42s} {gt:7.4f} {g[0]:+8.4f} [{g[1]:+.4f},{g[2]:+.4f}] {f16:7.4f} "
              f"{f[0]:+8.4f} [{f[1]:+.4f},{f[2]:+.4f}]")


if __name__ == "__main__":
    main()
