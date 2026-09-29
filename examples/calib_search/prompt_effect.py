#!/usr/bin/env python3
"""prompt_effect.py — does the calibration prompt change the FP8 scales?

Recomputes per-GEMM amax for a subset of candidate frames under every task
prompt (same images, state and noise seed), and compares against the amax
collected with each frame's own prompt. Reports how much the reduced (max)
scale vector moves when every frame is calibrated with a single prompt — which
is what ``online_rollout._calibrate_flashrt`` does.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from harness import Harness  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--pool", required=True)
    p.add_argument("--per_task", type=int, default=40)
    args = p.parse_args()

    out_dir = Path(args.pool).parent
    pool = np.load(args.pool)
    tasks = [str(t) for t in pool["tasks"]]
    A = np.load(out_dir / "amax.npz")
    names = [str(n) for n in A["names"]]
    rng = np.random.default_rng(0)
    cand = np.where(pool["split"] == 0)[0]
    sel = np.concatenate([rng.choice(cand[pool["task_id"][cand] == t], args.per_task,
                                     replace=False) for t in range(3)])

    h = Harness(args.ckpt, precision="fp8")
    i0 = int(sel[0])
    h.bootstrap(tasks[pool["task_id"][i0]], pool["images"][i0], pool["state_norm"][i0])
    assert h.scale_names() == names

    # amax[p, k] = frame sel[k] under prompt p; seed matches collect.py
    amax = np.stack([np.stack([
        h.frame_amax(tasks[pt], pool["images"][i], pool["state_norm"][i], seed=10_000 + int(i))
        for i in sel]) for pt in range(3)])
    own = A["amax"][sel]
    # sanity: recomputing with the frame's own prompt reproduces collect.py
    own_re = amax[pool["task_id"][sel], np.arange(len(sel))]
    print(f"own-prompt recompute max rel diff: "
          f"{np.max(np.abs(own_re - own) / np.maximum(own, 1e-12)):.2e}")

    grp = np.array(["vision" if n.startswith("vision") else "encoder" if n.startswith(
        "encoder") else "decoder" for n in names])
    print("\nper-frame |log(amax_prompt / amax_own)| — median / p99 / max, by group")
    for pt in range(3):
        r = np.abs(np.log(np.maximum(amax[pt], 1e-12) / np.maximum(own, 1e-12)))
        print(f"  prompt={tasks[pt]!r}")
        for g in ["vision", "encoder", "decoder"]:
            x = r[:, grp == g]
            print(f"    {g:8s} {np.median(x):.4f} / {np.percentile(x, 99):.4f} / {x.max():.4f}")

    print("\nreduced (max over frames) scale vector: single-prompt vs own-prompt")
    ref = own.max(0)
    for pt in range(3):
        r = np.log(amax[pt].max(0) / ref)
        print(f"  {tasks[pt]!r:30s} mean|Δlog| {np.abs(r).mean():.4f}  max|Δlog| "
              f"{np.abs(r).max():.4f}  worst={names[int(np.abs(r).argmax())]}")
    allp = amax.max((0, 1))
    r = np.log(allp / ref)
    print(f"  {'all three prompts':30s} mean|Δlog| {np.abs(r).mean():.4f}  max|Δlog| "
          f"{np.abs(r).max():.4f}")
    np.savez(out_dir / "prompt_effect.npz", sel=sel, amax=amax, own=own,
             names=np.array(names).astype("U"))


if __name__ == "__main__":
    main()
