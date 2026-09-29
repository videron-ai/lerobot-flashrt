#!/usr/bin/env python3
"""collect.py — per-frame FP8 amax for every pool frame, plus reference outputs.

Modes:
  --mode amax   FP8 model: per-GEMM dynamic amax for each pool frame (and for the
                frames of any --baseline_npz calibration caches).
  --mode ref    FP16 model: reference action chunks on the eval frames, with the
                same per-frame noise seed the FP8 search uses, so FP8-vs-FP16
                isolates quantization error from flow-matching sampling noise.

Usage (inside the FlashRT container):
    python examples/calib_search/collect.py --mode amax --ckpt ... --pool .../pool.npz
    python examples/calib_search/collect.py --mode ref  --ckpt ... --pool .../pool.npz
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from harness import Harness  # noqa: E402

DEFAULT_TASK = "Fold the T-shirt properly"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["amax", "ref"], required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--pool", required=True)
    p.add_argument("--baseline_npz", nargs="*", default=[],
                   help="existing flashrt_calib_N.npz caches to collect amax for")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    pool = np.load(args.pool)
    tasks = [str(t) for t in pool["tasks"]]
    N = len(pool["split"])
    out_dir = Path(args.pool).parent

    h = Harness(args.ckpt, precision="fp8" if args.mode == "amax" else "fp16")
    ev = np.where(pool["split"] == 1)[0]
    i0 = int(ev[0])
    h.bootstrap(tasks[pool["task_id"][i0]], pool["images"][i0], pool["state_norm"][i0])

    t0 = time.time()
    if args.mode == "amax":
        names = h.scale_names()
        # Repeatability check: same frame + seed twice must give identical amax.
        a = h.frame_amax(tasks[pool["task_id"][0]], pool["images"][0], pool["state_norm"][0], 0)
        b = h.frame_amax(tasks[pool["task_id"][0]], pool["images"][0], pool["state_norm"][0], 0)
        print(f"  {len(names)} scales; repeat max|Δ|/amax = "
              f"{np.max(np.abs(a - b) / np.maximum(a, 1e-12)):.2e}")

        amax = np.zeros((N, len(names)), np.float32)
        for i in range(N):
            amax[i] = h.frame_amax(tasks[pool["task_id"][i]], pool["images"][i],
                                   pool["state_norm"][i], seed=10_000 + i)
            if (i + 1) % 200 == 0:
                print(f"  amax {i + 1}/{N}  {time.time() - t0:.0f}s", flush=True)

        extra = {}
        for bi, path in enumerate(args.baseline_npz):
            blob = np.load(path)
            imgs, states = blob["images"], blob["states"]
            ba = np.stack([h.frame_amax(DEFAULT_TASK, imgs[j], states[j], seed=20_000 + j)
                           for j in range(len(imgs))])
            extra[f"baseline{bi}"] = ba
            extra[f"baseline{bi}_path"] = np.array(path)
            print(f"  baseline {path}: {len(imgs)} frames")
        out = args.out or out_dir / "amax.npz"
        np.savez(out, amax=amax, names=np.array(names).astype("U"), **extra)
    else:
        chunks = np.zeros((len(ev), 30, 32), np.float32)
        for k, i in enumerate(ev):
            chunks[k] = h.predict(tasks[pool["task_id"][i]], pool["images"][i],
                                  pool["state_norm"][i], seed=int(i), warm=True)
            if (k + 1) % 100 == 0:
                print(f"  ref {k + 1}/{len(ev)}  {time.time() - t0:.0f}s", flush=True)
        # Seed-repeatability check on the reference itself.
        again = h.predict(tasks[pool["task_id"][ev[0]]], pool["images"][ev[0]],
                          pool["state_norm"][ev[0]], seed=int(ev[0]), warm=True)
        print(f"  ref repeat max|Δ| = {np.abs(again - chunks[0]).max():.2e}")
        out = args.out or out_dir / "ref_fp16.npz"
        np.savez(out, chunks=chunks, eval_idx=ev)
    print(f"Saved {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
