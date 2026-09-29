#!/usr/bin/env python3
"""validate.py — write / verify a calibration cache through the deployed path.

``--write`` builds a ``flashrt_calib_N.npz`` from pool indices, in the exact
format ``online_rollout.py::_load_or_build_calibration`` reads (images
``(N, V, 224, 224, 3)`` uint8, normalized states ``(N, S)``, view_keys).

``--calib_npz`` then scores a cache end to end the way deployment uses it: a
freshly loaded FP8 model, ``set_prompt(task, state=obs[0].state)`` then
``model.calibrate(obs, percentile)`` — no scale hot-swapping — on the same eval
frames/seeds as ``search.py``. One cache per process: calibrate() only runs once.

Usage:
    python examples/calib_search/validate.py --ckpt C --pool P \\
        --write out.npz --idx_from_result greedy_q99_N16
    python examples/calib_search/validate.py --ckpt C --pool P --calib_npz out.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from harness import Harness, obs_dict  # noqa: E402
from search import Evaluator  # noqa: E402

DEPLOY_TASK = "Fold the T-shirt properly"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--pool", required=True)
    p.add_argument("--write", default=None, help="write a calibration cache here")
    p.add_argument("--idx_from_result", default=None,
                   help="search_results.jsonl candidate name whose idx to write")
    p.add_argument("--calib_npz", default=None, help="cache to validate")
    p.add_argument("--percentile", type=float, default=99.9)
    p.add_argument("--label", default=None)
    args = p.parse_args()

    out_dir = Path(args.pool).parent
    pool = np.load(args.pool)
    tasks = [str(t) for t in pool["tasks"]]

    if args.write:
        recs = [json.loads(l) for l in open(out_dir / "search_results.jsonl")]
        rec = [r for r in recs if r["name"] == args.idx_from_result][-1]
        idx = np.asarray(rec["idx"])
        path = Path(args.write)
        if path.exists():
            raise FileExistsError(f"{path} exists — refusing to overwrite")
        np.savez_compressed(
            path, images=pool["images"][idx], states=pool["state_norm"][idx],
            view_keys=pool["view_keys"],
            # provenance (ignored by online_rollout.py)
            src_episode=pool["episode"][idx], src_frame=pool["frame"][idx],
            src_task=np.array([tasks[t] for t in pool["task_id"][idx]]).astype("U"),
        )
        print(f"Wrote {path}: {len(idx)} frames, episodes {pool['episode'][idx].tolist()}")

    if args.calib_npz:
        blob = np.load(args.calib_npz)
        assert [str(v) for v in blob["view_keys"]] == [str(v) for v in pool["view_keys"]]
        obs = [obs_dict(blob["images"][i], blob["states"][i])
               for i in range(len(blob["images"]))]
        h = Harness(args.ckpt, precision="fp8")
        # Mirrors online_rollout._calibrate_flashrt exactly.
        h.model.set_prompt(DEPLOY_TASK, state=obs[0]["state"])
        h.model.calibrate(obs, percentile=args.percentile, verbose=False)
        h.model._current_prompt = DEPLOY_TASK
        ref = np.load(out_dir / "ref_fp16.npz")
        m = Evaluator(h, pool, ref, tasks).run(h.get_scales())
        per_frame_gt = m.pop("_per_frame_gt")
        per_frame_f16 = m.pop("_per_frame_f16")
        label = args.label or f"deployed:{args.calib_npz}"
        np.savez(out_dir / f"validate_{Path(args.calib_npz).stem}_"
                 f"{abs(hash(args.calib_npz)) % 10**6}.npz",
                 gt=per_frame_gt, f16=per_frame_f16, scales=h.get_scales())
        with open(out_dir / "search_results.jsonl", "a") as f:
            f.write(json.dumps({"name": label, "stage": "validate", **m,
                                "src": args.calib_npz, "n": len(obs)}) + "\n")
        print(f"  {label}: gt={m['gt_mae']:.4f} gt10={m['gt_mae10']:.4f} "
              f"f16={m['f16_mae']:.4f} [fold {m['gt_fold']:.4f} home {m['gt_home']:.4f} "
              f"unfold {m['gt_unfold']:.4f}]")


if __name__ == "__main__":
    main()
