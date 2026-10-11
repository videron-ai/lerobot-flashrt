#!/usr/bin/env python3
"""Decode the training dataset into compact FlashRT-ready arrays.

Produces (under OUT):
  eval/ep{E:03d}.npz  every frame of each held-out eval episode:
                      imgs (N,V,224,224,3) uint8, state_raw (N,16), action (N,16)
  pool.npz            calibration candidate pool, POOL_PER_EP frames per
                      non-eval episode: imgs, state_raw, ep, frame, task

Images go through the same resize_with_pad_torch + round-to-uint8 as
offline_rollout.extract_flashrt_inputs (the preprocessor is IDENTITY on
VISUAL, so doing it on CPU here is equivalent). State normalization happens
later on GPU with the checkpoint's own preprocessor.
"""
import argparse
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = "/nas/datasets_archive/openarm_bimanual_shirt_folding_gen3_345_episodes_fold_unfold_home_rightfirst"
REPO = "videron/openarm_bimanual_shirt_folding_gen3_345_episodes_fold_unfold_home_rightfirst"
VIEWS = ["observation.images.left_wrist", "observation.images.right_wrist",
         "observation.images.base"]  # checkpoint's VISUAL order


def episodes_table():
    e = pd.concat([pd.read_parquet(f, columns=["episode_index", "length", "tasks"])
                   for f in glob.glob(f"{ROOT}/meta/episodes/*/*.parquet")])
    e["task"] = e.tasks.map(lambda t: t[0])
    return e.sort_values("episode_index").reset_index(drop=True)


def decode(job):
    """job = (episode, [frame offsets]) -> (episode, offsets, imgs, state, action)"""
    import torch
    torch.set_num_threads(2)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.common.vla_utils import resize_with_pad_torch
    ep, offs = job
    ds = LeRobotDataset(REPO, root=ROOT, episodes=[ep])
    imgs, st, ac = [], [], []
    for o in offs:
        f = ds[o]
        assert int(f["episode_index"]) == ep and int(f["frame_index"]) == o
        views = []
        for k in VIEWS:
            im = resize_with_pad_torch(f[k].unsqueeze(0).float(), 224, 224)
            views.append(im.squeeze(0).permute(1, 2, 0).mul(255).round()
                         .clamp(0, 255).to(torch.uint8).numpy())
        imgs.append(np.stack(views))
        st.append(f["observation.state"].numpy())
        ac.append(f["action"].numpy())
    return ep, offs, np.stack(imgs), np.stack(st).astype(np.float32), np.stack(ac).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/tmp/calib_search_data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_fold", type=int, default=6)
    p.add_argument("--n_unfold", type=int, default=2)
    p.add_argument("--n_home", type=int, default=2)
    p.add_argument("--pool_per_ep", type=int, default=8)
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()
    os.makedirs(f"{a.out}/eval", exist_ok=True)

    e = episodes_table()
    rng = np.random.default_rng(a.seed)
    eval_eps = []
    for task, n in [("Fold the T-shirt properly", a.n_fold),
                    ("Unfold the T-shirt", a.n_unfold),
                    ("Return to home position", a.n_home)]:
        eval_eps += sorted(rng.choice(e[e.task == task].episode_index.values, n,
                                      replace=False).tolist())
    print("eval episodes:", eval_eps)
    length = dict(zip(e.episode_index, e.length))
    task_of = dict(zip(e.episode_index, e.task))

    # Eval: every frame, split into 150-frame jobs so workers share long episodes.
    jobs = []
    for ep in eval_eps:
        if os.path.exists(f"{a.out}/eval/ep{ep:03d}.npz"):
            continue
        L = int(length[ep])
        jobs += [(ep, list(range(s, min(s + 150, L)))) for s in range(0, L, 150)]
    # Pool: stratified in time — one random frame from each of pool_per_ep equal bins.
    pool_jobs = []
    for ep in e.episode_index:
        if ep in eval_eps:
            continue
        L = int(length[ep])
        edges = np.linspace(0, L, a.pool_per_ep + 1)
        offs = sorted({int(rng.integers(int(edges[i]), max(int(edges[i]) + 1, int(edges[i + 1]))))
                       for i in range(a.pool_per_ep)})
        pool_jobs.append((int(ep), offs))

    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        parts = {}
        for i, (ep, offs, im, st, ac) in enumerate(ex.map(decode, jobs)):
            parts.setdefault(ep, []).append((offs, im, st, ac))
            if sum(len(x[0]) for x in parts[ep]) == length[ep]:
                ch = sorted(parts.pop(ep), key=lambda x: x[0][0])
                np.savez(f"{a.out}/eval/ep{ep:03d}.npz",
                         imgs=np.concatenate([c[1] for c in ch]),
                         state_raw=np.concatenate([c[2] for c in ch]),
                         action=np.concatenate([c[3] for c in ch]),
                         task=task_of[ep])
                print(f"  eval ep{ep:03d} done ({length[ep]} frames) {time.time()-t0:.0f}s", flush=True)
        print(f"eval decode done in {time.time()-t0:.0f}s", flush=True)

        P = {"imgs": [], "state_raw": [], "ep": [], "frame": [], "task": []}
        for i, (ep, offs, im, st, ac) in enumerate(ex.map(decode, pool_jobs)):
            P["imgs"].append(im); P["state_raw"].append(st)
            P["ep"] += [ep] * len(offs); P["frame"] += offs
            P["task"] += [task_of[ep]] * len(offs)
            if (i + 1) % 50 == 0:
                print(f"  pool {i+1}/{len(pool_jobs)} eps {time.time()-t0:.0f}s", flush=True)
    np.savez(f"{a.out}/pool.npz", imgs=np.concatenate(P["imgs"]),
             state_raw=np.concatenate(P["state_raw"]), ep=np.array(P["ep"]),
             frame=np.array(P["frame"]), task=np.array(P["task"]),
             length=np.array([length[x] for x in P["ep"]]),
             eval_eps=np.array(eval_eps))
    print(f"pool: {len(P['ep'])} frames; total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
