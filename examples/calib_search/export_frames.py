#!/usr/bin/env python3
"""export_frames.py — dump a calibration cache as viewable PNGs.

Writes one PNG per frame (the three camera views side by side, exactly the
224×224 uint8 tensors FlashRT calibrates on) plus contact sheets and an
index.csv with the provenance of each frame.

Usage:
    python examples/calib_search/export_frames.py --calib_npz <cache.npz> --out <dir>
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--calib_npz", required=True)
    p.add_argument("--pool", default="rollout_outputs_calib_search/pool.npz",
                   help="for episode lengths (progress %); optional")
    p.add_argument("--out", required=True)
    p.add_argument("--per_sheet", type=int, default=8)
    args = p.parse_args()

    b = np.load(args.calib_npz)
    imgs = b["images"]                                  # (N, V, 224, 224, 3)
    views = [str(v).rsplit(".", 1)[-1] for v in b["view_keys"]]
    N, V = imgs.shape[:2]
    has_src = "src_episode" in b.files
    lengths = {}
    if has_src and Path(args.pool).exists():
        pool = np.load(args.pool)
        lengths = {int(e): int(l) for e, l in zip(pool["episode"], pool["length"])}

    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    pad, head = 4, 22
    rows = []
    tiles = []
    for i in range(N):
        if has_src:
            ep, fr, task = int(b["src_episode"][i]), int(b["src_frame"][i]), str(b["src_task"][i])
            prog = f"{100 * fr / lengths[ep]:.0f}%" if ep in lengths else "?"
            label = f"#{i:02d}  ep{ep:03d} frame {fr} ({prog})  {task}"
        else:
            ep, fr, task, prog = -1, -1, "", ""
            label = f"#{i:02d}"
        W = V * 224 + (V - 1) * pad
        tile = Image.new("RGB", (W, 224 + head), "white")
        d = ImageDraw.Draw(tile)
        d.text((4, 5), label, fill="black")
        for v in range(V):
            tile.paste(Image.fromarray(imgs[i, v]), (v * (224 + pad), head))
        fname = f"{i:02d}_ep{ep:03d}_f{fr:05d}.png"
        tile.save(out / "frames" / fname)
        tiles.append(tile)
        rows.append({"idx": i, "episode": ep, "frame": fr, "progress": prog, "task": task,
                     "file": f"frames/{fname}",
                     "state_norm": " ".join(f"{x:.3f}" for x in b["states"][i])})

    for s in range(0, N, args.per_sheet):
        chunk = tiles[s:s + args.per_sheet]
        W, H = chunk[0].size
        sheet = Image.new("RGB", (W, len(chunk) * (H + pad) + 20), "white")
        ImageDraw.Draw(sheet).text((4, 4), "views: " + " | ".join(views), fill="black")
        for k, t in enumerate(chunk):
            sheet.paste(t, (0, 20 + k * (H + pad)))
        sheet.save(out / f"sheet_{s // args.per_sheet + 1}.png")

    with open(out / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {N} frames, {(N + args.per_sheet - 1) // args.per_sheet} sheets, "
          f"index.csv to {out}")


if __name__ == "__main__":
    main()
