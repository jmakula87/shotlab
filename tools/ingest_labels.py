#!/usr/bin/env python
"""Ingest human ball labels (from tools/make_label_task.py) into a native-scale
training set.

The labeler emits, per frame: the crop offset (cx0,cy0), the human-confirmed ball
position IN the crop (px,py), radius, and present (ball / no-ball). Here we turn
each into a native 1280x1080 corridor tile (matching --tile inference), so the
ball keeps its true ~20px:

  * present -> tile around the ball + a YOLO box at (cx0+px, cy0+py).
  * absent  -> tile around the predicted spot with an EMPTY label (hard negative:
    the small orange things that AREN'T the ball -- shirt, rim, distant blobs).

These are REAL small-far-ball labels (and real negatives) -- what the first
retrain lacked. Combine with the close-ball set and retrain frozen-backbone,
no synthetic aug (the recipe that worked). Usage:
  python tools/ingest_labels.py --labels ~/Downloads/ball_labels.json \
     --clips-dir "data/raw/Camera 1" --val-clip 153054 --out dataset_ball_labeled
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.video_io import iter_frames

TILE_W = 1280


def _tile_x0(ball_x, w, tile_w):
    return int(min(max(ball_x - tile_w / 2, 0), max(0, w - tile_w)))


def _yolo_line(cx, cy, r, w, h, pad=1.7, cls=0):
    bw = bh = 2 * r * pad
    return (f"{cls} {cx / w:.6f} {cy / h:.6f} "
            f"{min(bw, w) / w:.6f} {min(bh, h) / h:.6f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--clips-dir", default="data/raw/Camera 1")
    ap.add_argument("--val-clip", default="153054", help="clip substring held out for val")
    ap.add_argument("--out", default="dataset_ball_labeled")
    args = ap.parse_args(argv)

    labs = json.load(open(os.path.expanduser(args.labels), encoding="utf-8"))
    by_clip = collections.defaultdict(dict)        # clip -> frame -> label
    for L in labs:
        by_clip[L["clip"]][int(L["frame"])] = L
    print(f"{len(labs)} labels across {len(by_clip)} clips")

    for split in ("train", "val"):
        os.makedirs(os.path.join(args.out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.out, "labels", split), exist_ok=True)

    contact = []
    counts = {"pos": 0, "neg": 0}
    for stem, frames in by_clip.items():
        clip = os.path.join(args.clips_dir, stem + ".mp4")
        if not os.path.exists(clip):
            print(f"  {stem}: clip not found at {clip}, skipped"); continue
        split = "val" if args.val_clip in stem else "train"
        img_d = os.path.join(args.out, "images", split)
        lbl_d = os.path.join(args.out, "labels", split)
        need = set(frames)
        npos = nneg = 0
        for idx, frame in iter_frames(clip):
            if idx not in need:
                continue
            h, w = frame.shape[:2]
            L = frames[idx]
            bx = L["cx0"] + L["px"]; by = L["cy0"] + L["py"]
            # negative frames have no ball; center the tile on the crop the human
            # judged (crop half-window = 200, the labeler's CROP/2).
            center_x = bx if L["present"] else L["cx0"] + 200
            x0 = _tile_x0(center_x, w, TILE_W)
            tile = frame[:, x0:x0 + TILE_W]
            th, tw = tile.shape[:2]
            stem_i = f"{stem}_{idx:06d}"
            cv2.imwrite(os.path.join(img_d, stem_i + ".jpg"), tile,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            with open(os.path.join(lbl_d, stem_i + ".txt"), "w") as f:
                if L["present"]:
                    f.write(_yolo_line(bx - x0, by, max(4.0, L["r"]), tw, th) + "\n")
                    npos += 1
                    if len(contact) < 60:
                        c = tile[max(0, int(by-2*L["r"])):int(by+2*L["r"]),
                                 max(0, int(bx-x0-2*L["r"])):int(bx-x0+2*L["r"])]
                        if c.size:
                            contact.append(cv2.resize(c, (96, 96)))
                else:
                    pass                    # empty file = hard negative
                    nneg += 1
        counts["pos"] += npos; counts["neg"] += nneg
        print(f"  {stem} -> {split}: {npos} ball + {nneg} negatives")

    with open(os.path.join(args.out, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(args.out)}\n")
        f.write("train: images/train\nval: images/val\nnc: 1\nnames: ['ball']\n")
    if contact:
        cols = 10; rows = (len(contact) + cols - 1) // cols
        sheet = np.zeros((rows * 96, cols * 96, 3), np.uint8)
        for i, t in enumerate(contact):
            r, c = divmod(i, cols); sheet[r*96:(r+1)*96, c*96:(c+1)*96] = t
        cv2.imwrite(os.path.join(args.out, "label_contact_sheet.jpg"), sheet)
    print(f"\nPOSITIVES {counts['pos']} · NEGATIVES {counts['neg']}  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
