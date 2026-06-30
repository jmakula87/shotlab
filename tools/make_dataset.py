#!/usr/bin/env python
"""Auto-build a YOLO training set for a ball detector from your OWN clips.

No manual boxing: we run the motion detector + tracker + rim-anchored shot
detection, and use the ball positions DURING TRACKED SHOTS as labels. Those are
the motion detector's reliable subset (they survived the RANSAC arc fit and
reach the rim), so they make clean weak-supervision labels. Fine-tuning distills
them into an appearance-based detector that also fires where motion alone fails.

Outputs a standard YOLO dataset under <out>/ (images/ + labels/ + data.yaml) and
a contact sheet so you can eyeball label quality before training.

Usage:
  python tools/make_dataset.py --clips "data/raw/Hoops/PXL_*.mp4" \
     --exclude 190656 191516 191606 --val-clip 192125 --stride 2 --out dataset_ball
"""

from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.phase1_ball.detect import MotionBallDetector
from shotlab.phase1_ball.pipeline import run_phase1
from shotlab.court import auto_calibrate, filter_shots_by_rim
from shotlab.video_io import iter_frames
from tools.clean_dataset import ball_colors   # red/blue label QA filter


def _yolo_line(cx, cy, r, w, h, pad=1.6, cls=0):
    """ball center+radius -> YOLO normalized box (slightly padded)."""
    bw, bh = 2 * r * pad, 2 * r * pad
    return (f"{cls} {cx / w:.6f} {cy / h:.6f} "
            f"{min(bw, w) / w:.6f} {min(bh, h) / h:.6f}")


def process(clip, calib, out_dir, split, stride, contact):
    base = os.path.splitext(os.path.basename(clip))[0]
    res = run_phase1(clip, detector=MotionBallDetector())
    shots, _ = filter_shots_by_rim(res.shots, calib)
    if not shots:
        return 0
    # frame -> ball box for all frames inside a real shot
    want = {}
    for s in shots:
        for f, x, y, r in zip(s.frames, s.xs, s.ys, s.radii):
            want[int(f)] = (float(x), float(y), float(r))
    keep = sorted(want)[::stride]
    keepset = set(keep)

    img_dir = os.path.join(out_dir, "images", split)
    lbl_dir = os.path.join(out_dir, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    n = 0
    for idx, frame in iter_frames(clip):
        if idx not in keepset:
            continue
        h, w = frame.shape[:2]
        x, y, r = want[idx]
        # label QA: keep only boxes that actually contain the red/blue ball
        crop = frame[max(0, int(y-r)):int(y+r), max(0, int(x-r)):int(x+r)]
        rf, bf = ball_colors(crop)
        if rf < 0.05 or bf < 0.03:
            continue
        stem = f"{base}_{idx:06d}"
        cv2.imwrite(os.path.join(img_dir, stem + ".jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        with open(os.path.join(lbl_dir, stem + ".txt"), "w") as f:
            f.write(_yolo_line(x, y, r, w, h) + "\n")
        if len(contact) < 48:
            crop = frame[max(0, int(y-2*r)):int(y+2*r), max(0, int(x-2*r)):int(x+2*r)]
            if crop.size:
                contact.append(cv2.resize(crop, (96, 96)))
        n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--val-clip", default=None,
                    help="filename substring of the clip to hold out for val")
    ap.add_argument("--stride", type=int, default=2,
                    help="keep every Nth in-shot frame (reduce near-duplicates)")
    ap.add_argument("--out", default="dataset_ball")
    args = ap.parse_args(argv)

    clips = sorted(glob.glob(args.clips))
    clips = [c for c in clips if not any(x in os.path.basename(c) for x in args.exclude)]
    if not clips:
        print("no clips matched")
        return 1

    contact = []
    total = {"train": 0, "val": 0}
    for c in clips:
        calib = auto_calibrate(c, os.path.basename(c))
        if calib is None:
            print(f"  {os.path.basename(c)}: no rim, skipped")
            continue
        split = "val" if (args.val_clip and args.val_clip in os.path.basename(c)) else "train"
        n = process(c, calib, args.out, split, args.stride, contact)
        total[split] += n
        print(f"  {os.path.basename(c)} -> {split}: {n} labeled frames")

    # data.yaml
    with open(os.path.join(args.out, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(args.out)}\n")
        f.write("train: images/train\nval: images/val\n")
        f.write("nc: 1\nnames: ['ball']\n")
    # contact sheet
    if contact:
        cols = 8
        rows = (len(contact) + cols - 1) // cols
        sheet = np.zeros((rows * 96, cols * 96, 3), np.uint8)
        for i, t in enumerate(contact):
            r, c = divmod(i, cols)
            sheet[r*96:(r+1)*96, c*96:(c+1)*96] = t
        cv2.imwrite(os.path.join(args.out, "label_contact_sheet.jpg"), sheet)

    print(f"\nTRAIN {total['train']} / VAL {total['val']} labeled frames")
    print(f"dataset: {args.out}  (check label_contact_sheet.jpg before training)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
