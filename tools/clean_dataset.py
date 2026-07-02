#!/usr/bin/env python
"""Remove bad auto-labels: keep only boxes whose crop actually looks like the
red/blue ball (contains BOTH red and blue). This drops the motion detector's
false positives -- the shooter's head (skin, no blue), foliage (green), and
gray motion blur -- before they pollute training.

Color is used ONLY to clean labels here; the trained CNN still learns appearance,
not a color rule.

Usage:
  python tools/clean_dataset.py --out dataset_ball [--red 0.05 --blue 0.03]
"""

from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np


def ball_colors(crop) -> tuple[float, float]:
    """Return (red_fraction, blue_fraction) of a crop."""
    if crop.size == 0:
        return 0.0, 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 80, 60), (12, 255, 255)) | \
          cv2.inRange(hsv, (168, 80, 60), (180, 255, 255))
    blue = cv2.inRange(hsv, (95, 50, 40), (135, 255, 255))
    n = crop.shape[0] * crop.shape[1]
    return float((red > 0).sum()) / n, float((blue > 0).sum()) / n


def orange_fraction(crop, sat_min: int = 90) -> float:
    """Fraction of a crop that is saturated basketball-orange (the 2026-07
    ball measures hue ~9, sat ~135 on this footage). Single-color QA is weaker
    than the old red-AND-blue rule, so the caller uses a higher threshold --
    the padded label box is ~30% ball by area when the label is right."""
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    orange = cv2.inRange(hsv, (3, sat_min, 60), (24, 255, 255))
    return float((orange > 0).sum()) / (crop.shape[0] * crop.shape[1])


def crop_ok(crop, ball: str, red=0.05, blue=0.03, orange=0.18,
            sat_min=90) -> bool:
    """Does this crop actually contain the ball we're training for?"""
    if ball == "redblue":
        rf, bf = ball_colors(crop)
        return rf >= red and bf >= blue
    return orange_fraction(crop, sat_min) >= orange


def box_from_label(line, w, h):
    cls, cx, cy, bw, bh = (float(v) for v in line.split())
    x0 = int((cx - bw / 2) * w); x1 = int((cx + bw / 2) * w)
    y0 = int((cy - bh / 2) * h); y1 = int((cy + bh / 2) * h)
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset_ball")
    ap.add_argument("--ball", choices=("redblue", "orange"), default="orange")
    ap.add_argument("--red", type=float, default=0.05)
    ap.add_argument("--blue", type=float, default=0.03)
    ap.add_argument("--orange", type=float, default=0.18)
    ap.add_argument("--sat-min", type=int, default=90,
                    help="min saturation for 'orange' (130 kills skin/leaves; "
                         "pair with a lower --orange like 0.05)")
    args = ap.parse_args(argv)

    kept_crops, removed = [], 0
    kept = 0
    for split in ("train", "val"):
        idir = os.path.join(args.out, "images", split)
        ldir = os.path.join(args.out, "labels", split)
        for img_path in glob.glob(os.path.join(idir, "*.jpg")):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(ldir, stem + ".txt")
            img = cv2.imread(img_path)
            ok = False
            if os.path.exists(lbl_path) and img is not None:
                h, w = img.shape[:2]
                with open(lbl_path) as f:
                    line = f.readline().strip()
                if line:
                    x0, y0, x1, y1 = box_from_label(line, w, h)
                    crop = img[y0:y1, x0:x1]
                    ok = crop_ok(crop, args.ball, args.red, args.blue,
                                 args.orange, args.sat_min)
                    if ok and len(kept_crops) < 48 and crop.size:
                        kept_crops.append(cv2.resize(crop, (96, 96)))
            if ok:
                kept += 1
            else:
                # remove the bad pair
                os.remove(img_path)
                if os.path.exists(lbl_path):
                    os.remove(lbl_path)
                removed += 1

    if kept_crops:
        cols = 8
        rows = (len(kept_crops) + cols - 1) // cols
        sheet = np.zeros((rows * 96, cols * 96, 3), np.uint8)
        for i, t in enumerate(kept_crops):
            r, c = divmod(i, cols)
            sheet[r*96:(r+1)*96, c*96:(c+1)*96] = t
        cv2.imwrite(os.path.join(args.out, "clean_contact_sheet.jpg"), sheet)

    print(f"kept {kept}, removed {removed}")
    for split in ("train", "val"):
        n = len(glob.glob(os.path.join(args.out, "images", split, "*.jpg")))
        print(f"  {split}: {n} clean labeled frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
