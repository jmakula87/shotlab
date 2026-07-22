#!/usr/bin/env python
"""Build a NATIVE-SCALE ball-training set from a far session (2026-07-21).

The problem this fixes: the existing dataset_ball balls are 26-59px (median 46px)
in 1920x1080 frames -- from CLOSE footage. A far session's ball is ~20px, a scale
the detector has NEVER SEEN, so it misses far shots and corridor tiling (native
crops) actually HURT (the model was trained on downscaled balls). This builder
teaches the small-far-ball scale:

  * POSITIVES: for each ballistic-verified ball frame (from the cached rim-anchored
    tracks), crop a native `tile_w` x H tile positioned on the ball (exactly the
    geometry --tile detection uses) so the ball keeps its ~20px, label it in crop
    coords.
  * COPY-PASTE AUG: lift each real ball patch (color-masked) and paste it onto
    no-ball corridor backgrounds at 12-26px across the flight band -- the standard
    small-object recall booster (Kisantal 2019 / Ghiasi 2021), and the main volume
    multiplier off a few hundred real frames.
  * HARD NEGATIVES: no-ball crops (shooter in the orange shirt, dribbling, rim,
    flowers) with EMPTY labels so the model learns orange-blob != ball.

Merge the output into dataset_ball (bigger balls) and retrain at imgsz=tile_w so
the model sees a RANGE of scales (13/20/31/46px) -> scale-robust, and --tile pays
off. Usage:
  python tools/make_dataset_native.py --clips "data/raw/Camera 1/PXL_20260720_*.mp4" \
     --exclude 150124 --val-clip 153054 --aug 800 --neg 300 --out dataset_ball_native
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.detect_cache import _path as track_path, deserialize_detection
from shotlab.video_io import iter_frames
from tools.clean_dataset import crop_ok

TILE_W = 1280
RNG = np.random.default_rng(20260721)   # fixed: Date/Math.random unavailable-safe


def _yolo_line(cx, cy, r, w, h, pad=1.6, cls=0):
    bw = bh = 2 * r * pad
    return (f"{cls} {cx / w:.6f} {cy / h:.6f} "
            f"{min(bw, w) / w:.6f} {min(bh, h) / h:.6f}")


def _tile_x0(ball_x, w, tile_w):
    """Left edge of a tile_w-wide crop centered on the ball, clamped in-frame."""
    return int(min(max(ball_x - tile_w / 2, 0), max(0, w - tile_w)))


def _ball_frames(clip):
    """frame_idx -> (cx, cy, r) for every ballistic-verified in-shot ball frame."""
    tj = track_path(clip)
    if not os.path.exists(tj):
        return {}
    with open(tj, encoding="utf-8") as f:
        _, shots = deserialize_detection(json.load(f))
    out = {}
    for s in shots:
        for fr, x, y, r in zip(s.frames, s.xs, s.ys, s.radii):
            out[int(fr)] = (float(x), float(y), float(r))
    return out


def _mask_patch(patch, ball):
    """Alpha for a ball patch from its orange chroma (soft edge) so copy-paste
    doesn't drop a hard rectangle onto the background."""
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    lo = np.array([3, 70, 60]); hi = np.array([28, 255, 255])
    m = cv2.inRange(hsv, lo, hi)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    m = cv2.GaussianBlur(m, (3, 3), 0).astype(np.float32) / 255.0
    return m


def build(clips, out_dir, val_sub, stride, n_aug, n_neg):
    img_tr = os.path.join(out_dir, "images", "train"); os.makedirs(img_tr, exist_ok=True)
    img_va = os.path.join(out_dir, "images", "val"); os.makedirs(img_va, exist_ok=True)
    lbl_tr = os.path.join(out_dir, "labels", "train"); os.makedirs(lbl_tr, exist_ok=True)
    lbl_va = os.path.join(out_dir, "labels", "val"); os.makedirs(lbl_va, exist_ok=True)

    contact, patches, bg_pool = [], [], []
    counts = {"pos": 0, "aug": 0, "neg": 0}

    def split_dirs(sub):
        va = val_sub and val_sub in sub
        return (img_va if va else img_tr, lbl_va if va else lbl_tr, "val" if va else "train")

    for clip in clips:
        base = os.path.splitext(os.path.basename(clip))[0]
        want = _ball_frames(clip)
        if not want:
            print(f"  {base}: no track cache -> skipped (run build_session first)")
            continue
        keep = set(sorted(want)[::stride])
        img_d, lbl_d, split = split_dirs(base)
        n_pos_clip = 0
        for idx, frame in iter_frames(clip):
            h, w = frame.shape[:2]
            if idx in keep:
                x, y, r = want[idx]
                qa = frame[max(0, int(y-r)):int(y+r), max(0, int(x-r)):int(x+r)]
                if not crop_ok(qa, "orange"):
                    continue
                x0 = _tile_x0(x, w, TILE_W)
                tile = frame[:, x0:x0 + TILE_W]
                th, tw = tile.shape[:2]
                cx = x - x0
                stem = f"{base}_{idx:06d}"
                cv2.imwrite(os.path.join(img_d, stem + ".jpg"), tile,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                with open(os.path.join(lbl_d, stem + ".txt"), "w") as f:
                    f.write(_yolo_line(cx, y, r, tw, th) + "\n")
                counts["pos"] += 1; n_pos_clip += 1
                # stash a tight ball patch for copy-paste (train clips only)
                if split == "train" and 6 <= r <= 40:
                    pr = int(r * 1.7)
                    pp = frame[max(0, int(y-pr)):int(y+pr), max(0, int(x-pr)):int(x+pr)]
                    if pp.size and pp.shape[0] > 4 and pp.shape[1] > 4:
                        patches.append((pp, r))
                if len(contact) < 48:
                    c = tile[max(0, int(y-2*r)):int(y+2*r),
                             max(0, int(cx-2*r)):int(cx+2*r)]
                    if c.size:
                        contact.append(cv2.resize(c, (96, 96)))
            elif split == "train" and len(bg_pool) < 400 and idx % 25 == 0:
                # no-ball frame -> a background for aug + a hard negative source
                x0 = int(RNG.integers(0, max(1, w - TILE_W)))
                bg_pool.append(frame[:, x0:x0 + TILE_W].copy())
        print(f"  {base} -> {split}: {n_pos_clip} positive frames")

    # ---- copy-paste augmentation: real ball patches onto no-ball corridors ----
    if patches and bg_pool:
        for i in range(n_aug):
            bg = bg_pool[int(RNG.integers(len(bg_pool)))].copy()
            th, tw = bg.shape[:2]
            patch, r0 = patches[int(RNG.integers(len(patches)))]
            scale = float(RNG.uniform(12, 26)) / (2 * r0)      # target 12-26px ball
            ph = max(5, int(patch.shape[0] * scale)); pw = max(5, int(patch.shape[1] * scale))
            pr = cv2.resize(patch, (pw, ph))
            alpha = _mask_patch(pr, "orange")[..., None]
            # place in the upper flight band, away from the very edges
            px = int(RNG.integers(10, max(11, tw - pw - 10)))
            py = int(RNG.integers(int(th * 0.05), max(int(th * 0.05) + 1, int(th * 0.72))))
            roi = bg[py:py+ph, px:px+pw]
            if roi.shape[:2] != pr.shape[:2]:
                continue
            bg[py:py+ph, px:px+pw] = (alpha * pr + (1 - alpha) * roi).astype(np.uint8)
            r_new = (pw + ph) / 4
            stem = f"aug_{i:06d}"
            cv2.imwrite(os.path.join(img_tr, stem + ".jpg"), bg,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            with open(os.path.join(lbl_tr, stem + ".txt"), "w") as f:
                f.write(_yolo_line(px + pw / 2, py + ph / 2, r_new, tw, th) + "\n")
            counts["aug"] += 1
            if len(contact) < 96:
                c = bg[max(0, py-6):py+ph+6, max(0, px-6):px+pw+6]
                if c.size:
                    contact.append(cv2.resize(c, (96, 96)))

    # ---- hard negatives: no-ball corridor crops, EMPTY labels ----
    for i in range(min(n_neg, len(bg_pool))):
        bg = bg_pool[int(RNG.integers(len(bg_pool)))]
        stem = f"neg_{i:06d}"
        cv2.imwrite(os.path.join(img_tr, stem + ".jpg"), bg,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        open(os.path.join(lbl_tr, stem + ".txt"), "w").close()   # empty = negative
        counts["neg"] += 1

    with open(os.path.join(out_dir, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(out_dir)}\n")
        f.write("train: images/train\nval: images/val\nnc: 1\nnames: ['ball']\n")
    if contact:
        cols = 8; rows = (len(contact) + cols - 1) // cols
        sheet = np.zeros((rows * 96, cols * 96, 3), np.uint8)
        for i, t in enumerate(contact):
            r, c = divmod(i, cols); sheet[r*96:(r+1)*96, c*96:(c+1)*96] = t
        cv2.imwrite(os.path.join(out_dir, "label_contact_sheet.jpg"), sheet)
    print(f"\nPOSITIVES {counts['pos']} · AUG {counts['aug']} · NEG {counts['neg']}  "
          f"({len(patches)} ball patches, {len(bg_pool)} backgrounds)")
    print(f"dataset: {out_dir}  (check label_contact_sheet.jpg before training)")
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--val-clip", default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--aug", type=int, default=800, help="copy-paste synthetic count")
    ap.add_argument("--neg", type=int, default=300, help="hard-negative count")
    ap.add_argument("--out", default="dataset_ball_native")
    args = ap.parse_args(argv)
    clips = sorted(glob.glob(args.clips))
    clips = [c for c in clips if not any(x in os.path.basename(c) for x in args.exclude)]
    if not clips:
        print("no clips matched"); return 1
    build(clips, args.out, args.val_clip, args.stride, args.aug, args.neg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
