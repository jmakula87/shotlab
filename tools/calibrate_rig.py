#!/usr/bin/env python
"""Calibrate cameras from checkerboard clip(s). The day-one S8 tool.

Mono (works TODAY with just the Pixel -- pins its true focal length):
  python tools/calibrate_rig.py --mono data/raw/calib_pixel.mp4
      -> data/calibration/intrinsics_<clip>.json

Stereo (when the S8 lands; both cameras rolling on the SAME board wave,
one clap after both start):
  python tools/calibrate_rig.py data/raw/calib_A.mp4 data/raw/calib_B.mp4
      -> data/calibration/stereo_rig.json

A = the wide/arc camera (becomes the world origin), B = the close body-cam.
Prints the reprojection RMS -- under ~1 px is a good rig; much above that,
re-film (board flat? ruler measured 6.000 in? varied tilts? both cams sharp?).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.video_io import probe
from shotlab.stereo import (corners_from_video, paired_corners_from_videos,
                            calibrate_intrinsics, calibrate_stereo)
from shotlab.sync import sync_clips

OUT_DIR = os.path.join("data", "calibration")


def mono(path: str) -> int:
    info = probe(path)
    views = corners_from_video(path)
    print(f"{os.path.basename(path)}: board found in {len(views)} sampled frames")
    if len(views) < 4:
        print("not enough board views -- wave the board slower / closer / better lit")
        return 1
    K, dist, rms = calibrate_intrinsics(views.values(), (info.width, info.height))
    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(OUT_DIR, f"intrinsics_{stem}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"K": K.tolist(), "dist": dist.tolist(), "rms_px": rms,
                   "image_size": [info.width, info.height]}, f, indent=2)
    print(f"focal {K[0,0]:.0f}px  rms {rms:.2f}px  -> {out}")
    return 0


def stereo(path_a: str, path_b: str) -> int:
    info_a, info_b = probe(path_a), probe(path_b)
    s = sync_clips(path_a, path_b)
    if s is None:
        print("no audio in one of the clips -- record with sound (the clap IS the sync)")
        return 1
    offset, conf = s
    print(f"audio sync: B runs {offset:+.3f}s behind A (confidence {conf:.2f})")
    if conf < 0.3:
        print("weak sync lock -- re-film with one clean loud clap after both start")
        return 1
    va, vb = paired_corners_from_videos(path_a, path_b, offset,
                                        info_a.fps, info_b.fps)
    print(f"board seen by BOTH cameras in {len(va)} paired frames")
    if len(va) < 4:
        print("not enough shared views -- keep the board visible to both cameras")
        return 1
    rig = calibrate_stereo(va, vb, (info_a.width, info_a.height),
                           (info_b.width, info_b.height))
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "stereo_rig.json")
    rig.save(out)
    print(f"baseline {rig.baseline_ft:.1f} ft  rms {rig.rms_px:.2f}px  -> {out}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+", help="one clip with --mono, else A B")
    ap.add_argument("--mono", action="store_true",
                    help="single-camera intrinsics only")
    args = ap.parse_args(argv)
    if args.mono or len(args.clips) == 1:
        return mono(args.clips[0])
    return stereo(args.clips[0], args.clips[1])


if __name__ == "__main__":
    raise SystemExit(main())
