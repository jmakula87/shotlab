#!/usr/bin/env python
"""Render per-shot review clips for one clip (ball trail + fitted arc + metrics),
for the dashboard's Shot-review view.

Usage:
  python tools/render_shots.py data/raw/Hoops/PXL_20260628_192125174.mp4 \
      --weights runs/detect/ball_finetune/weights/best.pt --stride 3
Writes data/out/<stem>/shots/shot_N_h264.mp4 + index.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.court import auto_calibrate, zone_for_release
from shotlab.phase1_ball.pipeline import run_phase1, metrics_for_shot
from shotlab.phase1_ball.overlay import render_shot_clip
from shotlab.video_io import to_h264, probe


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--weights", default="runs/detect/ball_finetune/weights/best.pt")
    ap.add_argument("--detector", default="yolo")
    ap.add_argument("--stride", default="auto")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--skip-done", action="store_true",
                    help="skip if this clip's shots/index.json already exists (resumable batch)")
    args = ap.parse_args(argv)

    stem = os.path.splitext(os.path.basename(args.video))[0]
    outdir = os.path.join("data", "out", stem, "shots")
    if args.skip_done and os.path.exists(os.path.join(outdir, "index.json")):
        print(f"skip {stem} (already rendered)"); return 0

    info = probe(args.video)
    if args.stride == "auto":
        stride = max(1, round(info.fps / 40), -(-info.n_frames // 7000))
    else:
        stride = int(args.stride)
    calib = auto_calibrate(args.video, os.path.basename(args.video))
    if calib is None:
        print("no rim detected"); return 1

    # cached detection (detect once, reuse) -- no re-detect on subsequent renders
    if args.detector == "yolo":
        from shotlab.detect_cache import detect_or_load
        print(f"loading/detecting shots (stride {stride}) ...")
        track, shots = detect_or_load(args.video, args.weights, calib, stride,
                                      args.max_frames, imgsz=640)
    else:
        from shotlab.phase1_ball.detect import MotionBallDetector
        det = MotionBallDetector()
        print(f"detecting shots (stride {stride}) ...")
        res = run_phase1(args.video, detector=det, calib=calib, stride=stride,
                         max_frames=args.max_frames)
        track, shots = res.track, res.shots
    os.makedirs(outdir, exist_ok=True)

    index = []
    for s in shots:
        m = metrics_for_shot(s, rim_x=calib.rim_x)
        z = zone_for_release((s.xs[s.ys.argmax()], s.ys.max()), calib)
        txt = [f"SHOT {s.index}", f"release {m.release_angle_deg}",
               f"entry {m.entry_angle_deg}", f"zone {z['zone']}"]
        raw = os.path.join(outdir, f"shot_{s.index}.mp4")
        render_shot_clip(args.video, s, track, raw, fps=info.fps, metrics_text=txt,
                         rim=(calib.rim_x, calib.rim_y, calib.rim_radius_px))
        h264 = to_h264(raw, os.path.join(outdir, f"shot_{s.index}_h264.mp4"))
        index.append({"shot": s.index, "file": os.path.basename(h264),
                      "release": m.release_angle_deg, "entry": m.entry_angle_deg,
                      "zone": z["zone"]})
        print(f"  shot {s.index} -> {os.path.basename(h264)}")

    with open(os.path.join(outdir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print(f"{len(index)} shot clips in {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
