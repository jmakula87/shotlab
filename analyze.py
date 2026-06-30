#!/usr/bin/env python
"""ShotLab CLI -- analyze a shooting-workout video.

Examples
--------
  # classical color detector (clean, well-lit footage or the synthetic clip)
  python analyze.py data/raw/synthetic_side.mp4 --detector color

  # YOLO detector (real footage); pass a fine-tuned basketball model for best results
  python analyze.py data/raw/myworkout.mp4 --detector yolo --weights basketball.pt --ball-class 0

Outputs (under data/out/<stem>/):
  <stem>_overlay.mp4   tracked trajectory + fitted arc + metrics
  <stem>_shots.csv     per-shot table
  <stem>_shots.json    same, machine-readable
"""

from __future__ import annotations

import argparse
import os
import sys

from shotlab.config import load_targets
from shotlab.report import (build_session_table, build_combined_table,
                            write_outputs, print_table)


def build_detector(args):
    if args.detector == "color":
        from shotlab.phase1_ball.detect import ColorBallDetector
        return ColorBallDetector()
    if args.detector == "motion":
        from shotlab.phase1_ball.detect import MotionBallDetector
        return MotionBallDetector()
    if args.detector == "motion+color":
        from shotlab.phase1_ball.detect import MotionColorBallDetector
        return MotionColorBallDetector()
    if args.detector == "roboflow":
        from shotlab.phase1_ball.detect_roboflow import RoboflowBallDetector
        return RoboflowBallDetector(model_id=args.weights, api_key=args.rf_key,
                                    ball_class=args.rf_class, conf=args.conf)
    from shotlab.phase1_ball.detect_yolo import YoloBallDetector
    roi = tuple(args.roi) if args.roi else None
    return YoloBallDetector(weights=args.weights, ball_class=args.ball_class,
                            conf=args.conf, imgsz=args.imgsz, roi=roi)


def main(argv=None):
    ap = argparse.ArgumentParser(description="ShotLab shot analysis")
    ap.add_argument("video")
    ap.add_argument("--detector",
                    choices=["color", "motion", "motion+color", "yolo", "roboflow"],
                    default="color")
    ap.add_argument("--weights", default="yolo11n.pt",
                    help="YOLO .pt weights, OR a Roboflow model id 'workspace/project/version'")
    ap.add_argument("--ball-class", type=int, default=32,
                    help="class id of the ball (32=COCO sports ball; 0 for most basketball models)")
    ap.add_argument("--rf-key", default=None,
                    help="Roboflow API key (or set ROBOFLOW_API_KEY)")
    ap.add_argument("--rf-class", default="ball",
                    help="Roboflow ball class NAME substring (default 'ball')")
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--roi", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                    help="restrict detection to a crop (the shooting lane)")
    ap.add_argument("--rim-x", type=float, default=None,
                    help="rim pixel x for entry-angle evaluation (optional)")
    ap.add_argument("--config", default=None, help="path to targets.yaml")
    ap.add_argument("--out", default="data/out")
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--pose", action="store_true",
                    help="run Phase 2 form analysis (pose estimation)")
    ap.add_argument("--handedness", choices=["right", "left"], default=None,
                    help="shooting hand (default: from config)")
    ap.add_argument("--camera-angle", choices=["side_on", "front_on"], default=None,
                    help="override config camera angle")
    ap.add_argument("--pose-variant", choices=["lite", "full", "heavy"],
                    default="full")
    ap.add_argument("--spin", action="store_true",
                    help="run Phase 3 backspin estimation (slow-mo >=120fps only)")
    args = ap.parse_args(argv)

    if not os.path.exists(args.video):
        print(f"Video not found: {args.video}", file=sys.stderr)
        return 2

    targets = load_targets(args.config)
    stem = os.path.splitext(os.path.basename(args.video))[0]
    out_dir = os.path.join(args.out, stem)

    print(f"[1/3] Detecting + tracking ball ({args.detector}) ...")
    from shotlab.phase1_ball.pipeline import run_phase1
    detector = build_detector(args)
    res = run_phase1(args.video, detector=detector, rim_x=args.rim_x)
    print(f"      video {res.info.width}x{res.info.height} @ {res.info.fps:.0f}fps,"
          f" {res.info.n_frames} frames; slow-mo={res.info.is_slowmo}")
    print(f"      shots detected: {len(res.shots)}")

    # ---- optional Phase 2: pose / form ----
    sess = targets.get("session", {})
    handedness = args.handedness or sess.get("handedness", "right")
    camera_angle = args.camera_angle or sess.get("camera_angle", "side_on")
    phase2 = None
    if args.pose:
        if not res.shots:
            print("[2/4] Pose skipped (no shots detected).")
        else:
            print(f"[2/4] Phase 2 pose/form ({camera_angle}, {handedness}-handed) ...")
            from shotlab.phase2_pose.pipeline import run_phase2
            phase2 = run_phase2(args.video, res.shots, res.track,
                                handedness=handedness, camera_angle=camera_angle,
                                variant=args.pose_variant)
            poses_found = len(phase2.poses)
            print(f"      pose frames: {poses_found}"
                  + ("" if poses_found else "  (no person detected -- check framing/lighting)"))

    # ---- optional Phase 3: spin (gated on fps) ----
    spins = None
    if args.spin and res.shots:
        min_fps = targets.get("spin", {}).get("min_fps_for_spin", 110)
        print(f"[+] Phase 3 spin (need >={min_fps}fps; clip is {res.info.fps:.0f}fps) ...")
        from shotlab.phase3_spin.pipeline import run_phase3
        spins = run_phase3(args.video, res.shots, res.track, min_fps=min_fps)
        statuses = {s.status for s in spins.values()}
        if statuses == {"skipped"}:
            print(f"      spin skipped: not slow-mo. {next(iter(spins.values())).note}")
        else:
            print(f"      spin estimated on {sum(1 for s in spins.values() if s.status=='ok')} shot(s).")

    step = "3/4" if args.pose else "2/3"
    print(f"[{step}] Building session table ...")
    if phase2 is not None or spins is not None:
        df = build_combined_table(res.metrics, phase2.forms if phase2 else [],
                                  targets, spins=spins)
    else:
        df = build_session_table(res.metrics, targets)
    paths = write_outputs(df, out_dir, stem)
    print()
    print_table(df)
    print()
    print(f"      table: {paths['csv']}")

    if not args.no_overlay:
        step = "4/4" if args.pose else "3/3"
        print(f"[{step}] Rendering overlay video ...")
        from shotlab.phase1_ball.overlay import render_overlay
        from shotlab.video_io import to_h264
        ov = os.path.join(out_dir, f"{stem}_overlay.mp4")
        render_overlay(args.video, res, ov,
                       poses=(phase2.poses if phase2 else None),
                       release_frames=(phase2.release_frames if phase2 else None))
        ov_play = to_h264(ov, os.path.join(out_dir, f"{stem}_overlay_h264.mp4"))
        print(f"      overlay: {ov_play}")
    else:
        print("      Skipped overlay (--no-overlay)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
