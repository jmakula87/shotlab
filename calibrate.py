#!/usr/bin/env python
"""Manual rim calibration -- the reliable way to anchor shots.

Auto rim-detection (orange blob) is fragile: a red/orange ball, a chair, or
foliage can fool it. Since the tripod only moves a few times in a session, you
calibrate each camera position ONCE by clicking the rim.

Usage:
    python calibrate.py data/raw/Hoops/PXL_20260628_191733118.mp4 \
        --name Hoops_left --apply-to 1917 1921 1922 1925

Controls (a window opens on a mid-clip frame):
    left-click rim CENTER, then left-click the rim EDGE (sets radius)
    n / p : next / previous sample frame      r : reset clicks
    s : save calibration                       q : quit without saving

Writes config/calibration_<name>.json. Pass --apply-to with filename substrings
to record which clips share this camera position (build_session can then map each
clip to the right calibration). With no GUI available, use --rim X Y --radius R.
"""

from __future__ import annotations

import argparse
import json
import os

import cv2

from shotlab.court import Calibration
from shotlab.video_io import probe


def _headless_save(args, info):
    calib = Calibration(
        session=args.name, image_w=info.width, image_h=info.height,
        rim_x=float(args.rim[0]), rim_y=float(args.rim[1]),
        rim_radius_px=float(args.radius),
        shot_gate_px=max(2.0 * float(args.radius), 90.0),
        note=f"manual (headless); applies to {args.apply_to}")
    path = os.path.join("config", f"calibration_{args.name}.json")
    calib.save(path)
    _save_map(args)
    print(f"saved {path}: rim ({args.rim[0]},{args.rim[1]}) r={args.radius}")


def _save_map(args):
    """Record which clip-substrings use this calibration."""
    map_path = os.path.join("config", "calibration_map.json")
    m = {}
    if os.path.exists(map_path):
        with open(map_path, encoding="utf-8") as f:
            m = json.load(f)
    for sub in (args.apply_to or []):
        m[sub] = args.name
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--name", required=True, help="calibration name (camera position)")
    ap.add_argument("--apply-to", nargs="*", default=[],
                    help="clip filename substrings sharing this camera position")
    ap.add_argument("--rim", nargs=2, type=float, metavar=("X", "Y"),
                    help="headless: rim center pixel")
    ap.add_argument("--radius", type=float, help="headless: rim half-width px")
    args = ap.parse_args(argv)

    info = probe(args.video)
    if args.rim and args.radius:
        _headless_save(args, info)
        return 0

    # interactive
    n = info.n_frames
    samples = [int(n * f) for f in (0.2, 0.35, 0.5, 0.65, 0.8)]
    si = 0
    clicks = []

    def load(idx):
        cap = cv2.VideoCapture(args.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        cap.release()
        return fr if ok else None

    frame = load(samples[si])
    win = f"calibrate: {os.path.basename(args.video)}"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
            clicks.append((x, y))

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        disp = frame.copy()
        if len(clicks) >= 1:
            cv2.circle(disp, clicks[0], 4, (0, 255, 0), -1)
        if len(clicks) == 2:
            r = int(((clicks[1][0]-clicks[0][0])**2 + (clicks[1][1]-clicks[0][1])**2) ** .5)
            cv2.circle(disp, clicks[0], r, (0, 255, 255), 2)
        cv2.putText(disp, "click rim CENTER then EDGE | s=save n/p=frame r=reset q=quit",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(win, disp)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('q'):
            break
        if k == ord('r'):
            clicks.clear()
        if k in (ord('n'), ord('p')):
            si = (si + (1 if k == ord('n') else -1)) % len(samples)
            frame = load(samples[si]); clicks.clear()
        if k == ord('s') and len(clicks) == 2:
            cx, cy = clicks[0]
            r = float(((clicks[1][0]-cx)**2 + (clicks[1][1]-cy)**2) ** .5)
            calib = Calibration(
                session=args.name, image_w=info.width, image_h=info.height,
                rim_x=float(cx), rim_y=float(cy), rim_radius_px=r,
                shot_gate_px=max(2.0 * r, 90.0),
                note=f"manual click; applies to {args.apply_to}")
            path = os.path.join("config", f"calibration_{args.name}.json")
            calib.save(path)
            _save_map(args)
            print(f"saved {path}")
            break
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
