#!/usr/bin/env python
"""Join per-release elbow flare (from the close-camera 3D analysis) onto the wide
camera's rim-anchored shots, so flare becomes a per-shot measurable the Shot
Explorer can filter/sort on.

Each flare reading is timestamped in the CLOSE clip; we audio-sync it to the WIDE
clip and drop it into whichever wide shot's flight window it lands in. Writes
<session>/flare_by_shot.json = {"<wide_clip>|<shot_in_clip>": flare_deg}.

Usage:
  python tools/join_flare_to_shots.py --session data/out/session_0710 \
      --a3d data/out/session_0710_3d/analysis3d.json \
      --wide-dir "data/raw/Camera 1" --close-dir "data/raw/Camera 2"
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.court import auto_calibrate
from shotlab.detect_cache import _path, deserialize_detection
from shotlab.sync import sync_clips
from shotlab.video_io import frame_times, probe

# wide clip <-> close clip pairing for the 07-10 two-camera session (filename order)
PAIRS = [("PXL_20260710_175751234.mp4", "20260710_135805"),
         ("PXL_20260710_180449842.mp4", "20260710_140431"),
         ("PXL_20260710_181146426.mp4", "20260710_141132"),
         ("PXL_20260710_181811930.mp4", "20260710_141758")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--a3d", required=True)
    ap.add_argument("--wide-dir", default=os.path.join("data", "raw", "Camera 1"))
    ap.add_argument("--close-dir", default=os.path.join("data", "raw", "Camera 2"))
    a = ap.parse_args()

    a3d = json.load(open(a.a3d, encoding="utf-8"))
    flare_shots = a3d["flare"]["shots"]
    # group flare readings by close clip -> [(frame, flare_deg)]
    by_close = {}
    for s in flare_shots:
        by_close.setdefault(str(s["clip"]), []).append((int(s["frame"]), s["flare_deg"]))

    out = {}
    for wide_name, close_name in PAIRS:
        wp = os.path.join(a.wide_dir, wide_name)
        cp = os.path.join(a.close_dir, close_name + ".mp4")
        if not (os.path.exists(wp) and os.path.exists(cp)):
            print(f"  missing pair {wide_name} / {close_name}"); continue
        offset, conf = sync_clips(wp, cp)          # wide_time = close_time + offset
        calib = auto_calibrate(wp, os.path.basename(a.session))
        track, shots = deserialize_detection(json.load(open(_path(wp))))
        wt = frame_times(wp)
        # wide shot -> (t_lo, t_hi) flight window (pad a bit past last detection)
        windows = []
        for sh in shots:
            f0, f1 = int(sh.frames[0]), int(sh.frames[-1])
            t0 = wt.get(f0, f0 / 30.0); t1 = wt.get(f1, f1 / 30.0) + 0.8
            windows.append((sh.index, t0, t1))
        ct = frame_times(cp)
        assigned = {}
        for frame, fl in by_close.get(close_name, []):
            wide_t = ct.get(frame, frame / 30.0) + offset
            hit = [idx for idx, t0, t1 in windows if t0 - 0.4 <= wide_t <= t1 + 0.4]
            if len(hit) == 1:
                assigned.setdefault(hit[0], []).append(fl)
        for idx, vals in assigned.items():
            out[f"{wide_name}|{idx}"] = round(float(np.median(vals)), 2)
        print(f"  {wide_name}: sync {offset:+.2f}s conf {conf:.2f}, "
              f"{len(assigned)} shots got flare", flush=True)

    dst = os.path.join(a.session, "flare_by_shot.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {dst} ({len(out)} shots with flare)")


if __name__ == "__main__":
    main()
