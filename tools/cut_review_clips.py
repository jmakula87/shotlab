#!/usr/bin/env python
"""Cut per-shot review clips from BOTH cameras for the Feel review view.

For every review candidate (verified make/miss) in a session: cut the wide
clip from ~3s before the flight (approach footwork) to ~1.5s after it
(landing), and the SAME real-time window from the paired close (S8) clip via
the audio-sync offset. Browser-playable H.264 + audio (the swish/clank is part
of feel). Sync offsets are cached to <session>/cam_sync.json.

Usage:
  python tools/cut_review_clips.py --session data/out/session_0710
  # single-camera sessions just get the wide angle:
  python tools/cut_review_clips.py --session data/out/session_0703 \
      --wide-dir data/raw/Hoops --no-close
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from shotlab.feelreview import (DEFAULT_PAIRS, close_window, review_candidates,
                                shot_windows)


def _cut(src: str, t0: float, t1: float, dst: str) -> bool:
    """Re-encode cut (stream-copy can't cut off-keyframe accurately)."""
    cmd = ["ffmpeg", "-y", "-ss", f"{t0:.3f}", "-to", f"{t1:.3f}", "-i", src,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart",
           "-loglevel", "error", dst]
    return subprocess.run(cmd).returncode == 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--wide-dir", default=os.path.join("data", "raw", "Camera 1"))
    ap.add_argument("--close-dir", default=os.path.join("data", "raw", "Camera 2"))
    ap.add_argument("--no-close", action="store_true",
                    help="single-camera session: skip the close angle")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    d = a.session
    df = pd.read_csv(os.path.join(d, "session_shots.csv"))
    tp = os.path.join(d, "make_truth.json")
    truth = json.load(open(tp, encoding="utf-8")) if os.path.exists(tp) else None
    cands = review_candidates(df, truth)
    outdir = os.path.join(d, "review_clips")
    os.makedirs(outdir, exist_ok=True)

    # audio-sync offsets, cached (each sync is a full audio pass)
    pairs = dict(DEFAULT_PAIRS)
    sync_p = os.path.join(d, "cam_sync.json")
    sync = json.load(open(sync_p, encoding="utf-8")) if os.path.exists(sync_p) else {}

    by_clip = {}
    for c in cands:
        by_clip.setdefault(c["clip"], []).append(c)

    n_wide = n_close = 0
    for clip, shots in sorted(by_clip.items()):
        wp = os.path.join(a.wide_dir, clip)
        if not os.path.exists(wp):
            print(f"  !! missing wide {wp} -- skipped ({len(shots)} shots)")
            continue
        windows = shot_windows(wp)
        cp = None
        offset = None
        if not a.no_close and clip in pairs:
            cp = os.path.join(a.close_dir, pairs[clip] + ".mp4")
            if not os.path.exists(cp):
                print(f"  !! missing close {cp} -- wide angle only")
                cp = None
            elif clip not in sync:
                from shotlab.sync import sync_clips
                off, conf = sync_clips(wp, cp)
                sync[clip] = {"offset": off, "conf": conf, "close": pairs[clip]}
                with open(sync_p, "w", encoding="utf-8") as f:
                    json.dump(sync, f, indent=1)
            if cp:
                offset = sync[clip]["offset"]
        stem = os.path.splitext(clip)[0]
        for c in shots:
            w = windows.get(c["shot_in_clip"])
            if w is None:
                print(f"  !! no window for {c['key']} -- skipped")
                continue
            wdst = os.path.join(outdir, f"{stem}_s{c['shot_in_clip']:03d}_wide.mp4")
            if a.overwrite or not os.path.exists(wdst):
                if _cut(wp, w[0], w[1], wdst):
                    n_wide += 1
            if cp:
                ct0, ct1 = close_window(w[0], w[1], offset)
                cdst = os.path.join(outdir,
                                    f"{stem}_s{c['shot_in_clip']:03d}_close.mp4")
                if a.overwrite or not os.path.exists(cdst):
                    if _cut(cp, ct0, ct1, cdst):
                        n_close += 1
        print(f"  {clip}: {len(shots)} shots"
              + (f" (sync {sync[clip]['offset']:+.2f}s)" if cp else " (wide only)"),
              flush=True)
    print(f"cut {n_wide} wide + {n_close} close clips -> {outdir} "
          f"({len(cands)} candidates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
