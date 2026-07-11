#!/usr/bin/env python
"""Apply the trained visual make/miss model to a session's shots and write
<session>/make_pred.json = {"<clip>|<shot>": {"made": bool, "prob": float}}.

The make/miss audit then shows the model's call + confidence and can surface the
shots it's UNSURE about first, so future sessions need far less hand-review.

Usage:
  python tools/apply_make_model.py --session data/out/session_0710 \
      --raw "data/raw/Camera 1" --model models/make_visual.joblib
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.court import auto_calibrate
from shotlab.detect_cache import _path, deserialize_detection
from shotlab.make import classify_make
from shotlab.video_io import probe
from shotlab import make_visual as mv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--raw", default=os.path.join("data", "raw", "Camera 1"))
    ap.add_argument("--model", default="models/make_visual.joblib")
    ap.add_argument("--clips", nargs="*", default=None,
                    help="clip filenames to process (default: all in the session csv)")
    a = ap.parse_args()

    import pandas as pd
    model = mv.load(a.model)
    df = pd.read_csv(os.path.join(a.session, "session_shots.csv"))
    clips = a.clips or sorted(df["clip"].astype(str).unique())
    preds = {}
    for clip in clips:
        vp = os.path.join(a.raw, clip)
        if not os.path.exists(vp):
            print(f"  missing {clip}"); continue
        calib = auto_calibrate(vp, os.path.basename(a.session))
        track, shots = deserialize_detection(json.load(open(_path(vp))))
        fps = probe(vp).fps
        rim = (calib.rim_x, calib.rim_y, calib.rim_radius_px)
        n = 0
        for s in shots:
            mr = classify_make(s, track, calib, fps=fps)
            if mr.rim_frame is None:
                continue
            sig = mv.extract_signals(vp, int(mr.rim_frame), rim)
            if sig is None:
                continue
            made, p = mv.predict(model, mv.features_from_signals(sig))
            preds[f"{clip}|{s.index}"] = {"made": bool(made), "prob": round(p, 3)}
            n += 1
        print(f"  {clip}: {n} shots predicted", flush=True)

    out = os.path.join(a.session, "make_pred.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2)
    # if the audit truth exists, report honest accuracy
    tpath = os.path.join(a.session, "make_truth.json")
    if os.path.exists(tpath):
        truth = json.load(open(tpath, encoding="utf-8"))
        both = [(preds[k]["made"], truth[k]) for k in preds
                if k in truth and truth[k] in ("make", "miss")]
        if both:
            acc = sum((m and t == "make") or (not m and t == "miss") for m, t in both) / len(both)
            print(f"\nvs your labels: {100*acc:.0f}% agreement on {len(both)} shots "
                  f"(in-sample — honest held-out was ~87%)")
    print(f"wrote {out} ({len(preds)} predictions)")


if __name__ == "__main__":
    main()
