#!/usr/bin/env python
"""Train the visual make/miss model (shotlab.make_visual) from a session's audited
labels (make_truth.json). Reports honest cross-validated accuracy (5-fold and
leave-one-clip-out) and saves the model.

Usage:
  python tools/train_make_model.py --session data/out/session_0710 \
      --raw "data/raw/Camera 1" --out models/make_visual.joblib
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
from shotlab.make import classify_make
from shotlab.video_io import probe
from shotlab import make_visual as mv


def build_matrix(session_dir, raw_dir):
    truth = json.load(open(os.path.join(session_dir, "make_truth.json"), encoding="utf-8"))
    clips = sorted({k.split("|")[0] for k in truth})
    X, y, keyclip = [], [], []
    for clip in clips:
        vp = os.path.join(raw_dir, clip)
        if not os.path.exists(vp):
            print(f"  missing clip {clip}"); continue
        calib = auto_calibrate(vp, os.path.basename(session_dir))
        track, shots = deserialize_detection(json.load(open(_path(vp))))
        fps = probe(vp).fps
        bymap = {s.index: s for s in shots}
        rim = (calib.rim_x, calib.rim_y, calib.rim_radius_px)
        n = 0
        for key, lab in truth.items():
            c, idx = key.split("|"); idx = int(idx)
            if c != clip or lab not in ("make", "miss"):
                continue
            s = bymap.get(idx)
            if s is None:
                continue
            mr = classify_make(s, track, calib, fps=fps)
            if mr.rim_frame is None:
                continue
            sig = mv.extract_signals(vp, int(mr.rim_frame), rim)
            if sig is None:
                continue
            X.append(mv.features_from_signals(sig))
            y.append(1 if lab == "make" else 0)
            keyclip.append(clip); n += 1
        print(f"  {clip}: {n} labeled shots featurized", flush=True)
    return np.array(X), np.array(y), np.array(keyclip)


def report_cv(X, y, clips):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, accuracy_score
    # 5-fold repeated
    aucs, accs = [], []
    for rep in range(10):
        skf = StratifiedKFold(5, shuffle=True, random_state=rep)
        p = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            m = mv.train(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y, p)); accs.append(accuracy_score(y, p > 0.5))
    print(f"5-fold CV:  AUC {np.mean(aucs):.3f}+-{np.std(aucs):.3f}  "
          f"acc {np.mean(accs):.3f}+-{np.std(accs):.3f}")
    # leave-one-clip-out (cross-clip generalization)
    p = np.zeros(len(y))
    for c in np.unique(clips):
        tr, te = clips != c, clips == c
        if tr.sum() < 10 or te.sum() == 0:
            continue
        m = mv.train(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
    print(f"leave-one-clip-out:  AUC {roc_auc_score(y, p):.3f}  "
          f"acc {accuracy_score(y, p > 0.5):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--raw", default=os.path.join("data", "raw", "Camera 1"))
    ap.add_argument("--out", default="models/make_visual.joblib")
    a = ap.parse_args()
    print("extracting features ...", flush=True)
    X, y, clips = build_matrix(a.session, a.raw)
    print(f"\n{len(y)} shots  ({int(y.sum())} make / {int((1-y).sum())} miss), "
          f"{X.shape[1]} features")
    if len(y) < 20:
        print("too few labeled shots to train reliably"); return 1
    report_cv(X, y, clips)
    model = mv.train(X, y)
    path = mv.save(model, a.out)
    print(f"\nsaved model -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
