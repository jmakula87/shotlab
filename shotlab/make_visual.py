"""Learned make/miss from the RIM + NET region -- what a human actually watches.

The tracked ball is lost right at the rim, so ball-trajectory geometry can't tell
make from miss (~coin flip) and the far mic makes audio weak (~0.6 AUC). A person
doesn't track the ball though -- they watch the NET whip and the ball drop THROUGH
and BELOW the rim (a make) vs carom off to the SIDE (a miss). This module extracts
exactly those cues from a small ROI around the (already-detected) rim and feeds a
model trained on your own audited make/miss labels.

Validated on session_0710 (74 labeled shots): leave-one-clip-out AUC 0.94,
accuracy 86% -- vs 0.49 for the old geometric heuristic. Discovered via a Fable
research pass; the winning cues are net-motion-vs-surroundings, net white-pixel
occlusion as the ball passes, and orange mass appearing BELOW the net vs to the
side. Everything is HSV color + frame differencing -- cheap, CPU-only, no ball
tracking at the rim required.
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = ["netVSflank", "netVSbase", "o_below", "o_side", "o_net",
                 "white_occl", "lateNetVSflank"]
PRE, POST = 20, 55          # frames sampled before/after the rim frame
_MODEL_PATH_DEFAULT = "models/make_visual.joblib"


def _orange_mask(bgr):
    import cv2
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return ((hsv[..., 0] >= 2) & (hsv[..., 0] <= 26) & (hsv[..., 1] >= 55)
            & (hsv[..., 2] >= 35)).astype(np.uint8)


def _white_mask(bgr):
    import cv2
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return ((hsv[..., 1] <= 70) & (hsv[..., 2] >= 140)).astype(np.uint8)


def extract_signals(clip_path, rim_frame, rim) -> dict | None:
    """Per-frame region signals over [rim_frame-PRE, rim_frame+POST]. `rim` =
    (rim_x, rim_y, rim_radius_px). Returns a dict of 1-D arrays, or None."""
    import cv2
    rx, ry, rr = float(rim[0]), float(rim[1]), float(rim[2])
    if rr <= 0 or rim_frame is None:
        return None
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        return None
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x0, x1 = int(max(0, rx - 4.5 * rr)), int(min(W, rx + 4.5 * rr))
    y0, y1 = int(max(0, ry - 3 * rr)), int(min(H, ry + 6 * rr))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx, dy = (xx - rx) / rr, (yy - ry) / rr
    regions = {
        "in_rim": (np.abs(dx) < 1.3) & (np.abs(dy) < 1.3),
        "net": (np.abs(dx) < 1.35) & (dy >= 0.2) & (dy < 2.6),
        "below_net": (np.abs(dx) < 2.5) & (dy >= 2.6),
        "side": (np.abs(dx) >= 1.6) & (np.abs(dx) < 4.5) & (np.abs(dy) < 2.5),
    }
    netbox = (int(rx - 1.5 * rr - x0), int(ry - y0),
              int(rx + 1.5 * rr - x0), int(ry + 2.8 * rr - y0))
    flankL = (np.abs(dx + 3.2) < 1.2) & (dy >= 0) & (dy < 2.6)
    flankR = (np.abs(dx - 3.2) < 1.2) & (dy >= 0) & (dy < 2.6)

    keys = ["o_in_rim", "o_net", "o_below_net", "o_side",
            "net_motion", "net_white_count", "flank_motion"]
    sig = {k: [] for k in keys}
    f_start = max(0, rim_frame - PRE); f_stop = rim_frame + POST + 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
    prev_gray = prev_white = None
    f = f_start
    try:
        while f < f_stop:
            ok, fr = cap.read()
            if not ok:
                break
            roi = fr[y0:y1, x0:x1]
            g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.int16)
            om = _orange_mask(roi); wm = _white_mask(roi)
            sig["o_in_rim"].append(float(om[regions["in_rim"]].sum()))
            sig["o_net"].append(float(om[regions["net"]].sum()))
            sig["o_below_net"].append(float(om[regions["below_net"]].sum()))
            sig["o_side"].append(float(om[regions["side"]].sum()))
            if prev_gray is not None:
                ad = np.abs(g - prev_gray)
                wmask = ((wm | prev_white) > 0)
                sub = ad[netbox[1]:netbox[3], netbox[0]:netbox[2]]
                subw = wmask[netbox[1]:netbox[3], netbox[0]:netbox[2]]
                sig["net_motion"].append(float(sub[subw].mean()) if subw.sum() > 20 else 0.0)
                sig["net_white_count"].append(float(subw.sum()))
                sig["flank_motion"].append(float(ad[flankL].mean() + ad[flankR].mean()) / 2)
            else:
                for k in ("net_motion", "net_white_count", "flank_motion"):
                    sig[k].append(0.0)
            prev_gray, prev_white = g, wm
            f += 1
    finally:
        cap.release()
    if len(sig["o_in_rim"]) < 10:
        return None
    return {k: np.asarray(v, float) for k, v in sig.items()}


def _win(a, t0, lo, hi):
    s = a[max(0, t0 + lo):min(len(a), t0 + hi)]
    return s if len(s) else np.array([0.0])


def features_from_signals(sig: dict) -> np.ndarray:
    """The 7 make/miss features from the per-frame signals (see FEATURE_NAMES)."""
    o = sig["o_in_rim"]
    lo, hi = max(0, PRE - 15), min(len(o), PRE + 16)
    t0 = (lo + int(o[lo:hi].argmax())) if (len(o[lo:hi]) and o[lo:hi].max() > 50) else PRE
    L = np.log1p
    nb = np.median(_win(sig["net_motion"], t0, -18, -6))
    fp = _win(sig["flank_motion"], t0, 0, 20).max()
    net_pk = _win(sig["net_motion"], t0, 0, 20).max()
    wbase = np.median(_win(sig["net_white_count"], t0, -18, -6))
    wmin = _win(sig["net_white_count"], t0, 0, 12).min()
    return np.array([
        L(net_pk) - L(fp),
        L(net_pk) - L(nb),
        L(_win(sig["o_below_net"], t0, 0, 30).max()),
        L(_win(sig["o_side"], t0, 2, 30).max()),
        L(_win(sig["o_net"], t0, 0, 15).max()),
        (wbase - wmin) / (wbase + 1.0),
        L(_win(sig["net_motion"], t0, 8, 30).max()) - L(fp),
    ], float)


def shot_features(clip_path, shot, calib, track=None) -> np.ndarray | None:
    """Full path: find the rim frame for `shot`, then extract the feature vector."""
    from .make import classify_make
    tr = track if track is not None else {int(f): c for f, c in
                                          zip(shot.frames, _fake_cands(shot))}
    mr = classify_make(shot, tr, calib)
    if mr.rim_frame is None:
        return None
    sig = extract_signals(clip_path, int(mr.rim_frame),
                          (calib.rim_x, calib.rim_y, calib.rim_radius_px))
    return features_from_signals(sig) if sig else None


def _fake_cands(shot):
    from .phase1_ball.detect import BallCandidate
    return [BallCandidate(int(f), float(x), float(y), float(r))
            for f, x, y, r in zip(shot.frames, shot.xs, shot.ys, shot.radii)]


# ---- model ----------------------------------------------------------------
def train(X, y):
    """Train the make/miss model (standardized gradient boosting) on features X
    and labels y (1=make). Returns the fitted sklearn pipeline."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    pipe = make_pipeline(StandardScaler(),
                         GradientBoostingClassifier(n_estimators=80, max_depth=2,
                                                    learning_rate=0.08, subsample=0.8,
                                                    random_state=0))
    return pipe.fit(np.asarray(X, float), np.asarray(y, int))


def save(model, path=_MODEL_PATH_DEFAULT):
    import os
    import joblib
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump({"model": model, "features": FEATURE_NAMES}, path)
    return path


def load(path=_MODEL_PATH_DEFAULT):
    import joblib
    d = joblib.load(path)
    return d["model"]


def predict(model, feats) -> tuple[bool, float]:
    """(made, probability-of-make) for one feature vector."""
    p = float(model.predict_proba(np.asarray(feats, float).reshape(1, -1))[0, 1])
    return p >= 0.5, p
