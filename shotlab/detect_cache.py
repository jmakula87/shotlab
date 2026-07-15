"""Cache a clip's ball DETECTION (track + shots) so it's computed once and
reused by the session build, per-shot rendering, comparisons, and re-analysis.

Detection (YOLO over thousands of frames) is ~80% of the work and was being re-run
from scratch every time. Caching it turns multi-minute re-detections into a
sub-second JSON read. The cache is keyed by the detection params (weights, imgsz,
stride, max_frames) + rim, so a param change triggers a fresh detect.

The track is small (a few thousand frame positions); the cache is ~100 KB/clip and
lives next to the metrics cache, so it travels into the session archive too.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .arc import ArcFit
from .phase1_ball.detect import BallCandidate
from .phase1_ball.track import Shot


def _path(video_path: str) -> str:
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join("data", "out", stem, f"{stem}_track.json")


def _weights_content_id(weights) -> str:
    """size:mtime of the weights file (or the summed sizes + newest mtime of an
    openvino export DIR). Catches a retrain/re-export into the same path, which
    a path-based key can't see."""
    p = str(weights)
    try:
        if os.path.isdir(p):
            tot, mt = 0, 0
            for name in sorted(os.listdir(p)):
                fp = os.path.join(p, name)
                if os.path.isfile(fp):
                    st = os.stat(fp)
                    tot += st.st_size
                    mt = max(mt, int(st.st_mtime))
            return f"{tot}:{mt}"
        st = os.stat(p)
        return f"{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return "absent"


def _weights_id(weights) -> str:
    """Identity for a weights path in cache signatures. The BASENAME alone is
    ambiguous -- every training run exports a dir literally named
    best_openvino_model, so switching models would silently reuse the old
    model's cached detections. The last three path parts disambiguate
    (e.g. ball_orange/weights/best_openvino_model); the appended CONTENT id
    (size:mtime) catches a retrain re-exported to the same path."""
    parts = os.path.normpath(str(weights)).split(os.sep)
    return "/".join(parts[-3:]) + "@" + _weights_content_id(weights)


def _params(video_path, weights, imgsz, stride, max_frames, calib) -> dict:
    from .video_io import video_id
    return {"weights": _weights_id(weights), "imgsz": int(imgsz),
            "stride": int(stride), "max_frames": max_frames,
            "rim": [round(calib.rim_x, 1), round(calib.rim_y, 1)],
            "video": video_id(video_path)}


def serialize_detection(track, shots, params) -> dict:
    """(track, shots) -> a JSON-able dict (shared by the whole-clip cache and the
    long-clip chunker's per-window caches)."""
    return {
        "params": params,
        "track": {str(int(f)): [round(c.cx, 2), round(c.cy, 2), round(c.r, 2),
                                round(c.conf, 3)] for f, c in track.items()},
        "shots": [{
            "index": int(s.index),
            "frames": [int(f) for f in s.frames],
            "coeffs": [float(x) for x in s.fit.coeffs],
            "n_used": int(s.fit.n_used),
            "rmse": float(s.fit.rmse_px),
            "direction": int(s.fit.direction),
            "rim_dist_px": s.meta.get("rim_dist_px"),
            # Which of `frames` are fit INLIERS. Without this, a round-trip marked
            # every point an inlier and release_angle/apex read off the walk-back
            # outliers (2026-07-06 audit D1). Aligned with `frames`.
            "inliers": ([int(bool(b)) for b in np.asarray(s.fit.inlier_mask).tolist()]
                        if getattr(s.fit, "inlier_mask", None) is not None
                        and len(np.asarray(s.fit.inlier_mask)) == len(s.frames)
                        else None),
        } for s in shots],
    }


def _recover_inlier_mask(sd, xs, ys) -> np.ndarray:
    """Which points are fit inliers. Prefer the serialized mask; for older caches
    that lack it, recompute from the stored (correct) coeffs by the same residual
    test the RANSAC fit uses -- the gross walk-back outliers sit far from the
    fitted parabola, so this recovers the real inliers without re-detecting."""
    saved = sd.get("inliers")
    if saved is not None and len(saved) == len(xs):
        m = np.array([bool(b) for b in saved])
        if m.sum() >= 3:
            return m
    hs = -np.asarray(ys, float)
    pred = np.polyval(np.asarray(sd["coeffs"], float), np.asarray(xs, float))
    thr = max(8.0, 3.0 * float(sd.get("rmse", 0.0)) + 1.0)
    m = np.abs(pred - hs) < thr
    return m if m.sum() >= 3 else np.ones(len(xs), bool)


def deserialize_detection(data):
    """Inverse of serialize_detection -> (track, shots). Rebuilds the fit's xs/hs
    from the INLIERS (not every point) so release_angle/apex match a fresh fit."""
    track = {int(k): BallCandidate(int(k), v[0], v[1], v[2], v[3])
             for k, v in data["track"].items()}
    shots = []
    for sd in data["shots"]:
        frames = np.array(sd["frames"])
        xs = np.array([track[int(f)].cx for f in frames])
        ys = np.array([track[int(f)].cy for f in frames])
        radii = np.array([track[int(f)].r for f in frames])
        mask = _recover_inlier_mask(sd, xs, ys)
        fit = ArcFit(coeffs=np.array(sd["coeffs"]),
                     inlier_mask=mask, xs=xs[mask], hs=(-ys)[mask],
                     n_used=int(mask.sum()), rmse_px=sd["rmse"],
                     direction=sd["direction"])
        shots.append(Shot(index=sd["index"], frames=frames, xs=xs, ys=ys,
                          radii=radii, fit=fit,
                          meta={"rim_dist_px": sd.get("rim_dist_px"),
                                "first_frame": int(frames[0]),
                                "last_frame": int(frames[-1])}))
    return track, shots


def save_detection(video_path, track, shots, params) -> str:
    path = _path(video_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialize_detection(track, shots, params), f)
    return path


def _load(video_path):
    path = _path(video_path)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    track, shots = deserialize_detection(data)
    return data["params"], track, shots


def detect_window(video_path, weights, calib, stride, start, stop, imgsz=640):
    """Detect ball + rim-anchored shots in a FRAME WINDOW [start, stop). Frame
    indices stay absolute. Uncached (the long-clip chunker caches the resulting
    records itself). Returns (track, shots)."""
    from .phase1_ball.pipeline import run_phase1
    from .phase1_ball.detect_yolo import YoloBallDetector
    det = YoloBallDetector(weights=weights, ball_class=0, conf=0.25, imgsz=imgsz)
    res = run_phase1(video_path, detector=det, calib=calib, stride=int(stride),
                     start_frame=int(start), max_frames=int(stop))
    return res.track, res.shots


def detect_or_load(video_path, weights, calib, stride, max_frames, imgsz=640,
                   use_cache=True):
    """Return (track, shots). Loads the cached detection if params match, else
    runs YOLO detection + rim-anchored shots and caches the result."""
    params = _params(video_path, weights, imgsz, stride, max_frames, calib)
    if use_cache:
        loaded = _load(video_path)
        if loaded is not None and loaded[0] == params:
            return loaded[1], loaded[2]
    from .phase1_ball.pipeline import run_phase1
    from .phase1_ball.detect_yolo import YoloBallDetector
    det = YoloBallDetector(weights=weights, ball_class=0, conf=0.25, imgsz=imgsz)
    res = run_phase1(video_path, detector=det, calib=calib, stride=int(stride),
                     max_frames=max_frames)
    save_detection(video_path, res.track, res.shots, params)
    return res.track, res.shots
