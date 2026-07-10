"""3D shot analysis from the 2-camera (or wide-only) footage -> a results JSON
the dashboard reads. Wraps the validated pieces so the heavy compute runs ONCE
(CLI) and the webpage just displays.

What it produces, with honest confidence:
  * per-shot metric ARC from the wide camera's known-diameter ball -- apex above
    release (ft), lateral (left/right) drift (ft), release angle in the vertical
    image plane, each gated by a GRAVITY self-check (reconstructed vertical accel
    vs -32.2). Focal-free -> trustworthy today.
  * session ELBOW FLARE from the close camera's metric pose world-landmarks
    (W7) -- median +/- spread, session-relative / model-biased -> LOW-MED conf.
  * optional camera TILT (W4) recovered from >=2 clean arcs, which upgrades the
    arc's depth & true release angle (else absolute depth is left out).

All the numbers here are things one wide camera + one close camera can support
WITHOUT a calibration board; the true metric stereo flare (W3) needs the two
cameras framed so the ball is co-visible, and is not computed here.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field

import numpy as np

from .arc3d import analyze_shot, BALL_DIAM_FT
from .video_io import frame_times, is_variable_fps, probe


def _contiguous_runs(frames, max_gap=5):
    frames = np.asarray(frames)
    if len(frames) == 0:
        return []
    breaks = np.where(np.diff(frames) > max_gap)[0]
    return np.split(np.arange(len(frames)), breaks + 1)


def wide_arcs(clip: str, weights: str, *, conf=0.12, imgsz=1280,
              start=0, stop=None, min_pts=8, max_shots=60):
    """Detect the ball in the wide clip (VFR-aware times) and return per-shot
    focal-free 3D arc metrics, each with a gravity self-check."""
    from .phase1_ball.detect_yolo import YoloBallDetector
    from .video_io import iter_frames
    info = probe(clip)
    W, H = info.width, info.height
    ts = frame_times(clip, start, stop)
    det = YoloBallDetector(weights=weights, ball_class=0, conf=conf, imgsz=imgsz)
    fr, us, vs, rs = [], [], [], []
    for idx, frame in iter_frames(clip, start=start, stop=stop):
        c = det.detect(idx, frame)
        if c:
            b = max(c, key=lambda z: getattr(z, "conf", 1.0))
            fr.append(idx); us.append(b.cx); vs.append(b.cy); rs.append(b.r)
    fr = np.array(fr); us = np.array(us); vs = np.array(vs); rs = np.array(rs)
    out = []
    for seg in _contiguous_runs(fr):
        if len(seg) < min_pts:
            continue
        f = fr[seg]
        t = np.array([ts.get(int(i), i / info.fps) for i in f])
        try:
            a = analyze_shot(us[seg], vs[seg], rs[seg], info.fps, W, H, times=t)
        except ValueError:
            continue
        trust = a.gravity_error_pct < 25
        out.append({
            "first_frame": int(f[0]), "last_frame": int(f[-1]),
            "n_points": int(a.n), "flight_s": round(float(t[-1] - t[0]), 2),
            "apex_above_release_ft": a.apex_above_release_ft,
            "lateral_drift_ft": round(float(a.X[-1] - a.X[0]), 2),
            "release_angle_deg": a.release_angle_deg,
            "entry_angle_deg": a.entry_angle_deg,
            "gravity_error_pct": a.gravity_error_pct,
            "trustworthy": bool(trust),
        })
        if len(out) >= max_shots:
            break
    return {"image_w": W, "image_h": H, "fps": round(info.fps, 2),
            "is_vfr": bool(is_variable_fps(clip)[0]), "shots": out}


def flare_from_close(clip: str, *, start=0, stop=None, variant="full",
                     min_gap_frames=15):
    """Elbow flare (deg) at each release from the close clip's metric world
    landmarks (W7). Returns per-shot flares + summary."""
    from .video_io import iter_frames
    from .phase2_pose.pose import PoseExtractor
    from .threed import elbow_flare
    UP = (0.0, -1.0, 0.0)
    ext = PoseExtractor(fps=probe(clip).fps, variant=variant, smooth=True)
    series = {}
    for idx, frame in iter_frames(clip, start=start, stop=stop):
        fp = ext.process_frame(idx, frame)
        if fp is None or fp.world is None:
            continue
        if all(fp.v(n) >= 0.5 for n in ("r_shoulder", "r_elbow", "r_wrist", "nose")):
            series[idx] = fp
    ext.close()
    idxs = sorted(series)
    rel = []
    for k in range(1, len(idxs) - 1):
        f = idxs[k]
        wy = series[f].pt("r_wrist")[1]
        if (wy < series[f].pt("nose")[1]
                and wy <= series[idxs[k - 1]].pt("r_wrist")[1]
                and wy < series[idxs[k + 1]].pt("r_wrist")[1]):
            if not rel or f - rel[-1] > min_gap_frames:
                rel.append(f)
    flares = []
    for f in rel:
        fp = series[f]
        fl = elbow_flare(fp.w("r_shoulder"), fp.w("r_elbow"), fp.w("r_wrist"), up=UP)
        flares.append({"frame": int(f), "flare_deg": fl.angle_deg})
    vals = np.array([x["flare_deg"] for x in flares]) if flares else np.array([])
    summary = None
    if len(vals):
        summary = {"n": int(len(vals)), "median_deg": round(float(np.median(vals)), 1),
                   "mean_deg": round(float(vals.mean()), 1),
                   "sd_deg": round(float(vals.std()), 1),
                   "min_deg": round(float(vals.min()), 1),
                   "max_deg": round(float(vals.max()), 1)}
    return {"shots": flares, "summary": summary,
            "confidence": "low-med",
            "note": "monocular world-landmark flare; session-relative, model-biased. "
                    "Magnitude is the coaching signal; sign is setup-dependent."}


@dataclass
class Analysis3D:
    wide: dict = field(default_factory=dict)      # from wide_arcs
    flare: dict = field(default_factory=dict)     # from flare_from_close
    camera_tilt: dict | None = None               # W4, if computed
    meta: dict = field(default_factory=dict)

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(path: str) -> "Analysis3D":
        with open(path, encoding="utf-8") as f:
            return Analysis3D(**json.load(f))
