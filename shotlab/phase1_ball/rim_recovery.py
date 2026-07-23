"""Rim-anchored backward recovery over the candidate CLOUD (experimental).

The 2026-07-23 review showed every residual miss (after greedy + beam) still has
ball detections near the rim -- the arc just doesn't survive assemble_track /
beam_tracks + the walk-back. This pass anchors on each near-rim cloud event that
produced NO shot, gathers the cloud in a backward window toward the shooter, and
RANSAC-fits a parabola directly (outlier-robust, so distractors don't matter). If
the inliers form a real launch->rim arc it emits a shot. Union this with greedy +
beam. Measured against the hand-count before trusting it (precision is the risk).
"""
from __future__ import annotations
import numpy as np

from .detect import BallCandidate
from .track import Shot
from ..arc import fit_parabola_ransac


def _rim_events(cloud, calib, conf_floor, max_gap=20):
    """Frames where a cloud candidate comes within the shot gate of the rim,
    grouped into events (return the representative closest-approach frame each)."""
    near = []
    for f in sorted(cloud):
        best = min((np.hypot(c.cx - calib.rim_x, c.cy - calib.rim_y)
                    for c in cloud[f] if c.conf >= conf_floor), default=np.inf)
        if best < calib.shot_gate_px:
            near.append((f, best))
    if not near:
        return []
    events, cur = [], [near[0]]
    for (f, d) in near[1:]:
        if f - cur[-1][0] <= max_gap:
            cur.append((f, d))
        else:
            events.append(cur); cur = [(f, d)]
    events.append(cur)
    return [min(ev, key=lambda t: t[1])[0] for ev in events]     # closest-approach frame


def recover_shots(cloud, calib, seen_rim_frames, *, conf_floor=0.05, window=40,
                  launch_drop=200.0, min_points=8, x_corridor=650.0, dedup=25):
    """Emit shots for near-rim cloud events not already covered by `seen_rim_frames`.
    RANSAC-fits the backward cloud window and applies the same shot validity gates
    as detect_shots_to_rim (downward arc, launch from below the rim, apex above rim,
    angle sanity). Returns a list of Shot (with a synthetic single-candidate track)."""
    shots = []
    emitted = list(seen_rim_frames)
    for ev in _rim_events(cloud, calib, conf_floor):
        if any(abs(ev - r) <= dedup for r in emitted):
            continue
        # gather the backward cloud window (toward the shooter), within an x-corridor
        pts = []
        for f in range(ev - window, ev + 6):
            for c in cloud.get(f, []):
                if c.conf >= conf_floor and abs(c.cx - calib.rim_x) < x_corridor:
                    pts.append((f, c.cx, c.cy, c.r))
        if len(pts) < min_points:
            continue
        pts.sort()
        fx = np.array([p[0] for p in pts]); x = np.array([p[1] for p in pts])
        y = np.array([p[2] for p in pts]); r = np.array([p[3] for p in pts])
        fit = fit_parabola_ransac(x, y, threshold_px=10.0)
        if fit is None or fit.coeffs[0] >= 0 or fit.n_used < min(min_points, 7):
            continue
        inl = fit.inliers if getattr(fit, "inliers", None) is not None else np.ones(len(x), bool)
        xi, yi, fi, ri = x[inl], y[inl], fx[inl], r[inl]
        if len(xi) < min_points:
            continue
        # the inlier arc must launch from >= launch_drop below the rim AND rise above it
        if (yi.max() - calib.rim_y) < 0.8 * launch_drop or yi.min() >= calib.rim_y:
            continue
        # must actually reach the rim, and not be a near-vertical toss
        if np.min(np.hypot(xi - calib.rim_x, yi - calib.rim_y)) >= calib.shot_gate_px:
            continue
        rel, ent = fit.release_angle_deg(), fit.entry_angle_deg(calib.rim_x)
        if min(rel, ent) > 78:
            continue
        order = np.argsort(fi)
        s = Shot(index=len(shots) + 1, frames=fi[order], xs=xi[order], ys=yi[order],
                 radii=ri[order], fit=fit,
                 meta={"first_frame": int(fi.min()), "last_frame": int(fi.max()),
                       "source": "rim_recovery"})
        shots.append(s)
        emitted.append(ev)
    return shots
