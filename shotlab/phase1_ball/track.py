"""Turn per-frame ball candidates into a single ball track, then split that
track into individual shots (parabolic flights).

The synthetic clip has clean gaps between shots; real footage has dribbling,
the ball at rest, and missed detections. The segmenter is therefore arc-driven:
it looks for contiguous runs of detections that actually form a rise-then-fall
parabola, rather than trusting raw gaps alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..arc import ArcFit, fit_parabola_ransac
from .detect import BallCandidate


@dataclass
class Shot:
    index: int
    frames: np.ndarray          # frame indices used
    xs: np.ndarray              # px
    ys: np.ndarray              # px (image, y-down)
    radii: np.ndarray
    fit: ArcFit
    meta: dict = field(default_factory=dict)


def assemble_track(cands_by_frame: dict[int, list[BallCandidate]],
                   gate_px: float = 120.0,
                   max_coast: int = 4) -> dict[int, BallCandidate]:
    """Pick one ball position per frame using confidence + motion continuity.

    Single forward pass with a constant-velocity predictor that RESETS after a
    detection gap. This is what makes multi-shot footage work: each shot's arc
    is tracked with motion continuity, but the predictor never tries to bridge
    the dead time between shots (which would reject the next shot's detections).

    At a reset the highest-confidence candidate seeds the new arc; with only one
    prior point we gate on position; with two we gate on the velocity prediction.
    """
    if not cands_by_frame:
        return {}

    chosen: dict[int, BallCandidate] = {}
    prev = prev_prev = None
    prev_f = None

    for f in sorted(cands_by_frame):
        cands = cands_by_frame.get(f, [])
        if not cands:
            continue

        gap = (f - prev_f) if prev_f is not None else None
        reset = prev is None or (gap is not None and gap > max_coast)

        if reset:
            # start a fresh arc from the most confident detection
            best = max(cands, key=lambda c: c.conf)
            prev_prev = None
        else:
            if prev_prev is not None:
                # constant-velocity prediction, time-scaled across any small gap
                step = max(1, f - prev_f)
                vx = (prev.cx - prev_prev.cx)
                vy = (prev.cy - prev_prev.cy)
                pred_x = prev.cx + vx * step
                pred_y = prev.cy + vy * step
            else:
                pred_x, pred_y = prev.cx, prev.cy
            best, bestd = None, gate_px
            for c in cands:
                d = np.hypot(c.cx - pred_x, c.cy - pred_y)
                score = d - 30 * c.conf       # prefer confident, nearby blobs
                if score < bestd:
                    bestd, best = score, c
            if best is None:
                # gate rejected everything -> treat as a fresh seed
                best = max(cands, key=lambda c: c.conf)
                prev_prev = None

        chosen[f] = best
        prev_prev, prev, prev_f = prev, best, f

    return chosen


def _runs(frames: list[int], max_gap: int) -> list[list[int]]:
    runs, cur = [], []
    for f in frames:
        if cur and f - cur[-1] > max_gap:
            runs.append(cur)
            cur = []
        cur.append(f)
    if cur:
        runs.append(cur)
    return runs


def segment_shots(track: dict[int, BallCandidate],
                  *, max_gap: int = 6,
                  min_points: int = 8,
                  min_height_span_px: float = 40.0,
                  threshold_px: float = 6.0) -> list[Shot]:
    """Split a track into shots. A run qualifies as a shot if its detections fit
    a downward-opening height parabola (real ball flight) over enough vertical
    range."""
    if not track:
        return []
    frames = sorted(track)
    shots: list[Shot] = []

    for run in _runs(frames, max_gap):
        if len(run) < min_points:
            continue
        xs = np.array([track[f].cx for f in run])
        ys = np.array([track[f].cy for f in run])
        rs = np.array([track[f].r for f in run])
        hs = -ys
        if hs.max() - hs.min() < min_height_span_px:
            continue
        fit = fit_parabola_ransac(xs, ys, threshold_px=threshold_px)
        if fit is None:
            continue
        # a<0 in h-space == concave-down == genuine rise-then-fall arc
        if fit.coeffs[0] >= 0:
            continue
        shots.append(Shot(
            index=len(shots) + 1,
            frames=np.array(run),
            xs=xs, ys=ys, radii=rs,
            fit=fit,
            meta={"first_frame": int(run[0]), "last_frame": int(run[-1])},
        ))
    return shots
