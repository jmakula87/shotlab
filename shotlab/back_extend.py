"""Extend a shot segment BACKWARD in time from a confirmed rim arrival.

Measured 2026-08-03 on the 07-29 hand-count: of 34 missed attempts, DETECTION=0
and TRACKER=0 -- the detector sees the ball at the rim every time and the track
reaches it every time. What fails is that the track does not reach back far enough
into the flight, so `detect_shots_to_rim` sees 2-6 points, or a drop of 105-153px
against its 160px bar, and discards a real shot.

Scaling that bar was tried and REJECTED (recall 85% -> 40%; see court.py's dead
letter): the bar is not a physical height, it is "how much of the flight was
tracked". So lengthen the track instead of moving the bar.

Backward search is much better conditioned than the forward association that built
the track: the endpoint is KNOWN (the ball demonstrably arrived at the rim), so we
extrapolate a ballistic arc into the past and accept the candidate that continues
it. A forward tracker has to guess which of many moving objects will become a
shot; here we already know one did.

Everything scale-sensitive is expressed in rim radii, because this codebase has
now been bitten twice by pixel constants tuned at one rim scale.
"""
from __future__ import annotations

import numpy as np

# search gate around the predicted position, in RIM RADII (never raw pixels)
GATE_RADII = 0.75
MAX_COAST = 6          # consecutive frames with no acceptable candidate
MAX_BACK = 120         # frames to search back: 4s at 30fps, longer than any flight
FIT_WINDOW = 12        # most recent accepted points used to re-fit the arc


def _fit(ts, xs, ys):
    """x linear in t, y quadratic in t -- the ballistic model, fit on what we have."""
    ts = np.asarray(ts, float); xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    t0 = ts[0]
    tt = ts - t0
    if len(ts) >= 3:
        fy = np.polyfit(tt, ys, 2)
    else:                                   # too few points for curvature
        fy = np.concatenate([[0.0], np.polyfit(tt, ys, 1)]) if len(ts) == 2 else \
             np.array([0.0, 0.0, ys[0]])
    fx = np.polyfit(tt, xs, 1) if len(ts) >= 2 else np.array([0.0, xs[0]])
    return t0, fx, fy


def _predict(t, t0, fx, fy):
    tt = t - t0
    return float(np.polyval(fx, tt)), float(np.polyval(fy, tt))


def extend_backward(seg_frames, seg_x, seg_y, cloud, rim_radius_px, *,
                    gate_radii: float = GATE_RADII, max_coast: int = MAX_COAST,
                    max_back: int = MAX_BACK, stop_below_y: float | None = None):
    """Walk backward from the earliest point of a segment, adding cloud candidates.

    `cloud` maps frame -> list of candidates with .cx/.cy/.r/.conf (the conf-0.01
    cloud). Returns (frames, xs, ys, radii) for the ADDED points, oldest first;
    empty when nothing could be added.

    `stop_below_y`: stop once the arc has descended past this image row (used to
    stop at the release height rather than running on into the dribble).
    """
    if not cloud or rim_radius_px <= 0 or len(seg_frames) == 0:
        return [], [], [], []
    gate = gate_radii * float(rim_radius_px)

    # newest-first working set, seeded with the segment we already have
    ts = [float(f) for f in seg_frames]
    xs = [float(v) for v in seg_x]
    ys = [float(v) for v in seg_y]

    f_start = int(min(seg_frames))
    added: list[tuple[int, float, float, float]] = []
    coast = 0
    for f in range(f_start - 1, max(-1, f_start - 1 - max_back), -1):
        use = min(len(ts), FIT_WINDOW)
        order = np.argsort(ts)[:use]                 # the EARLIEST points we hold
        t0, fx, fy = _fit([ts[i] for i in order], [xs[i] for i in order],
                          [ys[i] for i in order])
        px, py = _predict(float(f), t0, fx, fy)

        best, best_d = None, gate
        for c in cloud.get(f, ()) or ():
            d = float(np.hypot(c.cx - px, c.cy - py))
            if d < best_d:
                best, best_d = c, d
        if best is None:
            coast += 1
            if coast > max_coast:
                break
            continue
        coast = 0
        ts.append(float(f)); xs.append(float(best.cx)); ys.append(float(best.cy))
        added.append((f, float(best.cx), float(best.cy), float(getattr(best, "r", 0.0))))
        if stop_below_y is not None and best.cy > stop_below_y:
            break                                   # reached release height

    added.reverse()                                  # oldest first
    if not added:
        return [], [], [], []
    return ([a[0] for a in added], [a[1] for a in added],
            [a[2] for a in added], [a[3] for a in added])
