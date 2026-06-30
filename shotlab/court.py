"""Court calibration: rim location, rim-anchored shot detection, and zones.

The tripod is fixed for a session, so the rim sits at one pixel location across
all of that session's clips. Calibrating it once unlocks three things at once:

  1. Shot detection  -- a "shot" is a ball flight that reaches near the rim
     (dribbles never do), which is far more reliable than guessing from arc shape.
  2. Make/miss        -- classify from the trajectory through the rim (make.py).
  3. Zones/direction  -- classify each shot by the shooter's release position
     relative to the rim (left / center / right, near / far).

Calibration is stored as JSON per session (config/calibration_<session>.json) and
can be auto-detected (orange rim) or set by clicking (calibrate.py).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

import numpy as np

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise ImportError("opencv-python is required") from e


@dataclass
class Calibration:
    session: str
    image_w: int
    image_h: int
    rim_x: float
    rim_y: float
    rim_radius_px: float          # visible half-width of the rim
    shot_gate_px: float           # ball must pass within this of the rim to count
    handedness: str = "right"
    note: str = ""

    @property
    def rim(self) -> np.ndarray:
        return np.array([self.rim_x, self.rim_y])

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(path: str) -> "Calibration":
        with open(path, encoding="utf-8") as f:
            return Calibration(**json.load(f))


def detect_rim(frame_bgr, y_band=(0.08, 0.45), x_band=(0.08, 0.95)
               ) -> tuple[float, float, float] | None:
    """Auto-locate the orange rim. Returns (cx, cy, half_width_px) or None.

    Searches only the band where a backboard-mounted rim lives (upper-center by
    default). This is what makes it reliable on cluttered scenes: orange houses,
    the ball, and skin tones sit OUTSIDE this band (lower / at the edges), so
    they no longer get mistaken for the rim. Override the band per setup, or set
    the rim explicitly with calibrate.py.
    """
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 90, 90), (15, 255, 255))
    m2 = cv2.inRange(hsv, (165, 90, 90), (180, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    # zero out everything outside the search band
    y0, y1 = int(h * y_band[0]), int(h * y_band[1])
    x0, x1 = int(w * x_band[0]), int(w * x_band[1])
    band = np.zeros_like(mask)
    band[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    mask = band
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 60:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        blobs.append((a, x, y, bw, bh))
    if not blobs:
        return None
    # take the largest, then merge any nearby orange (the rim's other edge)
    blobs.sort(reverse=True)
    _, x, y, bw, bh = blobs[0]
    x0, x1, y0, y1 = x, x + bw, y, y + bh
    for a, bx, by, bbw, bbh in blobs[1:5]:
        if abs((by + bbh / 2) - (y + bh / 2)) < 40 and abs(bx - x) < 250:
            x0, x1 = min(x0, bx), max(x1, bx + bbw)
            y0, y1 = min(y0, by), max(y1, by + bbh)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half_w = max((x1 - x0) / 2, 15)
    return float(cx), float(cy), float(half_w)


def auto_calibrate(video_path: str, session: str, gate_mult: float = 2.0,
                   n_samples: int = 9) -> Calibration | None:
    """Auto-detect the rim by sampling several frames across the clip and taking
    the median detection (robust to a person briefly overlapping the rim, or one
    bad frame). The rim is re-detected PER CLIP because the tripod may be
    repositioned between clips."""
    from .video_io import probe, iter_frames
    info = probe(video_path)
    if info.n_frames <= 0:
        return None
    # sample frames spread through the middle of the clip
    idxs = set(int(info.n_frames * f) for f in
               [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.5][:n_samples])
    dets = []
    target = sorted(idxs)
    ti = 0
    for idx, frame in iter_frames(video_path):
        if ti >= len(target):
            break
        if idx == target[ti]:
            r = detect_rim(frame)
            if r is not None:
                dets.append(r)
            ti += 1
    if len(dets) < 3:
        return None
    arr = np.array(dets)
    cx, cy, half_w = (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])),
                      float(np.median(arr[:, 2])))
    return Calibration(
        session=session, image_w=info.width, image_h=info.height,
        rim_x=cx, rim_y=cy, rim_radius_px=half_w,
        shot_gate_px=max(gate_mult * half_w, 90.0),
        note=f"auto-detected rim (median of {len(dets)} frames)")


def shot_reaches_rim(shot, calib: Calibration) -> tuple[bool, float]:
    """Does this ball flight come near the rim? Returns (reached, min_dist_px).

    Checks both the tracked points and the fitted parabola sampled across its
    x-range (the ball is sometimes lost right at the rim against clutter)."""
    rim = calib.rim
    pts = np.stack([shot.xs, shot.ys], axis=1)
    d_pts = np.min(np.linalg.norm(pts - rim, axis=1)) if len(pts) else np.inf

    xs = np.linspace(shot.xs.min(), shot.xs.max(), 40)
    ys = -shot.fit.height_at(xs)
    d_fit = np.min(np.linalg.norm(np.stack([xs, ys], 1) - rim, axis=1))

    d = float(min(d_pts, d_fit))
    return d <= calib.shot_gate_px, d


def is_real_shot(shot, calib: Calibration) -> tuple[bool, str]:
    """A real shot = the ball launches from well BELOW the rim and rises UP to
    rim height, reaching the rim. This rejects the degenerate flights that pass a
    bare rim-distance test: near-horizontal rolls/passes at rim height (never came
    from below) and near-vertical bounces/retrievals (degenerate parabola fit).
    """
    reached, d = shot_reaches_rim(shot, calib)
    if not reached:
        return False, f"never reached rim (min {d:.0f}px)"

    ys = shot.ys                      # image y (down = larger)
    launch_y = ys.max()               # lowest point = release/launch
    apex_y = ys.min()                 # highest point of the flight
    below = launch_y - calib.rim_y    # how far below the rim it launched (px)
    apex_above = calib.rim_y - apex_y  # how far the apex sits above the rim (px)

    if below < 120:
        return False, "did not launch from below the rim (roll/pass)"
    if apex_y > calib.rim_y + 90:
        return False, "apex never reached rim height (low bounce/dribble)"
    if shot.fit.coeffs[0] >= 0:
        return False, "not a downward arc"
    # near-vertical at BOTH ends = a straight up/down bounce or toss, not a shot.
    # (real shots, even foreshortened, have an asymmetric, less-than-vertical end)
    rel = shot.fit.release_angle_deg()
    ent = shot.fit.entry_angle_deg(calib.rim_x)
    if min(rel, ent) > 75:
        return False, "near-vertical both ends (bounce/toss, not a shot)"
    return True, f"ok (rim {d:.0f}px, launched {below:.0f}px below)"


def filter_shots_by_rim(shots, calib: Calibration):
    """Keep only genuine shots (rim-anchored + launch/apex gates).
    Returns (kept_shots, rejected_count)."""
    kept = []
    for s in shots:
        ok, why = is_real_shot(s, calib)
        if ok:
            _, d = shot_reaches_rim(s, calib)
            s.meta["rim_dist_px"] = round(d, 1)
            kept.append(s)
    for i, s in enumerate(kept, 1):
        s.index = i
    return kept, len(shots) - len(kept)


def detect_shots_to_rim(track, calib: Calibration, *, max_rim_gap: int = 20,
                        launch_drop: float = 200.0, min_points: int = 8,
                        threshold_px: float = 8.0):
    """Find shots in a CONTINUOUS ball track (the case a good detector produces).

    Gap-based segmentation fails when the ball is tracked unbroken through
    dribbling AND shooting. Instead we anchor on the rim: each time the ball's
    path reaches the rim, we walk back to where it launched from well below the
    rim, and treat that ascending arc as a shot. This naturally ignores dribbling
    (which never reaches the rim) without needing detection gaps.
    """
    from .phase1_ball.track import Shot
    from .arc import fit_parabola_ransac

    if not track:
        return []
    frames = np.array(sorted(track))
    xy = np.array([[track[int(f)].cx, track[int(f)].cy] for f in frames])
    x, y = xy[:, 0], xy[:, 1]
    dist = np.hypot(x - calib.rim_x, y - calib.rim_y)
    fidx = {int(f): i for i, f in enumerate(frames)}

    near = frames[dist < calib.shot_gate_px]
    if len(near) == 0:
        return []
    # group near-rim frames into rim events
    events, cur = [], [near[0]]
    for a, b in zip(near, near[1:]):
        if b - a <= max_rim_gap:
            cur.append(b)
        else:
            events.append(cur); cur = [b]
    events.append(cur)

    shots, seen_launch = [], set()
    for ev in events:
        t_rim = min(ev, key=lambda f: dist[fidx[int(f)]])
        i = fidx[int(t_rim)]
        j = i
        while j > 0 and (y[j] - calib.rim_y) < launch_drop:
            j -= 1
        if int(frames[j]) in seen_launch:
            continue
        seen_launch.add(int(frames[j]))
        if (i - j + 1) < min_points or (y[j] - y[i]) < 0.8 * launch_drop:
            continue
        seg = slice(j, i + 1)
        f_seg = frames[seg]
        x_seg, y_seg = x[seg], y[seg]
        r_seg = np.array([track[int(f)].r for f in f_seg])
        fit = fit_parabola_ransac(x_seg, y_seg, threshold_px=threshold_px)
        if fit is None or fit.coeffs[0] >= 0:
            continue
        # reject non-shots that survive the rim test: near-vertical tosses/rebounds
        # (both ends ~vertical) and noisy fits with too few inlier points.
        if fit.n_used < 7:
            continue
        rel = fit.release_angle_deg()
        ent = fit.entry_angle_deg(calib.rim_x)
        if min(rel, ent) > 78:
            continue
        s = Shot(index=len(shots) + 1, frames=f_seg, xs=x_seg, ys=y_seg,
                 radii=r_seg, fit=fit,
                 meta={"rim_dist_px": round(float(dist[i]), 1),
                       "first_frame": int(f_seg[0]), "last_frame": int(f_seg[-1])})
        shots.append(s)
    return shots


def zone_for_release(release_xy, calib: Calibration) -> dict:
    """Classify a shot's origin relative to the rim (image-space proxy until a
    full court homography is calibrated).

      side:  left / center / right of the rim (shooter's release vs rim x)
      depth: a coarse near/mid/far from the rim, by image distance
    """
    rx, ry = release_xy
    dx = rx - calib.rim_x
    span = max(calib.image_w * 0.06, 1)
    if dx < -span:
        side = "left"
    elif dx > span:
        side = "right"
    else:
        side = "center"
    dist = float(np.hypot(dx, ry - calib.rim_y))
    # crude depth bins relative to frame width
    if dist < calib.image_w * 0.18:
        depth = "near"
    elif dist < calib.image_w * 0.33:
        depth = "mid"
    else:
        depth = "far"
    return {"side": side, "depth": depth, "zone": f"{depth}-{side}",
            "rim_dx_px": round(dx, 1), "rim_dist_px": round(dist, 1)}
