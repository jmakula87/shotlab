"""Single-camera metric arc from a KNOWN-diameter ball -- no calibration.

A regulation men's basketball is 29.5 in around, so its diameter is fixed:
D = 29.5/pi in = 0.782 ft. A sphere's silhouette has that same diameter from any
angle, so the ball's pixel radius r is a per-frame depth gauge: a ball twice as
far reads half as wide.

The quiet win: the ball's real-world HORIZONTAL (X) and VERTICAL (Y) positions in
feet are FOCAL-LENGTH-FREE. Deprojection multiplies a pixel offset by Z/f, and
the known diameter gives Z = f*D/(2r); the f cancels:

    X = (cx - cx0) * D / (2r)      Y = -(cy - cy0) * D / (2r)      [feet]

So from ONE camera, with only the ball's known size and the image center, we get
the true-feet arc, per-frame depth-corrected (the current arc uses one ppf for
the whole shot; this corrects each point as the ball recedes). DEPTH Z in
absolute feet still needs the focal, but depth *ratios* (drift shape) don't.

Free validation: with X,Y in real feet and time from the fps, the vertical
acceleration MUST come out to gravity, -32.2 ft/s^2. If it does, the whole
reconstruction is confirmed with nothing calibrated. `gravity_error_pct` reports
how close -- treat a shot with a large error as untrustworthy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

BALL_CIRCUM_IN = 29.5                       # regulation men's (size 7)
BALL_DIAM_FT = BALL_CIRCUM_IN / math.pi / 12.0     # ~0.782 ft
G_FT = 32.174                               # gravity, ft/s^2


@dataclass
class Arc3D:
    t: np.ndarray                # seconds from first point
    X: np.ndarray                # horizontal, feet (focal-free)
    Y: np.ndarray                # vertical, feet, up positive (focal-free)
    Z: np.ndarray | None         # depth, feet (needs focal); None if not given
    release_angle_deg: float
    entry_angle_deg: float | None
    apex_above_release_ft: float
    horiz_range_ft: float
    depth_drift_ft: float | None    # net toward/away-camera drift over the flight
    vert_accel_ft_s2: float         # measured; should be ~ -32.2
    gravity_error_pct: float        # |accel - g| / g * 100
    n: int


def deproject(xs, ys, radii, image_w, image_h,
              ball_diam_ft: float = BALL_DIAM_FT, focal_px: float | None = None):
    """Pixel ball track -> (X, Y, Z_or_None) in feet. X,Y are focal-free; Z is
    returned only if focal_px is supplied."""
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    r = np.asarray(radii, float)
    cx0, cy0 = image_w / 2.0, image_h / 2.0
    scale = ball_diam_ft / (2.0 * r)                 # feet per pixel, per frame
    X = (xs - cx0) * scale
    Y = -(ys - cy0) * scale
    Z = None
    if focal_px:
        Z = focal_px * ball_diam_ft / (2.0 * r)      # absolute depth, feet
    return X, Y, Z


def analyze_shot(xs, ys, radii, fps: float, image_w: int, image_h: int,
                 ball_diam_ft: float = BALL_DIAM_FT,
                 focal_px: float | None = None, times=None) -> Arc3D:
    """Full metric arc for one shot's ball track. Points must be in flight order.

    `times`: per-point capture time in SECONDS (from video_io.frame_times). Pass
    these -- the phones record variable frame rate and have dropped detections,
    so `frame_index / fps` mistimes the gravity fit. `fps` is only the fallback
    when times is None (constant-rate, gap-free)."""
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    radii = np.asarray(radii, float)
    n = len(xs)
    if n < 5:
        raise ValueError(f"need >=5 ball points, got {n}")
    if times is not None:
        t = np.asarray(times, float) - float(np.asarray(times, float)[0])
        if len(t) != n:
            raise ValueError(f"times has {len(t)} entries, need {n}")
    else:
        t = np.arange(n) / float(fps)
    X, Y, Z = deproject(xs, ys, radii, image_w, image_h, ball_diam_ft, focal_px)

    # gravity check: fit Y = c0 + c1 t + c2 t^2; vertical accel = 2*c2
    c2, c1, c0 = np.polyfit(t, Y, 2)
    accel = 2.0 * c2
    grav_err = abs(accel - (-G_FT)) / G_FT * 100.0

    # horizontal direction of travel (sign of net X change) -> release at the start
    direction = 1.0 if X[-1] >= X[0] else -1.0
    # release angle from the first ~3 points' slope in the real X-Y plane
    k = min(3, n - 1)
    dX = (X[k] - X[0]); dY = (Y[k] - Y[0])
    horiz = dX * direction
    rel_ang = math.degrees(math.atan2(dY, abs(horiz))) if horiz else float("nan")

    # entry angle from the last ~3 points (ball descending into the rim)
    dXe = (X[-1] - X[-1 - k]); dYe = (Y[-1] - Y[-1 - k])
    horiz_e = dXe * direction
    ent_ang = math.degrees(math.atan2(-dYe, abs(horiz_e))) if horiz_e else None

    apex = float(Y.max() - Y[0])
    horiz_range = float(abs(X[-1] - X[0]))
    depth_drift = float(Z[-1] - Z[0]) if Z is not None else None

    return Arc3D(t=t, X=X, Y=Y, Z=Z,
                 release_angle_deg=round(rel_ang, 1),
                 entry_angle_deg=round(ent_ang, 1) if ent_ang is not None else None,
                 apex_above_release_ft=round(apex, 2),
                 horiz_range_ft=round(horiz_range, 2),
                 depth_drift_ft=round(depth_drift, 2) if depth_drift is not None else None,
                 vert_accel_ft_s2=round(accel, 1),
                 gravity_error_pct=round(grav_err, 1),
                 n=n)
