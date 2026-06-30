"""
Trajectory geometry for Phase 1.

Everything in this module is pure math (numpy/scipy) and independent of which
ball detector we use. Given a set of ball detections (frame index + pixel x,y),
we fit a robust parabola to the flight path and derive:

  - release angle  (degrees above horizontal at the launch point)
  - apex height    (in pixels, and in feet if a px-per-foot scale is supplied)
  - entry angle    (degrees below horizontal as the ball descends to the rim)

Coordinate convention
---------------------
Image pixel coordinates have y growing DOWNWARD. We convert to a physical
"height" h = -y so that up is positive, then fit h = a*x^2 + b*x + c.

For a projectile filmed by a camera whose optical axis is perpendicular to the
plane of flight (the side-on tripod setup), horizontal screen distance is
proportional to time at constant horizontal velocity, so h-vs-x *is* the
parabola. Angles computed as atan(dh/dx) need NO distance calibration as long
as pixels are square (x and h share the same pixel unit) -- this is the big win
and why release/entry angle are reported with higher confidence than apex height
(which needs a px->feet scale).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ArcFit:
    """Result of fitting a parabola to one shot's ball detections."""

    coeffs: np.ndarray          # [a, b, c] for h = a*x^2 + b*x + c  (h = -y_px)
    inlier_mask: np.ndarray     # bool mask over the input points
    xs: np.ndarray              # input pixel x of inliers
    hs: np.ndarray              # input height (-y) of inliers
    n_used: int
    rmse_px: float              # fit residual on inliers, pixels
    direction: int = 1          # +1 if ball travels left->right, -1 otherwise
    meta: dict = field(default_factory=dict)

    # ---- derived geometry -------------------------------------------------
    def height_at(self, x: float) -> float:
        a, b, c = self.coeffs
        return a * x * x + b * x + c

    def slope_at(self, x: float) -> float:
        """dh/dx at x."""
        a, b, _ = self.coeffs
        return 2.0 * a * x + b

    def angle_at(self, x: float) -> float:
        """Signed angle of the flight path at x, degrees above horizontal."""
        return math.degrees(math.atan(self.slope_at(x)))

    @property
    def apex_x(self) -> float:
        a, b, _ = self.coeffs
        if abs(a) < 1e-12:
            return float(np.mean(self.xs))
        return -b / (2.0 * a)

    @property
    def apex_x_clamped(self) -> float:
        """Vertex x, clamped to the tracked x-range. For a full arc this is the
        true vertex; for a partial/near-vertical arc it avoids extrapolating the
        apex far outside the data (which produces absurd apex heights)."""
        return float(np.clip(self.apex_x, self.xs.min(), self.xs.max()))

    @property
    def apex_vertex_in_range(self) -> bool:
        """True if the analytic vertex sits within (a margin of) the tracked data
        -- i.e. the parabola fit is non-degenerate."""
        span = self.xs.max() - self.xs.min()
        margin = 0.3 * span + 1.0
        return self.xs.min() - margin <= self.apex_x <= self.xs.max() + margin

    @property
    def apex_height_px(self) -> float:
        """Apex height above the lowest tracked point, in pixels (clamped)."""
        return self.height_at(self.apex_x_clamped) - float(np.min(self.hs))

    def release_angle_deg(self) -> float:
        """Angle at the launch end of the tracked arc (always reported >0)."""
        x_release = self.xs.min() if self.direction > 0 else self.xs.max()
        return abs(self.angle_at(x_release))

    def entry_angle_deg(self, rim_x: float | None = None) -> float:
        """Angle as the ball descends to the rim.

        If rim_x is known (px), evaluate there; otherwise use the descending
        end of the tracked arc. Reported as a positive degrees-below-horizontal.
        """
        if rim_x is not None:
            x_eval = rim_x
        else:
            x_eval = self.xs.max() if self.direction > 0 else self.xs.min()
        return abs(self.angle_at(x_eval))


def fit_parabola_ransac(
    xs: np.ndarray,
    ys_px: np.ndarray,
    *,
    n_iters: int = 200,
    threshold_px: float = 6.0,
    min_inliers_frac: float = 0.5,
    rng: np.random.Generator | None = None,
) -> ArcFit | None:
    """Robustly fit h = -y = a*x^2 + b*x + c to noisy ball detections.

    RANSAC over 3-point samples, then least-squares refit on the inliers.
    Returns None if the data can't support a parabola (too few points or no
    consensus) -- callers should treat that as "no clean shot detected".
    """
    xs = np.asarray(xs, dtype=float)
    ys_px = np.asarray(ys_px, dtype=float)
    hs = -ys_px  # up is positive

    n = len(xs)
    if n < 4:
        return None

    if rng is None:
        rng = np.random.default_rng(0)  # deterministic by default

    best_inliers = None
    best_count = 0

    idx = np.arange(n)
    for _ in range(n_iters):
        sample = rng.choice(idx, size=3, replace=False)
        sx, sh = xs[sample], hs[sample]
        if len(np.unique(sx)) < 3:
            continue
        try:
            coeffs = np.polyfit(sx, sh, 2)
        except Exception:
            continue
        pred = np.polyval(coeffs, xs)
        resid = np.abs(pred - hs)
        inliers = resid < threshold_px
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < max(4, int(min_inliers_frac * n)):
        return None

    # Refit on all inliers for a stable estimate.
    coeffs = np.polyfit(xs[best_inliers], hs[best_inliers], 2)
    pred = np.polyval(coeffs, xs[best_inliers])
    rmse = float(np.sqrt(np.mean((pred - hs[best_inliers]) ** 2)))

    # Travel direction from the inlier point order (detections are time-ordered).
    in_x = xs[best_inliers]
    direction = 1 if in_x[-1] >= in_x[0] else -1

    return ArcFit(
        coeffs=coeffs,
        inlier_mask=best_inliers,
        xs=in_x,
        hs=hs[best_inliers],
        n_used=best_count,
        rmse_px=rmse,
        direction=direction,
    )


def estimate_px_per_foot_from_ball(ball_diameter_px: float) -> float:
    """A regulation men's basketball is ~9.51 in diameter = 0.7925 ft.

    Using the ball's own pixel size as a ruler gives a rough px->feet scale
    without any court calibration. It is approximate (the ball is only at the
    camera plane's depth when it's near it) -- apex height in feet is therefore
    flagged lower-confidence than the angles.
    """
    BALL_DIAMETER_FT = 9.51 / 12.0
    if ball_diameter_px <= 0:
        return float("nan")
    return ball_diameter_px / BALL_DIAMETER_FT
