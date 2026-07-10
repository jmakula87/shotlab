"""Gravity-constrained ballistic fit -> metric 3D shot arc from ONE camera.

The problem with the naive parabola (arc3d): it needs the ball's per-frame radius
to carry depth, and a contiguous clean run. Detections here are sparse and gappy,
and the ball is small so radius is noisy.

This instead fits the WHOLE flight as one rigid projectile with acceleration
PINNED to gravity:

    Pos(t) = P0 + V0*t + 1/2 * g_vec * t^2          (camera coords, feet)

and projects that 3D path through the camera intrinsics to the observed pixel
centers (and, softly, the observed radii). Six unknowns (P0, V0) fit from as few
as ~8 points scattered ANYWHERE along the arc -- no contiguity needed, which is
what makes it survive the intermittent YOLO recall. Because acceleration is not
fitted (it's fixed at g), the fit RESIDUAL is a real honesty gate: a track that
isn't a gravity projectile won't fit.

Camera convention: OpenCV (X right, Y DOWN, Z forward/away). A LEVEL camera has
g_vec = (0, +g, 0). Tilt is passed in (or recovered later from rim-PnP). Absolute
depth/scale needs a correct focal length; the focal-free vertical & lateral arc
(see `.reproj_ok` and arc3d cross-check) does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .arc3d import BALL_DIAM_FT, G_FT


@dataclass
class Ballistic3D:
    P0: np.ndarray               # release position, camera coords, feet
    V0: np.ndarray               # release velocity, ft/s
    g_vec: np.ndarray            # gravity used, ft/s^2
    release_speed_fps: float
    release_angle_deg: float     # above horizontal (true 3D)
    entry_angle_deg: float
    apex_above_release_ft: float
    depth_drift_ft: float        # +toward rim / away-from-camera over the flight
    lateral_drift_ft: float      # net left/right (camera-x) over the flight
    reproj_rmse_px: float        # fit residual -> honesty gate
    radius_consistency_pct: float  # focal-free: predicted vs observed ball size
    n: int
    t0: float
    t_end: float

    def pos(self, t):
        t = np.asarray(t, float)
        return self.P0 + np.outer(t, self.V0) + 0.5 * np.outer(t * t, self.g_vec)


def _project(P, fx, fy, cx0, cy0):
    Z = np.clip(P[:, 2], 1e-3, None)
    u = fx * P[:, 0] / Z + cx0
    v = fy * P[:, 1] / Z + cy0
    return u, v, Z


def fit_ballistic(t, u, v, r, K, ball_diam_ft: float = BALL_DIAM_FT,
                  g_vec=None, radius_weight: float = 0.3) -> Ballistic3D:
    """Fit one shot's pixel track (times seconds, u/v/r pixels) to a gravity
    projectile seen through K (3x3 intrinsics). Returns metric 3D arc."""
    t = np.asarray(t, float); t = t - t[0]
    u = np.asarray(u, float); v = np.asarray(v, float); r = np.asarray(r, float)
    n = len(t)
    if n < 6:
        raise ValueError(f"need >=6 points, got {n}")
    fx, fy = K[0, 0], K[1, 1]
    cx0, cy0 = K[0, 2], K[1, 2]
    if g_vec is None:
        g_vec = np.array([0.0, G_FT, 0.0])       # level camera, gravity = +Y (down)
    g_vec = np.asarray(g_vec, float)

    # initial guess: per-point depth from radius, back-projected
    Zi = fx * (ball_diam_ft / 2.0) / np.clip(r, 1e-3, None)
    Xi = (u - cx0) * Zi / fx
    Yi = (v - cy0) * Zi / fy
    # remove gravity to linearize for P0,V0 seed
    Xc = np.stack([Xi, Yi, Zi], 1) - 0.5 * np.outer(t * t, g_vec)
    A = np.stack([np.ones_like(t), t], 1)
    coef, *_ = np.linalg.lstsq(A, Xc, rcond=None)   # rows: [P0; V0]
    p0 = np.concatenate([coef[0], coef[1]])

    def resid(p):
        P0 = p[:3]; V0 = p[3:]
        P = P0 + np.outer(t, V0) + 0.5 * np.outer(t * t, g_vec)
        pu, pv, Z = _project(P, fx, fy, cx0, cy0)
        rp = fx * (ball_diam_ft / 2.0) / Z
        return np.concatenate([pu - u, pv - v, radius_weight * (rp - r)])

    sol = least_squares(resid, p0, method="lm", max_nfev=4000)
    P0, V0 = sol.x[:3], sol.x[3:]

    P = P0 + np.outer(t, V0) + 0.5 * np.outer(t * t, g_vec)
    pu, pv, Z = _project(P, fx, fy, cx0, cy0)
    reproj = float(np.sqrt(np.mean((pu - u) ** 2 + (pv - v) ** 2)))
    rp = fx * (ball_diam_ft / 2.0) / Z
    rad_pct = float(np.median(np.abs(rp - r) / np.clip(r, 1e-3, None)) * 100)

    # metrics
    up = -V0[1]                                   # world-up component (Y is down)
    horiz = float(np.hypot(V0[0], V0[2]))
    rel_ang = float(np.degrees(np.arctan2(up, horiz)))
    ve = V0 + g_vec * (t[-1] - t[0])              # velocity at end
    ent_ang = float(np.degrees(np.arctan2(-(-ve[1]), np.hypot(ve[0], ve[2]))))
    # apex: where vertical velocity = 0 (world-up), if within flight
    if abs(g_vec[1]) > 1e-6:
        t_apex = np.clip((-V0[1]) / (-g_vec[1]) * -1.0, t[0], t[-1]) if False else \
                 np.clip(V0[1] / g_vec[1] * -1.0, 0, t[-1])
    else:
        t_apex = t[-1]
    apex_pos = P0 + V0 * t_apex + 0.5 * g_vec * t_apex ** 2
    apex_up = float(-(apex_pos[1] - P0[1]))
    depth_drift = float(P[-1, 2] - P[0, 2])
    lateral_drift = float(P[-1, 0] - P[0, 0])

    return Ballistic3D(
        P0=P0, V0=V0, g_vec=g_vec,
        release_speed_fps=round(float(np.linalg.norm(V0)), 1),
        release_angle_deg=round(rel_ang, 1),
        entry_angle_deg=round(ent_ang, 1),
        apex_above_release_ft=round(apex_up, 2),
        depth_drift_ft=round(depth_drift, 2),
        lateral_drift_ft=round(lateral_drift, 2),
        reproj_rmse_px=round(reproj, 2),
        radius_consistency_pct=round(rad_pct, 1),
        n=n, t0=float(t[0]), t_end=float(t[-1]))


def intrinsics(image_w: int, image_h: int, focal_px: float) -> np.ndarray:
    """Simple pinhole K from a focal (principal point at image center)."""
    return np.array([[focal_px, 0, image_w / 2.0],
                     [0, focal_px, image_h / 2.0],
                     [0, 0, 1.0]])


def _grav_dir(pitch: float, roll: float) -> np.ndarray:
    """Gravity direction in camera coords for a camera tilted `pitch` up and
    `roll` about its optical axis. Level camera (0,0) -> (0,+g,0) (world-down is
    image-down). pitch>0 tips the lens up, tilting gravity partly into +Z."""
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    # start from world-down (0,1,0), rotate by roll about Z then pitch about X
    d = np.array([0.0, 1.0, 0.0])
    d = np.array([cr * d[0] - sr * d[1], sr * d[0] + cr * d[1], d[2]])   # roll (Z)
    d = np.array([d[0], cp * d[1] - sp * d[2], sp * d[1] + cp * d[2]])   # pitch (X)
    return G_FT * d


def fit_camera_tilt(shots, K, ball_diam_ft: float = BALL_DIAM_FT,
                    radius_weight: float = 0.3):
    """Recover the (fixed) camera tilt shared by several shots, from the physics
    alone -- no rim, no board. Each shot is a gravity projectile; the ONE tilt
    (pitch, roll) that makes every shot's reprojection consistent is the camera's
    orientation, which is what W2 needs to turn its arc into true depth & release
    angle.

    shots: list of (t, u, v, r) tuples (>=2 arcs with DIFFERENT launch directions
    are needed to observe tilt; a single arc is degenerate). Returns
    (pitch_deg, roll_deg, g_vec, per_shot_fits, rmse_px)."""
    shots = [(np.asarray(t, float) - np.asarray(t, float)[0], np.asarray(u, float),
              np.asarray(v, float), np.asarray(r, float)) for (t, u, v, r) in shots]
    if len(shots) < 2:
        raise ValueError("need >=2 shots to observe camera tilt")
    fx, fy = K[0, 0], K[1, 1]
    cx0, cy0 = K[0, 2], K[1, 2]

    # seed each shot's P0,V0 from a level-camera fit, then refine tilt jointly
    seeds = []
    for (t, u, v, r) in shots:
        f = fit_ballistic(t, u, v, r, K, ball_diam_ft, radius_weight=radius_weight)
        seeds.append(np.concatenate([f.P0, f.V0]))
    p0 = np.concatenate([[0.0, 0.0]] + seeds)      # pitch, roll, then per-shot 6

    def resid(p):
        pitch, roll = p[0], p[1]
        g = _grav_dir(pitch, roll)
        out = []
        for i, (t, u, v, r) in enumerate(shots):
            P0 = p[2 + 6 * i: 5 + 6 * i]; V0 = p[5 + 6 * i: 8 + 6 * i]
            P = P0 + np.outer(t, V0) + 0.5 * np.outer(t * t, g)
            pu, pv, Z = _project(P, fx, fy, cx0, cy0)
            rp = fx * (ball_diam_ft / 2.0) / Z
            out.append(np.concatenate([pu - u, pv - v, radius_weight * (rp - r)]))
        return np.concatenate(out)

    sol = least_squares(resid, p0, method="lm", max_nfev=8000)
    pitch, roll = float(sol.x[0]), float(sol.x[1])
    g = _grav_dir(pitch, roll)
    fits = []
    for i, (t, u, v, r) in enumerate(shots):
        fits.append(fit_ballistic(t, u, v, r, K, ball_diam_ft, g_vec=g,
                                  radius_weight=radius_weight))
    rmse = float(np.sqrt(np.mean(resid(sol.x)[: sum(2 * len(s[0]) for s in shots)] ** 2)))
    return round(np.degrees(pitch), 1), round(np.degrees(roll), 1), g, fits, round(rmse, 2)
