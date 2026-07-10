"""Ballistic fit + time-base handling. Mutation-checked: each assert fails if the
math it guards is broken."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.ballistic import fit_ballistic, intrinsics
from shotlab.arc3d import analyze_shot, BALL_DIAM_FT, G_FT

W, H, F = 1920, 1080, 1450.0
K = intrinsics(W, H, F)


def _project(P0, V0, g, t):
    P = P0 + np.outer(t, V0) + 0.5 * np.outer(t * t, g)
    Z = P[:, 2]
    u = F * P[:, 0] / Z + W / 2
    v = F * P[:, 1] / Z + H / 2
    r = F * (BALL_DIAM_FT / 2) / Z
    return u, v, r, Z


def test_recovers_known_projectile():
    """A clean projectile projected through K is recovered to tight tolerance."""
    P0 = np.array([0.4, -1.2, 15.0])
    V0 = np.array([1.2, -18.0, 12.0])
    g = np.array([0.0, G_FT, 0.0])
    t = np.linspace(0, 0.9, 18)
    u, v, r, _ = _project(P0, V0, g, t)
    fit = fit_ballistic(t, u, v, r, K)
    assert np.allclose(fit.P0, P0, atol=0.3), f"P0 {fit.P0} vs {P0}"
    assert np.allclose(fit.V0, V0, atol=0.6), f"V0 {fit.V0} vs {V0}"
    true_ang = np.degrees(np.arctan2(-V0[1], np.hypot(V0[0], V0[2])))
    assert abs(fit.release_angle_deg - true_ang) < 1.5
    assert fit.reproj_rmse_px < 0.5
    assert fit.radius_consistency_pct < 3.0


def test_gravity_gate_flags_nonprojectile():
    """A straight (constant-velocity, NO gravity) pixel track cannot be explained
    by a gravity projectile -> the reprojection residual must blow up."""
    t = np.linspace(0, 0.9, 18)
    # a path with zero vertical curvature in the image
    u = np.linspace(700, 1200, 18)
    v = np.linspace(500, 505, 18)
    r = np.full(18, 25.0)
    fit = fit_ballistic(t, u, v, r, K)
    assert fit.reproj_rmse_px > 3.0 or fit.radius_consistency_pct > 25.0, \
        "a non-projectile should not pass both gates"


def test_analyze_shot_uses_real_times_not_fps():
    """With gap/VFR times, passing the true timestamps must change the gravity
    result vs assuming a constant fps over the same index count."""
    # build a real projectile sampled at IRREGULAR times
    P0 = np.array([0.0, -1.0, 16.0]); V0 = np.array([0.5, -17.0, 10.0])
    g = np.array([0.0, G_FT, 0.0])
    t = np.array([0, .05, .07, .13, .21, .22, .30, .38, .5, .6, .75, .9])
    u, v, r, _ = _project(P0, V0, g, t)
    # correct times -> vertical accel should read close to gravity
    a_true = analyze_shot(u, v, r, fps=30, image_w=W, image_h=H, times=t)
    # wrong: pretend gap-free 30fps (arange) over the same 12 points
    a_wrong = analyze_shot(u, v, r, fps=30, image_w=W, image_h=H, times=None)
    assert a_true.gravity_error_pct < 20, a_true.gravity_error_pct
    assert a_wrong.gravity_error_pct > a_true.gravity_error_pct + 10, \
        (a_wrong.gravity_error_pct, a_true.gravity_error_pct)


if __name__ == "__main__":
    test_recovers_known_projectile()
    test_gravity_gate_flags_nonprojectile()
    test_analyze_shot_uses_real_times_not_fps()
    print("ok")
