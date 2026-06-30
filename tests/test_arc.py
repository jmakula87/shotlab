"""Validate the parabola/angle math against analytically-known trajectories."""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.arc import fit_parabola_ransac, estimate_px_per_foot_from_ball


def _synthetic_shot(release_deg, n=40, noise_px=0.0, seed=0, with_outliers=0):
    """Make image-space (x, y_px) points for a projectile launched at a known
    angle. Image y grows downward, so y_px = -h. We pick a launch speed/gravity
    that produce a nice arc across the frame."""
    rng = np.random.default_rng(seed)
    ang = math.radians(release_deg)
    v = 30.0
    vx = v * math.cos(ang)
    vy = v * math.sin(ang)
    g = 9.8
    t = np.linspace(0, 2 * vy / g, n)  # launch to same height
    x = vx * t
    h = vy * t - 0.5 * g * t * t        # height, up positive
    y_px = -h                            # image space
    if noise_px:
        x = x + rng.normal(0, noise_px, n)
        y_px = y_px + rng.normal(0, noise_px, n)
    if with_outliers:
        oi = rng.choice(n, with_outliers, replace=False)
        y_px[oi] += rng.uniform(-50, 50, with_outliers)
    return x, y_px


def test_release_angle_clean():
    for true_ang in (38, 45, 52, 60):
        x, y = _synthetic_shot(true_ang)
        fit = fit_parabola_ransac(x, y)
        assert fit is not None
        got = fit.release_angle_deg()
        assert abs(got - true_ang) < 1.0, f"release {true_ang} -> {got}"


def test_entry_equals_release_for_symmetric_arc():
    # Vacuum parabola is symmetric: entry angle == release angle.
    x, y = _synthetic_shot(50)
    fit = fit_parabola_ransac(x, y)
    assert abs(fit.entry_angle_deg() - fit.release_angle_deg()) < 1.0


def test_robust_to_noise_and_outliers():
    x, y = _synthetic_shot(48, noise_px=2.0, with_outliers=6, seed=3)
    fit = fit_parabola_ransac(x, y, threshold_px=6.0)
    assert fit is not None
    assert abs(fit.release_angle_deg() - 48) < 3.0
    assert fit.n_used >= 30  # most points recovered as inliers


def test_apex_is_positive_and_centered():
    x, y = _synthetic_shot(45, n=41)
    fit = fit_parabola_ransac(x, y)
    assert fit.apex_height_px > 0
    # apex x near the middle of the x-range for a symmetric arc
    assert x.min() < fit.apex_x < x.max()


def test_scale_helper():
    # 80 px ball -> px_per_foot; 0.7925 ft diameter
    s = estimate_px_per_foot_from_ball(80.0)
    assert abs(s - 80.0 / (9.51 / 12.0)) < 1e-6


def test_too_few_points_returns_none():
    assert fit_parabola_ransac(np.array([1, 2, 3.0]), np.array([1, 2, 3.0])) is None


if __name__ == "__main__":
    import traceback

    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for f in funcs:
        try:
            f()
            print(f"PASS {f.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL {f.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(funcs)} passed")
    sys.exit(0 if passed == len(funcs) else 1)
