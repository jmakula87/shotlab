"""Stereo-calibration validation against a synthetic ground-truth rig: two
known cameras (wide + close body-cam, like the real plan), a checkerboard
waved through known poses, corners projected exactly -- the solver must
recover the rig well enough to triangulate points back to real-feet truth."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.stereo import (board_points, calibrate_intrinsics,
                            calibrate_stereo, StereoRig, PATTERN, SQUARE_FT)
from shotlab.threed import Camera, triangulate

IMG = (1920, 1080)


def true_rig():
    """Ground truth like the real setup: A = wide cam ~35 ft out, B = close
    body-cam ~12 ft out, perpendicular-ish. Feet, world up = +Y."""
    K_a = Camera.intrinsics(1400, 960, 540)
    K_b = Camera.intrinsics(1250, 960, 540)
    cam_a = Camera.look_at([0, 5, -35], [0, 4, 0], [0, 1, 0], K_a)
    cam_b = Camera.look_at([14, 4, 2], [0, 4, 0], [0, 1, 0], K_b)
    return cam_a, cam_b


def rot(rx, ry, rz):
    cx, sx, cy, sy, cz, sz = (np.cos(rx), np.sin(rx), np.cos(ry),
                              np.sin(ry), np.cos(rz), np.sin(rz))
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def board_views(cam_a, cam_b, n_poses=12, seed=0):
    """Wave the board through varied poses; project the corners into both
    cameras. Returns (views_a, views_b) keyed by pose index."""
    rng = np.random.default_rng(seed)
    base = board_points()                      # (N,3), board plane z=0, feet
    center = base.mean(axis=0)
    views_a, views_b = {}, {}
    for i in range(n_poses):
        R = rot(*rng.uniform(-0.5, 0.5, 3))
        pos = np.array([rng.uniform(-4, 4), rng.uniform(2.5, 6.5),
                        rng.uniform(-3, 3)])
        world = (R @ (base - center).T).T + pos
        views_a[i] = cam_a.project(world)
        views_b[i] = cam_b.project(world)
    return views_a, views_b


def test_intrinsics_recovered():
    cam_a, cam_b = true_rig()
    va, _ = board_views(cam_a, cam_b)
    K, dist, rms = calibrate_intrinsics(va.values(), IMG)
    assert rms < 0.5, rms
    assert abs(K[0, 0] - 1400) / 1400 < 0.02, K[0, 0]      # focal within 2%
    assert np.all(np.abs(dist) < 0.05), dist               # ~no distortion


def test_stereo_recovers_baseline_and_pose():
    cam_a, cam_b = true_rig()
    va, vb = board_views(cam_a, cam_b)
    rig = calibrate_stereo(va, vb, IMG, IMG)
    true_baseline = np.linalg.norm(np.array([0, 5, -35]) - np.array([14, 4, 2]))
    assert rig.rms_px < 0.5, rig.rms_px
    assert abs(rig.baseline_ft - true_baseline) < 0.2, (rig.baseline_ft,
                                                        true_baseline)
    # relative rotation matches truth
    R_true = cam_b.R @ cam_a.R.T
    err = np.degrees(np.arccos(np.clip(
        (np.trace(rig.cam_b.R @ R_true.T) - 1) / 2, -1, 1)))
    assert err < 0.5, err


def test_triangulation_through_recovered_rig():
    """The point of it all: a 3D point seen by both real cameras, triangulated
    through the RECOVERED rig, must land at its true position (in cam-A
    coords, feet) -- elbow-flare offsets depend on this being metric."""
    cam_a, cam_b = true_rig()
    va, vb = board_views(cam_a, cam_b)
    rig = calibrate_stereo(va, vb, IMG, IMG)
    targets = np.array([[0.0, 6.5, 0.0],      # a release point
                        [1.2, 4.8, -0.6],     # an elbow
                        [-2.0, 5.5, 1.5]])
    px_a, px_b = cam_a.project(targets), cam_b.project(targets)
    got = triangulate(rig.cam_a.P, rig.cam_b.P, px_a, px_b)
    truth_in_a = (cam_a.R @ targets.T + cam_a.t.reshape(3, 1)).T
    assert np.max(np.abs(got - truth_in_a)) < 0.05, got     # < 0.6 inch


def test_rig_roundtrips_through_json(tmp_path=None):
    import tempfile
    cam_a, cam_b = true_rig()
    va, vb = board_views(cam_a, cam_b)
    rig = calibrate_stereo(va, vb, IMG, IMG)
    path = os.path.join(tempfile.gettempdir(), "shotlab_test_rig.json")
    rig.save(path)
    back = StereoRig.load(path)
    assert np.allclose(back.cam_b.R, rig.cam_b.R)
    assert abs(back.baseline_ft - rig.baseline_ft) < 1e-9
    # undistort with ~zero true distortion is ~identity INSIDE the field the
    # board covered (k1/k2 may trade off; they only cancel where calibrated)
    pts = np.array([[700.0, 400.0], [1200.0, 700.0]])
    assert np.max(np.abs(back.undistort(pts, "a") - pts)) < 2.0
    os.remove(path)


def test_too_few_views_raises():
    cam_a, cam_b = true_rig()
    va, vb = board_views(cam_a, cam_b, n_poses=3)
    try:
        calibrate_stereo(va, vb, IMG, IMG)
        assert False, "should have raised"
    except ValueError:
        pass


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} passed")
