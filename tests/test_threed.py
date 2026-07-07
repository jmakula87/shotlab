"""Validate the two-camera 3D core against SYNTHETIC ground truth.

Known 3D joints (with a KNOWN elbow flare) are projected into two virtual
cameras, then triangulated back -- we check we recover the 3D and the flare.
This is the stereo analog of the synthetic-clip arc test: it proves the math is
right before any real two-camera footage exists. No MediaPipe / no video needed.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.threed import (Camera, triangulate, triangulate_joints,
                            elbow_flare, shoulder_frame, release_point_spread)


# A 1080p-ish phone intrinsics, and two cameras ~90 deg apart looking at the shooter.
K = Camera.intrinsics(f=1500.0, cx=960.0, cy=540.0)
TARGET = (0.0, 1.4, 0.0)                       # roughly the shooter's chest
CAM1 = Camera.look_at((4.0, 1.4, 1.0), TARGET, up=(0, 1, 0), K=K)   # side-on, +X
CAM2 = Camera.look_at((-1.5, 1.4, 3.5), TARGET, up=(0, 1, 0), K=K)  # front-quarter


def test_look_at_centers_target():
    # the aim point projects to the principal point
    px = CAM1.project(TARGET)[0]
    assert abs(px[0] - 960) < 1e-6 and abs(px[1] - 540) < 1e-6


def test_look_at_is_a_proper_rotation_with_correct_orientation():
    # the old test only pinned the optical axis, which is invariant to roll/mirror
    # -- a det=-1 (mirror) or 180deg roll bug survived it (2026-07-07 audit).
    cam = Camera.look_at((0, 0, -5), (0, 0, 0), up=(0, 1, 0), K=K)
    assert abs(np.linalg.det(cam.R) - 1.0) < 1e-9        # proper rotation, not a reflection
    above = cam.project((0, 2, 0))[0]
    below = cam.project((0, -2, 0))[0]
    assert above[1] < below[1]                           # a point above -> higher (smaller y)


def test_triangulation_recovers_known_3d():
    pts = np.array([[0.1, 1.5, 0.2], [-0.3, 1.0, 0.4], [0.0, 3.05, 5.0]])
    p1 = CAM1.project(pts)
    p2 = CAM2.project(pts)
    rec = triangulate(CAM1.P, CAM2.P, p1, p2)
    assert np.allclose(rec, pts, atol=1e-6), rec - pts


def test_triangulation_robust_to_pixel_noise():
    rng = np.random.default_rng(0)
    pts = np.array([[0.2, 1.45, 0.1], [-0.2, 1.2, 0.3]])
    p1 = CAM1.project(pts) + rng.normal(0, 1.0, (2, 2))     # ~1px noise
    p2 = CAM2.project(pts) + rng.normal(0, 1.0, (2, 2))
    rec = triangulate(CAM1.P, CAM2.P, p1, p2)
    # 1px noise -> sub-cm 3D error at this geometry
    assert np.linalg.norm(rec - pts, axis=1).max() < 0.03


def _arm_with_flare(shoulder, rim, flare_deg, length=0.34, up=(0, 1, 0)):
    """Build an elbow position whose upper arm has a KNOWN flare angle: take a
    downward-and-forward in-plane direction, rotate it toward `side` by flare."""
    fwd, up_v, side = shoulder_frame(shoulder, rim, up)
    in_plane = -np.asarray(up_v, float) + 0.3 * fwd          # mostly down, a bit forward
    in_plane /= np.linalg.norm(in_plane)
    a = math.radians(flare_deg)
    arm_dir = math.cos(a) * in_plane + math.sin(a) * side
    return np.asarray(shoulder, float) + length * arm_dir


def test_elbow_flare_tucked_is_zero():
    shoulder = np.array([0.0, 1.4, 0.0])
    rim = np.array([0.0, 3.05, 5.0])
    elbow = _arm_with_flare(shoulder, rim, flare_deg=0.0)
    fr = elbow_flare(shoulder, elbow, rim)
    assert abs(fr.angle_deg) < 0.5, fr.angle_deg          # tucked -> ~0


def test_elbow_flare_recovers_known_angle_through_stereo():
    """The full path: 3D joints w/ known flare -> project to 2 cams -> triangulate
    -> recover flare. This is the real end-to-end proof."""
    shoulder = np.array([0.15, 1.45, 0.10])
    rim = np.array([0.0, 3.05, 5.0])
    for true_flare in (0.0, 10.0, 20.0, -15.0):
        elbow = _arm_with_flare(shoulder, rim, true_flare)
        wrist = elbow + np.array([0.0, -0.25, 0.05])
        j3d = {"shoulder": shoulder, "elbow": elbow, "wrist": wrist}
        j1 = {n: CAM1.project(p)[0] for n, p in j3d.items()}
        j2 = {n: CAM2.project(p)[0] for n, p in j3d.items()}
        rec = triangulate_joints(CAM1, CAM2, j1, j2)
        fr = elbow_flare(rec["shoulder"], rec["elbow"], rim)
        assert abs(fr.angle_deg - true_flare) < 0.5, (true_flare, fr.angle_deg)


def test_release_point_spread_tight_vs_loose():
    rng = np.random.default_rng(1)
    tight = rng.normal(0, 0.01, (20, 3))      # 1cm scatter
    loose = rng.normal(0, 0.08, (20, 3))      # 8cm scatter
    st = release_point_spread(tight)
    sl = release_point_spread(loose)
    assert st.rms_spread < sl.rms_spread
    assert st.rms_spread < 0.03 and sl.rms_spread > 0.08
    assert release_point_spread([[0, 0, 0]]) is None      # need >=2 shots


if __name__ == "__main__":
    import traceback
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in funcs:
        try:
            fn(); print(f"PASS {fn.__name__}"); passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(funcs)} passed")
    sys.exit(0 if passed == len(funcs) else 1)
