"""End-to-end two-camera fusion on synthetic ground truth: a shooter with a
KNOWN elbow flare seen by two cameras at different fps with a known sync
offset, rig calibrated from synthetic board views -- the fused 3D metrics must
recover the truth. This is the whole 2-cam pipeline minus real footage."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.stereo import board_points, calibrate_stereo
from shotlab.threed import Camera, elbow_flare, shoulder_frame
from shotlab.twocam import (frame_mapper, fuse_pose_tracks, shot_3d_metrics,
                            session_release_spread, Shot3D)
from shotlab.phase2_pose.pose import FramePose, L

IMG = (1920, 1080)
FPS_A, FPS_B = 30.0, 60.0
OFFSET_S = -2.5                    # B started 2.5 s BEFORE A


def true_rig():
    K_a = Camera.intrinsics(1400, 960, 540)
    K_b = Camera.intrinsics(1250, 960, 540)
    cam_a = Camera.look_at([0, 5, -35], [0, 4, 0], [0, 1, 0], K_a)
    cam_b = Camera.look_at([14, 4, 2], [0, 4, 0], [0, 1, 0], K_b)
    return cam_a, cam_b


def rot(rx, ry, rz):
    cx, sx, cy, sy, cz, sz = (np.cos(rx), np.sin(rx), np.cos(ry),
                              np.sin(ry), np.cos(rz), np.sin(rz))
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @
            np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @
            np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


def calibrated_rig():
    cam_a, cam_b = true_rig()
    rng = np.random.default_rng(0)
    base = board_points()
    center = base.mean(axis=0)
    va, vb = {}, {}
    for i in range(12):
        R = rot(*rng.uniform(-0.5, 0.5, 3))
        pos = np.array([rng.uniform(-4, 4), rng.uniform(2.5, 6.5),
                        rng.uniform(-3, 3)])
        world = (R @ (base - center).T).T + pos
        va[i], vb[i] = cam_a.project(world), cam_b.project(world)
    return cam_a, cam_b, calibrate_stereo(va, vb, IMG, IMG)


RIM_W = np.array([2.0, 10.0, 3.0])          # rim in world coords (feet)
FLARE_DEG = 12.0


def shooter_joints_world(t_s, flare_deg=FLARE_DEG):
    """The shooter mid-release with a KNOWN flare, drifting slightly so frames
    aren't identical. World coords, feet, +Y up."""
    S = np.array([0.0, 6.0, 0.0]) + 0.02 * np.sin(t_s) * np.array([1, 0, 1])
    fwd, up, side = shoulder_frame(S, RIM_W, up=(0, 1, 0))
    # orthonormal in-plane basis: fwd points UP at the rim, so raw `up` isn't
    # perpendicular to it -- a tucked vector mixing them isn't unit and the
    # nominal flare angle would be off. e1 = the plane's true vertical.
    e1 = np.cross(side, fwd)
    if np.dot(e1, up) < 0:
        e1 = -e1
    tucked = -e1 * np.cos(np.radians(20)) + fwd * np.sin(np.radians(20))
    a = np.radians(flare_deg)
    arm = tucked * np.cos(a) + side * np.sin(a)          # tucked _|_ side, unit
    elbow = S + arm * 1.0                                # 1 ft upper arm
    wrist = S + up * 0.9 + fwd * 0.5
    other_S = S - side * 1.2
    return {"r_shoulder": S, "r_elbow": elbow, "r_wrist": wrist,
            "l_shoulder": other_S, "l_hip": S + np.array([-0.6, -1.6, 0]),
            "r_hip": S + np.array([0.4, -1.6, 0])}


def frame_pose(cam, joints_world, idx, drop=()):
    xy = np.zeros((33, 2))
    vis = np.zeros(33)
    for name, X in joints_world.items():
        if name in drop:
            continue
        xy[L[name]] = cam.project(X)[0]
        vis[L[name]] = 1.0
    return FramePose(idx, xy, vis, np.zeros(33))


def pose_tracks(cam_a, cam_b, drop_b=()):
    poses_a = {f: frame_pose(cam_a, shooter_joints_world(f / FPS_A), f)
               for f in range(0, 90)}
    poses_b = {f: frame_pose(cam_b, shooter_joints_world(f / FPS_B + OFFSET_S),
                             f, drop=drop_b)
               for f in range(0, 500)}
    return poses_a, poses_b


def test_frame_mapper():
    to_b = frame_mapper(FPS_A, FPS_B, OFFSET_S)
    # A frame 60 = t 2.0 s = B time 4.5 s = B frame 270
    assert to_b(60) == 270, to_b(60)


def test_fused_joints_match_truth():
    cam_a, cam_b, rig = calibrated_rig()
    poses_a, poses_b = pose_tracks(cam_a, cam_b)
    fused = fuse_pose_tracks(poses_a, poses_b, rig, FPS_A, FPS_B, OFFSET_S)
    assert len(fused) > 80, len(fused)
    f = 60
    truth_world = shooter_joints_world(f / FPS_A)
    for name, X in fused[f].items():
        tw = truth_world[name]
        truth_a = cam_a.R @ tw + cam_a.t                # cam-A coords
        assert np.max(np.abs(X - truth_a)) < 0.05, (name, X, truth_a)


def test_flare_recovered():
    cam_a, cam_b, rig = calibrated_rig()
    poses_a, poses_b = pose_tracks(cam_a, cam_b)
    fused = fuse_pose_tracks(poses_a, poses_b, rig, FPS_A, FPS_B, OFFSET_S)
    rim_a = cam_a.R @ RIM_W + cam_a.t                   # rim in cam-A coords
    up_a = cam_a.R @ np.array([0, 1, 0])                # world up in cam-A
    m = shot_3d_metrics(fused, release_frame=60, rim_xyz=rim_a,
                        handedness="right", up=up_a)
    assert m.flare_deg is not None, m.note
    assert abs(abs(m.flare_deg) - FLARE_DEG) < 0.5, m.flare_deg
    assert m.release_point is not None


def test_flare_needs_stereo_visibility():
    cam_a, cam_b, rig = calibrated_rig()
    poses_a, poses_b = pose_tracks(cam_a, cam_b, drop_b=("r_elbow",))
    fused = fuse_pose_tracks(poses_a, poses_b, rig, FPS_A, FPS_B, OFFSET_S)
    rim_a = cam_a.R @ RIM_W + cam_a.t
    m = shot_3d_metrics(fused, 60, rim_a)
    assert m.flare_deg is None
    assert "stereo-visible" in m.note


def test_release_spread_over_shots():
    shots = [Shot3D(release_point=(0.5, 0.9, 0.1)),
             Shot3D(release_point=(0.55, 0.92, 0.08)),
             Shot3D(release_point=(0.45, 0.88, 0.12)),
             Shot3D(note="no fused pose at release")]
    sp = session_release_spread(shots)
    assert sp is not None and sp.n == 3
    assert 0.0 < sp.rms_spread < 0.1, sp.rms_spread


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} passed")
