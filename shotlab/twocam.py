"""Two-camera pose fusion: synced per-camera 2D poses -> 3D joint tracks ->
the metrics one camera can't measure (elbow flare, true release-point spread).

This is the ingest scaffold between the already-validated pieces:
  sync.py     -> offset_s between the clips
  stereo.py   -> StereoRig (metric cameras, feet)
  pose.py     -> per-camera FramePose dicts (frame_idx -> FramePose)
  threed.py   -> triangulation + flare/spread math

Frame alignment is nearest-frame (a 30 fps frame is 33 ms; joints move little
in half a frame except the wrist right at release -- fine for form metrics,
noted for anything faster). All 3D is in CAM-A coordinates, feet, and world-up
is taken from the rig's gravity hint (cam A is normally level, so -Y image up
~ world up; pass `up` explicitly if cam A is tilted).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .threed import Camera, triangulate, triangulate_joints, elbow_flare, \
    release_point_spread, shoulder_frame
from .stereo import StereoRig

# the joints worth triangulating for form work (BlazePose names)
FUSE_JOINTS = ["nose",
               "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
               "l_wrist", "r_wrist", "l_hip", "r_hip",
               "l_knee", "r_knee", "l_ankle", "r_ankle"]
MIN_VIS = 0.5


def frame_mapper(fps_a: float, fps_b: float, offset_s: float):
    """frame index in A -> the time-matched frame index in B (offset_s is
    sync.py's convention: B's clock runs behind A's by offset_s)."""
    def to_b(frame_a: int) -> int:
        return int(round((frame_a / fps_a - offset_s) * fps_b))
    return to_b


def _joints_2d(fp, names=FUSE_JOINTS, min_vis=MIN_VIS) -> dict:
    return {n: np.asarray(fp.pt(n), float) for n in names
            if fp.v(n) >= min_vis}


def fuse_pose_tracks(poses_a: dict, poses_b: dict, rig: StereoRig,
                     fps_a: float, fps_b: float, offset_s: float,
                     names=FUSE_JOINTS, min_vis=MIN_VIS,
                     search: int = 1) -> dict:
    """Per-frame 3D joints from the two cameras' 2D pose tracks.

    Returns {frame_a: {joint_name: (3,) cam-A coords, feet}} for every A frame
    where a time-matched B pose exists (within +/-`search` B frames) and at
    least one joint clears `min_vis` in BOTH views. Pixels are undistorted
    through the rig before triangulation."""
    to_b = frame_mapper(fps_a, fps_b, offset_s)
    out = {}
    for fa, fp_a in poses_a.items():
        fb0 = to_b(int(fa))
        fp_b = None
        for d in [0, 1, -1] if search else [0]:
            fp_b = poses_b.get(fb0 + d)
            if fp_b is not None:
                break
        if fp_b is None:
            continue
        ja = _joints_2d(fp_a, names, min_vis)
        jb = _joints_2d(fp_b, names, min_vis)
        common = [n for n in ja if n in jb]
        if not common:
            continue
        ua = rig.undistort([ja[n] for n in common], "a")
        ub = rig.undistort([jb[n] for n in common], "b")
        X = triangulate(rig.cam_a.P, rig.cam_b.P, ua, ub)
        out[int(fa)] = {n: X[i] for i, n in enumerate(common)}
    return out


def rim_3d(rig: StereoRig, rim_px_a, rim_px_b) -> np.ndarray:
    """The rim's 3D position (cam-A coords, feet) from its pixel location in
    each camera (cam B usually doesn't frame the rim -- if it ever does, this
    pins the flare plane; otherwise pass a surveyed/estimated point)."""
    ua = rig.undistort([rim_px_a], "a")
    ub = rig.undistort([rim_px_b], "b")
    return triangulate(rig.cam_a.P, rig.cam_b.P, ua, ub)[0]


@dataclass
class Shot3D:
    """3D form numbers for one shot (all feet / degrees, cam-A frame)."""
    flare_deg: float | None = None
    flare_offset_in: float | None = None
    release_point: tuple | None = None       # wrist at release, shoulder-relative
    note: str = ""


def shot_3d_metrics(joints3d: dict, release_frame: int, rim_xyz,
                    handedness: str = "right", up=(0, -1, 0)) -> Shot3D:
    """Elbow flare + shoulder-relative release point at the release frame.

    `up` defaults to (0,-1,0): cam-A coordinates have image-y DOWN, so world
    up is -Y when camera A sits level (the wide tripod cam). Pass the true up
    if cam A is tilted."""
    s = "r_" if handedness.lower().startswith("r") else "l_"
    fj = joints3d.get(int(release_frame))
    if fj is None:
        near = [f for f in joints3d if abs(f - release_frame) <= 2]
        if not near:
            return Shot3D(note="no fused pose at release")
        fj = joints3d[min(near, key=lambda f: abs(f - release_frame))]
    need = (s + "shoulder", s + "elbow", s + "wrist")
    if not all(n in fj for n in need):
        return Shot3D(note="shooting arm not stereo-visible at release")
    fl = elbow_flare(fj[s + "shoulder"], fj[s + "elbow"], rim_xyz, up=up)
    rel = np.asarray(fj[s + "wrist"], float) - np.asarray(fj[s + "shoulder"], float)
    return Shot3D(flare_deg=fl.angle_deg,
                  flare_offset_in=round(fl.offset * 12.0, 2),
                  release_point=tuple(round(float(v), 3) for v in rel))


def session_release_spread(shots: list, rim_xyz=None, shoulder=None, up=(0, -1, 0)):
    """release_point_spread over a session's Shot3D list (shoulder-relative
    points, so it's stance-independent). Pass rim+shoulder to split the spread
    into lateral/vertical/depth in the shooting frame."""
    pts = [s.release_point for s in shots if s.release_point is not None]
    if len(pts) < 2:
        return None
    frame = None
    if rim_xyz is not None and shoulder is not None:
        frame = shoulder_frame(shoulder, rim_xyz, up=up)
    return release_point_spread(np.asarray(pts, float), frame=frame)
