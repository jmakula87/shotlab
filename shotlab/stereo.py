"""Stereo calibration from checkerboard clips -> metric `threed.Camera` pair.

The protocol (both cameras rolling, per PROJECT_NOTES):
  1. print tools/make_checkerboard.py's board (verify the 6.000in ruler),
     mount it FLAT;
  2. walk it around the shooting area, tilting it through varied poses, held
     visible to BOTH cameras at once;
  3. this module finds the inner corners in sampled frames, calibrates each
     camera's intrinsics, then solves the relative pose (R, T) between them.

Because the square size is known in real inches, T comes out in real units --
that's what makes elbow-flare offsets and release-spread REAL inches/feet.

Conventions match shotlab.threed / OpenCV: x_cam = R @ X_world + t, camera
looks down +Z. Camera A is the world origin; camera B = (R @ X + T) off A.
All lengths are in FEET (the codebase's real-world unit).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import cv2
import numpy as np

from .threed import Camera

PATTERN = (9, 6)                 # inner corners of the printed board
SQUARE_FT = 0.9 / 12.0           # 0.9 in squares


def board_points(pattern=PATTERN, square_ft: float = SQUARE_FT) -> np.ndarray:
    """The board's corner grid in its own plane (z=0), in feet."""
    cols, rows = pattern
    pts = np.zeros((cols * rows, 3), np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_ft
    return pts


def find_corners(frame_bgr, pattern=PATTERN):
    """Inner corners in one frame, sub-pixel refined, or None."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCorners(
        gray, pattern,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    corners = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
    return corners.reshape(-1, 2)


def corners_from_video(path, pattern=PATTERN, stride: int = 15,
                       max_views: int = 25):
    """Sample a calibration clip -> {frame_idx: (N,2) corners}. Sampling every
    `stride` frames keeps the views diverse (a board barely moving between
    consecutive frames adds no constraint)."""
    from .video_io import iter_frames
    views = {}
    for idx, frame in iter_frames(path, start=0, stop=None):
        if idx % stride:
            continue
        c = find_corners(frame, pattern)
        if c is not None:
            views[idx] = c
            if len(views) >= max_views:
                break
    return views


def paired_corners_from_videos(path_a, path_b, offset_s: float,
                               fps_a: float, fps_b: float,
                               pattern=PATTERN, stride: int = 15,
                               max_views: int = 25):
    """Board views seen by BOTH cameras of one synced calibration take.

    Samples clip A every `stride` frames; wherever the board is found, seeks
    the time-matched frame of clip B (offset_s from shotlab.sync: B's clock
    runs behind A's by offset_s) and requires the board there too. Returns
    (views_a, views_b) sharing keys -- ready for calibrate_stereo."""
    from .video_io import iter_frames
    cap_b = cv2.VideoCapture(path_b)
    views_a, views_b = {}, {}
    try:
        for idx, frame in iter_frames(path_a, start=0, stop=None):
            if idx % stride:
                continue
            ca = find_corners(frame, pattern)
            if ca is None:
                continue
            idx_b = int(round((idx / fps_a - offset_s) * fps_b))
            if idx_b < 0:
                continue
            cap_b.set(cv2.CAP_PROP_POS_FRAMES, idx_b)
            ok, frame_b = cap_b.read()
            if not ok:
                continue
            cb = find_corners(frame_b, pattern)
            if cb is None:
                continue
            views_a[idx] = ca
            views_b[idx] = cb
            if len(views_a) >= max_views:
                break
    finally:
        cap_b.release()
    return views_a, views_b


def calibrate_intrinsics(corner_views, image_size, pattern=PATTERN,
                         square_ft: float = SQUARE_FT):
    """One camera's K + distortion from its board views.
    corner_views: iterable of (N,2) pixel corner arrays. Returns (K, dist,
    rms_px)."""
    corner_views = list(corner_views)
    if len(corner_views) < 4:
        raise ValueError(f"need >=4 usable board views, got {len(corner_views)}")
    obj = [board_points(pattern, square_ft)] * len(corner_views)
    img = [np.asarray(c, np.float32).reshape(-1, 1, 2) for c in corner_views]
    # k3 fixed: with a planar target k2/k3 trade off freely (both huge, net
    # effect ~zero inside the seen field) -- k1/k2 is plenty for phone lenses
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, tuple(image_size),
                                             None, None,
                                             flags=cv2.CALIB_FIX_K3)
    return K, dist, float(rms)


@dataclass
class StereoRig:
    cam_a: Camera            # world origin
    cam_b: Camera
    dist_a: np.ndarray       # lens distortion (undistort points before use)
    dist_b: np.ndarray
    rms_px: float            # stereo reprojection error
    baseline_ft: float       # distance between the two cameras

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        d = {k: getattr(self, k).tolist() if isinstance(getattr(self, k), np.ndarray)
             else getattr(self, k)
             for k in ("dist_a", "dist_b", "rms_px", "baseline_ft")}
        for name, cam in (("a", self.cam_a), ("b", self.cam_b)):
            d[f"K_{name}"] = cam.K.tolist()
            d[f"R_{name}"] = cam.R.tolist()
            d[f"t_{name}"] = cam.t.tolist()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)

    @staticmethod
    def load(path: str) -> "StereoRig":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        cams = {n: Camera(K=np.array(d[f"K_{n}"]), R=np.array(d[f"R_{n}"]),
                          t=np.array(d[f"t_{n}"])) for n in ("a", "b")}
        return StereoRig(cam_a=cams["a"], cam_b=cams["b"],
                         dist_a=np.array(d["dist_a"]),
                         dist_b=np.array(d["dist_b"]),
                         rms_px=d["rms_px"], baseline_ft=d["baseline_ft"])

    def undistort(self, pts, which: str) -> np.ndarray:
        """Pixel points -> undistorted pixel points for cam 'a' or 'b' (feed
        THESE to threed.triangulate, whose pinhole model has no distortion)."""
        K = (self.cam_a if which == "a" else self.cam_b).K
        dist = self.dist_a if which == "a" else self.dist_b
        p = np.asarray(pts, np.float32).reshape(-1, 1, 2)
        out = cv2.undistortPoints(p, K, dist, P=K)
        return out.reshape(-1, 2)


def calibrate_stereo(views_a: dict, views_b: dict, image_size_a, image_size_b,
                     pattern=PATTERN, square_ft: float = SQUARE_FT) -> StereoRig:
    """Full rig from the two cameras' board views (dicts frame_idx -> corners;
    the SAME board pose must have the same key in both -- use synced frame
    indices). Cam A becomes the world origin."""
    common = sorted(set(views_a) & set(views_b))
    if len(common) < 4:
        raise ValueError(f"need >=4 board poses seen by BOTH cameras, got {len(common)}")
    K_a, dist_a, _ = calibrate_intrinsics([views_a[k] for k in common],
                                          image_size_a, pattern, square_ft)
    K_b, dist_b, _ = calibrate_intrinsics([views_b[k] for k in common],
                                          image_size_b, pattern, square_ft)
    obj = [board_points(pattern, square_ft)] * len(common)
    img_a = [np.asarray(views_a[k], np.float32).reshape(-1, 1, 2) for k in common]
    img_b = [np.asarray(views_b[k], np.float32).reshape(-1, 1, 2) for k in common]
    rms, K_a, dist_a, K_b, dist_b, R, T, _, _ = cv2.stereoCalibrate(
        obj, img_a, img_b, K_a, dist_a, K_b, dist_b, tuple(image_size_a),
        flags=cv2.CALIB_FIX_INTRINSIC)
    cam_a = Camera(K=K_a, R=np.eye(3), t=np.zeros(3))
    cam_b = Camera(K=K_b, R=R, t=T.ravel())
    return StereoRig(cam_a=cam_a, cam_b=cam_b, dist_a=dist_a, dist_b=dist_b,
                     rms_px=float(rms), baseline_ft=float(np.linalg.norm(T)))
