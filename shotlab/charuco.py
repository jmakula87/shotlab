"""ChArUco stereo calibration -> metric `threed.Camera` pair.

Same job as shotlab.stereo, but for a ChArUco board (coarse checkerboard with an
ArUco code in every white square, from tools/make_charuco.py). Two things a plain
checkerboard can't do, both of which the WIDE / far camera needs:

  * every corner is self-identified, so a PARTIAL or steeply-angled view still
    contributes its visible corners (a plain board must be seen whole);
  * corners carry IDs, so the two cameras are matched corner-by-ID rather than
    by assuming both saw the identical full grid.

Board geometry is built in FEET (square_ft = square_in / 12), so the recovered
baseline T is in real feet -- what makes flare offsets and release spread real.
Conventions match shotlab.threed / OpenCV: x_cam = R @ X + t, camera looks +Z;
camera A is the world origin. Lengths in FEET.
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np

from .threed import Camera
from .stereo import StereoRig     # reuse the same save/load/undistort container


def load_spec(path: str = os.path.join("data", "calibration", "charuco_spec.json")):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def make_board(spec: dict):
    """Build the detector board in FEET (so triangulated lengths are feet)."""
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, spec.get("dict", "DICT_4X4_50")))
    square_ft = spec["square_in"] / 12.0
    marker_ft = square_ft * spec["marker_ratio"]
    board = cv2.aruco.CharucoBoard(
        (spec["squares_x"], spec["squares_y"]), square_ft, marker_ft, dictionary)
    return board


def detect(frame_bgr, detector):
    """(corners (N,2) float32, ids (N,) int) for one frame, or (None, None).
    Needs >=4 corners to be geometrically useful downstream."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    cc, ci, _, _ = detector.detectBoard(gray)
    if ci is None or len(ci) < 4:
        return None, None
    return cc.reshape(-1, 2).astype(np.float32), ci.reshape(-1).astype(int)


def corners_from_video(path, board, stride: int = 15, max_views: int = 30):
    """Sample a clip -> {frame_idx: (corners, ids)}."""
    from .video_io import iter_frames
    det = cv2.aruco.CharucoDetector(board)
    views = {}
    for idx, frame in iter_frames(path, start=0, stop=None):
        if idx % stride:
            continue
        cc, ci = detect(frame, det)
        if cc is not None:
            views[idx] = (cc, ci)
            if len(views) >= max_views:
                break
    return views


def paired_corners_from_videos(path_a, path_b, offset_s: float,
                               fps_a: float, fps_b: float, board,
                               stride: int = 15, max_views: int = 30):
    """Board views seen by BOTH cameras of one synced take (offset_s per
    shotlab.sync: B's clock runs behind A's by offset_s). Returns
    (views_a, views_b) sharing keys."""
    from .video_io import iter_frames
    det = cv2.aruco.CharucoDetector(board)
    cap_b = cv2.VideoCapture(path_b)
    views_a, views_b = {}, {}
    try:
        for idx, frame in iter_frames(path_a, start=0, stop=None):
            if idx % stride:
                continue
            cca, cia = detect(frame, det)
            if cca is None:
                continue
            idx_b = int(round((idx / fps_a - offset_s) * fps_b))
            if idx_b < 0:
                continue
            cap_b.set(cv2.CAP_PROP_POS_FRAMES, idx_b)
            ok, frame_b = cap_b.read()
            if not ok:
                continue
            ccb, cib = detect(frame_b, det)
            if ccb is None:
                continue
            views_a[idx] = (cca, cia)
            views_b[idx] = (ccb, cib)
            if len(views_a) >= max_views:
                break
    finally:
        cap_b.release()
    return views_a, views_b


def _obj_for_ids(board, ids: np.ndarray) -> np.ndarray:
    """3D board-plane positions (feet) for the given interior-corner IDs."""
    all_corners = board.getChessboardCorners()      # (Ncorners, 3), ID order
    return all_corners[ids].astype(np.float32)


def calibrate_intrinsics(views, image_size, board):
    """One camera's K + distortion from its ChArUco views (variable corner
    subsets are fine). views: iterable of (corners, ids). Returns (K, dist, rms)."""
    views = [v for v in views if v is not None and len(v[1]) >= 6]
    if len(views) < 4:
        raise ValueError(f"need >=4 usable ChArUco views (>=6 corners each), got {len(views)}")
    obj = [_obj_for_ids(board, ids).reshape(-1, 1, 3) for _, ids in views]
    img = [cc.reshape(-1, 1, 2).astype(np.float32) for cc, _ in views]
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, tuple(image_size), None,
                                             None, flags=cv2.CALIB_FIX_K3)
    return K, dist, float(rms)


def calibrate_stereo(views_a: dict, views_b: dict, image_size_a, image_size_b,
                     board) -> StereoRig:
    """Full rig from the two cameras' ChArUco views (dicts frame_idx ->
    (corners, ids); the SAME board pose must share a key -- use synced indices).
    For each shared pose only the corners seen by BOTH cameras (ID intersection)
    are used. Cam A becomes the world origin."""
    common = sorted(set(views_a) & set(views_b))
    if len(common) < 4:
        raise ValueError(f"need >=4 board poses seen by BOTH cameras, got {len(common)}")

    obj_list, img_a, img_b = [], [], []
    for k in common:
        cca, cia = views_a[k]
        ccb, cib = views_b[k]
        shared = np.intersect1d(cia, cib)
        if len(shared) < 4:
            continue                    # too little overlap this pose to constrain
        ia = {int(i): p for i, p in zip(cia, cca)}
        ib = {int(i): p for i, p in zip(cib, ccb)}
        pa = np.array([ia[int(s)] for s in shared], np.float32)
        pb = np.array([ib[int(s)] for s in shared], np.float32)
        obj_list.append(_obj_for_ids(board, shared).reshape(-1, 1, 3))
        img_a.append(pa.reshape(-1, 1, 2))
        img_b.append(pb.reshape(-1, 1, 2))
    if len(obj_list) < 4:
        raise ValueError(f"need >=4 poses with >=4 shared corners, got {len(obj_list)}")

    # per-camera intrinsics from their own (fuller) views, then fix them and
    # solve only the relative pose from the shared corners
    K_a, dist_a, _ = calibrate_intrinsics(views_a.values(), image_size_a, board)
    K_b, dist_b, _ = calibrate_intrinsics(views_b.values(), image_size_b, board)
    rms, K_a, dist_a, K_b, dist_b, R, T, _, _ = cv2.stereoCalibrate(
        obj_list, img_a, img_b, K_a, dist_a, K_b, dist_b, tuple(image_size_a),
        flags=cv2.CALIB_FIX_INTRINSIC)
    cam_a = Camera(K=K_a, R=np.eye(3), t=np.zeros(3))
    cam_b = Camera(K=K_b, R=R, t=T.ravel())
    return StereoRig(cam_a=cam_a, cam_b=cam_b, dist_a=dist_a, dist_b=dist_b,
                     rms_px=float(rms), baseline_ft=float(np.linalg.norm(T)))
