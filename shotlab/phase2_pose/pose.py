"""Phase 2 pose extraction.

MediaPipe BlazePose (33 keypoints) is the default per the 2026 survey: Apache-2.0,
real-time on CPU (this machine has no GPU), covers every joint we need plus feet.

Per the survey, temporal smoothing is MANDATORY -- per-frame jitter is worst on
exactly the fast/occluded frames we care about (the release). We run a One-Euro
filter on every landmark. The model's z (depth) is kept only as a qualitative
hint and never used for a reported metric.
"""

from __future__ import annotations

import math
import os
import urllib.request
from dataclasses import dataclass

import numpy as np

_MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}
_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "..", "models")


def ensure_pose_model(variant: str = "full") -> str:
    """Return a local path to the PoseLandmarker .task model, downloading it on
    first use (BlazePose lite/full/heavy)."""
    os.makedirs(_MODELS_DIR, exist_ok=True)
    path = os.path.join(_MODELS_DIR, f"pose_landmarker_{variant}.task")
    if not os.path.exists(path):
        urllib.request.urlretrieve(_MODEL_URLS[variant], path)
    return path

# BlazePose 33-landmark indices we use.
L = {
    "nose": 0,
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16,
    "l_index": 19, "r_index": 20,
    "l_hip": 23, "r_hip": 24,
    "l_knee": 25, "r_knee": 26,
    "l_ankle": 27, "r_ankle": 28,
}


@dataclass
class FramePose:
    frame_idx: int
    xy: np.ndarray        # (33, 2) pixel coords
    vis: np.ndarray       # (33,) visibility 0..1
    z: np.ndarray         # (33,) model depth (qualitative only)

    def pt(self, name: str) -> np.ndarray:
        return self.xy[L[name]]

    def v(self, name: str) -> float:
        return float(self.vis[L[name]])


class _OneEuro:
    """One-Euro filter (Casiez et al.) -- speed-adaptive smoothing. Tuned for
    pose landmarks: low lag on fast moves, strong smoothing when still."""

    def __init__(self, fps: float, min_cutoff=1.5, beta=0.05, d_cutoff=1.0):
        self.fps = max(fps, 1.0)
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev = None
        self._dx_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        dt = 1.0 / self.fps
        if self._x_prev is None:
            self._x_prev = x
            self._dx_prev = np.zeros_like(x)
            return x
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


class PoseExtractor:
    """Wraps MediaPipe PoseLandmarker (Tasks API, VIDEO mode) with One-Euro
    smoothing over a clip.

    MediaPipe 0.10.x ships only the Tasks API (legacy mp.solutions is gone), so
    we use PoseLandmarker. VIDEO mode gives tracking-based stabilization; we add
    One-Euro on top because the Tasks API dropped the old smoothing flag.
    """

    def __init__(self, fps: float, variant: str = "full",
                 min_det_conf: float = 0.5, min_track_conf: float = 0.5,
                 smooth: bool = True, model_path: str | None = None):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as e:  # pragma: no cover
            raise ImportError("mediapipe is required: pip install mediapipe") from e
        self._mp = mp
        path = model_path or ensure_pose_model(variant)
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf,
            output_segmentation_masks=False,
        )
        self._lm = vision.PoseLandmarker.create_from_options(options)
        self.fps = fps
        self.smooth = smooth
        self._filt_xy = _OneEuro(fps) if smooth else None

    def close(self):
        self._lm.close()

    def process_frame(self, frame_idx: int, frame_bgr: np.ndarray) -> FramePose | None:
        import cv2
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(round(frame_idx * 1000.0 / max(self.fps, 1.0)))
        res = self._lm.detect_for_video(mp_image, ts_ms)
        if not res.pose_landmarks:
            return None
        lms = res.pose_landmarks[0]
        xy = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=float)
        vis = np.array([lm.visibility for lm in lms], dtype=float)
        z = np.array([lm.z for lm in lms], dtype=float)
        if self.smooth:
            xy = self._filt_xy(xy)
        return FramePose(frame_idx, xy, vis, z)


# ---- geometry helpers ------------------------------------------------------

def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle at b for the a-b-c triple, in degrees."""
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba < 1e-6 or nbc < 1e-6:
        return float("nan")
    cosang = np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0)
    return float(math.degrees(math.acos(cosang)))


def side_keys(handedness: str) -> dict:
    """Return the landmark-name prefixes for the shooting/loading side."""
    s = "r" if handedness.lower().startswith("r") else "l"
    return {
        "shoulder": f"{s}_shoulder", "elbow": f"{s}_elbow", "wrist": f"{s}_wrist",
        "index": f"{s}_index", "hip": f"{s}_hip", "knee": f"{s}_knee",
        "ankle": f"{s}_ankle",
    }
