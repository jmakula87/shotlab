"""Two-camera 3D core (footage-independent).

The single-camera metrics are foreshortened because one view can't resolve the
depth axis. With TWO calibrated cameras seeing the same joints, we triangulate
the true 3D position of each joint and the depth-dependent metrics become real.

This module is the math foundation, built and validated against synthetic
ground truth BEFORE any real two-camera footage exists -- exactly how the
synthetic clip validated Phase 1's arc math. When the second camera (+ a
calibration clip) lands, real 2D joints + real camera geometry feed straight in.

Pipeline once footage exists:
  1. sync the two clips (one ball-bounce/clap frame)
  2. stereo-calibrate (the measured-marker clip) -> each Camera's K, R, t
  3. per shot, triangulate the matched joints -> 3D
  4. elbow_flare() + release_point_spread() on the 3D

Conventions: world is right-handed; a Camera looks down its own +Z, image y is
down. Projection is the standard pinhole P = K[R|t], x_cam = R @ X + t.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


@dataclass
class Camera:
    """Pinhole camera. K = intrinsics (3x3); (R, t) = world->camera extrinsics
    so that x_cam = R @ X_world + t and the camera looks down +Z_cam."""
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray

    @property
    def P(self) -> np.ndarray:
        return self.K @ np.hstack([self.R, self.t.reshape(3, 1)])

    def project(self, X) -> np.ndarray:
        """World point(s) -> pixel(s). Accepts (3,) or (N,3); returns (N,2)."""
        X = np.atleast_2d(np.asarray(X, float))
        Xc = (self.R @ X.T + self.t.reshape(3, 1)).T          # (N,3) camera frame
        proj = (self.K @ Xc.T).T
        return proj[:, :2] / proj[:, 2:3]

    @staticmethod
    def intrinsics(f, cx, cy) -> np.ndarray:
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], float)

    @staticmethod
    def look_at(position, target, up, K) -> "Camera":
        """Build a Camera at `position` aimed at `target`. `up` is a world-up
        hint (need not be exactly perpendicular -- it's re-orthogonalized)."""
        position = np.asarray(position, float)
        fwd = _unit(np.asarray(target, float) - position)     # camera +Z
        right = _unit(np.cross(np.asarray(up, float), fwd))    # camera +X
        down = np.cross(fwd, right)                            # camera +Y (img y down)
        R = np.vstack([right, down, fwd])                      # rows: world->cam
        t = -R @ position
        return Camera(K=np.asarray(K, float), R=R, t=t)


def triangulate(P1: np.ndarray, P2: np.ndarray, pts1, pts2) -> np.ndarray:
    """Linear (DLT) triangulation. Given two 3x4 projection matrices and matched
    pixels, recover the 3D point(s). Returns (N,3)."""
    pts1 = np.atleast_2d(np.asarray(pts1, float))
    pts2 = np.atleast_2d(np.asarray(pts2, float))
    out = []
    for (u1, v1), (u2, v2) in zip(pts1, pts2):
        A = np.array([
            u1 * P1[2] - P1[0],
            v1 * P1[2] - P1[1],
            u2 * P2[2] - P2[0],
            v2 * P2[2] - P2[1],
        ])
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        out.append(X[:3] / X[3])
    return np.array(out)


def triangulate_joints(cam1: Camera, cam2: Camera, joints1: dict,
                       joints2: dict) -> dict:
    """Triangulate every joint present (by name) in BOTH cameras' 2D dicts ->
    name -> 3D point."""
    names = [n for n in joints1 if n in joints2]
    if not names:
        return {}
    pts1 = np.array([joints1[n] for n in names], float)
    pts2 = np.array([joints2[n] for n in names], float)
    X = triangulate(cam1.P, cam2.P, pts1, pts2)
    return {n: X[i] for i, n in enumerate(names)}


@dataclass
class FlareResult:
    angle_deg: float        # upper-arm deviation out of the shoulder->rim vertical plane
    offset: float           # signed lateral elbow offset (same units as the 3D)
    note: str = ""


def shoulder_frame(shoulder, rim, up=(0, 1, 0)):
    """Right-handed frame at the shooting shoulder: forward (toward the rim), up
    (world vertical), side (their cross = the horizontal left/right axis). A
    perfectly tucked elbow lies in the forward-up plane; flare is motion along
    `side`."""
    shoulder = np.asarray(shoulder, float)
    fwd = _unit(np.asarray(rim, float) - shoulder)
    up = _unit(up)
    side = _unit(np.cross(up, fwd))           # horizontal, normal to the vertical plane
    return fwd, up, side


def elbow_flare(shoulder, elbow, rim, up=(0, 1, 0)) -> FlareResult:
    """True elbow flare from 3D joints: how far the upper arm (shoulder->elbow)
    swings out of the vertical plane through the shoulder pointing at the rim.

    ~0 deg = tucked (elbow under the ball); positive/negative = winged out to a
    side. The sign is camera-setup-dependent and gets pinned on real footage (an
    analog of LEFT_RIGHT_FLIP); the MAGNITUDE is the coaching signal. `offset` is
    the signed lateral elbow distance in whatever units the 3D is in (real inches
    once the stereo rig is metrically calibrated)."""
    shoulder = np.asarray(shoulder, float)
    elbow = np.asarray(elbow, float)
    _, _, side = shoulder_frame(shoulder, rim, up)
    arm = elbow - shoulder
    arm_u = _unit(arm)
    angle = float(np.degrees(np.arcsin(np.clip(np.dot(arm_u, side), -1.0, 1.0))))
    offset = float(np.dot(arm, side))
    return FlareResult(angle_deg=round(angle, 2), offset=round(offset, 4))


@dataclass
class ReleaseSpread:
    n: int
    rms_spread: float           # RMS distance of release points from their centroid
    std_lateral: float          # spread along each shoulder-frame axis
    std_vertical: float
    std_depth: float
    centroid: tuple

    def as_row(self) -> dict:
        return {"n": self.n, "rms_spread": round(self.rms_spread, 4),
                "std_lateral": round(self.std_lateral, 4),
                "std_vertical": round(self.std_vertical, 4),
                "std_depth": round(self.std_depth, 4)}


def release_point_spread(points_3d, *, frame=None) -> ReleaseSpread | None:
    """Consistency of the 3D release point across shots.

    `points_3d`: list of the ball's 3D position at release, ideally expressed
    RELATIVE to the shooting shoulder so it's comparable regardless of where you
    stood. `rms_spread` is the headline (tight cluster = repeatable release).
    Pass `frame` = (fwd, up, side) unit vectors to break the spread into
    lateral/vertical/depth; otherwise world axes are used."""
    P = np.asarray(points_3d, float)
    if P.ndim != 2 or len(P) < 2:
        return None
    centroid = P.mean(axis=0)
    dev = P - centroid
    rms = float(np.sqrt(np.mean(np.sum(dev ** 2, axis=1))))
    if frame is not None:
        fwd, up, side = (_unit(f) for f in frame)
        comp_side = dev @ side
        comp_up = dev @ up
        comp_fwd = dev @ fwd
    else:
        comp_side, comp_up, comp_fwd = dev[:, 0], dev[:, 1], dev[:, 2]
    return ReleaseSpread(
        n=len(P), rms_spread=rms,
        std_lateral=float(np.std(comp_side)),
        std_vertical=float(np.std(comp_up)),
        std_depth=float(np.std(comp_fwd)),
        centroid=tuple(round(float(c), 4) for c in centroid))
