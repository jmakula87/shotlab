"""Where the camera footage lives.

The 2026-07-29 close-cam clips were shot with the phone mounted INVERTED and the
files also carry a bogus -90 display-rotation flag, so cv2, ffmpeg and browsers all
render them rotated. They were corrected losslessly into `data/raw/Camera 2/upright/`
(a stream copy that rewrites the rotation flag; no re-encode).

Pose and flare read the CLOSE camera, and MediaPipe on a sideways body produces
garbage rather than an obvious failure -- so every close-cam tool must prefer
`upright/` when it exists. Three tools each carried their own default and could
silently disagree; this is the one place that decides.
"""
from __future__ import annotations

import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def wide_cam_dir(root: str | None = None) -> str:
    """The wide (rim-facing) camera's clip directory."""
    return os.path.join(root or _ROOT, "data", "raw", "Camera 1")


def close_cam_dir(root: str | None = None) -> str:
    """The close (body) camera's clip directory -- `upright/` when it exists.

    Falls back to the raw directory so older sessions, which need no correction,
    keep working unchanged.
    """
    base = os.path.join(root or _ROOT, "data", "raw", "Camera 2")
    upright = os.path.join(base, "upright")
    return upright if os.path.isdir(upright) else base


def close_cam_is_corrected(path: str) -> bool:
    """True when `path` is the rotation-corrected close-cam directory."""
    return os.path.basename(os.path.normpath(path)) == "upright"
