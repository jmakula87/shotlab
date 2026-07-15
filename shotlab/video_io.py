"""Video frame access and metadata (fps, resolution, slow-mo detection)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterator

import numpy as np

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise ImportError("opencv-python is required: pip install opencv-python") from e


@dataclass
class VideoInfo:
    path: str
    fps: float                  # EFFECTIVE fps for physics/time (real capture fps)
    n_frames: int
    width: int
    height: int
    container_fps: float = 0.0   # playback fps in the file (e.g. 30 for slow-mo)

    @property
    def duration_s(self) -> float:
        # real-time duration uses the true capture fps
        return self.n_frames / self.fps if self.fps else 0.0

    @property
    def is_slowmo(self) -> bool:
        # True high-fps capture (>=110), even when stored as a 30fps-playback file.
        return self.fps >= 110.0


def _android_capture_fps(path: str) -> float | None:
    """Read the real capture fps from the Android slow-mo metadata tag. Pixel
    phones save 120/240fps slow-mo as a 30fps-PLAYBACK file but stamp the true
    rate in format tag `com.android.capture.fps`. Returns None if absent."""
    import shutil
    import subprocess
    if shutil.which("ffprobe") is None:
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format_tags=com.android.capture.fps", "-of",
             "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, check=True).stdout.strip()
        return float(out) if out else None
    except Exception:
        return None


def probe(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    container_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    # Prefer the true capture fps when the file is a slow-mo (stored-at-30) clip.
    capture_fps = _android_capture_fps(path)
    fps = capture_fps if (capture_fps and capture_fps > container_fps + 1) else container_fps
    return VideoInfo(path, fps, n, w, h, container_fps=container_fps)


def to_h264(src: str, dst: str | None = None) -> str:
    """Transcode to browser-playable H.264 via ffmpeg (OpenCV's mp4v often
    won't decode in a browser/Streamlit). Returns dst, or src unchanged if
    ffmpeg is unavailable."""
    import shutil
    import subprocess

    if dst is None:
        base, _ = os.path.splitext(src)
        dst = base + "_h264.mp4"
    if shutil.which("ffmpeg") is None:
        return src
    cmd = ["ffmpeg", "-y", "-i", src, "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", "-loglevel", "error", dst]
    try:
        subprocess.run(cmd, check=True)
        return dst
    except Exception:
        return src


def iter_frames(path: str, start: int = 0, stop: int | None = None
                ) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, BGR frame) lazily."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    idx = start
    try:
        while True:
            if stop is not None and idx >= stop:
                break
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()


def frame_times(path: str, start: int = 0, stop: int | None = None) -> dict[int, float]:
    """{frame_index: presentation_time_seconds} straight from the container's PTS.

    These phones record VARIABLE frame rate (a Pixel clip measured 30 fps then
    24 fps mid-clip), so `frame_index / nominal_fps` is WRONG -- physics that
    assumes constant fps (arc gravity fits, sync frame-maps) silently distorts
    time. This reads the true per-frame timestamp. Uses grab() only (no color
    decode), so it's cheap enough to run over a whole clip.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    idx = start
    out: dict[int, float] = {}
    try:
        while True:
            if stop is not None and idx >= stop:
                break
            if not cap.grab():
                break
            out[idx] = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            idx += 1
    finally:
        cap.release()
    return out


def video_id(path: str) -> str:
    """Content identity of a video file for cache signatures: size + mtime.
    One stat() -- catches an in-place re-trim/re-transcode/overwrite that a
    basename-keyed cache can't see. Missing file -> 'absent' (stable key)."""
    try:
        st = os.stat(path)
        return f"{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return "absent"


def frame_times_cached(path: str, out_dir: str = os.path.join("data", "out")
                       ) -> dict[int, float] | None:
    """`frame_times()` with a JSON cache next to the clip's other caches, keyed
    on the video's content identity (the PTS scan is one grab() pass -- minutes
    on a long clip -- and never changes for the same file). Returns None when
    the container yields no usable PTS (all-zero / non-increasing), so callers
    fall back to frame/fps; that verdict is cached too."""
    stem = os.path.splitext(os.path.basename(path))[0]
    cpath = os.path.join(out_dir, stem, f"{stem}_pts.json")
    vid = video_id(path)
    if os.path.exists(cpath):
        try:
            with open(cpath, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("video") == vid:
                ts = data["times"]
                return {i: float(t) for i, t in enumerate(ts)} if ts else None
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass                       # corrupt/old cache -> rescan
    times = frame_times(path)
    vals = [times[k] for k in sorted(times)]      # frames scan in order from 0
    usable = len(vals) >= 2 and vals[-1] > vals[0] > -1e-9 and max(vals) > 0
    os.makedirs(os.path.dirname(cpath), exist_ok=True)
    with open(cpath, "w", encoding="utf-8") as f:
        json.dump({"video": vid,
                   "times": [round(v, 4) for v in vals] if usable else []}, f)
    return times if usable else None


def is_variable_fps(path: str, tol: float = 0.15) -> tuple[bool, float]:
    """(is_vfr, measured_fps) for a clip. measured_fps = 1/median(frame dt) from
    real PTS; is_vfr True when >5% of gaps deviate from the median by >tol."""
    ts = np.array(sorted(frame_times(path).values()))
    if len(ts) < 3:
        return False, 0.0
    dt = np.diff(ts)
    dt = dt[(dt > 0) & (dt < 1.0)]
    if len(dt) < 2:
        return False, 0.0
    med = float(np.median(dt))
    frac_off = float(np.mean(np.abs(dt - med) > tol * med))
    return frac_off > 0.05, (1.0 / med if med else 0.0)
