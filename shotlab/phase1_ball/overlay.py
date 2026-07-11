"""Render the tracked trajectory + fitted parabola + metrics as a video overlay
so the user can eyeball the tracking quality."""

from __future__ import annotations

import cv2
import numpy as np

from ..video_io import probe, iter_frames
from .pipeline import Phase1Result

_GREEN = (60, 220, 60)
_YELLOW = (40, 220, 240)
_RED = (60, 60, 235)
_WHITE = (240, 240, 240)


# BlazePose skeleton connections (by the L index map in phase2_pose.pose).
_SKELETON = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),   # arms + shoulders
    (11, 23), (12, 24), (23, 24),                        # torso
    (23, 25), (25, 27), (24, 26), (26, 28),              # legs
]


def _draw_skeleton(frame, fp):
    for a, b in _SKELETON:
        if fp.vis[a] >= 0.3 and fp.vis[b] >= 0.3:
            pa = tuple(int(v) for v in fp.xy[a])
            pb = tuple(int(v) for v in fp.xy[b])
            cv2.line(frame, pa, pb, (230, 200, 60), 2, cv2.LINE_AA)
    for i in range(33):
        if fp.vis[i] >= 0.3:
            cv2.circle(frame, tuple(int(v) for v in fp.xy[i]), 3,
                       (250, 250, 250), -1, cv2.LINE_AA)


def render_shot_clip(video_path: str, shot, track, out_path: str,
                     fps: float, pad: int = 8, metrics_text=None,
                     post_s: float = 1.5, rim=None) -> str:
    """Render a short overlay clip for ONE shot: the ball trail + fitted arc +
    a metrics caption. Used by the per-shot review / make-miss audit.

    Two things that matter for judging make vs miss:
      * the window runs PAST the last ball DETECTION by `post_s` seconds -- the
        ball is usually lost right at the rim, so without this the clip ends
        before you can see it drop through or bounce out;
      * playback fps is the window's REAL rate from the container timestamps
        (these phones record variable frame rate; writing at a nominal 30 made
        the clips play sped-up).
    `rim` = (rim_x, rim_y, rim_radius_px) draws the rim so in/out is visible."""
    from ..video_io import frame_times
    info = probe(video_path)
    lo = max(0, int(shot.frames[0]) - pad)
    post = int(round(post_s * (info.fps or 30.0)))
    hi = min(info.n_frames - 1, int(shot.frames[-1]) + post)

    # real-time playback fps from the container PTS over this window
    ts = frame_times(video_path, lo, hi + 1)
    tv = sorted(ts.values())
    real_fps = ((len(tv) - 1) / (tv[-1] - tv[0])) if len(tv) > 2 and tv[-1] > tv[0] \
        else (info.fps or 30.0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, min(real_fps, 60), (info.width, info.height))

    xs = np.linspace(shot.xs.min(), shot.xs.max(), 60)
    curve = np.stack([xs, -shot.fit.height_at(xs)], 1).astype(np.int32)
    shot_frames = set(int(f) for f in shot.frames)
    last_det = int(shot.frames[-1])

    for idx, frame in iter_frames(video_path, start=lo, stop=hi + 1):
        if rim is not None:
            cv2.circle(frame, (int(rim[0]), int(rim[1])), max(int(rim[2]), 6),
                       _YELLOW, 2, cv2.LINE_AA)
        cv2.polylines(frame, [curve], False, _YELLOW, 1, cv2.LINE_AA)
        for f in range(lo, idx + 1):
            c = track.get(f)
            if c is not None and f in shot_frames:
                cv2.circle(frame, (int(c.cx), int(c.cy)), 3, _RED, -1, cv2.LINE_AA)
        c = track.get(idx)
        if c is not None:
            cv2.circle(frame, (int(c.cx), int(c.cy)), max(int(c.r), 5), _GREEN, 2,
                       cv2.LINE_AA)
        if idx > last_det:      # past tracking -> tell the viewer to watch the rim
            cv2.putText(frame, "ball lost - watch the yellow rim (in or out?)",
                        (20, info.height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        _WHITE, 2, cv2.LINE_AA)
        for i, ln in enumerate(metrics_text or []):
            cv2.putText(frame, ln, (20, 40 + i * 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, _GREEN if i == 0 else _WHITE, 2, cv2.LINE_AA)
        vw.write(frame)
    vw.release()
    return out_path


def render_overlay(video_path: str, result: Phase1Result, out_path: str,
                   trail_len: int = 24, poses: dict | None = None,
                   release_frames: dict | None = None) -> str:
    info = probe(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, info.fps, (info.width, info.height))
    poses = poses or {}
    release_set = set((release_frames or {}).values())

    track = result.track
    # map frame -> shot for labeling + which fitted curve to draw
    frame_to_shot = {}
    for s in result.shots:
        for f in s.frames:
            frame_to_shot[int(f)] = s

    # precompute fitted-parabola polylines per shot (in image space)
    shot_curves = {}
    for s in result.shots:
        xs = np.linspace(s.xs.min(), s.xs.max(), 60)
        hs = s.fit.height_at(xs)
        ys = -hs
        shot_curves[s.index] = np.stack([xs, ys], 1).astype(np.int32)

    met_by_shot = {m.shot: m for m in result.metrics}

    for idx, frame in iter_frames(video_path):
        # draw skeleton if we have a pose for this frame
        fp = poses.get(idx)
        if fp is not None:
            _draw_skeleton(frame, fp)
        if idx in release_set:
            cv2.putText(frame, "RELEASE", (frame.shape[1] - 220, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 220, 60), 2,
                        cv2.LINE_AA)

        # draw the fitted curve for the active shot
        s = frame_to_shot.get(idx)
        if s is not None:
            cv2.polylines(frame, [shot_curves[s.index]], False, _YELLOW, 2,
                          cv2.LINE_AA)
            m = met_by_shot[s.index]
            lines = [
                f"SHOT {m.shot}",
                f"release {m.release_angle_deg} deg",
                f"entry   {m.entry_angle_deg} deg",
                f"apex    {m.apex_height_ft if m.apex_height_ft is not None else m.apex_height_px}"
                + ("ft" if m.apex_height_ft is not None else "px"),
            ]
            for i, ln in enumerate(lines):
                cv2.putText(frame, ln, (20, 40 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            _GREEN if i == 0 else _WHITE, 2, cv2.LINE_AA)

        # draw the detected-ball trail (recent frames)
        for f in range(max(0, idx - trail_len), idx + 1):
            c = track.get(f)
            if c is None:
                continue
            age = idx - f
            fade = max(0.2, 1.0 - age / trail_len)
            col = tuple(int(ch * fade) for ch in _RED)
            cv2.circle(frame, (int(c.cx), int(c.cy)), 3, col, -1, cv2.LINE_AA)
        # current ball outline
        c = track.get(idx)
        if c is not None:
            cv2.circle(frame, (int(c.cx), int(c.cy)), int(c.r), _GREEN, 2,
                       cv2.LINE_AA)

        vw.write(frame)

    vw.release()
    return out_path
