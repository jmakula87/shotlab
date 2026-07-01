"""Phase 2 orchestration: extract pose over the frames that matter for each
shot, then compute form metrics synced to the ball track."""

from __future__ import annotations

from dataclasses import dataclass

from ..video_io import probe, iter_frames
from .pose import PoseExtractor, FramePose
from .form import compute_form, ShotForm


@dataclass
class Phase2Result:
    poses: dict            # frame_idx -> FramePose
    forms: list            # list[ShotForm]
    release_frames: dict   # shot index -> release frame


def _needed_frames(shots, n_frames, pre=25, post=8) -> set[int]:
    need = set()
    for s in shots:
        lo = max(0, int(s.frames[0]) - pre)
        hi = min(n_frames - 1, int(s.frames[-1]) + post)
        need.update(range(lo, hi + 1))
    return need


def run_phase2(video_path: str, shots, ball_track, *,
               handedness="right", camera_angle="side_on",
               variant="full", smooth=True, rim_xy=None,
               px_per_foot=None) -> Phase2Result:
    info = probe(video_path)
    if not shots:
        return Phase2Result(poses={}, forms=[], release_frames={})

    need = _needed_frames(shots, info.n_frames)
    stop = (max(need) + 1) if need else 0          # don't decode past the last shot
    extractor = PoseExtractor(fps=info.fps, variant=variant, smooth=smooth)
    poses: dict[int, FramePose] = {}
    try:
        for idx, frame in iter_frames(video_path, stop=stop):
            if idx not in need:
                continue
            fp = extractor.process_frame(idx, frame)
            if fp is not None:
                poses[idx] = fp
    finally:
        extractor.close()

    forms, releases = [], {}
    for s in shots:
        sf = compute_form(s, ball_track, poses, info.fps,
                          handedness=handedness, camera_angle=camera_angle,
                          rim_xy=rim_xy, px_per_foot=px_per_foot)
        forms.append(sf)
        releases[s.index] = sf.release_frame
    return Phase2Result(poses=poses, forms=forms, release_frames=releases)
