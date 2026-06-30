"""Phase 1 end-to-end: video -> ball track -> shots -> per-shot arc metrics."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from ..arc import estimate_px_per_foot_from_ball
from ..video_io import probe, iter_frames
from .detect import BaseDetector, ColorBallDetector
from .track import assemble_track, segment_shots, Shot


@dataclass
class ShotArcMetrics:
    shot: int
    first_frame: int
    last_frame: int
    n_points: int
    release_angle_deg: float
    entry_angle_deg: float
    apex_height_px: float
    apex_height_ft: float | None
    fit_rmse_px: float
    px_per_foot: float | None
    direction: str

    def as_row(self) -> dict:
        return asdict(self)


def _ball_diameter_px(shot: Shot) -> float:
    # median diameter from the inlier radii = robust ruler
    return float(np.median(shot.radii) * 2.0)


def metrics_for_shot(shot: Shot, rim_x: float | None = None) -> ShotArcMetrics:
    fit = shot.fit
    diam = _ball_diameter_px(shot)
    ppf = estimate_px_per_foot_from_ball(diam)
    apex_px = fit.apex_height_px
    apex_ft = float(apex_px / ppf) if ppf and ppf == ppf else None
    return ShotArcMetrics(
        shot=shot.index,
        first_frame=int(shot.frames[0]),
        last_frame=int(shot.frames[-1]),
        n_points=int(fit.n_used),
        release_angle_deg=round(fit.release_angle_deg(), 1),
        entry_angle_deg=round(fit.entry_angle_deg(rim_x), 1),
        apex_height_px=round(apex_px, 1),
        apex_height_ft=round(apex_ft, 2) if apex_ft is not None else None,
        fit_rmse_px=round(fit.rmse_px, 2),
        px_per_foot=round(ppf, 1) if ppf == ppf else None,
        direction="L->R" if fit.direction > 0 else "R->L",
    )


@dataclass
class Phase1Result:
    info: object
    track: dict
    shots: list
    metrics: list


def run_phase1(video_path: str,
               detector: BaseDetector | None = None,
               rim_x: float | None = None,
               max_frames: int | None = None,
               calib=None,
               stride: int = 1,
               start_frame: int = 0) -> Phase1Result:
    """Detect + track the ball, then segment shots.

    If `calib` is given, shots are found by rim-anchored detection on the
    continuous track (the right method for a high-coverage detector like the
    fine-tuned YOLO). Otherwise we fall back to gap-based segmentation.

    `stride` detects every Nth frame -- useful on 120/240fps slow-mo where every
    frame is overkill and CPU inference is the bottleneck.

    `start_frame`/`max_frames` bound the decode to a frame WINDOW [start, max);
    frame indices in the returned track/shots stay ABSOLUTE, so windows from one
    clip can be merged (used by the long-clip auto-chunker).
    """
    info = probe(video_path)
    detector = detector or ColorBallDetector()

    cands_by_frame = {}
    for idx, frame in iter_frames(video_path, start=start_frame, stop=max_frames):
        if stride > 1 and (idx % stride):
            continue
        cands_by_frame[idx] = detector.detect(idx, frame)

    track = assemble_track(cands_by_frame)
    if calib is not None:
        from ..court import detect_shots_to_rim
        shots = detect_shots_to_rim(track, calib)
        rim_x = calib.rim_x
    else:
        shots = segment_shots(track)
    metrics = [metrics_for_shot(s, rim_x=rim_x) for s in shots]
    return Phase1Result(info=info, track=track, shots=shots, metrics=metrics)
