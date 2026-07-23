"""Phase 1 end-to-end: video -> ball track -> shots -> per-shot arc metrics."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from ..arc import estimate_px_per_foot_from_ball
from ..video_io import probe, iter_frames
from .detect import BaseDetector, ColorBallDetector, BallCandidate
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
               start_frame: int = 0,
               use_beam: bool = False,
               beam_greedy_conf: float = 0.25) -> Phase1Result:
    """Detect + track the ball, then segment shots.

    If `calib` is given, shots are found by rim-anchored detection on the
    continuous track (the right method for a high-coverage detector like the
    fine-tuned YOLO). Otherwise we fall back to gap-based segmentation.

    `stride` detects every Nth frame -- useful on 120/240fps slow-mo where every
    frame is overkill and CPU inference is the bottleneck.

    `start_frame`/`max_frames` bound the decode to a frame WINDOW [start, max);
    frame indices in the returned track/shots stay ABSOLUTE, so windows from one
    clip can be merged (used by the long-clip auto-chunker).

    `use_beam` (calibrated clips only): additionally run the multi-hypothesis beam
    tracker over the low-confidence candidate CLOUD and UNION its shots with the
    greedy ones. Validated 2026-07-23 across 3 hand-counted clips: recall 55%->80%
    at precision 0.96 (the greedy tracker fragments arcs on distractors; the beam
    recovers them). The CALLER must pass a low-conf detector (~0.01) so cands_by_frame
    is the cloud; the greedy pass uses the `beam_greedy_conf` (0.25) subset.
    """
    info = probe(video_path)
    detector = detector or ColorBallDetector()

    cands_by_frame = {}
    for idx, frame in iter_frames(video_path, start=start_frame, stop=max_frames):
        if stride > 1 and (idx % stride):
            continue
        cands_by_frame[idx] = detector.detect(idx, frame)

    greedy_cands = cands_by_frame
    if use_beam:                       # greedy uses the >=conf subset of the cloud
        greedy_cands = {f: [c for c in cs if c.conf >= beam_greedy_conf]
                        for f, cs in cands_by_frame.items()}
    track = assemble_track(greedy_cands)
    if calib is not None:
        from ..court import detect_shots_to_rim
        shots = detect_shots_to_rim(track, calib)
        rim_x = calib.rim_x
        if use_beam:
            shots, track = _union_beam(shots, track, cands_by_frame, calib)
    else:
        shots = segment_shots(track)
    metrics = [metrics_for_shot(s, rim_x=rim_x) for s in shots]
    return Phase1Result(info=info, track=track, shots=shots, metrics=metrics)


def _rim_frame(shot, calib) -> int:
    d = np.hypot(np.asarray(shot.xs) - calib.rim_x, np.asarray(shot.ys) - calib.rim_y)
    return int(np.asarray(shot.frames)[int(np.argmin(d))])


def _union_beam(greedy_shots, greedy_track, cloud, calib, tol: int = 25):
    """Union greedy shots with beam-tracker shots over the cloud. Greedy shots win
    on overlap (more stable); beam adds the fragmented-arc shots greedy dropped.
    `tol`=25f merges a shot's bounce-back re-approach (a miss produces two rim
    events ~20f apart) into one, while staying below the 31f minimum gap between
    distinct hand-counted attempts. Returns (unioned_shots, merged track)."""
    from ..court import detect_shots_to_rim
    from .track_beam import beam_tracks
    from .rim_recovery import recover_shots
    segs = beam_tracks(cloud, conf_floor=0.05, beam=24, max_coast=6)
    beam_shots = []
    for seg in segs:
        beam_shots.extend(detect_shots_to_rim(seg, calib))
    kept, rims = [], []
    for s in list(greedy_shots) + beam_shots:      # greedy first -> wins ties
        rf = _rim_frame(s, calib)
        if all(abs(rf - r) > tol for r in rims):
            kept.append(s); rims.append(rf)
    # rim-anchored recovery: near-rim cloud events with no shot yet, RANSAC-fit the
    # backward cloud window. Measured +8 shots / 111 at precision 0.99 across 3 clips
    # (the residual misses all HAD near-rim detections -- tracker-recoverable, 2026-07-23).
    for s in recover_shots(cloud, calib, rims):
        rf = _rim_frame(s, calib)
        if all(abs(rf - r) > tol for r in rims):
            kept.append(s); rims.append(rf)
    kept.sort(key=lambda s: _rim_frame(s, calib))
    for i, s in enumerate(kept, 1):
        s.index = i
    merged = dict(greedy_track)                    # greedy positions take priority
    for seg in segs:
        for f, c in seg.items():
            merged.setdefault(f, c)
    for s in kept:                                 # recovered shots' points for make/miss
        if s.meta.get("source") == "rim_recovery":
            for f, x, y, r in zip(s.frames, s.xs, s.ys, s.radii):
                merged.setdefault(int(f), BallCandidate(int(f), float(x), float(y), float(r), 0.5))
    return kept, merged
