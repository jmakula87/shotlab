"""Phase 3 orchestration: per-shot backspin estimation, gated on fps."""

from __future__ import annotations

from ..video_io import probe, iter_frames
from .spin import estimate_spin, SpinResult


def run_phase3(video_path: str, shots, ball_track, *, min_fps: float = 110.0
               ) -> dict:
    """Return {shot_index: SpinResult}. If the clip isn't slow-mo, every shot
    gets a 'skipped' result with an explanation (no guessing)."""
    info = probe(video_path)
    results: dict[int, SpinResult] = {}

    if info.fps < min_fps or not shots:
        for s in (shots or []):
            results[s.index] = SpinResult(
                status="skipped", backspin_rpm=None, confidence="na",
                fps=info.fps, n_pairs=0,
                note=(f"footage is {info.fps:.0f}fps; spin needs "
                      f">={min_fps:.0f}fps slow-mo. Re-film at 120-240fps."))
        return results

    for s in shots:
        lo, hi = int(s.frames[0]), int(s.frames[-1])
        frames_iter = iter_frames(video_path, start=lo, stop=hi + 1)
        results[s.index] = estimate_spin(frames_iter, ball_track, s, info.fps,
                                         min_fps=min_fps)
    return results
