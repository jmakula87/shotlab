"""Multi-hypothesis beam tracker over the candidate CLOUD (experimental).

The greedy `assemble_track` commits to one candidate per frame; when the detector's
pick flips between the ball and a stationary distractor, the ball's arc fragments and
the rim segmenter drops the shot (measured 2026-07-23: ~7/17 clip-1 misses are exactly
this -- a clean arc exists in the conf-0.01 cloud but greedy can't follow it). Feeding
the cloud to the greedy tracker made recall WORSE (it grabs the nearest noise).

This keeps a BEAM of live track hypotheses, each with a constant-velocity motion model,
and extends them through the cloud by minimizing acceleration (motion residual) rather
than committing frame-by-frame. A momentary distractor no longer derails a track: the
ball-following hypothesis stays in the beam and wins on total smoothness. Output is a
LIST of coherent track SEGMENTS (one dict per segment), each fed to detect_shots_to_rim.

Not wired into production; measured against the hand-counted eval first.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .detect import BallCandidate


@dataclass
class _Hypo:
    last_c: BallCandidate
    last_f: int
    vx: float | None
    vy: float | None
    cost: float
    miss: int
    path: dict = field(default_factory=dict)

    def predict(self, f: int):
        if self.vx is None:
            return self.last_c.cx, self.last_c.cy
        dt = f - self.last_f
        return self.last_c.cx + self.vx * dt, self.last_c.cy + self.vy * dt


def beam_tracks(cands_by_frame: dict[int, list[BallCandidate]], *,
                motion_gate: float = 90.0, max_coast: int = 4, beam: int = 16,
                conf_floor: float = 0.05, seed_conf: float = 0.30,
                conf_bonus: float = 25.0, coast_penalty: float = 40.0,
                size_penalty: float = 3.0, min_len: int = 6) -> list[dict]:
    """Return a list of track segments {frame: BallCandidate}, each a smooth path."""
    frames = sorted(cands_by_frame)
    active: list[_Hypo] = []
    done: list[dict] = []

    def finalize(h: _Hypo):
        if len(h.path) >= min_len:
            done.append(dict(h.path))

    for f in frames:
        cands = [c for c in cands_by_frame.get(f, []) if c.conf >= conf_floor]
        if not cands:
            continue
        next_active: list[_Hypo] = []
        claimed: set[int] = set()          # candidate ids extended this frame
        # extend each live hypothesis to its single best candidate (or coast)
        for h in active:
            px, py = h.predict(f)
            best, bestcost, besti = None, motion_gate, -1
            for i, c in enumerate(cands):
                d = np.hypot(c.cx - px, c.cy - py)
                sc = d - conf_bonus * c.conf + size_penalty * abs(c.r - h.last_c.r)
                if sc < bestcost:
                    bestcost, best, besti = sc, c, i
            if best is not None:
                claimed.add(besti)
                dt = f - h.last_f
                nvx = (best.cx - h.last_c.cx) / dt
                nvy = (best.cy - h.last_c.cy) / dt
                if h.vx is not None:                 # blend for stability
                    nvx = 0.5 * nvx + 0.5 * h.vx
                    nvy = 0.5 * nvy + 0.5 * h.vy
                np_ = dict(h.path); np_[f] = best
                next_active.append(_Hypo(best, f, nvx, nvy,
                                         h.cost + max(0.0, bestcost), 0, np_))
            elif h.miss + 1 <= max_coast:            # coast ONLY if no extension
                next_active.append(_Hypo(h.last_c, h.last_f, h.vx, h.vy,
                                         h.cost + coast_penalty, h.miss + 1,
                                         dict(h.path)))
            else:
                finalize(h)                          # coasted too long -> emit
        # seed new hypotheses from confident candidates not already claimed
        for i, c in enumerate(cands):
            if c.conf >= seed_conf and i not in claimed:
                next_active.append(_Hypo(c, f, None, None, 0.0, 0, {f: c}))
        # prune: lowest cost, but strongly favor longer established tracks
        next_active.sort(key=lambda h: h.cost - 8.0 * len(h.path))
        active = next_active[:beam]

    for h in active:
        finalize(h)

    # de-duplicate overlapping segments: greedily keep the longest, drop segments
    # that mostly overlap an already-kept one (the beam emits near-duplicates).
    done.sort(key=len, reverse=True)
    kept: list[dict] = []
    claimed: set[tuple[int, float]] = set()
    for seg in done:
        keys = {(f, round(c.cx)) for f, c in seg.items()}
        overlap = len(keys & claimed) / max(1, len(keys))
        if overlap < 0.5:
            kept.append(seg)
            claimed |= keys
    return kept
