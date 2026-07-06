"""Make/miss classification robustness (2026-07-06 audit D9): the verdict must
come from the ball's PASS through the rim in a short window, not from where the
tracker wandered ~1s later (a rebound / next possession), and the closest-rim
frame is exposed for audio timing (D8)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.court import Calibration
from shotlab.phase1_ball.detect import BallCandidate
from shotlab.make import classify_make


class FakeShot:
    def __init__(self, frames):
        self.frames = np.array(frames)


CAL = Calibration(session="t", image_w=800, image_h=600, rim_x=400, rim_y=200,
                  rim_radius_px=15.0, shot_gate_px=60.0)


def _track(seq):
    return {f: BallCandidate(f, x, y, 8.0, 0.9) for f, x, y in seq}


def test_make_despite_a_far_rebound_after():
    # ball drops straight down through the rim, THEN the tracker jumps to a
    # rebound far to the side. Old code trusted that last point -> called it a
    # miss; the windowed logic ignores the post-net jump and calls the make.
    track = _track([(0, 300, 100), (1, 350, 150), (2, 390, 190), (3, 400, 205),
                    (4, 400, 225), (5, 400, 245), (6, 700, 180), (7, 720, 170)])
    r = classify_make(FakeShot([0, 1, 2, 3]), track, CAL, fps=30)
    assert r.made is True, r
    assert r.rim_frame == 3, r.rim_frame          # closest approach (for audio, D8)


def test_miss_when_deflects_aside_in_window():
    track = _track([(0, 300, 100), (1, 350, 150), (2, 395, 195), (3, 410, 200),
                    (4, 450, 210), (5, 500, 215)])
    r = classify_make(FakeShot([0, 1, 2, 3]), track, CAL, fps=30)
    assert r.made is False, r


def test_na_when_never_reaches_rim():
    track = _track([(0, 100, 100), (1, 120, 120), (2, 140, 140), (3, 160, 160)])
    r = classify_make(FakeShot([0, 1, 2, 3]), track, CAL, fps=30)
    assert r.made is None and r.confidence == "na", r


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    p = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}"); p += 1
        except Exception:
            print(f"FAIL {fn.__name__}"); traceback.print_exc()
    print(f"\n{p}/{len(fns)} passed")
    sys.exit(0 if p == len(fns) else 1)
