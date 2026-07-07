"""court.shot_quality -- the is_real phantom flag (D2) had ZERO coverage: the
auditor made it return (False,...) unconditionally and the whole suite still
passed. These pin real→True, phantom→False (2026-07-07 test audit)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.court import Calibration, shot_quality
from shotlab.arc import fit_parabola_ransac
from shotlab.phase1_ball.track import Shot

CAL = Calibration(session="t", image_w=800, image_h=600, rim_x=400, rim_y=200,
                  rim_radius_px=15.0, shot_gate_px=60.0)


def _shot(frames, xs, ys):
    fit = fit_parabola_ransac(np.array(xs, float), np.array(ys, float), threshold_px=8.0)
    assert fit is not None
    return Shot(index=1, frames=np.array(frames), xs=np.array(xs, float),
                ys=np.array(ys, float), radii=np.full(len(frames), 12.0),
                fit=fit, meta={})


# a real arc: launches ~200px below the rim, rises to the rim, downward parabola
_XS = [150, 190, 230, 270, 310, 350, 390, 410]
_YS = [400, 330, 275, 235, 210, 196, 193, 200]


def test_clean_arc_is_real():
    s = _shot([100 + i * 2 for i in range(8)], _XS, _YS)
    ok, why = shot_quality(s, CAL, 30.0)
    assert ok is True, why


def test_over_long_flight_flagged():
    s = _shot([100 + i * 60 for i in range(8)], _XS, _YS)   # ~14s span
    ok, why = shot_quality(s, CAL, 30.0)
    assert ok is False and "long" in why, why


def test_stationary_near_rim_flagged():
    xs = [380, 390, 400, 410, 420, 430, 440, 450]
    ys = [210, 208, 206, 205, 206, 208, 210, 212]           # never launched from below
    s = _shot([100 + i * 2 for i in range(8)], xs, ys)
    ok, why = shot_quality(s, CAL, 30.0)
    assert ok is False, why


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
