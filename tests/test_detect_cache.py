"""The detection cache must round-trip the fit WITHOUT corrupting geometry.

2026-07-06 audit D1: deserialize rebuilt ArcFit with every point an inlier, so
release_angle/apex read off the walk-back outliers -- 17/114 shipped release
angles were wrong (one by 52deg). These tests pin the fix: a serialize ->
deserialize round-trip reproduces a fresh fit's release angle, and an old cache
(no stored inlier mask) self-heals by recomputing inliers from the coeffs.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.arc import fit_parabola_ransac
from shotlab.phase1_ball.detect import BallCandidate
from shotlab.phase1_ball.track import Shot
from shotlab.detect_cache import serialize_detection, deserialize_detection


def _shot_with_outlier():
    """A clean parabola of 20 points at frames 100.. plus a walk-back OUTLIER
    (an early, far-left stale detection) that a fresh RANSAC rejects."""
    fr = np.arange(100, 120)
    xs = np.linspace(200, 600, 20)                 # left -> right
    hs = -(0.002 * (xs - 400) ** 2) + 300          # a downward parabola in height
    ys = -hs
    # prepend a stale walk-back point far to the left and low (a real defect seen
    # on this footage): frame 99, x=150, y=514
    fr = np.concatenate([[99], fr])
    xs = np.concatenate([[150.0], xs])
    ys = np.concatenate([[514.0], ys])
    fit = fit_parabola_ransac(xs, ys, threshold_px=8.0)
    assert fit is not None
    track = {int(f): BallCandidate(int(f), float(x), float(y), 12.0, 0.9)
             for f, x, y in zip(fr, xs, ys)}
    shot = Shot(index=1, frames=fr, xs=xs, ys=ys, radii=np.full(len(fr), 12.0),
                fit=fit, meta={"rim_dist_px": 30.0})
    return track, shot, fit


def test_roundtrip_preserves_release_angle():
    track, shot, fit = _shot_with_outlier()
    data = serialize_detection(track, [shot], {"weights": "x"})
    _, shots2 = deserialize_detection(data)
    assert abs(shots2[0].fit.release_angle_deg() - fit.release_angle_deg()) < 0.5
    # the outlier must NOT be a fit inlier after the round-trip
    assert shots2[0].fit.inlier_mask.sum() == fit.inlier_mask.sum()
    assert bool(shots2[0].fit.inlier_mask[0]) is False       # the stale point


def test_old_cache_without_mask_self_heals():
    """A cache written before the mask existed still recovers the inliers from
    the (correct) stored coeffs, not from every point."""
    track, shot, fit = _shot_with_outlier()
    data = serialize_detection(track, [shot], {"weights": "x"})
    for sd in data["shots"]:                        # simulate an old cache
        sd.pop("inliers", None)
    _, shots2 = deserialize_detection(data)
    assert abs(shots2[0].fit.release_angle_deg() - fit.release_angle_deg()) < 1.0
    assert shots2[0].fit.inlier_mask.sum() < len(shot.frames)   # dropped the outlier


def test_weights_id_tracks_content():
    """A retrain re-exported to the SAME path (every export dir is literally
    named best_openvino_model) must change the weights identity -- path parts
    alone silently reuse the old model's detections (2026-07-15 audit)."""
    import tempfile
    import time
    from shotlab.detect_cache import _weights_id
    with tempfile.TemporaryDirectory() as td:
        wdir = os.path.join(td, "ball_orange", "weights", "best_openvino_model")
        os.makedirs(wdir)
        with open(os.path.join(wdir, "model.bin"), "wb") as f:
            f.write(b"x" * 1000)
        a = _weights_id(wdir)
        assert a.startswith("ball_orange/weights/best_openvino_model@"), a
        assert _weights_id(wdir) == a                 # stable while unchanged
        with open(os.path.join(wdir, "model.bin"), "wb") as f:
            f.write(b"y" * 2000)                      # re-export, same path
        assert _weights_id(wdir) != a
        # a plain .pt file works the same way
        pt = os.path.join(td, "best.pt")
        with open(pt, "wb") as f:
            f.write(b"z" * 10)
        b = _weights_id(pt)
        with open(pt, "wb") as f:
            f.write(b"z" * 20)
        assert _weights_id(pt) != b
    # a missing path (e.g. stock weights ultralytics resolves elsewhere) is a
    # stable, non-crashing key
    assert _weights_id("no_such_weights.pt").endswith("@absent")


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
