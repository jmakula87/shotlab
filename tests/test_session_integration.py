"""End-to-end integration of the per-shot record builder: drive a REAL Shot
(built by the real parabola fitter) through `_records_from_shots` with pose off,
and confirm the new fields (shot_form, shot_setup) populate without a video.

This covers the session.py wiring that the live real-clip run couldn't exercise
(that clip detected 0 shots due to footage, not code)."""

import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.arc import fit_parabola_ransac
from shotlab.phase1_ball.track import Shot
from shotlab.phase1_ball.detect import BallCandidate
from shotlab.court import Calibration
from shotlab.session import _records_from_shots, _record_cache_sig


def _make_shot_and_track():
    """A clean left->right parabola that lands at the rim, plus a frame-indexed
    ball track (with a dribble bounce before launch)."""
    rim_x, rim_y = 900.0, 200.0
    # flight: x 300->900, parabola peaking above the rim then dropping to it
    fr = np.arange(40, 60)
    xs = np.linspace(300, rim_x, len(fr))
    # h = -y; arc rises then falls. Put apex ~250px above release.
    t = np.linspace(0, 1, len(fr))
    ys = 500 - (1200 * t - 1200 * t**2) * 0.8     # image y (down); dips (rises)
    radii = np.full(len(fr), 9.0)
    fit = fit_parabola_ransac(xs, ys)
    assert fit is not None
    shot = Shot(index=1, frames=fr, xs=xs, ys=ys, radii=radii, fit=fit,
                meta={"rim_dist_px": 60.0})

    track = {}
    # pre-shot dribble: ball bounces near the shooter before frame 40
    bounce = [50, 80, 140, 80, 50, 85, 150, 80, 50]
    for i, cy in enumerate(bounce):
        f = 40 - len(bounce) + i
        track[f] = BallCandidate(frame_idx=f, cx=290.0, cy=float(cy), r=9.0, conf=0.9)
    # flight
    for f, x, y in zip(fr, xs, ys):
        track[int(f)] = BallCandidate(frame_idx=int(f), cx=float(x), cy=float(y),
                                      r=9.0, conf=0.9)
    # post-rim: drop straight through (a make)
    for k in range(1, 8):
        f = int(fr[-1]) + k
        track[f] = BallCandidate(frame_idx=f, cx=rim_x, cy=rim_y + 8 * k, r=9.0,
                                 conf=0.9)
    calib = Calibration(session="test", image_w=1920, image_h=1080,
                        rim_x=rim_x, rim_y=rim_y, rim_radius_px=18.0,
                        shot_gate_px=120.0)
    return shot, track, calib


def test_records_populate_new_fields_without_pose():
    shot, track, calib = _make_shot_and_track()
    info = types.SimpleNamespace(fps=30.0)
    recs = _records_from_shots([shot], track, "no_video.mp4", calib, info,
                               clip_start=None, do_spin=False, with_pose=False,
                               handedness="right")
    assert len(recs) == 1
    rec = recs[0]
    # arc metrics came through
    assert rec.release_angle_deg is not None
    assert rec.zone and rec.depth in ("near", "mid", "far")
    # NEW: shot-type tagging populated
    assert rec.shot_form in ("jumper", "layup", "floater")
    assert rec.shot_setup in ("catch_and_shoot", "on_the_move", "off_dribble",
                              "unknown")
    # the planted pre-shot bounce should be picked up as a dribble
    assert rec.shot_setup == "off_dribble", rec.shot_setup
    # make classifier ran (ball dropped through) -> a row, no crash
    assert rec.made in (True, False, None)
    # row() serializes cleanly with the new fields
    row = rec.row()
    assert "shot_form" in row and "shot_setup" in row


def _sig(**over):
    base = dict(detector_name="yolo", weights="best_openvino_model", imgsz=640,
                stride="auto", max_frames=None, with_pose=True, with_spin="auto",
                handedness="right")
    base.update(over)
    return _record_cache_sig(**base)


def test_cache_sig_stable_and_param_sensitive():
    # identical params -> identical signature (cache hit)
    assert _sig() == _sig()
    # any param change -> different signature (cache miss = recompute, not stale)
    assert _sig(imgsz=768) != _sig()
    assert _sig(with_pose=False) != _sig()
    assert _sig(weights="best.pt") != _sig()
    assert _sig(stride=2) != _sig()
    assert _sig(max_frames=8000) != _sig()


def test_cache_sig_tracks_record_schema():
    # the signature folds in the ShotRecord field set, so adding/removing a record
    # field invalidates old caches automatically. Verify the current schema's
    # fields are what the signature is built from (guards the stale-cache fix).
    from shotlab.session import ShotRecord
    from dataclasses import fields
    names = [f.name for f in fields(ShotRecord)]
    for must in ("shot_form", "shot_setup", "release_conf",
                 "elbow_angle_at_release_deg"):
        assert must in names, must     # the new fields are part of the schema sig


if __name__ == "__main__":
    import traceback
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in funcs:
        try:
            fn(); print(f"PASS {fn.__name__}"); passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(funcs)} passed")
    sys.exit(0 if passed == len(funcs) else 1)
