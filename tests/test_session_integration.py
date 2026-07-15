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


def test_real_time_uses_pts_and_falls_back():
    from shotlab.session import _real_time
    info = types.SimpleNamespace(fps=30.0, container_fps=30.0)
    # no PTS map -> the old frame/fps behavior, exactly
    assert _real_time(60, None, info) == 60 / 30.0
    # a VFR clip: PTS say frame 60 is at 2.5s, not the nominal 2.0s
    times = {60: 2.5}
    assert _real_time(60, times, info) == 2.5
    # frame missing from the map -> fallback, not a crash
    assert _real_time(61, times, info) == 61 / 30.0
    # slow-mo: PTS measure PLAYBACK time (stored 30fps); real capture time is
    # scaled by container/capture -- 4s of playback = 1s of real 120fps capture
    slomo = types.SimpleNamespace(fps=120.0, container_fps=30.0)
    assert abs(_real_time(120, {120: 4.0}, slomo) - 1.0) < 1e-9
    # info without container_fps (older callers/tests) -> scale of 1
    bare = types.SimpleNamespace(fps=30.0)
    assert _real_time(10, {10: 0.5}, bare) == 0.5


def test_abs_time_and_audio_window_use_pts():
    """The VFR fix end-to-end: abs_time comes from the PTS map, and the audio
    make/miss window is timed on the PTS rim moment (not frame/nominal-fps)."""
    import datetime as dt
    shot, track, calib = _make_shot_and_track()
    info = types.SimpleNamespace(fps=30.0, container_fps=30.0)
    # PTS: the clip actually ran at 20fps against a nominal 30 -- by the rim
    # frame the true time is ~1s later than frame/nominal-fps believes, which
    # is beyond the audio window's +0.7s reach (the real failure mode).
    times = {f: f / 20.0 for f in range(0, 80)}
    start = dt.datetime(2026, 7, 15, 18, 0, 0)
    recs = _records_from_shots([shot], track, "no_video.mp4", calib, info,
                               clip_start=start, do_spin=False, with_pose=False,
                               handedness="right", times=times)
    t_off = (dt.datetime.fromisoformat(recs[0].abs_time) - start).total_seconds()
    assert abs(t_off - 40 / 20.0) < 1e-6, t_off      # PTS time, not 40/30

    # audio window: a loud clang placed at the PTS rim time (not the nominal
    # time) must be found -> fused verdict flips the soft visual call to LOW
    # conf. With frame/fps timing the window would sit ~0.5s early at rim
    # frame ~59 -- put the clang beyond the window's +0.7s reach of that spot.
    sr = 8000
    n = sr * 6
    quiet = np.random.default_rng(0).normal(0, 0.01, n)
    rim_frame = 59                                   # last flight frame region
    t_pts = rim_frame / 20.0                         # ~2.95s; nominal ~1.97s
    clang = quiet.copy()
    lo = int(t_pts * sr)
    clang[lo:lo + sr // 10] += 0.9                   # sharp loud burst at PTS time
    recs_pts = _records_from_shots(
        [shot], track, "no_video.mp4", calib, info, clip_start=start,
        do_spin=False, with_pose=False, handedness="right",
        audio=(clang, sr), times=times)
    from shotlab.audio import audio_make_hint
    hint_at_pts = audio_make_hint(clang, sr, t_pts)
    hint_at_nominal = audio_make_hint(clang, sr, rim_frame / 30.0)
    # the burst is audible at the PTS time and NOT at the nominal time --
    # i.e. the timing choice changes the verdict, and we now use the PTS one
    assert hint_at_pts["made"] is False, hint_at_pts
    assert hint_at_nominal["made"] is not False, hint_at_nominal
    assert recs_pts[0].made is not None


def _seam_shot(f0, f1):
    """A minimal Shot spanning frames [f0, f1] (fit geometry irrelevant to the
    seam-dedup logic under test)."""
    fr = np.arange(f0, f1 + 1)
    xs = np.linspace(300, 900, len(fr))
    t = np.linspace(0, 1, len(fr))
    ys = 500 - (1200 * t - 1200 * t**2) * 0.8
    fit = fit_parabola_ransac(xs, ys)
    return Shot(index=1, frames=fr, xs=xs, ys=ys, radii=np.full(len(fr), 9.0),
                fit=fit, meta={})


def test_chunk_windows_overlap_past_their_end():
    from shotlab.session import _windows, _CHUNK_OVERLAP
    win = _windows(21000, 7000)
    # ownership spans stay disjoint + exhaustive (cache keys/resume unchanged)
    assert [(w0, w1) for w0, w1, _ in win] == [(0, 7000), (7000, 14000),
                                               (14000, 21000)]
    # ...but each window DETECTS past its end, far enough to hold a whole shot
    assert win[0][2] == 7000 + _CHUNK_OVERLAP
    assert win[1][2] == 14000 + _CHUNK_OVERLAP
    assert win[2][2] == 21000                    # clamped at the clip end
    # short clip: single window, no overshoot
    assert _windows(5000, 7000) == [(0, 5000, 5000)]


def test_seam_dedup_keeps_one_full_arc():
    """A shot straddling the 7000-frame seam: the first window (detecting to
    7000+overlap) sees the full arc; the second window starts mid-flight and
    sees a truncated tail. Exactly one shot must survive -- the full one --
    and disjoint neighbors must pass through untouched."""
    from shotlab.session import _merge_seam_pairs
    before = _seam_shot(6800, 6860)              # a normal shot, window 1 only
    full = _seam_shot(6950, 7010)                # straddles the seam, window 1
    dup = _seam_shot(7000, 7010)                 # window 2's truncated view
    after = _seam_shot(7300, 7360)               # a normal shot, window 2 only
    pairs = _merge_seam_pairs([], [("r_before", before), ("r_full", full)])
    pairs = _merge_seam_pairs(pairs, [("r_dup", dup), ("r_after", after)])
    assert [r for r, _ in pairs] == ["r_before", "r_full", "r_after"]
    # order independence at the seam: if the truncated view lands first, the
    # fuller arc still wins (records + shots stay paired)
    pairs2 = _merge_seam_pairs([], [("r_dup", dup)])
    pairs2 = _merge_seam_pairs(pairs2, [("r_full", full)])
    assert [r for r, _ in pairs2] == ["r_full"]


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


def test_cache_sig_tracks_video_content_and_calib():
    """2026-07-15 audit: the record cache is filed under the video's BASENAME,
    so the signature must carry the video's content identity and the effective
    calibration -- otherwise an in-place re-trim or a manual rim override
    silently reuses stale records."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        clip = os.path.join(td, "PXL_x.mp4")
        with open(clip, "wb") as f:
            f.write(b"a" * 100)
        s1 = _sig(video_path=clip)
        # same file, same content -> same sig
        assert _sig(video_path=clip) == s1
        # content changed in place (same basename!) -> different sig
        with open(clip, "wb") as f:
            f.write(b"b" * 200)
        assert _sig(video_path=clip) != s1
    # explicit calibration is part of the identity; a different rim differs
    from shotlab.court import Calibration
    c1 = Calibration(session="t", image_w=1920, image_h=1080, rim_x=900.0,
                     rim_y=200.0, rim_radius_px=18.0, shot_gate_px=120.0)
    c2 = Calibration(session="t", image_w=1920, image_h=1080, rim_x=700.0,
                     rim_y=200.0, rim_radius_px=18.0, shot_gate_px=120.0)
    assert _sig(calib=c1) != _sig()            # explicit != auto
    assert _sig(calib=c1) != _sig(calib=c2)    # rim moves -> new identity
    assert _sig(calib=c1) == _sig(calib=c1)


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
