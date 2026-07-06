"""Validate auto shot-type tagging (form + setup)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.shottype import (classify_form, classify_setup, detect_dribble,
                              classify_shot_type)


class FakeBall:
    def __init__(self, cy, r=8.0):
        self.cx, self.cy, self.r, self.conf = 100.0, cy, r, 0.9


def test_form_far_is_confident_jumper():
    form, conf = classify_form("far", apex_above_rim_ft=4.0, release_angle_deg=52)
    assert form == "jumper" and conf == "medium"
    form, conf = classify_form("mid", apex_above_rim_ft=3.0, release_angle_deg=48)
    assert form == "jumper" and conf == "medium"


def test_form_near_flat_low_is_layup():
    form, conf = classify_form("near", apex_above_rim_ft=0.8, release_angle_deg=22)
    assert form == "layup" and conf == "low"      # close, flat, no arc


def test_form_near_lobbed_is_floater_else_jumper():
    assert classify_form("near", 1.8, 40)[0] == "floater"   # close but lobbed
    assert classify_form("near", 3.0, 50)[0] == "jumper"    # close-range jumper


def test_setup_from_movement():
    assert classify_setup("set", None)[0] == "catch_and_shoot"
    assert classify_setup("left", None)[0] == "on_the_move"
    assert classify_setup("set", True)[0] == "off_dribble"   # dribble overrides
    assert classify_setup("unknown", None)[0] == "unknown"


def test_detect_dribble_bounce():
    # ball dips toward the floor and rebounds twice before release -> dribble
    rel = 60
    track = {}
    ys = [50, 70, 95, 70, 50, 75, 100, 72, 48, 40]   # two clear bounces
    for i, y in enumerate(ys):
        track[rel - len(ys) + 1 + i] = FakeBall(cy=y)
    dribbled, conf = detect_dribble(track, rel, fps=30)
    assert dribbled is True


def test_detect_dribble_catch_is_flat():
    rel = 60
    track = {rel - 8 + i: FakeBall(cy=50 + i * 0.2) for i in range(9)}  # ~flat
    dribbled, conf = detect_dribble(track, rel, fps=30)
    assert dribbled is False


def test_detect_dribble_sparse_is_unknown():
    track = {58: FakeBall(cy=50), 60: FakeBall(cy=52)}    # < min_samples
    dribbled, conf = detect_dribble(track, 60, fps=30)
    assert dribbled is None


def test_classify_shot_type_end_to_end():
    rel = 60
    track = {rel - 10 + i: FakeBall(cy=y) for i, y in
             enumerate([50, 80, 110, 80, 50, 85, 115, 80, 50, 45, 40])}
    st = classify_shot_type(depth="far", apex_above_rim_ft=4.0, release_angle_deg=50,
                            movement_dir="set", ball_track=track, rel_frame=rel,
                            fps=30)
    assert st.form == "jumper" and st.form_conf == "medium"
    assert st.setup == "off_dribble"      # bounces detected -> overrides 'set'


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
