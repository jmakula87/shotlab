"""Validate the universal (textbook) ideals + the grade helper."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.textbook import TEXTBOOK, grade, profile_block


def test_entry_angle_is_measurable_universal():
    e = TEXTBOOK["entry_angle_deg"]
    assert e["target"] == 45.0 and e["measurable_now"] is True


def test_flare_is_universal_but_needs_two_cameras():
    f = TEXTBOOK["elbow_flare_deg"]
    assert f["target"] == 0.0
    assert f["measurable_now"] is False        # can't see flare on 1 camera
    assert "2-camera" in f["needs"]


def test_grade_entry_on_target():
    within, delta, spec = grade("entry_angle_deg", 47.0)
    assert within is True and delta == 2.0


def test_grade_entry_off_target():
    within, delta, _ = grade("entry_angle_deg", 55.0)
    assert within is False and delta == 10.0


def test_grade_none_for_unmeasurable_or_missing():
    assert grade("elbow_flare_deg", 3.0) is None      # not measurable on 1 cam
    assert grade("entry_angle_deg", None) is None
    assert grade("knee_bend_deg", 120) is None        # not a textbook metric


def test_profile_block_separate_and_labeled():
    block = profile_block()
    assert "entry_angle_deg" in block and "elbow_flare_deg" in block
    assert block["entry_angle_deg"]["target"] == 45.0
    assert "why" in block["entry_angle_deg"]
    assert block["elbow_flare_deg"]["measurable_now"] is False


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
