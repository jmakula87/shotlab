"""Validate the universal (textbook) ideals + the grade helper."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.textbook import TEXTBOOK, grade, profile_block


def test_entry_and_release_targets():
    assert TEXTBOOK["entry_angle_deg"]["target"] == 45.0
    assert TEXTBOOK["release_angle_deg"]["target"] == 52.0


def test_arc_angles_are_calibration_gated():
    # both arc angles are measurable in pixels but foreshortened -> not trusted
    # against the target until calibration (NOT the 2nd camera)
    for m in ("entry_angle_deg", "release_angle_deg"):
        assert TEXTBOOK[m]["measurable_now"] is False
        assert TEXTBOOK[m]["blocked_by"] == "calibration"


def test_flare_needs_two_cameras():
    f = TEXTBOOK["elbow_flare_deg"]
    assert f["target"] == 0.0 and f["measurable_now"] is False
    assert f["blocked_by"] == "2nd camera"


def test_grade_none_while_calibration_gated():
    # nothing is measurable_now on the current rig -> grade returns None
    assert grade("entry_angle_deg", 47.0) is None
    assert grade("release_angle_deg", 52.0) is None
    assert grade("elbow_flare_deg", 3.0) is None
    assert grade("knee_bend_deg", 120) is None        # not a textbook metric


def test_profile_block_separate_and_labeled():
    block = profile_block()
    for m in ("entry_angle_deg", "release_angle_deg", "elbow_flare_deg"):
        assert m in block and "why" in block[m] and "blocked_by" in block[m]
    assert block["entry_angle_deg"]["target"] == 45.0
    assert block["release_angle_deg"]["target"] == 52.0


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
