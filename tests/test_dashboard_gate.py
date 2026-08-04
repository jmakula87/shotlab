"""Display surfaces must not show physically-implausible pose reads as numbers.

The session CSV deliberately keeps RAW values -- it is the analysis artifact, and
the 30-150 deg knee window is a CENSORING CHOICE rather than physics, with weak
evidence it discards signal along with artifacts. So the fix is not to gate at
write time; it is to gate at every DISPLAY.

Measured 2026-08-04 before the fix: 1081 implausible values across the built
sessions were reaching the dashboard as facts (305 release_vs_apex_s, 195
knee_bend_deg, 145 release_height_ft, ...), where they were also averaged into
summary numbers.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "dashboard"))

from shotlab.metric_ranges import VALID_RANGE


def _gate():
    from app import _gate_metrics
    return _gate_metrics


def test_implausible_values_become_absent_not_numbers():
    g = _gate()
    df = pd.DataFrame({"knee_bend_deg": [100.0, 178.0, 6.0],
                       "elbow_angle_at_release_deg": [160.0, 47.0, 170.0]})
    out = g(df)
    assert out["knee_bend_deg"].tolist()[0] == 100.0
    assert np.isnan(out["knee_bend_deg"].tolist()[1])      # straight leg, not a bend
    assert np.isnan(out["knee_bend_deg"].tolist()[2])      # 6 deg is impossible
    assert np.isnan(out["elbow_angle_at_release_deg"].tolist()[1])   # slewed frame


def test_a_bad_read_does_not_hide_the_rest_of_the_shot():
    """NaN, not a dropped row -- a bad knee must not also erase that shot's arc,
    outcome or video link."""
    g = _gate()
    df = pd.DataFrame({"knee_bend_deg": [178.0], "release_angle_deg": [52.0],
                       "clip": ["a.mp4"], "made": [True]})
    out = g(df)
    assert len(out) == 1
    assert out["release_angle_deg"].iloc[0] == 52.0
    assert out["clip"].iloc[0] == "a.mp4" and bool(out["made"].iloc[0]) is True


def test_ungated_columns_are_untouched():
    g = _gate()
    df = pd.DataFrame({"shot_in_clip": [1, 2], "zone": ["a", "b"],
                       "backspin_rpm": [9999.0, 1.0]})
    out = g(df)
    assert out["shot_in_clip"].tolist() == [1, 2]
    assert out["zone"].tolist() == ["a", "b"]
    assert "backspin_rpm" not in VALID_RANGE          # no gate defined -> passes
    assert out["backspin_rpm"].tolist() == [9999.0, 1.0]


def test_the_source_csv_is_not_mutated():
    """The artifact must keep its raw values; only the displayed copy is gated."""
    g = _gate()
    df = pd.DataFrame({"knee_bend_deg": [178.0]})
    out = g(df)
    assert df["knee_bend_deg"].iloc[0] == 178.0        # original untouched
    assert np.isnan(out["knee_bend_deg"].iloc[0])


def test_nan_and_missing_survive_as_nan():
    g = _gate()
    df = pd.DataFrame({"knee_bend_deg": [np.nan, None]})
    out = g(df)
    assert out["knee_bend_deg"].isna().all()


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    p = f = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}"); p += 1
        except Exception:
            traceback.print_exc(); print(f"FAIL {fn.__name__}"); f += 1
    print(f"\n{p}/{p + f} passed")
    sys.exit(1 if f else 0)
