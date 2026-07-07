"""The shared artifact gate (metric_ranges) -- the whole D4 arc had ZERO real
coverage (every other fixture was all-in-range, so a no-op'd gate() passed the
suite). These plant real artifacts and assert they're dropped (2026-07-07 test
audit)."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.metric_ranges import in_range, gate, VALID_RANGE


def test_in_range_rejects_artifacts_accepts_real():
    assert in_range("elbow_angle_at_release_deg", 160) is True
    assert in_range("elbow_angle_at_release_deg", 47) is False        # slewed mid-push
    assert in_range("knee_bend_deg", 100) is True
    assert in_range("knee_bend_deg", 172) is False                    # straight-leg pose
    assert in_range("release_height_ft", 6.5) is True
    assert in_range("release_height_ft", 34.9) is False               # impossible
    assert in_range("release_height_ft", -0.9) is False
    assert in_range("release_vs_apex_s", 0.0) is True
    assert in_range("release_vs_apex_s", -3.86) is False              # impossible
    assert in_range("knee_bend_deg", float("nan")) is False
    assert in_range("knee_bend_deg", float("inf")) is False


def test_every_gated_metric_drops_an_out_of_range_value():
    # for each gated column, a value just past each finite bound must be dropped
    for col, (lo, hi) in VALID_RANGE.items():
        if lo is not None:
            assert in_range(col, lo - 1) is False, col
        if hi is not None:
            assert in_range(col, hi + 1) is False, col


def test_gate_drops_artifact_rows_from_a_frame():
    df = pd.DataFrame({"knee_bend_deg": [100.0, 172.0, 95.0, np.nan, 30.5],
                       "other": [1, 2, 3, 4, 5]})
    g = gate(df, "knee_bend_deg")
    assert sorted(g["knee_bend_deg"].tolist()) == [30.5, 95.0, 100.0]   # 172 + NaN gone
    # a mean over the gated frame excludes the artifact
    assert abs(g["knee_bend_deg"].mean() - (30.5 + 95 + 100) / 3) < 1e-9
    # ungated mean would be pulled up by the 172 artifact -> proves the gate bites
    assert df["knee_bend_deg"].mean() > g["knee_bend_deg"].mean() + 15


def test_gate_is_noop_on_absent_or_ungated_column():
    df = pd.DataFrame({"x": [1, 2, 3]})
    assert len(gate(df, "not_a_column")) == 3
    df2 = pd.DataFrame({"n_points": [5, 999, 7]})   # not in VALID_RANGE -> all kept
    assert len(gate(df2, "n_points")) == 3


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
