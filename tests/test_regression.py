"""Regression guard: lock the analysis layer's output on a FIXED, deterministic
session table so unintended drift in the metric math gets caught. (The detection
pipeline needs video and is validated separately; this pins the pure analytics.)
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.session import consistency_stats, fatigue_trends, prescribe_target
from shotlab.correlate import correlate_makes


def _fixture():
    """A fixed 40-shot session (seed 42). Treat as golden input."""
    rng = np.random.default_rng(42)
    n = 40
    return pd.DataFrame({
        "elapsed_min": np.linspace(0, 25, n),
        "zone": (["center_mid"] * 20 + ["left_far"] * 20),
        "release_angle_deg": rng.normal(50, 6, n),
        "entry_angle_deg": rng.normal(45, 3, n),
        "knee_bend_deg": rng.normal(110, 8, n),
        "made": [True, False] * (n // 2),
    })


def test_consistency_is_stable():
    c = consistency_stats(_fixture()).set_index("metric")
    # golden: within-zone release-angle spread ~5.0 on this fixture
    assert abs(float(c.loc["release_angle_deg", "within_zone_std"]) - 5.0) < 0.1
    assert len(c) == 3


def test_prescribe_target_is_stable():
    assert prescribe_target(_fixture())["target_metric"] == "release_angle_deg"


def test_fatigue_slope_matches_a_planted_decline():
    # a metric that drops exactly 1.0/min -> slope ~ -1.0, trend 'declining'.
    # (The old test only checked the column EXISTED, so negating the fitted slope
    # survived it -- 2026-07-07 audit.)
    df = pd.DataFrame({"elapsed_min": np.arange(20, dtype=float), "zone": ["z"] * 20,
                       "knee_bend_deg": 120.0 - np.arange(20, dtype=float)})
    ft = fatigue_trends(df).set_index("metric")
    assert abs(float(ft.loc["knee_bend_deg", "slope_per_min"]) - (-1.0)) < 0.01
    assert ft.loc["knee_bend_deg", "trend"] == "declining"


def test_correlate_finds_no_driver_on_alternating_labels():
    df = _fixture()
    # made/miss alternate -> no metric should be a strong make-driver here
    assocs = correlate_makes(df.to_dict("records"), n_perm=200)
    assert all(a.confidence in ("low", "insufficient") for a in assocs)


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
