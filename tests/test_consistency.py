"""Validate cross-session consistency tracking (backlog #6: spread over time)."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.session import (consistency_stats, consistency_progress,
                             fatigue_breakdown, mean_drift,
                             prescribe_target, drill_effectiveness)


def _session_df(spread, n=40, seed=0):
    """A one-zone session whose release angle has the given std (spread)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "elapsed_min": np.linspace(0, 20, n),
        "zone": ["center_mid"] * n,
        "release_angle_deg": rng.normal(50, spread, n),
    })


def test_within_zone_std_recovers_spread():
    df = _session_df(spread=6.0, n=200)
    cs = consistency_stats(df, metrics=["release_angle_deg"])
    wz = cs.set_index("metric").loc["release_angle_deg", "within_zone_std"]
    assert 5.0 <= wz <= 7.0, wz          # recovers the planted ~6


def test_consistency_progress_flags_improvement():
    # session A is erratic (std 8), session B is tight (std 4) -> improving
    agg = pd.DataFrame([
        {"session": "session_A", "date": "2026-06-01", "std_release_angle": 8.0,
         "std_entry_angle": 6.0},
        {"session": "session_B", "date": "2026-06-08", "std_release_angle": 4.0,
         "std_entry_angle": 7.0},
    ])
    cp = consistency_progress(agg).set_index("metric")
    assert bool(cp.loc["release_angle", "improving"]) is True   # spread shrank
    assert cp.loc["release_angle", "delta"] == -4.0
    assert bool(cp.loc["entry_angle", "improving"]) is False    # spread grew


def test_consistency_progress_needs_two_sessions():
    agg = pd.DataFrame([{"session": "s", "date": "2026-06-01",
                         "std_release_angle": 8.0}])
    assert consistency_progress(agg).empty


def test_fatigue_breakdown_flags_the_fader():
    # knee bend degrades hard in the 2nd half; release angle stays put
    n = 24
    elapsed = np.linspace(0, 30, n)
    knee = np.where(elapsed <= 15, 110.0, 135.0) + np.random.default_rng(0).normal(0, 2, n)
    rel = 52 + np.random.default_rng(1).normal(0, 2, n)
    df = pd.DataFrame({"elapsed_min": elapsed, "knee_bend_deg": knee,
                       "release_angle_deg": rel})
    fb = fatigue_breakdown(df).set_index("metric")
    assert bool(fb.loc["knee_bend_deg", "fades_most"]) is True
    assert abs(fb.loc["knee_bend_deg", "change_in_sd"]) > \
           abs(fb.loc["release_angle_deg", "change_in_sd"])


def test_mean_drift_flags_creep():
    agg = pd.DataFrame([
        {"session": "a", "avg_release_angle": 52.0, "avg_entry_angle": 46.0},
        {"session": "b", "avg_release_angle": 49.0, "avg_entry_angle": 46.2},
        {"session": "c", "avg_release_angle": 45.0, "avg_entry_angle": 45.9},
    ])
    md = mean_drift(agg).set_index("metric")
    assert bool(md.loc["release_angle", "drifting"]) is True    # 52 -> 45 = big creep
    assert bool(md.loc["entry_angle", "drifting"]) is False     # basically flat
    assert md.loc["release_angle", "slope_per_session"] < 0
    assert mean_drift(agg.head(1)).empty                        # need >=2 sessions


def test_prescribe_target_picks_least_repeatable():
    rng = np.random.default_rng(0)
    n = 30
    df = pd.DataFrame({
        "elapsed_min": np.linspace(0, 20, n),
        "zone": ["center_mid"] * n,
        "entry_angle_deg": rng.normal(45, 1.0, n),      # tight (low CV)
        "release_angle_deg": rng.normal(50, 12.0, n),   # very scattered (high CV)
    })
    p = prescribe_target(df)
    assert p["target_metric"] == "release_angle_deg", p


def test_drill_effectiveness_tracks_followup():
    sessions = [
        {"name": "s1", "target_metric": "release_angle_deg",
         "stds": {"release_angle_deg": 12.0}},
        {"name": "s2", "target_metric": "knee_bend_deg",
         "stds": {"release_angle_deg": 7.0, "knee_bend_deg": 20.0}},   # release improved
        {"name": "s3", "target_metric": None,
         "stds": {"knee_bend_deg": 25.0}},                              # knee got worse
    ]
    de = drill_effectiveness(sessions)
    row1 = de[de["worked_on"] == "release_angle_deg"].iloc[0]
    assert bool(row1["improved"]) is True and row1["std_after"] == 7.0
    row2 = de[de["worked_on"] == "knee_bend_deg"].iloc[0]
    assert bool(row2["improved"]) is False        # 20 -> 25


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
