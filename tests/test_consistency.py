"""Validate cross-session consistency tracking (backlog #6: spread over time)."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.session import consistency_stats, consistency_progress


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
