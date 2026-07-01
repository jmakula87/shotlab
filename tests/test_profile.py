"""Validate the profile exporter: good-shot selection priority + ideal targets."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "tools"))

from export_profile import select_good, build_profile


def _df(n=20, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "shot_num": range(1, n + 1),
        "elbow_angle_at_release_deg": rng.normal(110, 6, n),
        "knee_bend_deg": rng.normal(115, 8, n),
        "tempo_dip_to_release_s": rng.normal(0.4, 0.05, n),
    })


def test_feel_tags_win_selection():
    df = _df()
    df["felt_good"] = [True] * 8 + [None] * 12
    good, method = select_good(df, session_dir="/nope", min_good=5)
    assert len(good) == 8 and "felt-good" in method


def test_falls_back_to_made():
    df = _df()
    df["made"] = [True] * 7 + [False] * 13         # no feel tags, no best_shots.csv
    good, method = select_good(df, session_dir="/nope", min_good=5)
    assert len(good) == 7 and "made" in method


def test_falls_back_to_all_when_sparse():
    df = _df()
    df["made"] = [True] * 2 + [False] * 18          # too few makes
    good, method = select_good(df, session_dir="/nope", min_good=5)
    assert len(good) == len(df) and "all shots" in method


def test_build_profile_ideal_is_mean_of_good():
    df = _df()
    df["felt_good"] = [True] * 10 + [None] * 10
    prof = build_profile(df, session_dir="/nope")
    good = df.iloc[:10]
    assert abs(prof["ideal"]["elbow_angle_at_release_deg"]
               - round(float(good["elbow_angle_at_release_deg"].mean()), 2)) < 0.01
    # tolerance respects the floor (>= 6 for elbow)
    assert prof["tolerance"]["elbow_angle_at_release_deg"] >= 6.0
    assert prof["n_good"] == 10


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
