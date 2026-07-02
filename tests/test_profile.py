"""Validate the profile exporter: good-shot selection priority + ideal targets."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "tools"))

from export_profile import select_good, select_form_good, build_profile


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


def test_profile_has_separate_textbook_block():
    """Universal ideals ship in a SEPARATE `textbook` block, never blended into
    the personal `ideal` (so a pro number can't override your own norm)."""
    df = _df()
    df["felt_good"] = [True] * 10 + [None] * 10
    prof = build_profile(df, session_dir="/nope")
    assert "textbook" in prof and "entry_angle_deg" in prof["textbook"]
    assert prof["textbook"]["entry_angle_deg"]["target"] == 45.0
    # the personal ideal must NOT contain the textbook entry-angle target
    # unless it came from the user's own shots (this df has no entry angle)
    assert "entry_angle_deg" not in prof["ideal"]


def test_form_ideals_survive_when_best_arc_shots_lack_pose():
    """The 2026-07-02 regression: the arc-ranked 'best' pool is far/clean-arc
    shots with NO pose, so elbow/knee were dropped. Form ideals must instead
    come from the pose-reliable pool."""
    rng = np.random.default_rng(3)
    n = 40
    df = pd.DataFrame({
        "shot_num": range(1, n + 1),
        "made": [True] * 20 + [False] * 20,
        "release_angle_deg": rng.normal(50, 5, n),
        "entry_angle_deg": rng.normal(45, 5, n),
        # pose present on only the FIRST 15 shots (the close, pose-readable ones)
        "elbow_angle_at_release_deg": [rng.normal(110, 6) for _ in range(15)]
                                      + [np.nan] * 25,
        "knee_bend_deg": [rng.normal(115, 8) for _ in range(15)] + [np.nan] * 25,
    })
    # best_shots.csv = 10 far shots with NO pose (shots 26..35)
    prof = build_profile(df, session_dir="/nope")
    # arc ideals always populate; form ideals populate from the pose pool
    assert "release_angle_deg" in prof["ideal"]
    assert "elbow_angle_at_release_deg" in prof["ideal"], prof["ideal"]
    assert "knee_bend_deg" in prof["ideal"]
    assert prof["n_form"] >= 5


def test_form_good_prefers_pose_present_made_shots():
    df = _df(n=30, seed=4)
    df["made"] = [True] * 12 + [False] * 18
    df.loc[df.index[:6], "knee_bend_deg"] = np.nan   # 6 made shots lack pose
    good, method = select_form_good(df, min_good=5)
    assert good["knee_bend_deg"].notna().all()       # only pose-present shots
    assert "pose-reliable" in method


def test_form_good_conf_orders_pool():
    df = _df(n=20, seed=5)
    df["made"] = [True] * 20
    df["release_conf"] = (["low"] * 10 + ["high"] * 10)
    good, _ = select_form_good(df, min_good=5)
    # high-confidence shots should sort to the front
    assert good.iloc[0]["release_conf"] == "high"


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
