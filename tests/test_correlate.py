"""Validate the make-correlation engine on data with a KNOWN planted signal."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.correlate import correlate_makes, summarize_make_drivers


def _rows(n_made, n_miss, seed=0):
    """Makes have entry_angle ~+5deg vs misses (planted); release_angle is noise
    (no signal). knee_bend present but with too few of one outcome to qualify."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_made):
        rows.append({"made": True,
                     "entry_angle_deg": float(rng.normal(47, 2)),
                     "release_angle_deg": float(rng.normal(52, 3)),
                     "knee_bend_deg": float(rng.normal(110, 5))})
    for i in range(n_miss):
        rows.append({"made": False,
                     "entry_angle_deg": float(rng.normal(42, 2)),
                     "release_angle_deg": float(rng.normal(52, 3)),
                     "knee_bend_deg": float(rng.normal(110, 5))})
    return rows


def test_detects_planted_entry_angle_signal():
    rows = _rows(25, 25)
    assocs = {a.metric: a for a in correlate_makes(rows, n_perm=500)}
    ea = assocs["entry_angle_deg"]
    assert ea.direction == "higher"          # makes have higher entry angle
    assert ea.confidence == "medium"         # significant + large effect
    assert ea.p_perm is not None and ea.p_perm < 0.05
    assert ea.cohen_d > 0.8                  # big planted effect
    # the noise metric should NOT come out significant
    ra = assocs["release_angle_deg"]
    assert ra.confidence in ("low", "insufficient")
    # entry angle ranks first (largest real effect)
    assert correlate_makes(rows, n_perm=500)[0].metric == "entry_angle_deg"


def test_insufficient_sample_is_flagged_not_invented():
    rows = _rows(30, 3)                       # only 3 misses -> below min_n
    assocs = {a.metric: a for a in correlate_makes(rows, min_n=8, n_perm=300)}
    ea = assocs["entry_angle_deg"]
    assert ea.confidence == "insufficient"
    assert ea.p_perm is None                  # no test run on too-few data
    assert "need" in ea.note


def test_missing_metric_reports_insufficient():
    rows = [{"made": True, "entry_angle_deg": 47.0},
            {"made": False, "entry_angle_deg": 42.0}]
    assocs = {a.metric: a for a in correlate_makes(rows, n_perm=100)}
    # apex_height_ft absent from every row -> no pairs -> insufficient
    assert assocs["apex_height_ft"].confidence == "insufficient"
    assert assocs["apex_height_ft"].n_made == 0


def test_nan_values_treated_as_missing():
    """A metric that is NaN for every shot (no pose/spin) must NOT produce a
    spurious finding -- it should read as insufficient, like a missing column."""
    rows = _rows(20, 20)
    for r in rows:
        r["backspin_rpm"] = float("nan")
    assocs = {a.metric: a for a in correlate_makes(rows, n_perm=300)}
    bs = assocs["backspin_rpm"]
    assert bs.confidence == "insufficient"
    assert bs.n_made == 0 and bs.n_miss == 0
    assert bs.p_perm is None          # no bogus p from NaN comparisons


def test_summary_is_honest_when_empty():
    rows = [{"made": None, "entry_angle_deg": 47.0}]   # nothing classified
    txt = summarize_make_drivers(correlate_makes(rows, n_perm=100))
    assert "Not enough" in txt


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
