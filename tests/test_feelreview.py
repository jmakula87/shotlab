"""Feel review core: candidate filtering (verified make/miss only), the
close-cam window mapping (sync-offset sign!), save/resume, and the CSV join
(felt_good semantics: good->True, off->False, okay->None)."""

import json
import os
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.feelreview import (FAULTS, FEEL, MISS_DIR, MOVEMENT, SETUP,
                                apply_review, close_window, load_review,
                                review_candidates, save_entry)


def _df():
    return pd.DataFrame([
        {"clip": "A.mp4", "shot_in_clip": 1, "shot_num": 1, "made": True},
        {"clip": "A.mp4", "shot_in_clip": 2, "shot_num": 2, "made": False},
        {"clip": "A.mp4", "shot_in_clip": 3, "shot_num": 3, "made": None},
        {"clip": "B.mp4", "shot_in_clip": 1, "shot_num": 4, "made": True},
    ])


def test_candidates_use_truth_and_drop_nonshots():
    truth = {"A.mp4|1": "make", "A.mp4|2": "notshot", "A.mp4|3": "miss"}
    # B.mp4|1 unlabeled -> not reviewable when truth exists
    cands = review_candidates(_df(), truth)
    assert [c["key"] for c in cands] == ["A.mp4|1", "A.mp4|3"]
    assert [c["made"] for c in cands] == [True, False]


def test_candidates_fall_back_to_heuristic():
    # no ground truth: made True/False qualify, None (unclassifiable) doesn't
    cands = review_candidates(_df(), None)
    assert [c["key"] for c in cands] == ["A.mp4|1", "A.mp4|2", "B.mp4|1"]


def test_close_window_offset_sign():
    # wide_time = close_time + offset  =>  close_time = wide_time - offset.
    # A +16.5s offset (wide started later) puts wide's 20-25s at close 3.5-8.5.
    assert close_window(20.0, 25.0, 16.5) == (3.5, 8.5)
    # negative offset -> close time is LATER than wide time
    assert close_window(20.0, 25.0, -13.7) == (33.7, 38.7)
    # clamp at the close clip's start, keep the end
    t0, t1 = close_window(1.0, 6.0, 5.0)
    assert t0 == 0.0 and t1 == 1.0


def test_save_resume_and_apply():
    with tempfile.TemporaryDirectory() as td:
        _df().to_csv(os.path.join(td, "session_shots.csv"), index=False)
        save_entry(td, "A.mp4|1", {"feel": "good", "movement": "moving left",
                                   "setup": "off the dribble",
                                   "tags": ["rushed", "elbow flared"],
                                   "miss_dir": None, "note": "", "made": True})
        save_entry(td, "A.mp4|2", {"feel": "off", "movement": None,
                                   "setup": None, "tags": [],
                                   "miss_dir": "short", "note": "tired legs",
                                   "made": False})
        assert len(load_review(td)) == 2               # resume state persists
        n = apply_review(td)
        assert n == 2
        df = pd.read_csv(os.path.join(td, "session_shots.csv"))
        r1 = df[df["shot_num"] == 1].iloc[0]
        assert r1["felt_good"] == True                  # noqa: E712 (csv bool)
        assert r1["review_movement"] == "moving left"
        assert r1["review_tags"] == "rushed;elbow flared"
        r2 = df[df["shot_num"] == 2].iloc[0]
        assert r2["felt_good"] == False                 # noqa: E712
        assert r2["miss_dir"] == "short"
        assert r2["review_note"] == "tired legs"
        r3 = df[df["shot_num"] == 3].iloc[0]            # unreviewed: untouched
        assert pd.isna(r3["feel"]) and pd.isna(r3["felt_good"])
        # idempotent
        assert apply_review(td) == 2


def test_vocabulary_sane():
    # the UI builds itself from these; a typo'd duplicate would silently merge
    all_tags = [t for opts in FAULTS.values() for t in opts]
    assert len(all_tags) == len(set(all_tags))
    assert len(set(FEEL)) == 3 and "good" in FEEL and "off" in FEEL
    assert set(MISS_DIR) >= {"short", "long", "left", "right"}
    assert len(set(MOVEMENT)) == len(MOVEMENT)
    assert len(set(SETUP)) == len(SETUP)


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
