"""The blind labeller's non-GUI contract.

What actually matters here is not the video playback, it is that (a) the keys
match the format every other tool reads, (b) already-labelled shots are skipped
by default so a resumed pass does not re-ask, and (c) the label store keeps its
string-valued contract -- provenance lives in a sidecar, because six readers
across five files consume make_truth.json and would break if labels became dicts.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.label_shots import _shots, _load, KEYS


def _session(tmp, rows, clips_present):
    os.makedirs(os.path.join(tmp, "review_clips"), exist_ok=True)
    with open(os.path.join(tmp, "session_shots.csv"), "w", encoding="utf-8") as f:
        f.write("clip,shot_in_clip\n")
        for c, i in rows:
            f.write(f"{c},{i}\n")
    for name in clips_present:
        open(os.path.join(tmp, "review_clips", name), "wb").close()
    return tmp


def test_keys_match_the_format_other_tools_read():
    with tempfile.TemporaryDirectory() as tmp:
        _session(tmp, [("PXL_A.mp4", 1)], ["PXL_A_s001_wide.mp4"])
        got = _shots(tmp, os.path.join(tmp, "review_clips"), True, {})
        assert len(got) == 1
        # "<clip filename>|<shot_in_clip>" -- same as make_truth.json elsewhere
        assert got[0][0] == "PXL_A.mp4|1", got[0][0]


def test_already_labelled_shots_are_skipped_by_default():
    with tempfile.TemporaryDirectory() as tmp:
        _session(tmp, [("PXL_A.mp4", 1), ("PXL_A.mp4", 2)],
                 ["PXL_A_s001_wide.mp4", "PXL_A_s002_wide.mp4"])
        truth = {"PXL_A.mp4|1": "make"}
        assert len(_shots(tmp, os.path.join(tmp, "review_clips"), True, truth)) == 1
        # ...but --all revisits everything
        assert len(_shots(tmp, os.path.join(tmp, "review_clips"), False, truth)) == 2


def test_shots_without_a_review_clip_are_not_offered():
    with tempfile.TemporaryDirectory() as tmp:
        _session(tmp, [("PXL_A.mp4", 1), ("PXL_A.mp4", 2)], ["PXL_A_s001_wide.mp4"])
        got = _shots(tmp, os.path.join(tmp, "review_clips"), True, {})
        assert [g[2] for g in got] == [1], got


def test_load_survives_a_corrupt_store():
    # a half-written json must not wipe a labelling session
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "x.json")
        open(p, "w", encoding="utf-8").write("{not json")
        assert _load(p, {"fallback": 1}) == {"fallback": 1}
        assert _load(os.path.join(tmp, "missing.json"), {}) == {}


def test_the_four_label_keys_are_the_documented_ones():
    assert set(KEYS.values()) == {"make", "miss", "notshot", "unsure"}
    assert KEYS[ord("m")] == "make" and KEYS[ord("n")] == "miss"


def test_label_store_stays_string_valued():
    """Guards the contract: make_truth.json maps key -> STRING. If someone makes
    labels dicts to carry provenance, train_make_model's `lab not in ("make",
    "miss")` check silently drops every row and the model trains on nothing."""
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "make_truth.json")
        json.dump({"PXL_A.mp4|1": "make"}, open(p, "w", encoding="utf-8"))
        store = _load(p, {})
        assert all(isinstance(v, str) for v in store.values())


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
