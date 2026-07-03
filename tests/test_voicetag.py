"""Validate the voice-tag parser + shot-assignment (the STT engine is validated
on real audio separately)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.voicetag import parse_phrase, assign_to_shots, vocab, REASONS


def test_plain_outcomes():
    assert parse_phrase("good") == ("good", [])
    assert parse_phrase("bad") == ("bad", [])
    assert parse_phrase("money") == ("good", [])
    assert parse_phrase("brick") == ("bad", [])


def test_bad_with_reason():
    o, r = parse_phrase("bad, flare")
    assert o == "bad" and r == ["flare"]
    o, r = parse_phrase("bad off hand")
    assert o == "bad" and "off_hand" in r


def test_off_hand_does_not_falsetrigger_bad_alone():
    # "off" is NOT a bad-synonym, so "off hand" without "bad" only sets the
    # reason -- but it still needs an outcome word to be a tag
    o, r = parse_phrase("off hand")
    assert o is None and r == ["off_hand"]


def test_order_independent_and_bad_wins():
    o, r = parse_phrase("flare, that was bad")
    assert o == "bad" and r == ["flare"]


def test_unrelated_speech_is_no_tag():
    assert parse_phrase("what time is it") == (None, [])
    assert parse_phrase("") == (None, [])


def test_vocab_covers_all_keywords():
    v = set(vocab())
    for kws in REASONS.values():
        assert kws <= v


def test_assign_phrase_to_most_recent_shot():
    shots = [{"id": 1, "t": 10.0}, {"id": 2, "t": 20.0}, {"id": 3, "t": 30.0}]
    phrases = [{"t": 11.5, "text": "good"},          # -> shot 1
               {"t": 21.0, "text": "bad, flare"},    # -> shot 2
               {"t": 31.2, "text": "bad off hand"}]  # -> shot 3
    tags = assign_to_shots(phrases, shots)
    assert tags[1]["outcome"] == "good"
    assert tags[2] == {"outcome": "bad", "reasons": ["flare"], "text": "bad, flare"}
    assert tags[3]["reasons"] == ["off_hand"]


def test_phrase_too_far_after_shot_is_dropped():
    shots = [{"id": 1, "t": 10.0}]
    tags = assign_to_shots([{"t": 20.0, "text": "good"}], shots, max_gap_s=4.0)
    assert tags == {}                                # 10s later -> not this shot


def test_correction_overwrites():
    shots = [{"id": 1, "t": 10.0}]
    phrases = [{"t": 11.0, "text": "good"}, {"t": 12.5, "text": "bad, flare"}]
    tags = assign_to_shots(phrases, shots)
    assert tags[1]["outcome"] == "bad"               # the later call wins


def test_accepts_prebuilt_outcome():
    shots = [{"id": 1, "t": 10.0}]
    tags = assign_to_shots([{"t": 11.0, "outcome": "good", "reasons": []}], shots)
    assert tags[1]["outcome"] == "good"


def test_apply_tags_to_session_writes_felt_good(tmp_path=None):
    """End-to-end (minus STT): injected phrases -> felt_good + felt_reasons in
    the session CSV, mapped to the right shots by clip-relative time."""
    import tempfile
    import pandas as pd
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "tools"))
    from voicetag_session import apply_tags_to_session

    d = tempfile.mkdtemp()
    clip = "PXL_20260701_181500000.mp4"          # clip starts 18:15:00
    # two shots at +12s and +40s into the clip
    df = pd.DataFrame({
        "clip": [clip, clip],
        "shot_in_clip": [1, 2],
        "abs_time": ["2026-07-01T18:15:12", "2026-07-01T18:15:40"],
        "made": [None, None],
    })
    df.to_csv(os.path.join(d, "session_shots.csv"), index=False)

    phrases = {clip: [{"t": 13.5, "text": "good"},          # -> shot 1
                      {"t": 41.0, "text": "bad, flare"}]}    # -> shot 2
    s = apply_tags_to_session(d, phrases_by_clip=phrases)
    assert s["good"] == 1 and s["bad"] == 1 and s["reasons"] == {"flare": 1}

    out = pd.read_csv(os.path.join(d, "session_shots.csv"))
    assert bool(out.loc[0, "felt_good"]) is True
    assert bool(out.loc[1, "felt_good"]) is False
    assert out.loc[1, "felt_reasons"] == "flare"


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
