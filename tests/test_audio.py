"""Validate the audio make/miss heuristic + fusion on synthetic sound."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.audio import audio_make_hint, rim_contact_ratio, fuse_make

SR = 16000


def _clip(seconds=2.0, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.01, int(SR * seconds))      # quiet ambient noise


def test_loud_clang_reads_as_miss():
    x = _clip()
    # a loud transient 1.0s in (the rim moment) = rim/backboard clang
    t = 1.0
    burst = int(0.05 * SR)
    i = int(t * SR)
    x[i:i + burst] += np.hanning(burst) * 1.0
    hint = audio_make_hint(x, SR, rim_time_s=t)
    assert hint["made"] is False, hint
    assert rim_contact_ratio(x, SR, t) > 4


def test_clean_swish_reads_as_make():
    x = _clip(seed=1)                                   # no burst = clean/quiet
    hint = audio_make_hint(x, SR, rim_time_s=1.0)
    assert hint["made"] is True, hint


def test_fusion_agreement_and_conflict():
    # agreement bumps confidence
    made, conf = fuse_make(True, "low", {"made": True})
    assert made is True and conf == "medium"
    # conflict -> keep visual, drop to low
    made, conf = fuse_make(True, "medium", {"made": False})
    assert made is True and conf == "low"
    # no visual, audio present -> use audio (low)
    made, conf = fuse_make(None, "na", {"made": False})
    assert made is False and conf == "low"
    # neither -> na
    assert fuse_make(None, "na", {"made": None}) == (None, "na")


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
