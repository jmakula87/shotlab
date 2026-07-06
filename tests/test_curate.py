"""Curation drops the right shots: manual excludes/layups (exclude.json) AND
auto-flagged degenerate detections (is_real == False; 2026-07-06 audit D2)."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.curate import apply_excludes, save_excludes


def _df():
    return pd.DataFrame({
        "shot_num": [1, 2, 3, 4, 5],
        "is_real": [True, False, True, True, "False"],   # 2 & 5 are phantoms
        "made": [True, True, False, True, True],
    })


def test_drops_auto_flagged_phantoms(tmpdir=None):
    d = os.path.join(os.path.dirname(__file__), "_curate_tmp")
    os.makedirs(d, exist_ok=True)
    # no exclude.json -> only is_real=False dropped (shots 2 and 5)
    out = apply_excludes(_df(), d)
    assert set(out["shot_num"]) == {1, 3, 4}, sorted(out["shot_num"])


def test_combines_with_manual_excludes():
    d = os.path.join(os.path.dirname(__file__), "_curate_tmp")
    os.makedirs(d, exist_ok=True)
    save_excludes(d, exclude=[3], layups=[4])
    out = apply_excludes(_df(), d)          # drop is_real(2,5) + exclude(3) + layup(4)
    assert set(out["shot_num"]) == {1}, sorted(out["shot_num"])
    os.remove(os.path.join(d, "exclude.json"))


def test_noop_without_is_real_or_excludes():
    df = pd.DataFrame({"shot_num": [1, 2], "made": [True, False]})
    out = apply_excludes(df, "/no/such/dir")
    assert len(out) == 2


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
