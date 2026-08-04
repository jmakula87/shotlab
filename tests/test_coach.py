"""coach.py had ZERO test coverage. The load-bearing behavior after the fix arc:
recommend_drills must NOT prescribe a drill whose premise contradicts the
measured make-direction (the "coached the opposite of the data" regression), and
generate_review's money spot is the best MAKE%, not the tightest spread
(2026-07-07 test audit)."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.coach import recommend_drills, generate_review


def _session(follow_makes, follow_misses):
    """20 shots, follow-through hold differs by outcome as given; other metrics flat."""
    rows = []
    for i in range(10):
        j = (i - 5) * 0.01        # within-group jitter so cohen_d is finite
        rows.append({"shot_num": 2 * i + 1, "zone": "z", "made": True, "elapsed_min": i,
                     "follow_through_hold_s": follow_makes + j, "release_conf": "high",
                     "entry_angle_deg": 47.0})
        rows.append({"shot_num": 2 * i + 2, "zone": "z", "made": False, "elapsed_min": i,
                     "follow_through_hold_s": follow_misses + j, "release_conf": "high",
                     "entry_angle_deg": 47.0})
    return pd.DataFrame(rows)


def test_no_backwards_drill_when_direction_contradicts_premise():
    # makes hold SHORTER than misses -> the "hold it longer" drill's premise is
    # wrong for this session, so it must NOT be prescribed.
    df = _session(follow_makes=0.40, follow_misses=0.72)
    assert not any("Freeze the follow" in d for d in recommend_drills(df))


def test_never_prescribes_the_follow_through_drill_from_make_direction():
    """UPDATED 2026-08-04. This test used to assert the OPPOSITE: that makes
    holding LONGER should prescribe "hold it longer". That is a reversed causal
    arrow. follow_through_hold_s is measured entirely AFTER the ball leaves the
    hand, over a 1.0s window, while the ball needs ~1s to reach the rim --
    P(hold>=t) is identical for makes and misses through t=10 frames and separates
    only at 0.6-0.8s post-release, when the ball is at the rim. A miss sends the
    shooter to rebound; a make does not. So "makes hold longer" is an EFFECT of
    making, and prescribing it coaches the shooter to mimic that effect.
    It must not be prescribed in EITHER direction."""
    assert not any("Freeze the follow" in d
                   for d in recommend_drills(_session(follow_makes=0.72,
                                                      follow_misses=0.40)))
    assert not any("Freeze the follow" in d
                   for d in recommend_drills(_session(follow_makes=0.40,
                                                      follow_misses=0.72)))


def test_money_spot_is_best_make_pct_not_tightest_spread():
    # spot A: tight entry spread but 1/6 makes; spot B: looser but 4/6 makes.
    rows = []
    for i in range(6):
        rows.append({"shot_num": i + 1, "zone": "A", "made": i == 0, "elapsed_min": i,
                     "entry_angle_deg": 45.0 + (i % 2) * 0.2})          # very tight
    for i in range(6):
        rows.append({"shot_num": 100 + i, "zone": "B", "made": i < 4, "elapsed_min": i,
                     "entry_angle_deg": 40.0 + i * 3})                  # loose
    rev = generate_review(pd.DataFrame(rows))
    money = " ".join(rev["strengths"])
    assert "money spot" in money and "**B**" in money, money


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
