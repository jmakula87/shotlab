"""The close->wide join must be ONE-TO-ONE.

Each close-cam release used to pick its own nearest wide shot independently, which
is not injective: measured 2026-08-04, 141 releases claimed only 112 distinct wide
shots, with 14 taking two each. Duplicates then enter analyses as independent
observations -- pseudo-replication, which makes permutation p-values
anticonservative. These tests bite on that specific defect.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.flare_report import assign_one_to_one


def test_two_releases_cannot_claim_the_same_shot():
    # THE regression. Both releases sit near the single wide shot at t=10.0; the
    # naive nearest-match gives both `made` from it. Only the closer may win.
    a = assign_one_to_one([9.9, 10.3], [10.0])
    assert len(a) == 1, a
    assert 0 in a and 1 not in a, a          # 9.9 is nearer than 10.3
    assert a[0][0] == 0


def test_each_wide_shot_used_at_most_once():
    ev = [1.0, 1.2, 5.0, 5.1, 9.0]
    wide = [1.05, 5.05, 9.02]
    a = assign_one_to_one(ev, wide)
    assert len(a) == 3, a
    assert len({j for j, _d in a.values()}) == 3   # no wide shot reused
    assert len(a.keys()) == len(set(a.keys()))     # no release reused


def test_unmatched_beyond_max_dist_is_dropped_not_snapped():
    a = assign_one_to_one([0.0, 100.0], [0.1])
    assert list(a) == [0], a                 # the far one gets nothing, not a match


def test_closest_pair_wins_globally_not_in_input_order():
    # Release 0 is nearer to wide 0, but release 1 is NEARER STILL. Processing in
    # input order would hand wide 0 to release 0 and strand release 1.
    a = assign_one_to_one([10.4, 10.05], [10.0])
    assert list(a) == [1], a
    assert a[1][0] == 0


def test_empty_inputs_are_safe():
    assert assign_one_to_one([], [1.0]) == {}
    assert assign_one_to_one([1.0], []) == {}
    assert assign_one_to_one([], []) == {}


def test_distance_is_reported_for_auditing():
    a = assign_one_to_one([10.25], [10.0])
    assert abs(a[0][1] - 0.25) < 1e-9, a


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
