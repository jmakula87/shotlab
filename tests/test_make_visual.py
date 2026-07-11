"""Visual make/miss features + model. Mutation-checked: a make-like net/ball
signature must score differently from a miss-like one, and the model roundtrips."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab import make_visual as mv


def _signals(make: bool, n=76):
    """Synthetic per-frame signals. A MAKE: net whips + ball passes through the
    net (white occlusion) + orange mass appears BELOW the net. A MISS: little net
    motion, orange caroms to the SIDE."""
    rng = np.random.default_rng(0 if make else 1)
    t0 = mv.PRE
    z = lambda: rng.normal(3, 0.5, n).clip(0)
    sig = {"o_in_rim": np.zeros(n), "o_net": z(), "o_below_net": z(),
           "o_side": z(), "net_motion": rng.normal(5, 1, n).clip(0),
           "net_white_count": np.full(n, 120.0), "flank_motion": rng.normal(5, 1, n).clip(0)}
    sig["o_in_rim"][t0] = 400.0                      # ball reaches the rim at t0
    if make:
        sig["net_motion"][t0:t0 + 12] += 60          # net whips
        sig["o_below_net"][t0 + 3:t0 + 20] += 800     # ball drops below the net
        sig["o_net"][t0:t0 + 8] += 400
        sig["net_white_count"][t0:t0 + 8] = 20        # net occluded as ball passes
    else:
        sig["o_side"][t0 + 2:t0 + 20] += 900          # ball caroms to the side
    return sig


def test_make_vs_miss_features_separate():
    fm = mv.features_from_signals(_signals(True))
    fs = mv.features_from_signals(_signals(False))
    assert fm.shape == (len(mv.FEATURE_NAMES),) and np.all(np.isfinite(fm))
    # netVSflank, o_below higher on the make; o_side higher on the miss
    i = mv.FEATURE_NAMES.index
    assert fm[i("o_below")] > fs[i("o_below")] + 1.0
    assert fs[i("o_side")] > fm[i("o_side")] + 1.0
    assert fm[i("netVSflank")] > fs[i("netVSflank")]


def test_model_roundtrip(tmp_path=None):
    # a separable synthetic set the model must learn
    X, y = [], []
    for k in range(40):
        X.append(mv.features_from_signals(_signals(k % 2 == 0)) + np.random.default_rng(k).normal(0, 0.1, 7))
        y.append(1 if k % 2 == 0 else 0)
    X = np.array(X); y = np.array(y)
    model = mv.train(X, y)
    made, p = mv.predict(model, mv.features_from_signals(_signals(True)))
    assert made is True and p > 0.5
    made2, p2 = mv.predict(model, mv.features_from_signals(_signals(False)))
    assert made2 is False and p2 < 0.5
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), "m.joblib")
    mv.save(model, path)
    m2 = mv.load(path)
    assert mv.predict(m2, X[0])[1] == mv.predict(model, X[0])[1]


if __name__ == "__main__":
    test_make_vs_miss_features_separate()
    test_model_roundtrip()
    print("ok")
