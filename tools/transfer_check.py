"""Does the make/miss model transfer between sessions, and what fixes it?

The standing claim was "make/miss is SESSION-SPECIFIC, every session needs its
own labels". That is true of the SHIPPED model and false as a statement about the
features. Measured 2026-08-04 on two sessions, both directions:

    transfer            GBM raw      z-scored + logistic     majority
    0720 -> 0729         64.2%            78.1% (AUC .840)     53%
    0729 -> 0720         50.0%            78.4% (AUC .845)     51%

50.0% is exactly chance. A per-session z-score costs 1-4 points WITHIN a session
(LOCO 81.0 vs 84.7 on 0729, 79.5 vs 80.7 on 0720) and buys 14-28 points ACROSS
sessions. So the cue DIRECTIONS are session-invariant; the GBM's split THRESHOLDS
are not -- absolute pixel masses shift with exposure, white balance and rim ROI,
and a tree splits on absolute values while a standardised linear model does not.

Why the z-score is legitimate on an unlabelled session: it uses only that
session's FEATURE distribution, never its labels. It is transductive -- it needs
the session's shots as a batch, and needs enough of them to estimate mean/sd, so
it is not meaningful for a handful of shots.

⚠️ TWO SESSIONS. This is a replicated observation, not an established law, and
78% is still well below the ~85% a within-session re-fit reaches. It is a much
better ZERO-LABEL starting point, not a reason to stop labelling. Re-run this on
the next session BEFORE relying on it -- that check is pre-registered.

Run (SYSTEM python): python -X utf8 tools/transfer_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import shotlab.make_visual as mv

FEAT_CACHE = ROOT / "data" / "out" / "make_feats"
SESSIONS = {
    "0720": ["PXL_20260720_151519220", "PXL_20260720_152319112",
             "PXL_20260720_153054813"],
    "0729": ["PXL_20260729_155320813", "PXL_20260729_155914855-001",
             "PXL_20260729_160743954", "PXL_20260729_161439291-002"],
}


def zscore(X):
    """Label-free per-session standardisation. Zero-variance columns pass through
    rather than exploding -- a constant feature carries no information either way."""
    s = X.std(0)
    s[s < 1e-9] = 1.0
    return (X - X.mean(0)) / s


def load(clips):
    X, y = [], []
    for c in clips:
        p = FEAT_CACHE / f"{c}.json"
        if not p.exists():
            raise SystemExit(f"missing feature cache {p}\n"
                             f"build it with tools/label_curve.py first")
        d = json.load(open(p))
        X += d["X"]; y += d["y"]
    return np.array(X, float), np.array(y, int)


def _fit_zlr(X, y):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=2000).fit(zscore(X), y)


def transfer(a, b):
    from sklearn.metrics import roc_auc_score
    Xa, ya = load(SESSIONS[a]); Xb, yb = load(SESSIONS[b])
    gbm = (mv.train(Xa, ya).predict(Xb) == yb).mean()
    p = _fit_zlr(Xa, ya).predict_proba(zscore(Xb))[:, 1]
    zlr = ((p > 0.5).astype(int) == yb).mean()
    return gbm, zlr, roc_auc_score(yb, p), max(yb.mean(), 1 - yb.mean()), len(yb)


def loco(name):
    """Leave-one-clip-out WITHIN a session: the cost of switching model family."""
    clips = SESSIONS[name]
    d = {c: json.load(open(FEAT_CACHE / f"{c}.json")) for c in clips}
    g = z = n = 0
    for held in clips:
        Xtr = np.array([r for c in clips if c != held for r in d[c]["X"]], float)
        ytr = np.array([r for c in clips if c != held for r in d[c]["y"]], int)
        Xte = np.array(d[held]["X"], float); yte = np.array(d[held]["y"], int)
        g += int((mv.train(Xtr, ytr).predict(Xte) == yte).sum())
        z += int(((_fit_zlr(Xtr, ytr).predict_proba(zscore(Xte))[:, 1] > 0.5)
                  .astype(int) == yte).sum())
        n += len(yte)
    return g / n, z / n, n


def main():
    print("CROSS-SESSION transfer (the case a brand-new session is actually in)")
    print(f"  {'':<16}{'GBM raw':>10}{'z + logistic':>15}{'AUC':>8}"
          f"{'majority':>10}{'n':>6}")
    print("  " + "-" * 65)
    for a, b in (("0720", "0729"), ("0729", "0720")):
        g, z, auc, base, n = transfer(a, b)
        print(f"  {a} -> {b:<9}{g:>9.1%}{z:>15.1%}{auc:>8.3f}{base:>10.1%}{n:>6}")

    print("\nWITHIN-session leave-one-clip-out (what switching costs where you"
          " already have labels)")
    print(f"  {'session':<16}{'GBM raw':>10}{'z + logistic':>15}{'delta':>9}{'n':>6}")
    print("  " + "-" * 56)
    for s in SESSIONS:
        g, z, n = loco(s)
        print(f"  {s:<16}{g:>9.1%}{z:>15.1%}{z-g:>+9.1%}{n:>6}")

    print("\n  Read: the z-scored linear model gives up a little where you have"
          "\n  labels and gains a lot where you do not. Use it for ZERO-LABEL"
          "\n  pre-labelling on a new session; re-fit once real labels exist.")
    print("  ⚠️ Two sessions. Re-run this on the next one before relying on it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
