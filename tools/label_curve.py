"""How many labels does a NEW session actually need?

Make/miss does not transfer across sessions (07-29 leave-one-clip-out 89%, the
0720 model 55% on it), so every session needs its own labels and its own re-fit.
That makes labelling cost the gate on everything downstream. Before building any
labelling UI, measure the thing that decides its design: the learning curve.

Three protocols, because they answer different UX questions and DISAGREE:

  A. RANDOM POOL  -- owner labels k shots sampled from across the whole session.
     Evaluated on a FIXED held-out set so the curves are comparable at every k.
  B. WHOLE CLIPS  -- owner labels the first clip(s) only. This is leave-one-clip-
     out's family and is the honest analogue of "label the first few minutes",
     but it confounds sample SIZE with clip DIVERSITY.
  C. UNCERTAINTY  -- active learning: label what the model is least sure of.
     Compared against A at identical k on the SAME fixed test set, which is the
     only fair comparison; scoring active learning on its own biased labelled
     pool is the classic way to fake a win.

Run (SYSTEM python): python -X utf8 tools/label_curve.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools import eval_ablations as E, rim_segments as rs
from shotlab.phase1_ball.track import assemble_track
from shotlab.court import detect_shots_to_rim
from shotlab.phase1_ball.pipeline import _union_beam, _rim_frame
import shotlab.make_visual as mv

CLIPS_0729 = ["PXL_20260729_155320813", "PXL_20260729_155914855-001",
              "PXL_20260729_160743954", "PXL_20260729_161439291-002"]
FEAT_CACHE = ROOT / "data" / "out" / "make_feats"


def clip_features(clip, tol=30):
    """(X, y) of make_visual features + hand-counted labels for one clip. Cached.

    Same construction as tools/refit_make_model.py, parametrised over the clip so
    it works for any hand-counted session.
    """
    FEAT_CACHE.mkdir(parents=True, exist_ok=True)
    cache = FEAT_CACHE / f"{clip}.json"
    if cache.exists():
        d = json.load(open(cache))
        return np.array(d["X"], float), np.array(d["y"], int)
    calib = rs.calib_at(rs.load_rims(clip), 0)
    # via eval_ablations' own loader, NOT a hand-rolled json.load: the cache
    # carries a params identity header (added 08-02) and the loader re-detects on
    # mismatch. Reading the file directly both crashes on the header and would
    # silently accept a cache built under a different detector config.
    raw = E._detect_full_clip(clip)
    gtrack = assemble_track(E._cands_at_conf(raw, 0.25))
    greedy = detect_shots_to_rim(gtrack, calib)
    shots, track = _union_beam(greedy, gtrack, E._cands_at_conf(raw, 0.01), calib)
    att = E.load_attempts(clip)
    vp = str(E.CLIP_DIR / f"{clip}.mp4")
    used, X, y = set(), [], []
    for s in shots:
        rf = _rim_frame(s, calib)
        near = [a for a in att
                if abs(a["rim_frame"] - rf) <= tol and a["attempt_id"] not in used]
        if not near:
            continue
        a = min(near, key=lambda a: abs(a["rim_frame"] - rf))
        used.add(a["attempt_id"])
        feats = mv.shot_features(vp, s, calib, track=track)
        if feats is None:
            continue
        X.append([float(v) for v in feats])
        y.append(1 if a["outcome"] == "make" else 0)
    json.dump({"X": X, "y": y}, open(cache, "w"))
    print(f"  featurised {clip}: {len(y)} shots", flush=True)
    return np.array(X, float), np.array(y, int)


def _fit_score(Xtr, ytr, Xte, yte):
    """Accuracy of a model fitted on (Xtr,ytr), scored on the held-out set.
    Returns None when a fold has one class -- the model is undefined there, and
    silently scoring it would reward degenerate all-one-class predictions."""
    if len(np.unique(ytr)) < 2:
        return None
    return float((mv.train(Xtr, ytr).predict(Xte) == yte).mean())


def main():
    data = {c: clip_features(c) for c in CLIPS_0729}
    X = np.vstack([data[c][0] for c in CLIPS_0729])
    y = np.concatenate([data[c][1] for c in CLIPS_0729])
    clip = np.concatenate([np.full(len(data[c][1]), i)
                           for i, c in enumerate(CLIPS_0729)])
    print(f"\n07-29: {len(y)} labelled shots, {int(y.sum())} makes "
          f"({y.mean():.0%}), {len(CLIPS_0729)} clips")
    maj = max(y.mean(), 1 - y.mean())
    print(f"majority-class baseline = {maj:.0%}  (any curve must beat THIS, not 50%)\n")

    rng = np.random.default_rng(0)
    N_TEST, N_REP = 40, 40

    # ---- A vs C: random vs uncertainty, identical k, identical fixed test set --
    print("A/C: label k shots from the session, predict the rest")
    print(f"    fixed held-out test set of {N_TEST}, {N_REP} repeats, mean +/- sd")
    print(f"    {'k':>5}{'random':>16}{'uncertainty':>16}{'delta':>9}")
    print("    " + "-" * 48)
    ks = [8, 12, 16, 20, 30, 40, 60, 80]
    for k in ks:
        ra, ua = [], []
        for rep in range(N_REP):
            r = np.random.default_rng(1000 + rep)
            idx = r.permutation(len(y))
            te, pool = idx[:N_TEST], idx[N_TEST:]
            if k > len(pool):
                continue
            # --- random ---
            sel = pool[:k]
            s = _fit_score(X[sel], y[sel], X[te], y[te])
            if s is not None:
                ra.append(s)
            # --- uncertainty: seed randomly, then add least-confident ---
            seed_n = min(8, k)
            lab = list(pool[:seed_n])
            rest = list(pool[seed_n:])
            while len(lab) < k and rest:
                if len(np.unique(y[lab])) < 2:
                    lab.append(rest.pop(0))       # cannot fit yet; take next
                    continue
                m = mv.train(X[lab], y[lab])
                p = m.predict_proba(X[rest])[:, 1]
                j = int(np.argmin(np.abs(p - 0.5)))   # closest to the boundary
                lab.append(rest.pop(j))
            s = _fit_score(X[lab], y[lab], X[te], y[te])
            if s is not None:
                ua.append(s)
        if ra and ua:
            d = np.mean(ua) - np.mean(ra)
            print(f"    {k:>5}   {np.mean(ra):.0%} +/- {np.std(ra):.0%}"
                  f"      {np.mean(ua):.0%} +/- {np.std(ua):.0%}   {d:+.0%}")

    # ---- B: whole clips, the "label the first few minutes" scenario -----------
    print("\nB: label WHOLE clips, predict the unlabelled clips")
    print(f"    {'clips labelled':>16}{'shots':>8}{'accuracy':>11}")
    print("    " + "-" * 37)
    from itertools import combinations
    for ncl in (1, 2, 3):
        accs, ns = [], []
        for tr in combinations(range(len(CLIPS_0729)), ncl):
            m = np.isin(clip, tr)
            s = _fit_score(X[m], y[m], X[~m], y[~m])
            if s is not None:
                accs.append(s); ns.append(int(m.sum()))
        if accs:
            print(f"    {ncl:>16}{int(np.mean(ns)):>8}   {np.mean(accs):.0%} "
                  f"+/- {np.std(accs):.0%}   (n={len(accs)} combos)")
    print("\n  B is the honest analogue of 'label the first N minutes', but it")
    print("  confounds SAMPLE SIZE with CLIP DIVERSITY -- compare it to A at the")
    print("  same shot count to see which of the two is actually doing the work.")

    # ---- D: WHEN IS LABELLING WORTH IT AT ALL? -------------------------------
    # A brand-new session can be pre-labelled for FREE at ~78% by the z-scored
    # transfer model (tools/transfer_check.py), which needs zero labels from it.
    # So the operative question is not "how many labels reach 89%" but "at what k
    # does labelling this session finally beat the free option". Below that k,
    # every label collected is wasted effort.
    print("\nD: k labels from THIS session vs the FREE zero-label transfer model")
    from sklearn.linear_model import LogisticRegression

    def _z(X):
        s = X.std(0); s[s < 1e-9] = 1.0
        return (X - X.mean(0)) / s

    other = {}
    for c in ["PXL_20260720_151519220", "PXL_20260720_152319112",
              "PXL_20260720_153054813"]:
        p = FEAT_CACHE / f"{c}.json"
        if p.exists():
            d = json.load(open(p))
            other.setdefault("X", []).extend(d["X"])
            other.setdefault("y", []).extend(d["y"])
    if not other:
        print("    07-20 feature cache missing -- skipped")
        return 0
    Xo, yo = np.array(other["X"], float), np.array(other["y"], int)
    zlr_other = LogisticRegression(max_iter=2000).fit(_z(Xo), yo)

    print(f"    {'k labels':>9}{'GBM refit':>12}{'zLR refit':>12}"
          f"{'free transfer':>15}{'worth it?':>11}")
    print("    " + "-" * 60)
    for k in [0, 8, 16, 24, 32, 48, 64, 80]:
        g, z, t = [], [], []
        for rep in range(N_REP):
            r = np.random.default_rng(2000 + rep)
            idx = r.permutation(len(y))
            te, pool = idx[:N_TEST], idx[N_TEST:]
            if k > len(pool):
                continue
            # the free option: trained on the OTHER session, z-scored on THIS one
            t.append(float((zlr_other.predict(_z(X[te])) == y[te]).mean()))
            if k == 0:
                continue
            sel = pool[:k]
            if len(np.unique(y[sel])) < 2:
                continue
            g.append(float((mv.train(X[sel], y[sel]).predict(X[te]) == y[te]).mean()))
            z.append(float((LogisticRegression(max_iter=2000)
                            .fit(_z(X[sel]), y[sel]).predict(_z(X[te])) == y[te]).mean()))
        tr = np.mean(t) if t else float("nan")
        gm = np.mean(g) if g else float("nan")
        zm = np.mean(z) if z else float("nan")
        best = max([v for v in (gm, zm) if v == v], default=float("nan"))
        verdict = "--" if k == 0 else ("yes" if best > tr else "NO")
        print(f"    {k:>9}{(f'{gm:.0%}' if gm == gm else '--'):>12}"
              f"{(f'{zm:.0%}' if zm == zm else '--'):>12}{tr:>15.0%}{verdict:>11}")
    print("\n    'NO' means the labels collected so far are WORSE than the free")
    print("    zero-label transfer model -- effort spent for negative return.")


if __name__ == "__main__":
    raise SystemExit(main())
