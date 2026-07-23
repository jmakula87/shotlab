"""Re-fit the make/miss visual model on the NEW hand-count ground truth, leave-one-
clip-out. The shipped model was trained on session_0710; this tests whether a model
trained on THIS footage (89 matched shots across the 3 clips, corrected rims) is
robust across clips -- the honest bar before wiring make_visual into production.

For each matched shot it extracts make_visual features + the true make/miss label,
then LOCO-trains (train on 2 clips, test on the held-out clip). Features are cached
to the scratchpad so re-runs are fast. With --save, also fits on all 89 and writes
models/make_visual_0720.joblib.

Run (SYSTEM python): python -X utf8 tools/refit_make_model.py [--save]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools import eval_ablations as E, rim_segments as rs
from shotlab.phase1_ball.track import assemble_track
from shotlab.court import detect_shots_to_rim
from shotlab.phase1_ball.pipeline import _union_beam, _rim_frame
import shotlab.make_visual as mv

CLIPS = ["PXL_20260720_151519220", "PXL_20260720_152319112", "PXL_20260720_153054813"]
FEAT_CACHE = ROOT / "data" / "out" / "make_feats"


def clip_features(clip, tol=30):
    """(X, y) of make_visual features + labels for matched shots on one clip. Cached."""
    FEAT_CACHE.mkdir(parents=True, exist_ok=True)
    cache = FEAT_CACHE / f"{clip}.json"
    if cache.exists():
        d = json.load(open(cache))
        return np.array(d["X"], float), np.array(d["y"], int)
    calib = rs.calib_at(rs.load_rims(clip), 0)
    raw = {int(k): v for k, v in json.load(open(E.CAND_CACHE / f"{clip}_cloud01.json")).items()}
    gtrack = assemble_track(E._cands_at_conf(raw, 0.25))
    greedy = detect_shots_to_rim(gtrack, calib)
    shots, track = _union_beam(greedy, gtrack, E._cands_at_conf(raw, 0.01), calib)
    att = E.load_attempts(clip)
    vp = str(E.CLIP_DIR / f"{clip}.mp4")
    used, X, y = set(), [], []
    for s in shots:
        rf = _rim_frame(s, calib)
        near = [a for a in att if abs(a["rim_frame"] - rf) <= tol and a["attempt_id"] not in used]
        if not near:
            continue
        a = min(near, key=lambda a: abs(a["rim_frame"] - rf)); used.add(a["attempt_id"])
        feats = mv.shot_features(vp, s, calib, track=track)
        if feats is None:
            continue
        X.append([float(v) for v in feats]); y.append(1 if a["outcome"] == "make" else 0)
    json.dump({"X": X, "y": y}, open(cache, "w"))
    return np.array(X, float), np.array(y, int)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", help="also fit on all 89 -> models/make_visual_0720.joblib")
    args = ap.parse_args(argv)

    data = {c: clip_features(c) for c in CLIPS}
    for c, (X, y) in data.items():
        print(f"{c[-6:]}: {len(y)} labeled shots ({int(y.sum())} makes)")

    print("\nLEAVE-ONE-CLIP-OUT (train on the other 2, test on held-out):")
    tot_correct = tot_n = 0
    # compare to the SHIPPED model too, as a baseline
    shipped = mv.load()
    for held in CLIPS:
        Xtr = np.vstack([data[c][0] for c in CLIPS if c != held])
        ytr = np.concatenate([data[c][1] for c in CLIPS if c != held])
        Xte, yte = data[held]
        model = mv.train(Xtr, ytr)
        pred = model.predict(Xte)
        acc = (pred == yte).mean()
        mr = ((pred == 1) & (yte == 1)).sum() / max(1, (yte == 1).sum())
        sp = (shipped.predict(Xte) == yte).mean()
        tot_correct += int((pred == yte).sum()); tot_n += len(yte)
        print(f"  held-out {held[-6:]}: refit acc {(pred==yte).sum()}/{len(yte)}={acc:.0%} "
              f"(make-recall {mr:.0%})  |  shipped-model acc {sp:.0%}")
    print(f"\nLOCO aggregate (refit): {tot_correct}/{tot_n} = {tot_correct/tot_n:.0%}")

    if args.save:
        X = np.vstack([data[c][0] for c in CLIPS]); y = np.concatenate([data[c][1] for c in CLIPS])
        model = mv.train(X, y)
        out = ROOT / "models" / "make_visual_0720.joblib"
        mv.save(model, str(out)); print(f"saved model trained on all {len(y)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
