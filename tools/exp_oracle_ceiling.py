"""STEP-1 experiment 1a: the ORACLE CEILING — how many shots does the REAL tracker
(assemble_track + segment_shots) recover if the detector NEVER misses a labeled ball
frame?  That delta is the maximum any temporal model / tracker fix could ever buy.

For each labeled clip we build the per-frame candidate stream three ways and run the
ACTUAL pipeline (shotlab.phase1_ball.track) on each:

  baseline  = YOLO @ conf 0.25 (what the pipeline uses today)
  cloud     = YOLO @ conf 0.01 (the low-conf candidate cloud Step 2 would keep)
  oracle    = baseline PLUS a synthetic high-conf candidate at the GROUND-TRUTH
              center on every labeled ball-present frame (detector never misses)
  oracle+n  = oracle, but each injected center jittered +-4px (localization noise
              a real temporal model would have) -> does RANSAC still fit?

We only have ground truth on the labeled flight-window frames, so this measures shot
RECOVERY within the labeled footage (each flight window ~ one shot), NOT a full-clip
shot count.  Read it as: oracle-minus-baseline = shots the tracker drops purely
because the detector missed frames = the prize a better detector/temporal model buys.
If that delta is ~0, Step 2's expensive candidate-cloud/RANSAC fusion buys little and
the lever is elsewhere (film closer).  Large delta => the fusion build is worth it.

Run:  <system-python> -X utf8 tools/exp_oracle_ceiling.py
"""
from __future__ import annotations
import json, random, collections
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "data" / "labels" / "ball_labels_0720.json"
CLIP_DIR = ROOT / "data" / "raw" / "Camera 1"
WEIGHTS = str(ROOT / "runs" / "detect" / "ball_gpu_kaggle" / "weights" / "best.onnx")

import sys
sys.path.insert(0, str(ROOT))
from shotlab.phase1_ball.detect_yolo import YoloBallDetector
from shotlab.phase1_ball.detect import BallCandidate
from shotlab.phase1_ball.track import assemble_track, segment_shots

random.seed(1234)                      # deterministic +-4px jitter
NOISE_PX = 4.0


def run_pipeline(cands_by_frame):
    """The REAL tracker + segmenter. Returns (shots, n_points_in_shots, n_fit_fails)."""
    track = assemble_track(cands_by_frame)
    shots = segment_shots(track)
    npts = sum(len(s.frames) for s in shots)
    return shots, npts


def main():
    all_labels = json.load(open(LABELS))
    present = [L for L in all_labels if L.get("present")]
    by_clip_present = collections.defaultdict(list)
    by_clip_all = collections.defaultdict(list)
    for L in all_labels:
        by_clip_all[L["clip"]].append(L)
    for L in present:
        by_clip_present[L["clip"]].append(L)

    det025 = YoloBallDetector(WEIGHTS, ball_class=None, conf=0.25, imgsz=1280, tiles=None)
    det001 = YoloBallDetector(WEIGHTS, ball_class=None, conf=0.01, imgsz=1280, tiles=None)
    print("provider:", getattr(det025, "active_provider", "?"))

    totals = collections.Counter()
    print(f"\n{'clip':>14s}  {'labeled':>7s} {'GTballs':>7s} | "
          f"{'base':>4s} {'cloud':>5s} {'oracle':>6s} {'orc+n':>5s}   (shots)")

    for clip in sorted(by_clip_all):
        Ls_all = by_clip_all[clip]
        gt = {L["frame"]: (L["cx0"] + L["px"], L["cy0"] + L["py"], L["r"])
              for L in by_clip_present[clip]}
        want = {L["frame"] for L in Ls_all}          # every labeled frame (present+absent)
        path = CLIP_DIR / f"{clip}.mp4"
        if not path.exists():
            print(f"  MISSING {path}"); continue
        maxf = max(want)
        cap = cv2.VideoCapture(str(path))

        base, cloud, oracle, oraclen = {}, {}, {}, {}
        idx = 0
        while idx <= maxf:
            ok, fr = cap.read()
            if not ok:
                break
            if idx in want:
                c025 = det025.detect(idx, fr)
                c001 = det001.detect(idx, fr)
                base[idx] = list(c025)
                cloud[idx] = list(c001)
                oracle[idx] = list(c025)
                oraclen[idx] = list(c025)
                if idx in gt:
                    gx, gy, r = gt[idx]
                    oracle[idx].append(BallCandidate(idx, gx, gy, r, 0.99))
                    jx = gx + random.uniform(-NOISE_PX, NOISE_PX)
                    jy = gy + random.uniform(-NOISE_PX, NOISE_PX)
                    oraclen[idx].append(BallCandidate(idx, jx, jy, r, 0.99))
            idx += 1
        cap.release()

        rb, _ = run_pipeline(base)
        rc, _ = run_pipeline(cloud)
        ro, _ = run_pipeline(oracle)
        rn, _ = run_pipeline(oraclen)
        totals["base"] += len(rb); totals["cloud"] += len(rc)
        totals["oracle"] += len(ro); totals["oraclen"] += len(rn)
        totals["labeled"] += len(want); totals["gtballs"] += len(gt)
        print(f"{clip:>14s}  {len(want):>7d} {len(gt):>7d} | "
              f"{len(rb):>4d} {len(rc):>5d} {len(ro):>6d} {len(rn):>5d}")

    print("-" * 70)
    print(f"{'TOTAL':>14s}  {totals['labeled']:>7d} {totals['gtballs']:>7d} | "
          f"{totals['base']:>4d} {totals['cloud']:>5d} "
          f"{totals['oracle']:>6d} {totals['oraclen']:>5d}")
    dc = totals["cloud"] - totals["base"]
    do = totals["oracle"] - totals["base"]
    dn = totals["oraclen"] - totals["base"]
    print(f"\nDELTA vs baseline:  cloud(@0.01) {dc:+d}   ORACLE {do:+d}   oracle+4px {dn:+d}")
    print("\nINTERPRETATION:")
    print("  ORACLE delta  = ceiling of shots a perfect detector/temporal model buys.")
    print("  cloud delta   = shots the CURRENT detector already sees sub-threshold that")
    print("                  the (now-fixed) tracker can stitch -> cheap Step-2 win.")
    print("  oracle+4px    = how much of the ceiling survives realistic localization noise.")
    print("  If ORACLE delta ~ 0 -> detection isn't the bottleneck here; lever = film closer.")


if __name__ == "__main__":
    main()
