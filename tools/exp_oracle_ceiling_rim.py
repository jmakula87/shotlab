"""STEP-1 experiment 1a (RIM-ANCHORED follow-up): size the attempt-detection prize
through the PRODUCTION shot segmenter.

The original 1a (tools/exp_oracle_ceiling.py) ran the oracle candidate streams through
the GAP-BASED `segment_shots`. But production shot detection is the RIM-ANCHORED
`detect_shots_to_rim` (court.py): it walks a continuous track and cuts a shot each time
the ball rises to the rim from well below. Gap-based segmentation is confounded here --
injecting oracle balls FILLS gaps and can MERGE flights, which is likely why the original
oracle recovered FEWER shots than baseline. This variant re-measures the ceiling through
the segmenter we actually ship, so the "prize a perfect detector buys" number is honest.

Same four candidate streams as 1a:
  baseline  = YOLO @ conf 0.25 (today's pipeline)
  cloud     = YOLO @ conf 0.01 (low-conf candidate cloud)
  oracle    = baseline + a synthetic GT-center candidate on every labeled ball frame
  oracle+n  = oracle with +-4px localization jitter

Rim is auto-detected PER CLIP (detect_rim, median over the decoded labeled frames) --
the tripod may move between clips. Only labeled flight-window frames carry candidates,
so this measures shot RECOVERY within labeled footage, same scope as 1a.

Run:  <system-python> -X utf8 tools/exp_oracle_ceiling_rim.py
"""
from __future__ import annotations
import json, random, collections
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "data" / "labels" / "ball_labels_0720.json"
CLIP_DIR = ROOT / "data" / "raw" / "Camera 1"
WEIGHTS = str(ROOT / "runs" / "detect" / "ball_gpu_kaggle" / "weights" / "best.onnx")

import sys
sys.path.insert(0, str(ROOT))
from shotlab.phase1_ball.detect_yolo import YoloBallDetector
from shotlab.phase1_ball.detect import BallCandidate
from shotlab.phase1_ball.track import assemble_track
from shotlab.court import Calibration, detect_rim, detect_shots_to_rim

random.seed(1234)                      # deterministic +-4px jitter
NOISE_PX = 4.0


def run_pipeline(cands_by_frame, calib):
    """The REAL tracker + the PRODUCTION rim-anchored segmenter."""
    track = assemble_track(cands_by_frame)
    shots = detect_shots_to_rim(track, calib)
    return shots


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
    print(f"\n{'clip':>14s}  {'labeled':>7s} {'GTballs':>7s}  {'rim(x,y,r,gate)':>22s} | "
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
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        base, cloud, oracle, oraclen = {}, {}, {}, {}
        rim_dets = []
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
                # opportunistically detect the (static) rim from frames we already decode
                rd = detect_rim(fr)
                if rd is not None:
                    rim_dets.append(rd)
            idx += 1
        cap.release()

        if len(rim_dets) < 3:
            print(f"  {clip}: rim not found ({len(rim_dets)} dets) -> SKIP"); continue
        arr = np.array(rim_dets)
        cx, cy, half_w = (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])),
                          float(np.median(arr[:, 2])))
        calib = Calibration(
            session=clip, image_w=W, image_h=H,
            rim_x=cx, rim_y=cy, rim_radius_px=half_w,
            shot_gate_px=max(2.0 * half_w, 90.0),
            note=f"auto rim, median of {len(rim_dets)} labeled frames")

        rb = run_pipeline(base, calib)
        rc = run_pipeline(cloud, calib)
        ro = run_pipeline(oracle, calib)
        rn = run_pipeline(oraclen, calib)
        totals["base"] += len(rb); totals["cloud"] += len(rc)
        totals["oracle"] += len(ro); totals["oraclen"] += len(rn)
        totals["labeled"] += len(want); totals["gtballs"] += len(gt)
        rim_s = f"({cx:.0f},{cy:.0f},{half_w:.0f},{calib.shot_gate_px:.0f})"
        print(f"{clip:>14s}  {len(want):>7d} {len(gt):>7d}  {rim_s:>22s} | "
              f"{len(rb):>4d} {len(rc):>5d} {len(ro):>6d} {len(rn):>5d}")

    print("-" * 88)
    print(f"{'TOTAL':>14s}  {totals['labeled']:>7d} {totals['gtballs']:>7d}  {'':>22s} | "
          f"{totals['base']:>4d} {totals['cloud']:>5d} "
          f"{totals['oracle']:>6d} {totals['oraclen']:>5d}")
    dc = totals["cloud"] - totals["base"]
    do = totals["oracle"] - totals["base"]
    dn = totals["oraclen"] - totals["base"]
    print(f"\nDELTA vs baseline:  cloud(@0.01) {dc:+d}   ORACLE {do:+d}   oracle+4px {dn:+d}")
    print("\nINTERPRETATION (rim-anchored / production segmenter):")
    print("  ORACLE delta ~ 0 -> a perfect detector buys ~no extra shots through the")
    print("     shipping segmenter either -> the lever is attempt-detection/film-closer,")
    print("     NOT detection or temporal fusion. Confirms 1a on the production path.")
    print("  ORACLE delta > 0 -> the gap-based 1a UNDERSOLD the prize; a better detector")
    print("     does buy shots once segmentation is rim-anchored -> reconsider fusion.")


if __name__ == "__main__":
    main()
