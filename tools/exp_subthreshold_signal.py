"""STEP-1 experiment 1b+1c: is the flight ball detectable SUB-THRESHOLD, and does
TILING help when the ball is genuinely tiny?  (Codex+Fable review, 2026-07-22)

For every human-labeled ball-present frame (ground-truth center = cx0+px, cy0+py,
radius r), run the CANONICAL onnx detector at conf 0.01 (plain AND tiled) plus the
stateful MotionBallDetector, and ask: does ANY candidate land within 1.5*r of the
true center?  Stratified by ball size (far/mid/close) and, decisively, restricted
to the frames the CURRENT pipeline (conf 0.25, plain) already MISSES.

Read this as: if the sub-threshold / tiled / motion configs recover most of the
conf-0.25 misses -> the signal is already there and the TRACKER is the bottleneck
(no new model needed). If they recover ~nothing -> the floor is real at the logit
level and only THEN is a temporal model justified.

Run:  python -X utf8 tools/exp_subthreshold_signal.py
"""
from __future__ import annotations
import json, math, collections
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "data" / "labels" / "ball_labels_0720.json"
CLIP_DIR = ROOT / "data" / "raw" / "Camera 1"
WEIGHTS = str(ROOT / "runs" / "detect" / "ball_gpu_kaggle" / "weights" / "best.onnx")

import sys
sys.path.insert(0, str(ROOT))
from shotlab.phase1_ball.detect_yolo import YoloBallDetector
from shotlab.phase1_ball.detect import MotionBallDetector

CONFS = [0.01, 0.10, 0.25]           # sweep (detector runs once at 0.01, filter up)
SIZE_BUCKETS = [("far r<=12", 0, 12), ("mid 12-20", 12, 20), ("close >20", 20, 1e9)]


def hit(cands, gx, gy, r, conf_thr):
    """Any candidate above conf_thr within 1.5*r of the GT center?"""
    tol = 1.5 * max(r, 6)
    for c in cands:
        if c.conf >= conf_thr and math.hypot(c.cx - gx, c.cy - gy) <= tol:
            return True
    return False


def bucket(r):
    for name, lo, hi in SIZE_BUCKETS:
        if lo < r <= hi or (lo == 0 and r <= hi):
            return name
    return SIZE_BUCKETS[-1][0]


def main():
    labels = [L for L in json.load(open(LABELS)) if L.get("present")]
    by_clip = collections.defaultdict(list)
    for L in labels:
        by_clip[L["clip"]].append(L)

    det_plain = YoloBallDetector(WEIGHTS, ball_class=None, conf=0.01, imgsz=1280, tiles=None)
    det_tiled = YoloBallDetector(WEIGHTS, ball_class=None, conf=0.01, imgsz=1280, tiles="auto")
    print("provider:", getattr(det_plain, "active_provider", "?"))

    # records: per labeled frame, hit flags for each config
    rows = []
    for clip, Ls in by_clip.items():
        path = CLIP_DIR / f"{clip}.mp4"
        if not path.exists():
            print(f"  MISSING clip {path}"); continue
        want = {L["frame"]: L for L in Ls}
        maxf = max(want)
        cap = cv2.VideoCapture(str(path))
        det_motion = MotionBallDetector()      # fresh state per clip
        idx = 0
        while idx <= maxf:
            ok, fr = cap.read()
            if not ok:
                break
            mcands = det_motion.detect(idx, fr)         # feed EVERY frame (stateful)
            if idx in want:
                L = want[idx]
                gx, gy, r = L["cx0"] + L["px"], L["cy0"] + L["py"], L["r"]
                pc = det_plain.detect(idx, fr)
                tc = det_tiled.detect(idx, fr)
                row = {"clip": clip, "frame": idx, "r": r, "bucket": bucket(r)}
                for t in CONFS:
                    row[f"plain@{t}"] = hit(pc, gx, gy, r, t)
                    row[f"tiled@{t}"] = hit(tc, gx, gy, r, t)
                row["motion"] = hit(mcands, gx, gy, r, 0.0)
                rows.append(row)
            idx += 1
        cap.release()
        print(f"  {clip}: evaluated {sum(1 for r in rows if r['clip']==clip)} present frames")

    n = len(rows)
    print(f"\n=== overall hit-rate on {n} labeled ball-present frames ===")
    cols = [f"plain@{t}" for t in CONFS] + [f"tiled@{t}" for t in CONFS] + ["motion"]
    def rate(subset, col):
        return (100 * sum(r[col] for r in subset) / len(subset)) if subset else 0.0
    print(f"{'config':12s} {'all':>6s}  " + "  ".join(f"{b.split()[0]:>7s}" for b,_,_ in SIZE_BUCKETS))
    for col in cols:
        line = f"{col:12s} {rate(rows,col):>5.0f}% "
        for bname, _, _ in SIZE_BUCKETS:
            sub = [r for r in rows if r["bucket"] == bname]
            line += f" {rate(sub,col):>6.0f}%"
        print(line)

    # THE decisive table: among frames the CURRENT pipeline (plain@0.25) MISSES,
    # what fraction does each other config recover?
    missed = [r for r in rows if not r["plain@0.25"]]
    print(f"\n=== recovery of the {len(missed)} frames the CURRENT pipeline (plain@0.25) MISSES ===")
    print(f"(of those, {sum(1 for r in missed if r['bucket']=='far r<=12')} are far r<=12)")
    for col in ["plain@0.01", "tiled@0.01", "tiled@0.25", "motion"]:
        recov = rate(missed, col)
        farsub = [r for r in missed if r["bucket"] == "far r<=12"]
        print(f"  {col:12s} recovers {recov:>4.0f}% of all misses   {rate(farsub,col):>4.0f}% of far misses")
    print("\nINTERPRETATION: high recovery => signal exists, TRACKER is the bottleneck")
    print("                near-zero    => real logit-level floor, temporal model justified")


if __name__ == "__main__":
    main()
