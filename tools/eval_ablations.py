"""Full-clip, HAND-COUNTED attempt evaluation with staged ablations.

The decisive experiment both adversarial reviewers (Codex + Fable, 2026-07-22)
specified after retiring the Step-1 oracle family: stop quoting unmatched
single-digit output counts inside detector-selected label windows. Instead:

  1. The owner hand-counts every attempt in a FULL clip (fresh, NOT seeded from
     detections): tools/hand_count.py -> process/handcount/<clip>_attempts.csv.
  2. The owner sets ONE (or, for a clip where the camera moved, a few
     frame-ranged) manually-verified rim(s): tools/verify_rim.py -> config/rim_<clip>.json.
  3. This runs the production shot pipeline over the full clip and MATCHES its
     output to the hand count, reporting recall AND precision -- with real
     denominators, split by rim-reached vs airball attempts.

Conditions (Codex's 5-stage ablation):
  C1 baseline: YOLO@0.25 -> assemble_track -> detect_shots_to_rim   [runnable now]
  C2 cloud:    YOLO@0.01 -> assemble_track -> detect_shots_to_rim   [runnable now]
  C3 oracle-assoc: baseline cands + GT-nearest association -> segmenter [needs dense GT]
  C4 oracle-track: GT-only track -> segmenter                          [needs dense GT]
  C5 arc-only:     GT attempt windows -> arc fit                       [needs dense GT]
C1 vs C2 sizes the cheap detection lever; C2->C4 sizes tracking; C4->C5 sizes
segmentation/RANSAC. C3-C5 need dense per-attempt ball labels the current
label set (flight-windows-only, selection-biased) does not provide; they are
stubbed until a dense GT track is supplied via --gt-track.

Run (SYSTEM python for ONNX-DirectML):
  python -X utf8 tools/eval_ablations.py --clip PXL_20260720_151519220
  python -X utf8 tools/eval_ablations.py --selftest      # matching logic, no YOLO
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools import rim_segments as rs

CLIP_DIR = ROOT / "data" / "raw" / "Camera 1"
HANDCOUNT_DIR = ROOT / "process" / "handcount"
WEIGHTS = str(ROOT / "runs" / "detect" / "ball_gpu_kaggle" / "weights" / "best.onnx")
CAND_CACHE = ROOT / "data" / "out" / "eval_cands"


# ------------------------------------------------------------------ matching
def match(produced, attempts, tol):
    """Global nearest matching between produced shots and hand-counted attempts.

    produced: list of {"shot": int, "rim_frame": int}
    attempts: list of {"attempt_id","rim_frame","outcome","reached"}
    A pair is eligible if |rim_frame difference| <= tol; assign closest-first,
    each shot/attempt used once. Returns (tp_pairs, fp_shots, fn_attempts).
    """
    pairs = []
    for a in attempts:
        for p in produced:
            d = abs(int(p["rim_frame"]) - int(a["rim_frame"]))
            if d <= tol:
                pairs.append((d, a["attempt_id"], p["shot"]))
    pairs.sort()
    used_a, used_p, tp = set(), set(), []
    for d, aid, sid in pairs:
        if aid in used_a or sid in used_p:
            continue
        used_a.add(aid); used_p.add(sid)
        tp.append({"attempt_id": aid, "shot": sid, "frame_err": d})
    fp = [p for p in produced if p["shot"] not in used_p]
    fn = [a for a in attempts if a["attempt_id"] not in used_a]
    return tp, fp, fn


def report(name, produced, attempts, tol):
    tp, fp, fn = match(produced, attempts, tol)
    reached = [a for a in attempts if a.get("reached") == "rim"]
    airball = [a for a in attempts if a.get("reached") == "airball"]
    tp_ids = {t["attempt_id"] for t in tp}
    tp_reached = sum(1 for a in reached if a["attempt_id"] in tp_ids)
    tp_air = sum(1 for a in airball if a["attempt_id"] in tp_ids)
    n_prod = len(produced)
    prec = len(tp) / n_prod if n_prod else float("nan")
    rec_all = len(tp) / len(attempts) if attempts else float("nan")
    rec_reached = tp_reached / len(reached) if reached else float("nan")
    print(f"\n[{name}]  produced={n_prod}  attempts={len(attempts)} "
          f"(rim={len(reached)} airball={len(airball)})")
    print(f"   matched(TP)={len(tp)}  false_pos={len(fp)}  missed(FN)={len(fn)}")
    print(f"   precision={prec:.2f}  recall_all={rec_all:.2f}  "
          f"recall_rim-reached={rec_reached:.2f}  "
          f"recall_airball={tp_air}/{len(airball)}  <- airball recall sizes the "
          f"attempt-detection prize (rim-gated pipeline is blind to these)")
    if fn:
        miss = ", ".join(f"{a['attempt_id']}@{a['rim_frame']}({a.get('reached','?')})"
                         for a in fn[:12])
        print(f"   MISSED attempts: {miss}{' ...' if len(fn) > 12 else ''}")
    return {"condition": name, "produced": n_prod, "tp": len(tp), "fp": len(fp),
            "fn": len(fn), "precision": prec, "recall_all": rec_all,
            "recall_rim": rec_reached, "recall_airball": tp_air,
            "n_airball": len(airball)}


# ------------------------------------------------------------------ io
def load_attempts(clip):
    p = HANDCOUNT_DIR / f"{clip}_attempts.csv"
    if not p.exists():
        raise SystemExit(f"no hand count at {p} -- run tools/hand_count.py first")
    out = []
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append({"attempt_id": int(row["attempt_id"]),
                        "rim_frame": int(row["rim_frame"]),
                        "outcome": row.get("outcome", "").strip(),
                        "reached": row.get("reached", "").strip()})
    return out


# ------------------------------------------------------------------ pipeline
def _shot_rim_frame(shot, calib):
    d = np.hypot(np.asarray(shot.xs) - calib.rim_x, np.asarray(shot.ys) - calib.rim_y)
    return int(np.asarray(shot.frames)[int(np.argmin(d))])


def _segment_full_clip(cands_by_frame, rim_doc, n_frames):
    """Production segmentation over a full-clip candidate stream, honoring
    frame-ranged rims (camera-move safe). Returns produced-shot event list."""
    from shotlab.phase1_ball.track import assemble_track
    from shotlab.court import detect_shots_to_rim
    track = assemble_track(cands_by_frame)
    produced, sid = [], 0
    for f0, f1, calib in rs.segments(rim_doc, n_frames):
        sub = {f: c for f, c in track.items() if f0 <= f < f1}
        if not sub:
            continue
        for s in detect_shots_to_rim(sub, calib):
            sid += 1
            produced.append({"shot": sid, "rim_frame": _shot_rim_frame(s, calib),
                             "first_frame": int(s.frames[0]),
                             "last_frame": int(s.frames[-1])})
    return produced


def _detect_full_clip(clip):
    """Detect the ball every frame at conf 0.01 (cloud). Baseline @0.25 is a
    subset (filter by conf), so ONE YOLO pass serves both conditions. Cached."""
    import cv2
    from shotlab.phase1_ball.detect_yolo import YoloBallDetector
    CAND_CACHE.mkdir(parents=True, exist_ok=True)
    cache = CAND_CACHE / f"{clip}_cloud01.json"
    if cache.exists():
        print(f"  using cached candidates {cache.name}")
        raw = json.load(open(cache))
        return {int(k): v for k, v in raw.items()}
    path = CLIP_DIR / f"{clip}.mp4"
    if not path.exists():
        raise SystemExit(f"clip not found: {path}")
    det = YoloBallDetector(WEIGHTS, ball_class=None, conf=0.01, imgsz=1280, tiles=None)
    print(f"  provider: {getattr(det, 'active_provider', '?')}  detecting {path.name} ...")
    cap = cv2.VideoCapture(str(path))
    out, idx = {}, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        cs = det.detect(idx, fr)
        out[idx] = [[float(c.cx), float(c.cy), float(c.r), float(c.conf)] for c in cs]
        if idx % 1000 == 0:
            print(f"    frame {idx}")
        idx += 1
    cap.release()
    json.dump({str(k): v for k, v in out.items()}, open(cache, "w"))
    print(f"  cached {idx} frames -> {cache.name}")
    return out


def _cands_at_conf(raw, conf):
    from shotlab.phase1_ball.detect import BallCandidate
    out = {}
    for f, lst in raw.items():
        keep = [BallCandidate(int(f), x, y, r, c) for x, y, r, c in lst if c >= conf]
        out[int(f)] = keep
    return out


def run(clip, tol):
    from shotlab.video_io import probe
    rim_doc = rs.load_rims(clip)
    if rim_doc is None:
        raise SystemExit(f"no verified rim at {rs.rim_path(clip)} -- run tools/verify_rim.py first")
    attempts = load_attempts(clip)
    info = probe(str(CLIP_DIR / f"{clip}.mp4"))
    n_frames = info.n_frames
    print(f"clip {clip}: {n_frames} frames, {len(attempts)} hand-counted attempts, "
          f"{len(rim_doc['rims'])} rim segment(s), match tol +-{tol}f")
    raw = _detect_full_clip(clip)
    rows = []
    for name, conf in [("C1 baseline@0.25", 0.25), ("C2 cloud@0.01", 0.01)]:
        cands = _cands_at_conf(raw, conf)
        produced = _segment_full_clip(cands, rim_doc, n_frames)
        rows.append(report(name, produced, attempts, tol))
    print("\n(C3 oracle-assoc / C4 oracle-track / C5 arc-only need a dense per-attempt "
          "GT ball track -- supply via --gt-track once labeled; stubbed for now.)")
    out = HANDCOUNT_DIR / f"{clip}_eval.json"
    json.dump({"clip": clip, "tol": tol, "n_attempts": len(attempts), "conditions": rows},
              open(out, "w"), indent=2)
    print(f"\nwrote {out}")


# ------------------------------------------------------------------ selftest
def _selftest():
    attempts = [
        {"attempt_id": 1, "rim_frame": 100, "outcome": "make", "reached": "rim"},
        {"attempt_id": 2, "rim_frame": 500, "outcome": "miss", "reached": "rim"},
        {"attempt_id": 3, "rim_frame": 900, "outcome": "miss", "reached": "airball"},
        {"attempt_id": 4, "rim_frame": 1300, "outcome": "make", "reached": "rim"},
    ]
    # produced: hits 1 (off by 8), 2 (off by 40 -> beyond tol=30 -> miss), a bogus
    # rim event near the airball (attempt 3, off by 5 -> counts as a match), and a
    # pure false positive at 2000.
    produced = [
        {"shot": 1, "rim_frame": 108},
        {"shot": 2, "rim_frame": 905},
        {"shot": 3, "rim_frame": 2000},
    ]
    tp, fp, fn = match(produced, attempts, tol=30)
    ids = sorted(t["attempt_id"] for t in tp)
    assert ids == [1, 3], ids
    assert sorted(p["shot"] for p in fp) == [3], fp
    assert sorted(a["attempt_id"] for a in fn) == [2, 4], fn
    # tie-break: closest pair wins uniquely
    a2 = [{"attempt_id": 1, "rim_frame": 100, "reached": "rim"},
          {"attempt_id": 2, "rim_frame": 110, "reached": "rim"}]
    p2 = [{"shot": 1, "rim_frame": 105}]     # equidistant-ish; only one can match
    tp2, fp2, fn2 = match(p2, a2, tol=30)
    assert len(tp2) == 1 and len(fn2) == 1, (tp2, fn2)
    r = report("selftest", produced, attempts, 30)
    assert r["tp"] == 2 and r["fp"] == 1 and r["fn"] == 2
    assert r["recall_airball"] == 1 and r["n_airball"] == 1
    print("\neval_ablations selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", help="clip stem, e.g. PXL_20260720_151519220")
    ap.add_argument("--tol", type=int, default=30,
                    help="match tolerance in frames between produced & hand-counted rim frame")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        _selftest(); return 0
    if not args.clip:
        ap.error("--clip required (or --selftest)")
    run(args.clip, args.tol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
