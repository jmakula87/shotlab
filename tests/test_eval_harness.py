"""Coverage for the full-clip attempt-evaluation harness (tools/eval_ablations.py,
tools/rim_segments.py, tools/hand_count.py) built after the 2026-07-22 dual review.

The reviewers' lesson: a green suite that never exercises the consequential logic
is worthless (assemble_track had zero tests). So this bites on the parts that
would silently give wrong recall/precision numbers: the produced<->attempt
matcher, the frame-ranged rim resolution (camera-move safe), and the hand-count
CSV round-trip that feeds the eval.
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import rim_segments as rs
from tools import eval_ablations as E
from tools import hand_count as HC

PASS = 0
TOTAL = 0


def check(name, cond):
    global PASS, TOTAL
    TOTAL += 1
    if cond:
        PASS += 1
    else:
        print(f"FAIL: {name}")


# --- matcher: TP/FP/FN, tolerance boundary, one-to-one, airball accounting ---
attempts = [
    {"attempt_id": 1, "rim_frame": 100, "outcome": "make", "reached": "rim"},
    {"attempt_id": 2, "rim_frame": 500, "outcome": "miss", "reached": "rim"},
    {"attempt_id": 3, "rim_frame": 900, "outcome": "miss", "reached": "airball"},
    {"attempt_id": 4, "rim_frame": 1300, "outcome": "make", "reached": "rim"},
]
produced = [
    {"shot": 1, "rim_frame": 108},   # matches #1 (err 8)
    {"shot": 2, "rim_frame": 905},   # matches #3 airball (err 5)
    {"shot": 3, "rim_frame": 2000},  # false positive
]
tp, fp, fn = E.match(produced, attempts, tol=30)
check("matcher TP set", sorted(t["attempt_id"] for t in tp) == [1, 3])
check("matcher FP set", [p["shot"] for p in fp] == [3])
check("matcher FN set", sorted(a["attempt_id"] for a in fn) == [2, 4])

# tolerance is inclusive at the boundary, exclusive past it
check("tol boundary inclusive",
      len(E.match([{"shot": 9, "rim_frame": 130}], [attempts[0]], tol=30)[0]) == 1)
check("tol boundary exclusive",
      len(E.match([{"shot": 9, "rim_frame": 131}], [attempts[0]], tol=30)[0]) == 0)

# one produced shot cannot satisfy two attempts (no double counting)
two = [{"attempt_id": 1, "rim_frame": 100, "reached": "rim"},
       {"attempt_id": 2, "rim_frame": 108, "reached": "rim"}]
tp2, _, fn2 = E.match([{"shot": 1, "rim_frame": 104}], two, tol=30)
check("one-to-one matching", len(tp2) == 1 and len(fn2) == 1)

# report() computes the airball recall that sizes the attempt-detection prize
r = E.report("t", produced, attempts, 30)
check("report tp/fp/fn", r["tp"] == 2 and r["fp"] == 1 and r["fn"] == 2)
check("report airball recall", r["recall_airball"] == 1 and r["n_airball"] == 1)
check("report precision", abs(r["precision"] - 2 / 3) < 1e-9)

# --- frame-ranged rims: the camera-move case must resolve to the right rim -----
doc = {"clip": "T", "image_w": 1920, "image_h": 1080, "rims": [
    {"rim_x": 1134, "rim_y": 470, "rim_radius_px": 16, "shot_gate_px": 90,
     "f0": 0, "f1": 2000, "note": "early"},
    {"rim_x": 1244, "rim_y": 461, "rim_radius_px": 16, "shot_gate_px": 90,
     "f0": 2000, "f1": None, "note": "late"}]}
check("rim early", rs.calib_at(doc, 100).rim_x == 1134)
check("rim late", rs.calib_at(doc, 5000).rim_x == 1244)
check("rim boundary [f0,f1)", rs.calib_at(doc, 2000).rim_x == 1244)
segs = rs.segments(doc, 6000)
check("segments split at camera move",
      [(a, b, c.rim_x) for a, b, c in segs] == [(0, 2000, 1134), (2000, 6000, 1244)])

# a later-added overlapping rim supersedes (last match wins) -> correction workflow
doc_ov = rs.add_rim("T", 1920, 1080, rim_x=900, rim_y=460, rim_radius_px=18, f0=0)
doc_ov["rims"].append({"rim_x": 950, "rim_y": 460, "rim_radius_px": 18,
                       "shot_gate_px": 90, "f0": 0, "f1": None, "note": "fix"})
check("later rim supersedes on overlap", rs.calib_at(doc_ov, 10).rim_x == 950)

# --- hand-count CSV round-trip + eval reads the same shape --------------------
tmp = Path(tempfile.mkdtemp())
HC.HANDCOUNT_DIR = tmp
E.HANDCOUNT_DIR = tmp
att = []
HC.add_attempt(att, 500, "miss", "rim")
HC.add_attempt(att, 100, "make", "rim")     # earlier frame sorts first
HC.add_attempt(att, 900, "miss", "airball")
check("attempts sort by frame", [int(a["rim_frame"]) for a in att] == [100, 500, 900])
check("attempt ids stable under sort", [int(a["attempt_id"]) for a in att] == [2, 1, 3])
HC.save_attempts("T", att)
rows = E.load_attempts("T")
check("eval reads hand-count CSV", len(rows) == 3 and rows[0]["outcome"] == "make")
check("airball survives round-trip", rows[2]["reached"] == "airball")

print(f"{PASS}/{TOTAL} passed")
sys.exit(0 if PASS == TOTAL else 1)
