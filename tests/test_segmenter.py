"""Coverage for assemble_track + detect_shots_to_rim -- the tracker/segmenter the
2026-07-22 dual review found had ZERO tests (a green suite protected none of it).

Every assertion is built to FAIL on a reverted implementation:
- velocity normalization (the /dt_prev fix): a distractor is parked exactly where
  the OLD un-normalized predictor would land, so picking it flips the test.
- the walk-back gap-stop (new fix): two arcs across a dead-ball void; the buggy
  walk-back fabricates a shot spanning the void.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.phase1_ball.detect import BallCandidate
from shotlab.phase1_ball.track import assemble_track
from shotlab.court import Calibration, detect_shots_to_rim

PASS = 0
TOTAL = 0


def check(name, cond):
    global PASS, TOTAL
    TOTAL += 1
    if not cond:
        print(f"FAIL: {name}")
    else:
        PASS += 1


def C(f, x, y, conf=0.9, r=10.0):
    return BallCandidate(int(f), float(x), float(y), float(r), float(conf))


# --- assemble_track: constant-velocity prediction beats a static distractor ----
# ball moves +10px/frame; at f2 a distractor sits at the PREVIOUS position (0px
# from prev, 10px from the velocity prediction). A predictor that ignores velocity
# (pred=prev) would wrongly grab the distractor.
tr = assemble_track({0: [C(0, 100, 300)], 1: [C(1, 110, 300)],
                     2: [C(2, 120, 300), C(2, 110, 300)]})
check("velocity prediction beats static distractor", tr[2].cx == 120)

# --- assemble_track: PER-FRAME velocity normalization (the /dt_prev fix) --------
# prev_prev@f0=(100), prev@f2=(120): spanned 2 frames -> 10px/frame. At f4 (step 2)
# the true ball is at 140. The OLD bug (raw displacement 20, x step) predicts 160;
# a distractor is parked at 160 so the un-normalized predictor grabs it.
tr2 = assemble_track({0: [C(0, 100, 300)], 2: [C(2, 120, 300)],
                      4: [C(4, 140, 300), C(4, 160, 300)]})
check("per-frame velocity normalization picks 140 not 160", tr2[4].cx == 140)

# --- assemble_track: reset after a gap > max_coast re-seeds on max-conf ---------
# shot A races right; after a long void, shot B appears far away where A's velocity
# could never reach. B must still be seeded/tracked (the predictor must NOT try to
# bridge the void, which would reject B).
cands = {0: [C(0, 100, 300)], 1: [C(1, 300, 300)],        # A: +200/frame
         80: [C(80, 900, 500, conf=0.95), C(80, 100, 100, conf=0.4)],
         81: [C(81, 905, 500)]}
trR = assemble_track(cands)
check("reset seeds max-conf candidate after gap", trR[80].cx == 900)
check("reset continues tracking the new arc", 81 in trR and trR[81].cx == 905)


# --- detect_shots_to_rim: a valid ascending arc reaching the rim = one shot -----
def make_arc(f0, x0, x1, rimx, rimy, apex_y, launch_y, n=21):
    """A downward-opening (in image space, ball rises then falls) flight from a
    launch well below the rim up to near the rim. h = -y is a downward parabola."""
    xs = np.linspace(x0, x1, n)
    # parabola in image-y with min (apex) at x1 (near rim) and launch_y at x0
    A = (launch_y - apex_y) / (x0 - x1) ** 2
    ys = apex_y + A * (xs - x1) ** 2
    return {int(f0 + i): [C(f0 + i, xs[i], ys[i])] for i in range(n)}


calib = Calibration(session="T", image_w=1920, image_h=1080,
                    rim_x=500.0, rim_y=200.0, rim_radius_px=16.0, shot_gate_px=90.0)
arc = make_arc(100, 300, 500, 500, 200, apex_y=195, launch_y=470)
shots = detect_shots_to_rim({f: c[0] for f, c in arc.items()}, calib)
check("clean ascending arc -> exactly one shot", len(shots) == 1)
if shots:
    s = shots[0]
    check("shot frames stay within the arc",
          int(s.frames[0]) >= 100 and int(s.frames[-1]) <= 120)

# a flight that never launches from below the rim (all near rim height) is not a shot
flat = {int(100 + i): C(100 + i, 300 + i * 10, 205) for i in range(21)}
check("near-rim-height roll is not a shot", len(detect_shots_to_rim(flat, calib)) == 0)

# apex-below-rim gate: an arc whose highest point stays BELOW the rim (a post-miss
# bounce/roll) is not a shot, even if it launches from below and reaches the gate.
# Same launch, but the apex sits 20px BELOW the rim (min y = rim_y+20 > rim_y).
bounce = make_arc(100, 300, 500, 500, 200, apex_y=220, launch_y=470)
check("apex-below-rim bounce is rejected",
      len(detect_shots_to_rim({f: c[0] for f, c in bounce.items()}, calib)) == 0)


# --- detect_shots_to_rim: walk-back must NOT cross a dead-ball void (the fix) ---
# Both arcs lie on ONE parabola P (launch x=100/y=470 -> rim x=500/y=195), so the
# cross-void segment RANSAC-fits cleanly -- the bug is not masked by a failed fit:
#   arcA = lower part of P (x 100..280), never reaches the rim (no rim event)
#   void (no detections)
#   arcB = upper part of P (x 340..500), reaches the rim
# Buggy walk-back: from B's rim event it marches back across the void into A and
# fabricates ONE shot spanning ~1900 frames. Fixed: it stops at the void, B alone
# never launched from below the rim, so no phantom cross-possession shot.
A_coef = (470 - 195) / (100 - 500) ** 2


def P(x):
    return 195 + A_coef * (x - 500) ** 2


xsA = np.linspace(100, 280, 19)
arcA = {int(100 + i): C(100 + i, xsA[i], P(xsA[i])) for i in range(19)}
xsB = np.linspace(340, 500, 17)
arcB = {int(2000 + i): C(2000 + i, xsB[i], P(xsB[i])) for i in range(17)}
track = {**arcA, **arcB}
sB = detect_shots_to_rim(track, calib)
spans_void = any(int(s.frames[-1]) - int(s.frames[0]) > 500 for s in sB)
check("walk-back does not fabricate a shot spanning the dead-ball void", not spans_void)
check("all produced shots stay within one arc",
      all(int(s.frames[-1]) - int(s.frames[0]) < 200 for s in sB))

print(f"{PASS}/{TOTAL} passed")
sys.exit(0 if PASS == TOTAL else 1)
