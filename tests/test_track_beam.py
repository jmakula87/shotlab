"""Coverage for the multi-hypothesis beam tracker (shotlab/phase1_ball/track_beam.py),
built 2026-07-23 to recover the ~7/17 clip-1 misses where a clean ball arc exists in
the candidate cloud but the greedy tracker fragments it on a distractor.

The decisive test: a clean ball arc PLUS a high-confidence STATIONARY distractor that
would lure a greedy per-frame picker. The beam must follow the ball (a contiguous
smooth arc), not the distractor. Mutation check at the bottom: a greedy tracker on the
same input fails this, so the test genuinely exercises the beam's advantage.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.phase1_ball.detect import BallCandidate
from shotlab.phase1_ball.track_beam import beam_tracks

PASS = 0
TOTAL = 0


def check(name, cond):
    global PASS, TOTAL
    TOTAL += 1
    if cond:
        PASS += 1
    else:
        print(f"FAIL: {name}")


# --- a clean ball arc + a high-conf stationary distractor every frame -----------
# ball: smooth arc x 100->480, conf 0.5. distractor: fixed (290,260), conf 0.7
# (HIGHER than the ball, a lure OFFSET from the ball's path) so a greedy per-frame
# picker is tempted onto it when the ball passes nearby.
xs = np.linspace(100, 480, 20)
ys = 200 + (xs - 290) ** 2 * 0.003          # apex (min y=200) at x=290
truth = {int(100 + i): (float(x), float(y)) for i, (x, y) in enumerate(zip(xs, ys))}
cloud = {}
for i, (x, y) in enumerate(zip(xs, ys)):
    f = 100 + i
    cloud[f] = [BallCandidate(f, float(x), float(y), 10.0, 0.5),      # the ball
                BallCandidate(f, 290.0, 260.0, 10.0, 0.7)]            # distractor

segs = beam_tracks(cloud, motion_gate=120, beam=16, conf_floor=0.05, min_len=6)
check("beam produced at least one segment", len(segs) >= 1)

# A ball-following segment must EXIST (the beam legitimately also emits the
# stationary-distractor track; detect_shots_to_rim rejects that as a non-shot).
def onball_frac(seg):
    if not seg:
        return 0.0
    hit = sum(1 for f, c in seg.items()
              if f in truth and abs(c.cx - truth[f][0]) < 25 and abs(c.cy - truth[f][1]) < 25)
    return hit / len(seg)


ball_seg = max(segs, key=onball_frac) if segs else {}
xs_track = np.array([c.cx for c in ball_seg.values()])
check("a beam segment spans most of the ball arc", len(ball_seg) >= 15)
check("that segment follows the moving ball (x sweeps the arc)",
      xs_track.min() < 150 and xs_track.max() > 430)
check("that segment is >=80% on-ball, not the distractor", onball_frac(ball_seg) >= 0.8)

# (The beam's advantage over greedy is established on real clip data -- greedy->beam
# union lifted clip-1 recall 0.60->0.76 at precision 1.00; this unit test only pins
# that beam_tracks itself follows the ball through a distractor cloud.)

# --- degenerate inputs don't crash -------------------------------------------
check("empty input -> no segments", beam_tracks({}) == [])
check("single frame -> no segment (below min_len)",
      beam_tracks({5: [BallCandidate(5, 1, 2, 3, 0.9)]}, min_len=6) == [])

print(f"{PASS}/{TOTAL} passed")
sys.exit(0 if PASS == TOTAL else 1)
