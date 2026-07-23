"""Coverage for the rim-anchored backward recovery pass (shotlab/phase1_ball/
rim_recovery.py), built 2026-07-23 to recover residual misses whose ball IS
detected near the rim but whose arc doesn't survive tracking. Measured +8/111 at
precision 0.99 across the 3 hand-counted clips.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.phase1_ball.detect import BallCandidate
from shotlab.court import Calibration
from shotlab.phase1_ball.rim_recovery import recover_shots

PASS = 0
TOTAL = 0


def check(name, cond):
    global PASS, TOTAL
    TOTAL += 1
    if cond:
        PASS += 1
    else:
        print(f"FAIL: {name}")


calib = Calibration(session="t", image_w=1920, image_h=1080, rim_x=500.0,
                    rim_y=200.0, rim_radius_px=40.0, shot_gate_px=90.0)


def arc_cloud(f0, x0, x1, apex_y, launch_y, n=22, distractor=None):
    """A clean ascending arc in the cloud (+ optional stationary distractor)."""
    xs = np.linspace(x0, x1, n)
    A = (launch_y - apex_y) / (x0 - x1) ** 2
    ys = apex_y + A * (xs - x1) ** 2
    cloud = {}
    for i in range(n):
        f = int(f0 + i)
        cloud[f] = [BallCandidate(f, float(xs[i]), float(ys[i]), 10.0, 0.6)]
        if distractor is not None:
            cloud[f].append(BallCandidate(f, distractor[0], distractor[1], 10.0, 0.8))
    return cloud


# a real launch->rim arc buried in a distractor cloud is recovered
cloud = arc_cloud(100, 300, 500, apex_y=195, launch_y=470, distractor=(700, 500))
shots = recover_shots(cloud, calib, [])
check("recovers a valid launch->rim arc from the cloud", len(shots) == 1)
if shots:
    check("recovered shot is tagged", shots[0].meta.get("source") == "rim_recovery")

# if the rim event is already covered, it is NOT re-emitted
check("does not re-emit an already-seen rim event",
      len(recover_shots(cloud, calib, [120])) == 0)

# an arc whose apex stays BELOW the rim (a bounce/roll) is rejected
bounce = arc_cloud(100, 300, 500, apex_y=230, launch_y=470)   # apex below rim (y=230>200)
check("rejects an apex-below-rim bounce", len(recover_shots(bounce, calib, [])) == 0)

# a short cloud (too few points) yields nothing
short = {100 + i: [BallCandidate(100 + i, 400 + i, 300, 10, 0.6)] for i in range(4)}
check("too-few-points -> no shot", len(recover_shots(short, calib, [])) == 0)

# empty cloud is safe
check("empty cloud -> no shot", recover_shots({}, calib, []) == [])

print(f"{PASS}/{TOTAL} passed")
sys.exit(0 if PASS == TOTAL else 1)
