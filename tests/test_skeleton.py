#!/usr/bin/env python
"""Unit tests for the ideal-skeleton geometry (profile v2).

The heavy end-to-end path (decode clip -> pose -> average) is exercised by
running tools/export_profile.py on a real session; here we pin the pure math:
canonicalization is translation/scale invariant (so shots at different spots and
sizes average cleanly) and the display mapping stays in range and keeps shape.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.skeleton import _canonical, _nominal   # noqa: E402
from shotlab.phase2_pose.pose import L               # noqa: E402

_p = 0


def check(cond, msg):
    global _p
    print(("PASS" if cond else "FAIL") + "  " + msg)
    assert cond, msg
    _p += 1


class FP:                       # minimal FramePose stand-in
    def __init__(self, xy, vis):
        self.xy = np.asarray(xy, float)
        self.vis = np.asarray(vis, float)


def _make_pose(cx, cy, scale, w, h):
    """A simple 33-joint pose placed at (cx,cy) in pixels, torso size ~scale."""
    xy = np.zeros((33, 2))
    xy[L["l_shoulder"]] = [cx - 0.15 * scale, cy]
    xy[L["r_shoulder"]] = [cx + 0.15 * scale, cy]
    xy[L["l_hip"]] = [cx - 0.10 * scale, cy + 0.5 * scale]
    xy[L["r_hip"]] = [cx + 0.10 * scale, cy + 0.5 * scale]
    xy[L["r_elbow"]] = [cx + 0.30 * scale, cy - 0.10 * scale]
    xy[L["r_wrist"]] = [cx + 0.35 * scale, cy - 0.35 * scale]
    xy[L["r_knee"]] = [cx + 0.10 * scale, cy + 0.8 * scale]
    xy[L["r_ankle"]] = [cx + 0.10 * scale, cy + 1.1 * scale]
    return FP(xy, np.ones(33))


W, H = 1920, 1080

# same body geometry, different frame position AND size
a = _make_pose(400, 500, 300, W, H)
b = _make_pose(1200, 300, 500, W, H)

ca, _ = _canonical(a, W, H)
cb, _ = _canonical(b, W, H)

# shoulder midpoint lands at the origin
s_mid = (ca[L["l_shoulder"]] + ca[L["r_shoulder"]]) / 2
check(np.allclose(s_mid, [0, 0], atol=1e-9), "canonical centers shoulders at origin")

# shoulder->hip length normalizes to 1
h_mid = (ca[L["l_hip"]] + ca[L["r_hip"]]) / 2
check(abs(np.hypot(*(s_mid - h_mid)) - 1.0) < 1e-9, "shoulder->hip length == 1")

# translation + scale invariance: identical body -> identical canonical shape.
# Compare only joints _make_pose actually sets (unset joints sit at pixel (0,0),
# which canonicalizes differently per frame -- not part of the invariance claim).
_set = [L[n] for n in ("l_shoulder", "r_shoulder", "l_hip", "r_hip",
                       "r_elbow", "r_wrist", "r_knee", "r_ankle")]
check(np.allclose(ca[_set], cb[_set], atol=1e-6),
      "canonical is translation + scale invariant")

# degenerate torso -> None (no divide-by-zero)
flat = FP(np.zeros((33, 2)), np.ones(33))
check(_canonical(flat, W, H) is None, "degenerate torso returns None")

# display mapping stays in [0,1] and preserves relative shape (affine)
disp = _nominal(ca)
check(disp[_set].min() >= 0 and disp[_set].max() <= 1,
      "display coords stay within [0,1]")
# elbow is to the right of and above the shoulder in both spaces (shape kept)
check((disp[L["r_elbow"]][0] > disp[L["r_shoulder"]][0]) and
      (disp[L["r_elbow"]][1] < disp[L["r_hip"]][1]),
      "display mapping preserves limb layout")

print(f"\n{_p}/{_p} passed")
