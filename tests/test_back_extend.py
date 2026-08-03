"""Backward track extension must recover flight the forward tracker dropped -- and
must NOT wander onto a distractor.

Built after measuring that every one of 34 missed 07-29 attempts was the segmenter
seeing too little of the flight (detection and tracking both scored zero misses).
These tests are synthetic on purpose: a known parabola means the right answer is
known exactly, so a regression shows up as a number rather than as a vibe.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.back_extend import extend_backward

PASS = 0
TOTAL = 0


def check(name, cond):
    global PASS, TOTAL
    TOTAL += 1
    if cond:
        PASS += 1
    else:
        print(f"FAIL: {name}")


class C:
    def __init__(self, cx, cy, r=20.0, conf=0.9):
        self.cx, self.cy, self.r, self.conf = cx, cy, r, conf


RR = 120.0                      # a 4K rim


def arc(n=40, f0=1000):
    """A real-looking flight: x linear, y quadratic, apex above the rim."""
    f = np.arange(f0, f0 + n)
    t = f - f0
    x = 300.0 + 22.0 * t
    y = 1400.0 - 60.0 * t + 1.6 * t ** 2
    return f, x, y


def cloud_from(f, x, y, distractor=None, drop=()):
    cl = {}
    for fi, xi, yi in zip(f, x, y):
        if fi in drop:
            cl.setdefault(int(fi), [])
            continue
        cl.setdefault(int(fi), []).append(C(xi, yi))
    if distractor is not None:
        dx, dy = distractor
        for fi in f:
            cl.setdefault(int(fi), []).append(C(dx, dy, conf=0.95))
    return cl


# --- recovers the flight the tracker dropped -------------------------------
f, x, y = arc()
cl = cloud_from(f, x, y)
keep = 8                                        # tracker only held the last 8 frames
af, ax, ay, ar = extend_backward(f[-keep:], x[-keep:], y[-keep:], cl, RR)
check("recovers the dropped early flight", len(af) >= 25)
check("added points are oldest-first", list(af) == sorted(af))
check("added points precede the segment", max(af) < f[-keep])
if len(af):
    err = max(abs(px - x[list(f).index(fi)]) + abs(py - y[list(f).index(fi)])
              for fi, px, py in zip(af, ax, ay))
    check("recovered positions match the true arc", err < 1e-6)

# --- does not wander onto a stationary distractor --------------------------
cl_d = cloud_from(f, x, y, distractor=(2200.0, 1500.0))
af2, ax2, ay2, _ = extend_backward(f[-keep:], x[-keep:], y[-keep:], cl_d, RR)
check("ignores a bright stationary distractor",
      all(abs(px - 2200.0) > 1.0 for px in ax2))
check("still recovers the arc with a distractor present", len(af2) >= 25)

# --- bridges a short dropout, stops at a long one --------------------------
cl_gap = cloud_from(f, x, y, drop=set(range(1020, 1024)))     # 4-frame hole
af3, *_ = extend_backward(f[-keep:], x[-keep:], y[-keep:], cl_gap, RR)
check("bridges a 4-frame dropout", min(af3) < 1020)

cl_big = cloud_from(f, x, y, drop=set(range(1012, 1026)))     # 14 > MAX_COAST
af4, *_ = extend_backward(f[-keep:], x[-keep:], y[-keep:], cl_big, RR)
check("stops at a dropout longer than the coast limit", min(af4) >= 1026)

# --- the gate scales with the rim ------------------------------------------
# a 4K-sized wander is inside a 4K gate but outside a 0720-sized one
off = cloud_from(f, x, y)
for fi in list(off):
    if fi < 1032:
        off[fi] = [C(x[list(f).index(fi)] + 60.0, y[list(f).index(fi)])]
a_big, *_ = extend_backward(f[-keep:], x[-keep:], y[-keep:], off, 120.0)
a_small, *_ = extend_backward(f[-keep:], x[-keep:], y[-keep:], off, 30.0)
check("a large rim admits a proportionally larger wander", len(a_big) > len(a_small))

# --- degenerate inputs are safe --------------------------------------------
check("empty cloud is safe", extend_backward(f[-4:], x[-4:], y[-4:], {}, RR)[0] == [])
check("zero rim radius is safe", extend_backward(f[-4:], x[-4:], y[-4:], cl, 0.0)[0] == [])
check("empty segment is safe", extend_backward([], [], [], cl, RR)[0] == [])

# --- stop_below_y halts at release height ----------------------------------
af5, _, ay5, _ = extend_backward(f[-keep:], x[-keep:], y[-keep:], cl, RR,
                                 stop_below_y=1300.0)
check("stops once below the release row", len(af5) and max(ay5) >= 1300.0)
check("...and stops EARLIER than an unbounded search", len(af5) < len(af))

print(f"{PASS}/{TOTAL} passed")
sys.exit(0 if PASS == TOTAL else 1)
