"""Coverage for tools/guiview.py -- the display scaling the review GUIs use.

Why this is worth a test file: `verify_rim` turns two CLICKS into the rim center
and radius, and the rim's radius is the unit for make/miss thresholds and
apex-height feet. A display/image scale mismatch is therefore not a cosmetic bug
-- it is the 8px-radius defect that already corrupted make/miss once, and it
reappears the moment footage stops fitting on screen (the 07-29 wide cam is 4K).
So bite on the round trip and on the rim geometry, not just the scale factor.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import guiview as gv

PASS = 0
TOTAL = 0


def check(name, cond):
    global PASS, TOTAL
    TOTAL += 1
    if cond:
        PASS += 1
    else:
        print(f"FAIL: {name}")


# --- scale factor ---
check("720p fits -> no scaling", gv.display_scale(1280, 720) == 1.0)
check("never upscales a small frame", gv.display_scale(640, 360) == 1.0)
check("degenerate size is safe", gv.display_scale(0, 0) == 1.0)
# 1080p is scaled as well: a full-size window overflows a 1080p desktop, the
# window manager resizes it, and then click coords are backend-dependent again.
check("1080p is scaled to fit the desktop",
      abs(gv.display_scale(1920, 1080) - 1600 / 1920) < 1e-9)

s4k = gv.display_scale(3840, 2160)
check("4K is width-limited to max_w", abs(s4k - 1600 / 3840) < 1e-9)
check("tall frame is height-limited", abs(gv.display_scale(1000, 3000) - 900 / 3000) < 1e-9)
check("portrait 1080x1920 scales by height", abs(gv.display_scale(1080, 1920) - 900 / 1920) < 1e-9)

# --- click round trip: a click lands on the pixel it was drawn from ---
ok = True
for ix, iy in [(0, 0), (615.5, 231.0), (1920, 1080), (3839, 2159)]:
    dx, dy = gv.to_display_xy(ix, iy, s4k)
    rx, ry = gv.to_image_xy(dx, dy, s4k)
    if abs(rx - ix) > 1 / s4k or abs(ry - iy) > 1 / s4k:
        ok = False
check("image->display->image round trip within one display pixel", ok)
check("scale 1.0 is identity", gv.to_image_xy(615.5, 231.0, 1.0) == (615.5, 231.0))
check("scale 0 does not divide by zero", gv.to_image_xy(10, 20, 0.0) == (10.0, 20.0))

# --- THE consequential case: edge-to-edge rim span survives the round trip ---
# Clicking the rim's left and right edge on a 4K frame must recover the true
# radius, not a downscaled one. A raw (unmapped) display click would yield
# radius*scale ~= 13px instead of 40px -- the exact make/miss corruption.
left, right = (900.0, 500.0), (980.0, 500.0)      # true span 80px -> radius 40
dl = gv.to_display_xy(*left, s4k)
dr = gv.to_display_xy(*right, s4k)
il = gv.to_image_xy(*dl, s4k)
ir = gv.to_image_xy(*dr, s4k)
radius = ((ir[0] - il[0]) ** 2 + (ir[1] - il[1]) ** 2) ** 0.5 / 2.0
check("mapped clicks recover the true rim radius", abs(radius - 40.0) < 1.0)

raw_radius = ((dr[0] - dl[0]) ** 2 + (dr[1] - dl[1]) ** 2) ** 0.5 / 2.0
check("unmapped display clicks WOULD be wrong (guards the guard)", raw_radius < 20.0)

# --- frame resizing ---
try:
    import numpy as np
    fr = np.zeros((2160, 3840, 3), dtype=np.uint8)
    d = gv.to_display(fr, s4k)
    check("4K frame resized to fit", d.shape[1] == 1600 and d.shape[0] == 900)
    check("no copy when the frame already fits", gv.to_display(fr, 1.0) is fr)
    small = np.zeros((10, 10, 3), dtype=np.uint8)
    check("tiny frame never resized to zero", gv.to_display(small, 0.01).shape[0] >= 1)
except ImportError:                                   # pragma: no cover
    print("NOTE numpy/cv2 unavailable -- skipped frame-resize checks")

print(f"{PASS}/{TOTAL} passed")
raise SystemExit(0 if PASS == TOTAL else 1)
