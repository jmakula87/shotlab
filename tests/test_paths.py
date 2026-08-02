"""Close-cam footage must resolve to the ROTATION-CORRECTED clips.

The 07-29 close cam was mounted inverted and its files carry a bogus -90 rotation
flag, so every renderer shows them sideways; corrected copies live in
data/raw/Camera 2/upright/. MediaPipe on a sideways body degrades quietly rather
than failing, so reading the wrong directory produces plausible, wrong pose. Three
tools each had their own default; this pins the single resolver they now share.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab import paths

PASS = 0
TOTAL = 0


def check(name, cond):
    global PASS, TOTAL
    TOTAL += 1
    if cond:
        PASS += 1
    else:
        print(f"FAIL: {name}")


with tempfile.TemporaryDirectory() as td:
    cam2 = os.path.join(td, "data", "raw", "Camera 2")
    os.makedirs(cam2)

    # no upright/ yet -> the raw directory, so older sessions keep working
    check("falls back to the raw close-cam dir",
          paths.close_cam_dir(td) == cam2)
    check("...which is not flagged as corrected",
          paths.close_cam_is_corrected(paths.close_cam_dir(td)) is False)

    os.makedirs(os.path.join(cam2, "upright"))
    check("prefers upright/ once it exists",
          paths.close_cam_dir(td) == os.path.join(cam2, "upright"))
    check("...and is flagged as corrected",
          paths.close_cam_is_corrected(paths.close_cam_dir(td)) is True)

    check("wide cam is unaffected",
          paths.wide_cam_dir(td) == os.path.join(td, "data", "raw", "Camera 1"))
    check("a trailing separator doesn't fool the corrected check",
          paths.close_cam_is_corrected(os.path.join(cam2, "upright") + os.sep) is True)

# the tools must all resolve through this module rather than hard-coding a path
_tools = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
for name in ("flare_report.py", "cut_review_clips.py", "join_flare_to_shots.py"):
    src = open(os.path.join(_tools, name), encoding="utf-8").read()
    check(f"{name} resolves the close cam via shotlab.paths",
          "paths.close_cam_dir(" in src)

print(f"{PASS}/{TOTAL} passed")
sys.exit(0 if PASS == TOTAL else 1)
