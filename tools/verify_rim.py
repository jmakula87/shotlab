"""Set ONE manually-verified rim -- or several frame-ranged rims when the camera
moved mid-clip (the 2026-07-22 review found clip1 has a 110px within-clip rim
shift). Writes config/rim_<clip>.json (see tools/rim_segments.py), consumed by
tools/eval_ablations.py.

WHY manual: auto rim-detection (court.detect_rim, orange blob) is a confound --
on this footage it disagreed with the cached rim by 110/225/59px. The tripod
moves only a few times, so clicking the rim is cheap and removes the confound.

GUI (a window opens on the clip):
  navigate:  d/a = +/-1 frame,  e/q = +/-30,  c/z = +/-300,  g = jump to frame#
  set rim:   left-click the rim's LEFT edge, then its RIGHT edge. Center = the
             midpoint, radius = HALF the span. (Do NOT click center+near-center --
             that gave a 4-6x-too-small radius that corrupts make/miss + apex
             scaling.) The rim is ADDED as soon as both edges land (no ENTER); a
             ball-diameter sanity check prints if the radius looks wrong. It covers
             from the current frame to clip end; add a 2nd rim later only if the
             camera moved mid-clip (it supersedes for later frames).
  r = clear a half-click   x = delete last added rim   s = save   ESC/close = save & quit
Headless (no GUI): --rim X Y --radius R [--f0 N]   appends one entry, f0..end.

Usage:
  python -X utf8 tools/verify_rim.py --clip PXL_20260720_151519220
  python -X utf8 tools/verify_rim.py --clip C --rim 1134 470 --radius 16 --f0 0
  python -X utf8 tools/verify_rim.py --selftest
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools import rim_segments as rs
from shotlab.video_io import probe

CLIP_DIR = ROOT / "data" / "raw" / "Camera 1"


def _median_ball_diameter(clip):
    """Median detected ball diameter (px) from the eval candidate cloud, if cached
    -- used only for a rim-radius sanity check. Returns None if unavailable."""
    import json
    import numpy as np
    cache = ROOT / "data" / "out" / "eval_cands" / f"{clip}_cloud01.json"
    if not cache.exists():
        return None
    raw = json.load(open(cache))
    rads = [c[2] for cs in raw.values() for c in cs if c[3] >= 0.4]
    return float(np.median(rads)) * 2.0 if rads else None


def _clip_path(clip):
    p = CLIP_DIR / f"{clip}.mp4"
    if not p.exists():
        raise SystemExit(f"clip not found: {p}")
    return p


def headless_add(clip, x, y, r, f0):
    info = probe(str(_clip_path(clip)))
    doc = rs.add_rim(clip, info.width, info.height, rim_x=x, rim_y=y,
                     rim_radius_px=r, f0=f0, f1=None, note="manual (headless)")
    p = rs.save_rims(doc)
    print(f"saved {p}: rim ({x},{y}) r={r} from frame {f0} to end "
          f"({len(doc['rims'])} rim(s) total)")


def gui(clip):
    import cv2
    path = _clip_path(clip)
    info = probe(str(path))
    cap = cv2.VideoCapture(str(path))
    # start FRESH each run (do not append to a prior file) -- re-clicking a rim
    # simply replaces it. For a mid-clip camera move, click all positions in this
    # one session (first at frame 0, then navigate + click the later one).
    doc = {"clip": clip, "image_w": info.width, "image_h": info.height, "rims": []}
    state = {"frame": info.n_frames // 3, "clicks": []}

    def on_mouse(ev, mx, my, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN and len(state["clicks"]) < 2:
            state["clicks"].append((mx, my))

    win = f"verify_rim {clip}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    def read(fno):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        ok, fr = cap.read()
        return fr if ok else None

    while True:
        fr = read(state["frame"])
        if fr is None:
            state["frame"] = max(0, state["frame"] - 1); continue
        disp = fr.copy()
        # existing rims covering this frame
        c = rs.calib_at(doc, state["frame"])
        if c is not None:
            cv2.circle(disp, (int(c.rim_x), int(c.rim_y)), int(c.rim_radius_px),
                       (0, 255, 0), 2)
            cv2.circle(disp, (int(c.rim_x), int(c.rim_y)),
                       int(c.shot_gate_px), (0, 180, 0), 1)
        # AUTO-COMMIT: click the rim's LEFT edge then RIGHT edge -> center is the
        # midpoint, radius is HALF the span. (Clicking center+near-center gave a
        # 4-6x-too-small radius that corrupted make/miss + apex-height scaling.)
        if len(state["clicks"]) == 2:
            (x1, y1), (x2, y2) = state["clicks"]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            rad = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 / 2.0
            f0 = 0 if not doc["rims"] else state["frame"]   # first rim covers frame 0
            doc = rs.add_rim(clip, info.width, info.height, rim_x=cx, rim_y=cy,
                             rim_radius_px=rad, f0=f0, f1=None,
                             note=f"manual edge-to-edge @f{state['frame']}")
            state["clicks"] = []
            print(f"added rim center=({cx:.0f},{cy:.0f}) r={rad:.0f} from frame {f0}")
            bd = _median_ball_diameter(clip)
            if bd:
                ratio = rad / bd
                if ratio < 0.3 or ratio > 2.0:
                    print(f"  ⚠️ SANITY: rim half-width {rad:.0f}px vs median ball "
                          f"diameter {bd:.0f}px (ratio {ratio:.2f}). A real rim half-"
                          f"width is ~0.5-1.0x the ball diameter -- re-click the EDGES "
                          f"if this looks off ('x' deletes the last rim).")
                else:
                    print(f"  ✓ sanity: rim/ball ratio {ratio:.2f} (plausible)")
            print("  press 's' or close the window to save")
        for i, (cx, cy) in enumerate(state["clicks"]):
            cv2.circle(disp, (int(cx), int(cy)), 4, (0, 0, 255), -1)
        txt = f"f {state['frame']}/{info.n_frames}  rims={len(doc['rims'])} (click LEFT then RIGHT rim edge)"
        cv2.putText(disp, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)
        cv2.imshow(win, disp)
        k = cv2.waitKey(30) & 0xFF
        if k in (27,) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        elif k == ord('d'): state["frame"] = min(info.n_frames - 1, state["frame"] + 1)
        elif k == ord('a'): state["frame"] = max(0, state["frame"] - 1)
        elif k == ord('e'): state["frame"] = min(info.n_frames - 1, state["frame"] + 30)
        elif k == ord('q'): state["frame"] = max(0, state["frame"] - 30)
        elif k == ord('c'): state["frame"] = min(info.n_frames - 1, state["frame"] + 300)
        elif k == ord('z'): state["frame"] = max(0, state["frame"] - 300)
        elif k == ord('r'): state["clicks"] = []
        elif k == ord('x') and doc["rims"]:
            popped = doc["rims"].pop(); print("deleted", popped)
        elif k == ord('g'):
            try:
                state["frame"] = max(0, min(info.n_frames - 1,
                                            int(input("jump to frame #: "))))
            except (ValueError, EOFError):
                pass
        elif k == ord('s'):
            p = rs.save_rims(doc); print(f"saved {p} ({len(doc['rims'])} rim(s))")
    cap.release()
    cv2.destroyAllWindows()
    if doc["rims"]:
        p = rs.save_rims(doc); print(f"saved {p} ({len(doc['rims'])} rim(s))")


def _selftest():
    import tempfile, os
    # exercise add/save/load/resolve without a video or GUI
    orig = rs.CONFIG_DIR
    try:
        rs.CONFIG_DIR = Path(tempfile.mkdtemp())
        doc = rs.add_rim("T", 1920, 1080, rim_x=1134, rim_y=470, rim_radius_px=16, f0=0)
        rs.save_rims(doc)
        doc = rs.add_rim("T", 1920, 1080, rim_x=1244, rim_y=461, rim_radius_px=16, f0=2000)
        p = rs.save_rims(doc)
        loaded = rs.load_rims("T")
        assert len(loaded["rims"]) == 2
        assert rs.calib_at(loaded, 100).rim_x == 1134
        assert rs.calib_at(loaded, 3000).rim_x == 1244
        segs = rs.segments(loaded, 5000)
        assert [s[2].rim_x for s in segs] == [1134, 1244], segs
        os.remove(p)
        print("verify_rim selftest OK")
    finally:
        rs.CONFIG_DIR = orig


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip")
    ap.add_argument("--rim", nargs=2, type=float, metavar=("X", "Y"))
    ap.add_argument("--radius", type=float)
    ap.add_argument("--f0", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        _selftest(); return 0
    if not args.clip:
        ap.error("--clip required (or --selftest)")
    if args.rim and args.radius:
        headless_add(args.clip, args.rim[0], args.rim[1], args.radius, args.f0)
    else:
        gui(args.clip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
