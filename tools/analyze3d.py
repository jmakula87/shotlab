#!/usr/bin/env python
"""Compute the 3D shot analysis (wide-camera metric arcs + close-camera elbow
flare) and write <out>/analysis3d.json for the dashboard to display.

The heavy compute (YOLO ball detection + MediaPipe pose) runs here, ONCE, so the
webpage only has to read the JSON.

Usage:
  python tools/analyze3d.py --wide "data/raw/Camera 1/PXL_...mp4" \
      --close "data/raw/Camera 2/2026...mp4" \
      --weights runs/detect/ball_finetune/weights/best.pt \
      --out data/out/session_0710_3d \
      --wide-window 4200 5800 --close-window 300 3300
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.analysis3d import Analysis3D, wide_arcs, flare_from_close


def main():
    ap = argparse.ArgumentParser(description="3D shot analysis -> analysis3d.json")
    ap.add_argument("--wide", help="wide clip (ball/arc/rim)")
    ap.add_argument("--close", help="close clip (body -> elbow flare)")
    ap.add_argument("--weights", default="runs/detect/ball_finetune/weights/best.pt")
    ap.add_argument("--out", required=True, help="session out dir")
    ap.add_argument("--wide-window", nargs=2, type=int, default=None,
                    metavar=("START", "STOP"))
    ap.add_argument("--close-window", nargs=2, type=int, default=None,
                    metavar=("START", "STOP"))
    ap.add_argument("--imgsz", type=int, default=1280)
    a = ap.parse_args()

    res = Analysis3D(meta={"wide": a.wide, "close": a.close})
    if a.wide:
        ws, wp = (a.wide_window or (0, None))
        print(f"[wide] detecting ball + fitting arcs on {a.wide} ...", flush=True)
        res.wide = wide_arcs(a.wide, a.weights, imgsz=a.imgsz, start=ws, stop=wp)
        n = len(res.wide.get("shots", []))
        good = sum(s["trustworthy"] for s in res.wide.get("shots", []))
        print(f"[wide] {n} arcs, {good} pass the gravity check "
              f"(VFR={res.wide.get('is_vfr')})", flush=True)
    if a.close:
        cs, cp = (a.close_window or (0, None))
        print(f"[close] posing + measuring elbow flare on {a.close} ...", flush=True)
        res.flare = flare_from_close(a.close, start=cs, stop=cp)
        s = res.flare.get("summary")
        if s:
            print(f"[close] flare median {s['median_deg']} deg (n={s['n']}, "
                  f"sd {s['sd_deg']})", flush=True)

    out = os.path.join(a.out, "analysis3d.json")
    res.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
