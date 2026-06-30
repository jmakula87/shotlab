#!/usr/bin/env python
"""Render a side-by-side comparison of two shots' key phases (annotated stills).

Usage:
  python tools/compare_shots.py \
     --a data/raw/Hoops_0629/PXL_20260629_153042129.mp4 --shot-a 1 \
     --b data/raw/Hoops_0629/PXL_20260629_154649605.mp4 --shot-b 3 \
     --labels good weak --out data/out/comparisons/good_vs_weak.png
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.compare import compare_shots

DEFAULT_W = "runs/detect/ball_finetune/weights/best_openvino_model"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--shot-a", type=int, required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--shot-b", type=int, required=True)
    ap.add_argument("--labels", nargs=2, default=["A", "B"])
    ap.add_argument("--weights", default=DEFAULT_W)
    ap.add_argument("--handedness", default="right")
    ap.add_argument("--out", default="data/out/comparisons/comparison.png")
    args = ap.parse_args(argv)

    out = compare_shots(args.a, args.shot_a, args.b, args.shot_b,
                        weights=args.weights, out_path=args.out,
                        handedness=args.handedness, labels=tuple(args.labels))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
