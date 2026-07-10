#!/usr/bin/env python
"""Generate the printable stereo-calibration checkerboard (letter size).

The PDF is drawn in true inches, so printed at 100% / "Actual size" (NOT "fit
to page") every square is exactly SQUARE_IN inches -- that known size is what
makes the stereo calibration metric. A ruler bar is printed on the sheet:
measure it after printing; if it isn't exactly 6.000 in, the print scaled and
the sheet must be reprinted.

Usage:
  python tools/make_checkerboard.py            -> data/calibration/checkerboard_9x6.pdf

Mount the print on something FLAT (foam board / clipboard); any warp becomes
calibration error.
"""
from __future__ import annotations

import os
import sys

# pattern: inner corners the detector finds = (COLS-1, ROWS-1) = (9, 6)
COLS, ROWS = 10, 7


def main(out_dir: str = os.path.join("data", "calibration"),
         square_in: float = 0.9, page_w: float = 11.0, page_h: float = 8.5,
         ruler_in: float = 6.0) -> str:
    SQUARE_IN, PAGE_W, PAGE_H = square_in, page_w, page_h
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAGE_W); ax.set_ylim(0, PAGE_H)
    ax.axis("off")

    grid_w, grid_h = COLS * SQUARE_IN, ROWS * SQUARE_IN
    x0, y0 = (PAGE_W - grid_w) / 2, (PAGE_H - grid_h) / 2 + 0.35
    for r in range(ROWS):
        for c in range(COLS):
            if (r + c) % 2 == 0:
                ax.add_patch(Rectangle((x0 + c * SQUARE_IN, y0 + r * SQUARE_IN),
                                       SQUARE_IN, SQUARE_IN, facecolor="black",
                                       edgecolor="none"))

    # scale-verification ruler with inch ticks
    rx, ry = (PAGE_W - ruler_in) / 2, y0 - 0.6 * (SQUARE_IN if SQUARE_IN > 1 else 0.9)
    ax.plot([rx, rx + ruler_in], [ry, ry], color="black", lw=1.5)
    for i in range(int(ruler_in) + 1):
        ax.plot([rx + i, rx + i], [ry - 0.08, ry + 0.08], color="black", lw=1.2)
    ax.text(rx + ruler_in / 2, ry - 0.35,
            f"this bar must measure exactly {ruler_in:.3f} in after printing "
            f"(print at 100% / Actual size)  ·  squares = {SQUARE_IN:.2f} in  ·  "
            f"inner corners = {COLS-1}x{ROWS-1}",
            ha="center", va="top", fontsize=9)

    os.makedirs(out_dir, exist_ok=True)
    tag = f"{COLS-1}x{ROWS-1}_{SQUARE_IN:.2f}in".replace(".", "p")
    out = os.path.join(out_dir, f"checkerboard_{tag}.pdf")
    fig.savefig(out, format="pdf")
    fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    print(f"wrote {out} (+ .png preview)")
    print(f"board {COLS*SQUARE_IN:.1f}x{ROWS*SQUARE_IN:.1f} in; solver pattern "
          f"({COLS-1}, {ROWS-1}); calibrate with --square-in {SQUARE_IN}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="printable stereo-calibration checkerboard")
    ap.add_argument("--square-in", type=float, default=0.9, help="square size, inches")
    ap.add_argument("--page-w", type=float, default=11.0, help="page width, inches")
    ap.add_argument("--page-h", type=float, default=8.5, help="page height, inches")
    ap.add_argument("--ruler-in", type=float, default=6.0, help="verify-ruler length")
    ap.add_argument("--out-dir", default=os.path.join("data", "calibration"))
    a = ap.parse_args()
    main(a.out_dir, a.square_in, a.page_w, a.page_h, a.ruler_in)
