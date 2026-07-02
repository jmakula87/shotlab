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
SQUARE_IN = 0.9
PAGE_W, PAGE_H = 11.0, 8.5            # letter, landscape


def main(out_dir: str = os.path.join("data", "calibration")) -> str:
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

    # scale-verification ruler: 6.000 in with inch ticks
    rx, ry = (PAGE_W - 6.0) / 2, y0 - 0.55
    ax.plot([rx, rx + 6.0], [ry, ry], color="black", lw=1.5)
    for i in range(7):
        ax.plot([rx + i, rx + i], [ry - 0.08, ry + 0.08], color="black", lw=1.2)
    ax.text(rx + 3.0, ry - 0.30,
            "this bar must measure exactly 6.000 in after printing "
            f"(print at 100% / Actual size)  ·  squares = {SQUARE_IN:.1f} in  ·  "
            f"inner corners = {COLS-1}x{ROWS-1}",
            ha="center", va="top", fontsize=9)

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"checkerboard_{COLS-1}x{ROWS-1}.pdf")
    fig.savefig(out, format="pdf")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300)
    plt.close(fig)
    print(f"wrote {out} (+ .png preview)")
    print(f"pattern for the solver: inner corners ({COLS-1}, {ROWS-1}), "
          f"square {SQUARE_IN} in")
    return out


if __name__ == "__main__":
    main(*sys.argv[1:])
