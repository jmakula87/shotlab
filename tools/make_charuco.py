#!/usr/bin/env python
"""Generate a printable ChArUco calibration board (coarse, big squares).

A ChArUco board is a checkerboard with an ArUco code printed inside every white
square. Because each code is self-identifying, the detector recognizes corners
even from a partial or steeply-angled view -- which is exactly what a FAR / WIDE
camera needs. Big squares (few of them) survive the distance; the codes survive
the angle.

The image is written at a true DPI so, printed at 100% / "Actual size", every
square is exactly --square-in inches. Measure the printed ruler bar to confirm
the print didn't scale; that known square size is what makes calibration metric.

Usage:
  python tools/make_charuco.py                      # 4x3 squares, 5.0 in, DICT_4X4_50
  python tools/make_charuco.py --square-in 6 --squares-x 3 --squares-y 3

Print big (a copy shop / plotter), mount FLAT (foam board), and film it filling
as much of BOTH cameras' frames as you can, tilted to several angles.
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

DICT = cv2.aruco.DICT_4X4_50   # 4x4 codes: chunky bits, easy to read at distance


def main(square_in: float = 5.0, squares_x: int = 4, squares_y: int = 3,
         marker_ratio: float = 0.72, dpi: int = 150,
         out_dir: str = os.path.join("data", "calibration")) -> str:
    if squares_x * squares_y // 2 > 50:
        raise SystemExit("too many squares for DICT_4X4_50 (max 50 markers)")
    marker_in = square_in * marker_ratio
    dictionary = cv2.aruco.getPredefinedDictionary(DICT)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_in, marker_in,
                                   dictionary)

    margin_in = 0.5
    wpx = int(round((squares_x * square_in + 2 * margin_in) * dpi))
    hpx = int(round((squares_y * square_in + 2 * margin_in) * dpi))
    img = board.generateImage((wpx, hpx), marginSize=int(round(margin_in * dpi)),
                              borderBits=1)

    # scale-verification ruler: a 6.000 in bar with inch ticks, under the board
    img = cv2.copyMakeBorder(img, 0, int(0.9 * dpi), 0, 0, cv2.BORDER_CONSTANT,
                             value=255)
    y = hpx + int(0.45 * dpi)
    x0 = int(margin_in * dpi)
    cv2.line(img, (x0, y), (x0 + 6 * dpi, y), 0, 2)
    for i in range(7):
        cv2.line(img, (x0 + i * dpi, y - 8), (x0 + i * dpi, y + 8), 0, 2)
    cv2.putText(img, f"6.000 in ruler -- print at 100% / Actual size  |  "
                f"squares = {square_in:.2f} in  |  {squares_x}x{squares_y} ChArUco "
                f"DICT_4X4_50",
                (x0, y + int(0.32 * dpi)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5 * dpi / 150, 0, max(1, dpi // 150))

    os.makedirs(out_dir, exist_ok=True)
    tag = f"{squares_x}x{squares_y}_{square_in:.1f}in".replace(".", "p")
    out_png = os.path.join(out_dir, f"charuco_{tag}.png")
    out_pdf = os.path.join(out_dir, f"charuco_{tag}.pdf")
    cv2.imwrite(out_png, img)
    _png_to_pdf(out_png, out_pdf, wpx, hpx + int(0.9 * dpi), dpi)

    # machine-readable board spec the calibrator reads back (single source of truth)
    spec = {"dict": "DICT_4X4_50", "squares_x": squares_x, "squares_y": squares_y,
            "square_in": square_in, "marker_ratio": marker_ratio}
    with open(os.path.join(out_dir, "charuco_spec.json"), "w") as f:
        json.dump(spec, f, indent=2)

    board_w, board_h = squares_x * square_in, squares_y * square_in
    print(f"wrote {out_pdf} (+ .png)")
    print(f"board {board_w:.1f} x {board_h:.1f} in  ({squares_x*squares_y} squares, "
          f"{square_in:.1f} in each)")
    print(f"interior corners the solver uses: {(squares_x-1)*(squares_y-1)}")
    print(f"spec -> {os.path.join(out_dir, 'charuco_spec.json')}")
    return out_pdf


def _png_to_pdf(png: str, pdf: str, wpx: int, hpx: int, dpi: int) -> None:
    """Wrap the PNG in a true-inch PDF page so it prints at exact scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    win, hin = wpx / dpi, hpx / dpi
    fig = plt.figure(figsize=(win, hin))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(cv2.imread(png, cv2.IMREAD_GRAYSCALE), cmap="gray",
              vmin=0, vmax=255, aspect="auto")
    fig.savefig(pdf, format="pdf", dpi=dpi)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="printable ChArUco calibration board")
    ap.add_argument("--square-in", type=float, default=5.0)
    ap.add_argument("--squares-x", type=int, default=4)
    ap.add_argument("--squares-y", type=int, default=3)
    ap.add_argument("--marker-ratio", type=float, default=0.72)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--out-dir", default=os.path.join("data", "calibration"))
    a = ap.parse_args()
    main(a.square_in, a.squares_x, a.squares_y, a.marker_ratio, a.dpi, a.out_dir)
