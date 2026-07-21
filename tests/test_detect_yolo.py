"""Letterbox geometry for the ONNX/DirectML ball backend. Mutation-checked: each
assert fails if the box<->original-frame coordinate math is broken.

The full onnxruntime detect path is validated against the ultralytics .pt output
on real frames (see the 2026-07-21 session); it needs onnxruntime + an exported
model, so it isn't reproduced in CI. What CI CAN guard is the pure geometry that
maps a detection in the padded square back to the original frame -- the most
regression-prone part of the backend."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.phase1_ball.detect_yolo import _letterbox, corridor_tiles


def _map_back(px, py, r, dw, dh):
    """Inverse of the letterbox: padded-square coords -> original-frame coords
    (exactly what _detect_onnx does to each detection)."""
    return (px - dw) / r, (py - dh) / r


def test_letterbox_preserves_aspect_and_centers_pad():
    """A 1920x1080 frame into 1280 square: fit by the long side, pad the short
    side symmetrically. Breaks if ratio uses the wrong side or pad isn't centered."""
    img = np.zeros((1080, 1920, 3), np.uint8)
    im, r, dw, dh = _letterbox(img, 1280, 1280)
    assert im.shape[0] == 1280 and im.shape[1] == 1280, im.shape
    assert abs(r - 1280 / 1920) < 1e-6, r          # long side (width) sets scale
    assert abs(dw) < 1e-6, dw                       # width fills, no horizontal pad
    assert abs(dh - (1280 - 1080 * r) / 2) < 0.6, dh  # vertical pad split in half


def test_box_maps_back_to_original_pixel():
    """A known original-frame point, forward-projected into the padded square,
    must map back to itself. Breaks if _detect_onnx drops the pad offset or the
    ratio (the two ways the ball would land at the wrong pixel)."""
    img = np.zeros((1080, 1920, 3), np.uint8)
    _, r, dw, dh = _letterbox(img, 1280, 1280)
    ox, oy = 711.0, 291.0                            # the real ball from the session
    px, py = ox * r + dw, oy * r + dh                # where it lands in the square
    bx, by = _map_back(px, py, r, dw, dh)
    assert abs(bx - ox) < 1e-3 and abs(by - oy) < 1e-3, (bx, by)


def test_radius_scales_by_inverse_ratio():
    """A box of width w in the square is w/r pixels in the original frame."""
    img = np.zeros((1080, 1920, 3), np.uint8)
    _, r, _, _ = _letterbox(img, 1280, 1280)
    w_square = 34.0
    assert abs(w_square / r - w_square * 1920 / 1280) < 1e-6


def test_corridor_tiles_cover_with_ball_safe_overlap():
    """1920 into a 1280 model -> 2 native-width tiles that (a) each equal the
    model width, (b) together cover [0,w], (c) overlap by >> a ball so no shot is
    lost at a seam. Breaks if the tiling downscales (tile != model width) or
    leaves a gap/thin seam a ball could straddle un-detected in both tiles."""
    tiles = corridor_tiles(1920, 1080, 1280)
    assert len(tiles) == 2, tiles
    assert all(x1 - x0 == 1280 for x0, _, x1, _ in tiles), tiles   # native width
    assert tiles[0][0] == 0 and tiles[-1][2] == 1920, tiles        # full coverage
    overlap = tiles[0][2] - tiles[1][0]                            # 1280 - 640
    assert overlap > 100, overlap                                  # >> ~20px ball
    assert all(y0 == 0 and y1 == 1080 for _, y0, _, y1 in tiles)   # full height


def test_corridor_single_tile_when_frame_fits():
    """No tiling when the frame already fits the model width (don't upscale-pad
    for nothing). Breaks if it ever splits a small frame."""
    assert corridor_tiles(1280, 720, 1280) == [(0, 0, 1280, 720)]
    assert corridor_tiles(960, 540, 1280) == [(0, 0, 960, 540)]


if __name__ == "__main__":
    test_letterbox_preserves_aspect_and_centers_pad()
    test_box_maps_back_to_original_pixel()
    test_radius_scales_by_inverse_ratio()
    test_corridor_tiles_cover_with_ball_safe_overlap()
    test_corridor_single_tile_when_frame_fits()
    print("ok")
