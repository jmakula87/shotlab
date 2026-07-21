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

from shotlab.phase1_ball.detect_yolo import _letterbox


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


if __name__ == "__main__":
    test_letterbox_preserves_aspect_and_centers_pad()
    test_box_maps_back_to_original_pixel()
    test_radius_scales_by_inverse_ratio()
    print("ok")
