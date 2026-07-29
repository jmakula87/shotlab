"""Shared display scaling for the OpenCV review GUIs (hand_count, verify_rim).

WHY: the 2026-07-29 wide cam is 3840x2160, which does not fit on a screen. An
unscaled `cv2.imshow` window gets resized by the window manager, and whether the
mouse callback then reports IMAGE coordinates or WINDOW coordinates depends on the
highgui backend. `verify_rim` CLICKS the rim, so a silent display/image scale
mismatch would corrupt the rim center and radius -- exactly the defect that once
shipped an 8px radius and broke make/miss thresholds plus apex-height scaling.

So don't rely on the backend: downscale explicitly, draw overlays in DISPLAY space
(so text stays legible at any resolution), and map clicks back to image space with
the known factor.
"""
from __future__ import annotations


def display_scale(w: int, h: int, max_w: int = 1600, max_h: int = 900) -> float:
    """Factor to multiply IMAGE coords by to get DISPLAY coords. Never upscales:
    returns 1.0 for a frame that already fits."""
    if w <= 0 or h <= 0:
        return 1.0
    return min(1.0, max_w / float(w), max_h / float(h))


def to_display(frame, scale: float):
    """Resize a frame for display. Identity at scale >= 1."""
    if scale >= 1.0:
        return frame
    import cv2
    h, w = frame.shape[:2]
    return cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=cv2.INTER_AREA)


def to_image_xy(mx: float, my: float, scale: float) -> tuple[float, float]:
    """DISPLAY-space click -> IMAGE-space coords (the inverse of `scale`)."""
    if scale <= 0:
        return float(mx), float(my)
    return mx / scale, my / scale


def to_display_xy(ix: float, iy: float, scale: float) -> tuple[int, int]:
    """IMAGE-space point -> DISPLAY-space, for drawing overlays after `to_display`."""
    return int(round(ix * scale)), int(round(iy * scale))


def _selftest():
    assert display_scale(1280, 720) == 1.0             # fits -> no scaling
    assert display_scale(0, 0) == 1.0                  # degenerate -> safe
    # 1080p is scaled too (a full-size window overflows a 1080p desktop and gets
    # resized by the window manager -- the ambiguity this module removes)
    assert abs(display_scale(1920, 1080) - 1600 / 1920) < 1e-9
    s = display_scale(3840, 2160)
    assert abs(s - 1600 / 3840) < 1e-9, s              # width-limited
    assert abs(display_scale(1000, 3000) - 900 / 3000) < 1e-9   # height-limited

    # a click anywhere maps back to the pixel it was drawn from (round trip)
    for ix, iy in [(0, 0), (615.5, 231.0), (3839, 2159)]:
        dx, dy = to_display_xy(ix, iy, s)
        rx, ry = to_image_xy(dx, dy, s)
        assert abs(rx - ix) <= 1 / s and abs(ry - iy) <= 1 / s, (ix, iy, rx, ry)

    # the rim geometry that matters: edge-to-edge span survives the round trip
    left, right = (900.0, 500.0), (980.0, 500.0)
    dl = to_display_xy(*left, s); dr = to_display_xy(*right, s)
    il = to_image_xy(*dl, s); ir = to_image_xy(*dr, s)
    span = ((ir[0] - il[0]) ** 2 + (ir[1] - il[1]) ** 2) ** 0.5
    assert abs(span - 80.0) < 2.0, span                # radius 40 recovered, not 13

    import numpy as np
    fr = np.zeros((2160, 3840, 3), dtype=np.uint8)
    d = to_display(fr, s)
    assert d.shape[1] == 1600 and d.shape[0] == 900, d.shape
    assert to_display(fr, 1.0) is fr                   # no copy when it fits
    print("guiview selftest OK")


if __name__ == "__main__":
    _selftest()
