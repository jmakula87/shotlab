"""Ball detection.

Two backends behind one interface so the rest of the pipeline never cares which
one ran:

  ColorBallDetector  -- classical HSV-orange + circularity. No ML deps, fast on
                        CPU, great on clean/well-lit footage and the synthetic
                        clip. Used as a fallback and for quick iteration.
  YoloBallDetector   -- ML detector (see detect_yolo.py). Robust to clutter,
                        motion blur, varied lighting. The production path.

A detector returns zero or more BallCandidate per frame; the tracker decides
which candidate is the real ball.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BallCandidate:
    frame_idx: int
    cx: float
    cy: float
    r: float          # radius in px
    conf: float       # 0..1


class BaseDetector:
    name = "base"

    def detect(self, frame_idx: int, frame_bgr: np.ndarray) -> list[BallCandidate]:
        raise NotImplementedError


class ColorBallDetector(BaseDetector):
    """Detect an orange basketball by HSV color + shape.

    Defaults target a basketball-orange ball. Tune hsv_lo/hsv_hi for your ball
    and lighting (or just use the YOLO backend on messy footage).
    """

    name = "color"

    def __init__(self,
                 hsv_lo=(5, 80, 80), hsv_hi=(28, 255, 255),
                 min_radius=6, max_radius=60,
                 min_circularity=0.55):
        self.hsv_lo = np.array(hsv_lo, np.uint8)
        self.hsv_hi = np.array(hsv_hi, np.uint8)
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_circularity = min_circularity

    def detect(self, frame_idx, frame_bgr):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lo, self.hsv_hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < np.pi * self.min_radius ** 2:
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
            if not (self.min_radius <= r <= self.max_radius):
                continue
            perim = cv2.arcLength(c, True)
            if perim <= 0:
                continue
            circularity = 4 * np.pi * area / (perim * perim)   # 1.0 == perfect circle
            if circularity < self.min_circularity:
                continue
            fill = area / (np.pi * r * r)        # how solid the blob is
            conf = float(np.clip(0.5 * circularity + 0.5 * fill, 0, 1))
            out.append(BallCandidate(frame_idx, float(x), float(y), float(r), conf))
        out.sort(key=lambda b: b.conf, reverse=True)
        return out


class MotionBallDetector(BaseDetector):
    """Detect the ball by MOTION (background subtraction), not color or class.

    Built for real outdoor footage where the ball is small, backlit (a dark
    silhouette, not orange), and the background is cluttered (trees, houses) --
    the exact case where ColorBallDetector drowns in false positives and stock
    YOLO goes blind. The ball moves fast while the scene is mostly static, so a
    background-subtractor isolates it. Residual movers (swaying leaves, the
    shooter) are filtered by size/shape here and by the RANSAC arc fit downstream.

    STATEFUL: must be applied to frames strictly in order (run_phase1 does this).
    """

    name = "motion"

    def __init__(self, min_radius=7, max_radius=34, min_circularity=0.55,
                 var_threshold=40, history=200):
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_circularity = min_circularity
        self._mog = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False)

    def detect(self, frame_idx, frame_bgr):
        fg = self._mog.apply(frame_bgr)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < np.pi * self.min_radius ** 2 * 0.6:
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
            if not (self.min_radius <= r <= self.max_radius):
                continue
            perim = cv2.arcLength(c, True)
            if perim <= 0:
                continue
            circularity = 4 * np.pi * area / (perim * perim)
            if circularity < self.min_circularity:
                continue
            fill = area / (np.pi * r * r)
            conf = float(np.clip(0.6 * circularity + 0.4 * fill, 0, 1))
            out.append(BallCandidate(frame_idx, float(x), float(y), float(r), conf))
        out.sort(key=lambda b: b.conf, reverse=True)
        return out


class MotionColorBallDetector(BaseDetector):
    """Require a candidate to be BOTH moving AND orange -- the robust choice for
    cluttered outdoor footage (leafy background, no clean sky) when the ball is
    FRONT-LIT (sun behind the camera, so the ball reads orange not silhouette).

    Motion (MOG2) removes the static clutter (driveway, house, the tan ground
    that fools color); the orange-hue gate removes the wind-swayed leaves that
    fool pure motion. Their intersection is almost always just the ball.

    STATEFUL: apply to frames strictly in order.
    """

    name = "motion+color"

    def __init__(self, hsv_lo=(3, 70, 70), hsv_hi=(30, 255, 255),
                 min_radius=6, max_radius=40, min_circularity=0.5,
                 min_orange_frac=0.25, var_threshold=40, history=200):
        self.hsv_lo = np.array(hsv_lo, np.uint8)
        self.hsv_hi = np.array(hsv_hi, np.uint8)
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_circularity = min_circularity
        self.min_orange_frac = min_orange_frac
        self._mog = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False)

    def detect(self, frame_idx, frame_bgr):
        fg = self._mog.apply(frame_bgr)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        orange = cv2.inRange(hsv, self.hsv_lo, self.hsv_hi)
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < np.pi * self.min_radius ** 2 * 0.6:
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
            if not (self.min_radius <= r <= self.max_radius):
                continue
            perim = cv2.arcLength(c, True)
            if perim <= 0:
                continue
            circularity = 4 * np.pi * area / (perim * perim)
            if circularity < self.min_circularity:
                continue
            # fraction of the blob that is orange
            mask = np.zeros(orange.shape, np.uint8)
            cv2.drawContours(mask, [c], -1, 255, -1)
            inter = cv2.bitwise_and(orange, mask)
            orange_frac = (inter > 0).sum() / max((mask > 0).sum(), 1)
            if orange_frac < self.min_orange_frac:
                continue
            conf = float(np.clip(0.4 * circularity + 0.6 * orange_frac, 0, 1))
            out.append(BallCandidate(frame_idx, float(x), float(y), float(r), conf))
        out.sort(key=lambda b: b.conf, reverse=True)
        return out
