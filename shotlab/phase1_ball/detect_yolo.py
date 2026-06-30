"""YOLO ball-detection backend (the production path).

Per the 2026 model survey, the right approach for a small, fast, motion-blurred
basketball is a FINE-TUNED detector feeding RANSAC -- not a multi-object tracker.
Stock COCO "sports ball" (class 32) is documented as unreliable, so:

  * Default model = yolo11n.pt (auto-downloads; the survey's battle-tested
    safe fallback) restricted to the sports-ball class. Works out of the box,
    decent on clean side-on footage, weak on small/blurred/cluttered frames.
  * For real accuracy, pass a basketball-specific weights file (a Roboflow
    Universe basketball model, ~96% mAP@50, or your own fine-tune). Set
    `ball_class` to that model's ball class id (often 0).

License note: the `ultralytics` package is AGPL-3.0. Fine for personal/local
use (your case). If you ever distribute this closed-source, swap to an
Apache-2.0 detector (RT-DETR / D-FINE / RF-DETR) behind this same interface.
"""

from __future__ import annotations

import numpy as np

from .detect import BaseDetector, BallCandidate

# COCO class id for "sports ball"
_COCO_SPORTS_BALL = 32


class YoloBallDetector(BaseDetector):
    name = "yolo"

    def __init__(self,
                 weights: str = "yolo11n.pt",
                 ball_class: int | None = _COCO_SPORTS_BALL,
                 conf: float = 0.20,
                 imgsz: int = 960,
                 device: str | None = None,
                 roi: tuple[int, int, int, int] | None = None):
        """
        weights:    path to a .pt model (auto-downloads known names).
        ball_class: class id to keep; None = keep all detections.
        conf:       confidence threshold (low, because the ball is small/fast).
        imgsz:      inference size; >=960 helps small-ball recall.
        roi:        optional (x0,y0,x1,y1) crop (the shooting lane) to cut
                    false positives and speed up CPU inference.
        """
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "ultralytics is required for the YOLO backend: "
                "pip install ultralytics") from e
        self._model = YOLO(weights)
        self.ball_class = ball_class
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.roi = roi

    def detect(self, frame_idx, frame_bgr):
        x0 = y0 = 0
        img = frame_bgr
        if self.roi is not None:
            x0, y0, x1, y1 = self.roi
            img = frame_bgr[y0:y1, x0:x1]

        res = self._model.predict(img, imgsz=self.imgsz, conf=self.conf,
                                  device=self.device, verbose=False)[0]
        out = []
        if res.boxes is None:
            return out
        for b in res.boxes:
            cls = int(b.cls[0])
            if self.ball_class is not None and cls != self.ball_class:
                continue
            conf = float(b.conf[0])
            bx0, by0, bx1, by1 = (float(v) for v in b.xyxy[0])
            cx = x0 + (bx0 + bx1) / 2
            cy = y0 + (by0 + by1) / 2
            r = (abs(bx1 - bx0) + abs(by1 - by0)) / 4  # avg half-side
            out.append(BallCandidate(frame_idx, cx, cy, r, conf))
        out.sort(key=lambda c: c.conf, reverse=True)
        return out
