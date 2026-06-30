"""Roboflow ready-made basketball detector backend.

For users who don't want to train anything: Roboflow Universe hosts basketball
detectors (ball / hoop / player) that run LOCALLY via the `inference` package
once a free API key downloads the weights. Your video never leaves the machine.

Usage (wired through analyze.py / build_session.py):
    pip install inference
    set ROBOFLOW_API_KEY=...        # free key from roboflow.com
    --detector roboflow --weights <workspace/project/version> --ball-class basketball

This wraps the model behind the same BaseDetector interface, so the tracker,
arc fit, rim-anchoring and session analytics all work unchanged.
"""

from __future__ import annotations

import os

import numpy as np

from .detect import BaseDetector, BallCandidate


class RoboflowBallDetector(BaseDetector):
    name = "roboflow"

    def __init__(self, model_id: str, api_key: str | None = None,
                 ball_class: str | None = "ball", conf: float = 0.20):
        """
        model_id:   Roboflow model id "workspace/project/version".
        api_key:    Roboflow API key (or set ROBOFLOW_API_KEY env var).
        ball_class: keep only predictions whose class name CONTAINS this string
                    (case-insensitive); None keeps all. Many basketball models
                    label the ball "ball" or "basketball".
        """
        try:
            from inference import get_model
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Roboflow backend needs the inference package: pip install inference"
            ) from e
        key = api_key or os.environ.get("ROBOFLOW_API_KEY")
        if not key:
            raise ValueError("Set ROBOFLOW_API_KEY or pass api_key=")
        self._model = get_model(model_id=model_id, api_key=key)
        self.ball_class = (ball_class or "").lower()
        self.conf = conf

    def detect(self, frame_idx, frame_bgr):
        # inference expects RGB or a path; pass the numpy array (it handles BGR
        # arrays from cv2 fine via its preprocessing).
        res = self._model.infer(frame_bgr, confidence=self.conf)[0]
        out = []
        for p in getattr(res, "predictions", []):
            name = str(getattr(p, "class_name", "")).lower()
            if self.ball_class and self.ball_class not in name:
                continue
            cx, cy = float(p.x), float(p.y)
            r = (float(p.width) + float(p.height)) / 4.0
            out.append(BallCandidate(frame_idx, cx, cy, r, float(p.confidence)))
        out.sort(key=lambda b: b.conf, reverse=True)
        return out
