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

GPU on AMD/Windows (2026-07-21): the ultralytics/OpenVINO path is CPU-only here
(OpenVINO's GPU plugin is Intel-only; torch on Windows has no CUDA/ROCm). But
this box has a Radeon RX 9070 XT, reachable via DirectML. So when `weights`
points at an exported `.onnx`, we run it through onnxruntime with the
DmlExecutionProvider (falling back to CPU) -- ~20x faster at imgsz 1280, which
is what makes high-res small-ball detection cheap. The ONNX decode below was
validated to produce byte-for-byte the same detections as the ultralytics .pt
path (see the 2026-07-21 session): same ball at (711,291) conf 0.80 on the
frame the 640 model missed. Export with:
    YOLO('best.pt').export(format='onnx', imgsz=1280, opset=17)
"""

from __future__ import annotations

import numpy as np

from .detect import BaseDetector, BallCandidate

# COCO class id for "sports ball"
_COCO_SPORTS_BALL = 32


def _letterbox(img, new_h, new_w, color=(114, 114, 114)):
    """Ultralytics-style aspect-preserving resize + centered pad. Returns the
    padded image plus (ratio, pad_w, pad_h) to map boxes back to the original."""
    import cv2
    h, w = img.shape[:2]
    r = min(new_h / h, new_w / w)
    nw, nh = round(w * r), round(h * r)
    dw, dh = (new_w - nw) / 2, (new_h - nh) / 2
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)
    return im, r, dw, dh


def corridor_tiles(w, h, tile_w):
    """Native-resolution horizontal tiles covering a `w`x`h` frame with tiles of
    width `tile_w` (the model's input width). Idea: letterboxing a 1920px frame
    into a 1280 model shrinks a 20px ball to ~13px, but a `tile_w`-wide crop feeds
    the model at ratio 1.0 (no downscale) so the ball keeps its native pixels.
    Tiles overlap so a ball near a seam lands whole in at least one tile; a
    cross-tile NMS dedups the overlap. Full height (one row) because h (1080) <
    tile_w (1280) already letterboxes 1:1.

    ⚠️ MEASURED 2026-07-21: this HURTS with the current orange model. It was
    fine-tuned on DOWNSCALED balls (~13px); fed native ~20px balls it detects in
    FEWER frames at LOWER conf (clip-2 sample: 30 vs 47 frames, conf 0.62 vs 0.72
    -> the session went 11 shots -> 2 with --tile). Tiling and retraining are
    COUPLED: this only pays off once the detector is retrained on native-scale
    court crops (consult idea #3). Kept as opt-in infra for that retrain; do NOT
    enable --tile on the current weights."""
    if w <= tile_w:
        return [(0, 0, w, h)]
    import math
    overlap = tile_w // 4                          # >> a ball; a seam-straddler
    n = math.ceil((w - overlap) / (tile_w - overlap))  # 1920/1280 -> 2 tiles
    xs = [round(i * (w - tile_w) / (n - 1)) for i in range(n)]  # evenly spaced
    return [(x, 0, x + tile_w, h) for x in xs]


class YoloBallDetector(BaseDetector):
    name = "yolo"

    def __init__(self,
                 weights: str = "yolo11n.pt",
                 ball_class: int | None = _COCO_SPORTS_BALL,
                 conf: float = 0.20,
                 imgsz: int = 960,
                 device: str | None = None,
                 roi: tuple[int, int, int, int] | None = None,
                 tiles=None):
        """
        weights:    path to a .pt model (auto-downloads known names).
        ball_class: class id to keep; None = keep all detections.
        conf:       confidence threshold (low, because the ball is small/fast).
        imgsz:      inference size; >=960 helps small-ball recall.
        roi:        optional (x0,y0,x1,y1) crop (the shooting lane) to cut
                    false positives and speed up CPU inference.
        tiles:      None (whole frame), 'auto' (native-res corridor tiling --
                    detect on model-width crops so a far ball keeps its pixels;
                    built lazily from the first frame), or an explicit list of
                    (x0,y0,x1,y1) boxes. Ignores `roi` when set.
        """
        self.ball_class = ball_class
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.roi = roi
        self.tiles = tiles
        self._tile_boxes = None if tiles == "auto" else tiles

        # ONNX weights -> onnxruntime (DirectML GPU, CPU fallback). Anything else
        # -> ultralytics (.pt / OpenVINO dir), the CPU path.
        self.backend = "onnx" if str(weights).lower().endswith(".onnx") else "ultralytics"
        if self.backend == "onnx":
            self._init_onnx(weights)
            self._model_w = self._in_w
        else:
            try:
                from ultralytics import YOLO
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "ultralytics is required for the YOLO backend: "
                    "pip install ultralytics") from e
            self._model = YOLO(weights)
            self._model_w = self.imgsz

    def _init_onnx(self, weights):
        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "onnxruntime is required for the .onnx backend: "
                "pip install onnxruntime-directml  (or onnxruntime)") from e
        avail = ort.get_available_providers()
        providers = [p for p in ("DmlExecutionProvider", "CUDAExecutionProvider",
                                 "CPUExecutionProvider") if p in avail]
        self._sess = ort.InferenceSession(weights, providers=providers)
        self._in_name = self._sess.get_inputs()[0].name
        shp = self._sess.get_inputs()[0].shape           # [1,3,H,W]
        self._in_h = int(shp[2]) if isinstance(shp[2], int) else self.imgsz
        self._in_w = int(shp[3]) if isinstance(shp[3], int) else self.imgsz
        self.active_provider = self._sess.get_providers()[0]

    def detect(self, frame_idx, frame_bgr):
        H, W = frame_bgr.shape[:2]
        if self.tiles == "auto" and self._tile_boxes is None:
            self._tile_boxes = corridor_tiles(W, H, self._model_w)
        if self._tile_boxes:
            cands = []
            for (tx0, ty0, tx1, ty1) in self._tile_boxes:
                cands.extend(self._detect_region(frame_idx, frame_bgr,
                                                 tx0, ty0, tx1, ty1))
            return self._merge_nms(cands)
        if self.roi is not None:
            x0, y0, x1, y1 = self.roi
        else:
            x0, y0, x1, y1 = 0, 0, W, H
        return self._detect_region(frame_idx, frame_bgr, x0, y0, x1, y1)

    def _detect_region(self, frame_idx, frame_bgr, x0, y0, x1, y1):
        img = frame_bgr[y0:y1, x0:x1]
        if self.backend == "onnx":
            return self._detect_onnx(frame_idx, img, x0, y0)
        return self._detect_ultra(frame_idx, img, x0, y0)

    def _merge_nms(self, cands, iou=0.45):
        """Cross-tile dedup: a ball in a tile overlap is detected twice; keep the
        higher-conf box. Same NMS the per-region decode uses."""
        import cv2
        if not cands:
            return cands
        boxes = [[c.cx - c.r, c.cy - c.r, 2 * c.r, 2 * c.r] for c in cands]
        scores = [c.conf for c in cands]
        idx = cv2.dnn.NMSBoxes(boxes, scores, self.conf, iou)
        out = [cands[i] for i in np.array(idx).flatten()]
        out.sort(key=lambda c: c.conf, reverse=True)
        return out

    def _detect_ultra(self, frame_idx, img, x0, y0):
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

    def _detect_onnx(self, frame_idx, img, x0, y0, iou=0.45):
        """onnxruntime path. Letterbox -> infer -> decode YOLOv8/11 head
        ([1, 4+nc, N]) -> NMS -> BallCandidate in original-frame coords. Validated
        to match the ultralytics .pt output exactly."""
        import cv2
        im, r, dw, dh = _letterbox(img, self._in_h, self._in_w)
        blob = im[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob)
        pred = self._sess.run(None, {self._in_name: blob})[0][0].T  # [N, 4+nc]
        nc = pred.shape[1] - 4
        cls_scores = pred[:, 4:]
        cls_id = cls_scores.argmax(1)
        conf = cls_scores.max(1)
        keep = conf >= self.conf
        if self.ball_class is not None:
            keep &= (cls_id == self.ball_class)
        pred, conf, cls_id = pred[keep], conf[keep], cls_id[keep]
        out = []
        if not len(pred):
            return out
        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        boxes_xywh = np.stack([cx - w / 2, cy - h / 2, w, h], 1)
        idx = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), conf.tolist(), self.conf, iou)
        for i in np.array(idx).flatten():
            ox = x0 + (cx[i] - dw) / r
            oy = y0 + (cy[i] - dh) / r
            orad = (w[i] + h[i]) / 4 / r
            out.append(BallCandidate(frame_idx, float(ox), float(oy),
                                     float(orad), float(conf[i])))
        out.sort(key=lambda c: c.conf, reverse=True)
        return out
