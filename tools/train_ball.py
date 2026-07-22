#!/usr/bin/env python
"""Fine-tune a ball detector on YOUR auto-labeled dataset.

Starts from the trusted, already-installed Ultralytics base model (yolo11n) --
NOT an untrusted third-party weights file -- and adapts it to your red/blue ball
and leafy background. CPU-friendly (nano model, small dataset).

Usage:
  python tools/train_ball.py --data dataset_ball/data.yaml --epochs 60 --imgsz 768

The fine-tuned weights land at runs/detect/<name>/weights/best.pt -- a TRUSTED
file you produced locally. Use it with:
  python analyze.py <clip> --detector yolo --weights runs/detect/<name>/weights/best.pt --ball-class 0
"""

from __future__ import annotations

import argparse


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_ball/data.yaml")
    ap.add_argument("--base", default="yolo11n.pt", help="trusted base model")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=768,
                    help="higher helps the small ball; slower on CPU")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", default="ball_finetune")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--freeze", type=int, default=None,
                    help="freeze the first N layers (preserve a good base model's "
                         "appearance features; adapt only the head to new scale)")
    ap.add_argument("--mosaic", type=float, default=1.0)
    args = ap.parse_args(argv)

    # GPU on AMD/Windows: MIOpen can't compile BatchNorm for gfx1201, so route BN
    # through pure-torch primitives (conv stays on the GPU). No-op on CPU/non-ROCm.
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tools.rocm_bn_patch import maybe_apply
    on_rocm_gpu = maybe_apply(args.device)
    # ROCm-Windows preview is unstable with AMP; force it off there.
    amp = not on_rocm_gpu

    from ultralytics import YOLO
    model = YOLO(args.base)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        amp=amp,
        name=args.name,
        freeze=args.freeze,     # None = full fine-tune; N = keep base features
        patience=15,            # early stop if val plateaus
        # augmentation that helps a small single-class set:
        mosaic=args.mosaic, close_mosaic=10,
        hsv_h=0.02, hsv_s=0.5, hsv_v=0.4,   # color jitter (your ball is red/blue)
        fliplr=0.5, scale=0.5, translate=0.1,
        degrees=5.0,
        verbose=True,
    )
    metrics = model.val(data=args.data)
    print("\n=== validation ===")
    try:
        print(f"mAP50: {metrics.box.map50:.3f}   mAP50-95: {metrics.box.map:.3f}")
    except Exception:
        pass
    print(f"weights: runs/detect/{args.name}/weights/best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
