"""Package the ShotLab ball dataset + base weights into ONE zip for Kaggle.

Upload the resulting zip as a Kaggle Dataset (or a new version of an existing
one); Kaggle auto-extracts it so the notebook sees, under /kaggle/input/<name>/:
    dataset_ball/{images,labels}/{train,val}
    dataset_ball_labeled/{images,labels}/{train,val}
    ball_orange_best.pt          <- the base model to fine-tune from

This mirrors dataset_ball_human.yaml (the two real dirs, no synthetic aug) so the
cloud run reproduces the local `--data dataset_ball_human.yaml --base
runs/detect/ball_orange/weights/best.pt --freeze 10` config. See
process/KAGGLE_TRAINING.md for the full workflow.

Run:  python tools/pack_kaggle_dataset.py
Out:  data/kaggle/shotlab_ball_dataset.zip
"""
from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "kaggle"
OUT_ZIP = OUT_DIR / "shotlab_ball_dataset.zip"

DATA_DIRS = ["dataset_ball", "dataset_ball_labeled"]
BASE_WEIGHTS_SRC = ROOT / "runs" / "detect" / "ball_orange" / "weights" / "best.pt"
BASE_WEIGHTS_ARCNAME = "ball_orange_best.pt"

# only ship what training needs (images + labels), skip stray caches/contact sheets
KEEP_SUFFIXES = {".jpg", ".jpeg", ".png", ".txt"}


def main() -> int:
    # preflight: everything must exist before we build a partial archive
    missing = [d for d in DATA_DIRS if not (ROOT / d).is_dir()]
    if missing:
        sys.exit(f"missing dataset dirs: {missing} (run the labeling/ingest step first)")
    if not BASE_WEIGHTS_SRC.exists():
        sys.exit(f"missing base weights: {BASE_WEIGHTS_SRC}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_img = n_lbl = 0
    # ZIP_STORED: images are already-compressed jpg; storing is far faster and the
    # size delta is negligible.
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_STORED) as z:
        for d in DATA_DIRS:
            src_root = ROOT / d
            for p in sorted(src_root.rglob("*")):
                if not p.is_file() or p.suffix.lower() not in KEEP_SUFFIXES:
                    continue
                z.write(p, p.relative_to(ROOT).as_posix())
                if p.suffix.lower() == ".txt":
                    n_lbl += 1
                else:
                    n_img += 1
        z.write(BASE_WEIGHTS_SRC, BASE_WEIGHTS_ARCNAME)

    size_mb = OUT_ZIP.stat().st_size / 2**20
    print(f"wrote {OUT_ZIP}")
    print(f"  images: {n_img}  labels: {n_lbl}  + {BASE_WEIGHTS_ARCNAME}")
    print(f"  size: {size_mb:.0f} MB")
    print("\nNext: upload this zip to Kaggle as a Dataset (kaggle.com/datasets -> New)")
    print("then run kaggle/shotlab_train.ipynb against it. See process/KAGGLE_TRAINING.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
