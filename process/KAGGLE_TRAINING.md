# ShotLab GPU training on Kaggle (the chosen training path, 2026-07-22)

**Why cloud, not local ROCm:** two independent reviews (Codex + Fable, see
`GPU_TRAINING_CONSULT_2026_07_22.md`) converged: for an OCCASIONAL 40-epoch
fine-tune of a nano model on ~2000 images, free cloud CUDA has a far better
reliability-to-effort ratio than WSL2+ROCm — stock CUDA is the most-tested path
(no monkeypatches, no silent-op surface like DirectML), it's ~30-60 min per run,
zero maintenance, and zero risk of another hard-lock. Native-Windows ROCm is out
(AMD's own 7.2 docs: "No ML training support" on Windows). WSL2+ROCm remains a
viable *optional* local-capability project (`WSL_ROCM_SETUP.md`), not the default.

GPU **inference** stays local (ONNX/DirectML on the RX 9070 XT, 5 ms/frame). Only
*training* goes to the cloud; the trained `best.pt`/`best.onnx` come back here.

---

## The loop (per filming session)

### 1. Label (local, as today)
`make_label_task.py` -> confirm/fix in the browser -> `ingest_labels.py`. This
refreshes `dataset_ball_labeled/`.

### 2. Package (local)
```bash
python tools/pack_kaggle_dataset.py
# -> data/kaggle/shotlab_ball_dataset.zip  (~884 MB: both real dirs + base weights)
```

### 3. Upload to Kaggle (browser, once per session)
- kaggle.com -> **Datasets -> New Dataset** (first time) or open the existing
  **shotlab-ball-dataset** -> **New Version**.
- Drag in `shotlab_ball_dataset.zip`. Kaggle auto-extracts it. Create/Update.
- (Privacy: keep the dataset **Private**. It contains frames of you shooting.)

### 4. Run the notebook (browser)
- First time: **Notebooks -> New**, then File -> Import -> `kaggle/shotlab_train.ipynb`
  (committed in this repo). Or open your saved copy.
- **Settings -> Accelerator -> GPU T4 x1** (or P100).
- **Add Input ->** the shotlab-ball-dataset (attach the version you just uploaded).
- **Run All.** ~30-60 min. The last cell asserts `mAP50 > 0.1` (the cloud analogue
  of the DirectML box/dfl-loss check — a backend that learns nothing fails here).

### 5. Retrieve + verify (local — the check that actually matters)
- Download `best.pt` and `best.onnx` from the notebook's **Output** tab.
- Drop `best.pt` into `runs/detect/ball_gpu_kaggle/weights/` (or wherever you keep
  canonical weights) and `best.onnx` alongside it.
- **Verify vs the golden CPU-trained baseline** (`ball_human`), per the reviewers'
  correctness ladder — don't trust a checkpoint you didn't watch train:
  ```bash
  # run the new ONNX through the proven local DirectML inference on a labeled holdout
  # and compare detections / mAP to the ball_human baseline
  python -X utf8 build_session.py --clips "data/raw/Camera 1/PXL_*.mp4" \
      --detector yolo --weights runs/detect/ball_gpu_kaggle/weights/best.onnx \
      --imgsz 1280 ...   # eyeball ball tracks vs ball_human on the same clip
  ```
  Cloud CUDA is known-correct, so this is a formality — but run it once: it's the
  end-to-end gate that would have caught the DirectML corruption.

---

## Notes / gotchas
- **Batch size:** notebook defaults to `batch=8` (safe on a 16 GB T4 at imgsz 1280).
  Bump to 16 on a P100.
- **ultralytics is pinned to 8.4.104** in the notebook to match local; Kaggle's
  CUDA torch is kept (the notebook asserts `torch.cuda.is_available()`).
- **Availability:** free GPU quota is ~30 GPU-hrs/week on Kaggle; a run needs ~1.
  Free tiers aren't guaranteed a GPU instantly — if none is offered, wait or retry.
- **The zip is gitignored** (`data/` is). Only the packaging script + notebook are
  tracked. Regenerate the zip each session from fresh labels.
- If you'd rather stay fully offline: CPU training still works
  (`train_ball.py --device cpu`, ~11 min/epoch), and WSL2+ROCm is the optional
  local-GPU route (`WSL_ROCM_SETUP.md`).
