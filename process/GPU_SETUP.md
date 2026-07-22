# GPU on ShotLab — Radeon RX 9070 XT (RDNA4 / gfx1201), Windows

This box has an **AMD Radeon RX 9070 XT (16 GB)** + Ryzen 9 7900X. The GPU is
usable for BOTH inference and training, but by two different paths (CUDA is
NVIDIA-only, so everything here is the AMD route). Established 2026-07-22.

> ## ⛔ FROZE THE MACHINE — 2026-07-22. Do NOT re-run GPU *training* as-is.
> The ROCm training path below **hard-locked the whole computer** on its first
> attempt (Kernel-Power 41, dirty shutdown at 10:10:08 — required a manual power
> cycle). No WHEA hardware error, no TDR (Event 4101), no BSOD dump → a full
> **ROCm-Windows GPU compute deadlock** that took the display/UI down with it.
> Trigger: `ball_human_gpu` (yolo11n, imgsz 1280, **batch 48**, device 0) at the
> start of epoch 1 — **while a CPU training job was still running** (extra
> CPU+RAM pressure). See the freeze post-mortem in `PROJECT_NOTES.md`
> (Session log 2026-07-22 — GPU-training freeze).
>
> **Correction to the claim below:** the "~3 min/epoch, verified" line was NOT
> substantiated — **no GPU run ever completed an epoch** (neither the batch-16
> smoke nor the batch-48 full run wrote a `results.csv`; the second froze the
> box). GPU *training* on this hardware is **UNVERIFIED and currently unsafe**.
> **GPU *inference* (DirectML/ONNX, §1) is a separate, proven-stable path and is
> NOT implicated.** Mitigations (TDR/watchdog/isolation/WSL2) pending the
> 2026-07-22 Codex+Fable consult (`process/GPU_FREEZE_CONSULT_2026_07_22.md`).

---

## 1. Inference (detection) — DirectML/ONNX — the default, in the MAIN env

Runs in the normal Python 3.13 environment (no special setup beyond
`onnxruntime-directml`, already in requirements). Export the detector to ONNX and
point `--weights` at the `.onnx`; `shotlab/phase1_ball/detect_yolo.py` runs it via
onnxruntime's `DmlExecutionProvider` on the Radeon.

```bash
# one-time: export a trained detector to ONNX (fixed 1280 input)
python -X utf8 -c "from ultralytics import YOLO; YOLO('runs/detect/<name>/weights/best.pt').export(format='onnx', imgsz=1280)"

# then just use the .onnx as weights -> detection runs on the GPU (~20x vs CPU)
python -X utf8 build_session.py --clips "data/raw/Camera 1/PXL_*.mp4" \
    --detector yolo --weights runs/detect/<name>/weights/best.onnx \
    --imgsz 1280 --stride 2 --chunk-frames 7000 --pose --shooter-height 70in --out data/out/session
```

Measured: **6.2 ms/frame on the GPU vs 121 ms on CPU (19.6×)** at imgsz 1280.
Works on Python 3.13. Nothing else needed.

> ### ⚠️ 2026-07-22 regression + fix: don't let plain `onnxruntime` co-exist
> The main env had **both** `onnxruntime` (CPU) and `onnxruntime-directml`
> installed (a later `torch` install dragged in the plain CPU package). They share
> the `onnxruntime` import namespace, the CPU build shadows the DirectML one, and
> `get_available_providers()` silently drops `DmlExecutionProvider` → **all
> detection quietly fell back to CPU (~20× slower) with no error.** Fix:
> ```bash
> python -m pip uninstall -y onnxruntime onnxruntime-directml
> python -m pip install onnxruntime-directml   # the DirectML build INCLUDES a CPU fallback provider
> ```
> Verify it's actually on the GPU (should print DmlExecutionProvider + ~5 ms):
> ```bash
> python -X utf8 -c "import onnxruntime as ort; print(ort.get_available_providers())"
> ```
> Re-verified after the fix: **5.1 ms/inference at imgsz 1280 on the RX 9070 XT.**
> **Never `pip install onnxruntime` (plain) into this env** — it's the CPU build
> and it kills GPU detection. Only `onnxruntime-directml` belongs here.

---

## 2. Training — ROCm 7.13 + BatchNorm bypass — in an ISOLATED Python 3.12 env

Training needs PyTorch on the GPU. On AMD/Windows that means **ROCm**, which needs
**Python 3.12** (the main env is 3.13, too new) — so training lives in a separate
venv, `.venv_rocm713` (gitignored, ~several GB). The working 3.13 env is untouched.

### The one gotcha: MIOpen can't compile BatchNorm for gfx1201

On ROCm-Windows (both 7.2.1 and 7.13.0 preview), MIOpen's BatchNorm kernels fail
to JIT-compile for gfx1201 — HIPRTC dies with `'type_traits' file not found`
(a header-packaging bug, [ROCm/ROCm #6150](https://github.com/ROCm/ROCm/issues/6150)).
Everything else works, **including convolution**. So we route BatchNorm through
pure-torch primitives (`tools/rocm_bn_patch.py`) — conv stays GPU-accelerated, BN
never calls MIOpen. `train_ball.py` applies this AUTOMATICALLY when `--device` is a
GPU on a ROCm torch (no-op on CPU). Remove the patch once AMD ships a ROCm-Windows
build whose MIOpen BN compiles for gfx1201.

### One-time setup (already done; here for reproduction / a fresh machine)

```bash
# 1. isolated Python 3.12 venv
py -3.12 -m venv .venv_rocm713
./.venv_rocm713/Scripts/python.exe -m pip install --upgrade pip

# 2. AMD ROCm 7.13.0 preview PyTorch for gfx120X (RX 9070 XT), cp312
./.venv_rocm713/Scripts/python.exe -m pip install --no-cache-dir \
    --index-url https://repo.amd.com/rocm/whl/gfx120X-all/ \
    "torch==2.11.0+rocm7.13.0" "torchvision==0.26.0+rocm7.13.0"

# 3. ultralytics + opencv (PyPI; does NOT downgrade the ROCm torch)
./.venv_rocm713/Scripts/python.exe -m pip install ultralytics opencv-python
```
Requires the AMD driver ≥ Adrenalin 26.2.2 (this box is on a June-2026 driver).

### Sanity check (should print BN_TRAIN_OK)

```bash
./.venv_rocm713/Scripts/python.exe -c "
import torch, torch.nn as nn
from tools.rocm_bn_patch import maybe_apply; maybe_apply(0)
x=torch.randn(8,16,32,32,device='cuda',requires_grad=True)
nn.BatchNorm2d(16).cuda().train()(x).sum().backward(); torch.cuda.synchronize()
print('BN_TRAIN_OK')"
```

### Train on the GPU — same train_ball.py, run with the venv python + `--device 0`

```bash
./.venv_rocm713/Scripts/python.exe -X utf8 tools/train_ball.py \
    --data dataset_ball_human.yaml \
    --base runs/detect/ball_orange/weights/best.pt \
    --imgsz 1280 --epochs 40 --batch 48 --freeze 10 --device 0 --name ball_gpu
```

`train_ball.py` auto-applies the BN bypass and forces `amp=False` on ROCm. Then
export the result to ONNX (step 1) for GPU inference.

⚠️ **UNVERIFIED / froze the machine (2026-07-22).** The command above (batch 48)
is exactly what hard-locked the box on its first epoch — see the warning at the
top of this file. The earlier "~3 min/epoch steady-state, first epoch slower —
ROCm kernel JIT" figure was an **extrapolation from the first-epoch iteration
rate, NOT a completed epoch** — no GPU run ever wrote a `results.csv`. Treat GPU
training as unproven on this hardware until the freeze is mitigated (consult
pending). CPU training (`--device cpu`, ~11 min/epoch measured) is the
guaranteed-safe fallback and is what produced `ball_human`.

### If ROCm ever regresses / you're on a fresh machine that won't cooperate
Fallbacks (from the 2026-07-22 Codex+Fable consult, `BALLTRACK_CONSULT` sibling):
newer ROCm (7.14+) native-Windows wheels; **WSL2 + Linux ROCm** (more mature
MIOpen — highest reliability). CPU training always works as the guaranteed
fallback.

### ⛔ torch-directml training — TESTED AND RULED OUT (2026-07-22, silent wrongness)
Built an isolated `.venv_directml` (py3.12, torch 2.4.1 + `torch-directml`
0.2.5) and drove ultralytics onto the DirectML device. **It is NOT viable — not
because it crashes, but because it silently computes WRONG gradients.** Findings:
- The `privateuseone` device is accepted only via a `select_device` monkeypatch,
  and training then needs CPU-fallback shims for unimplemented ops
  (`unique(return_counts)`, `bincount`, `scatter_add_`) plus neutered CUDA-only
  memory helpers. With those, a full 1-epoch train runs at ~4.2 it/s on the GPU.
- **But the loss is corrupt.** Same batch/model, CPU vs DirectML:
  CPU `box=3.26 cls=5.15 dfl=3.01` (total 22.85) vs
  DirectML `box=0.0 cls=0.20 dfl=0.0` (total 0.39). The TaskAlignedAssigner
  produces **zero positive matches** on DirectML → box/dfl losses are exactly 0 →
  the model would train to "completion" and learn nothing about localization.
- Validation also breaks (`BatchNorm` under `inference_mode` →
  "Cannot set version_counter for inference tensor").
- Reproducers: `scratchpad/dml_train_smoke.py`, `scratchpad/dml_loss_check.py`.

Conclusion: once DirectML is seen returning *silently* wrong results, no op on it
is trustworthy for training. **Do not use torch-directml for training.** DirectML
is fine for *inference* (§1) because that path is validated against known-good
detections; training correctness is far more sensitive.

---

## TL;DR going forward
- **Detection → GPU by default (SAFE):** `export(format='onnx')`, use the `.onnx`
  via DirectML. This path is proven and did NOT cause the freeze.
- **Training → CLOUD (chosen 2026-07-22, Codex+Fable consult).** Occasional YOLO
  fine-tuning goes to **free cloud CUDA (Kaggle)** — most-tested path, ~30-60 min,
  zero freeze risk. See `KAGGLE_TRAINING.md` (`tools/pack_kaggle_dataset.py` +
  `kaggle/shotlab_train.ipynb`). **CPU** (`--device cpu`, ~11 min/epoch) is the
  offline fallback. **WSL2+Linux ROCm** is an optional local-GPU side-quest
  (`WSL_ROCM_SETUP.md`), not the default. **Native-Windows ROCm is REJECTED** —
  AMD's own 7.2 docs state "No ML training support" on Windows, and it froze the
  box. **torch-directml is ruled out** — silent wrong gradients (box/dfl=0); see §2.
- Detection env: only `onnxruntime-directml` (NOT plain `onnxruntime`) — see the
  regression box in §1.
- Keep `.venv_rocm713`; it's the (currently unsafe) ROCm training env.
  `.venv_directml` is a proven dead end (kept only for the reproducers; safe to delete).
