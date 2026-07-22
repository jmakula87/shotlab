# GPU-training path consult — Codex + Fable (2026-07-22)

**Question:** is WSL2+Linux-ROCm the best way to GPU-accelerate ShotLab's
OCCASIONAL YOLO fine-tuning (RX 9070 XT / gfx1201), or is there a better option?
Two independent adversarial reviews, run in parallel (read-only repo access for
Codex; the runbook docs for Fable). Prompt: `scratchpad/gpu_consult_prompt.md`.

## Both converged (high confidence)
1. **Free cloud CUDA (Kaggle preferred, Colab fine) is the best default — not
   WSL2.** For an occasional 40-epoch nano-model fine-tune on ~2000 images: ~30-60
   min on a free T4/P100, stock CUDA = most-tested path (no monkeypatches, no
   silent-op surface), zero maintenance, zero hard-lock risk. Fable prefers Kaggle
   (persistent versioned datasets, 30 GPU-hr/wk).
2. **Native-Windows ROCm: REJECT.** Codex: AMD's ROCm 7.2 Windows limitations page
   states *"No ML training support."* Fable: our OWN `GPU_SETUP.md` records the
   MIOpen BN bug hits "both 7.2.1 and 7.13.0", so "stable 7.2 fixes Windows" is
   self-contradicted.
3. **WSL2+ROCm: viable but NOT "the reliable route"** — an optional local-capability
   project, time-boxed (~90 min), second to cloud.
4. **CPU overnight: honorable known-correct fallback** (~7 hr/40 ep), not a joke.

## Divergence (immaterial)
Paid cloud: Codex #2 (dependable when free is unavailable), Fable #5 (skip; free
covers ~1 GPU-hr/session). Both agree it only matters if free quota annoys.

## Corrections they forced on our own docs
- **Freeze post-mortem was over-confident.** We asserted "ROCm compute deadlock,
  NOT hardware." Fable: we never isolated a **PSU power-transient** under
  simultaneous full CPU+GPU load (RDNA4 spikes + 7900X full-load is a classic
  transient trigger). If the cause is power, WSL2 does NOT protect us (same
  silicon/watts) — only cloud sidesteps it. Corrected in `GPU_SETUP.md`.
- **WSL runbook version error:** production ROCDXG-WSL is **ROCm 7.2.1 + Adrenalin
  26.2.2**, not base 7.2 / 26.1.1; don't mix the legacy runtime-swap steps with
  ROCDXG. Corrected in `WSL_ROCM_SETUP.md`.
- WSL still traverses `/dev/dxg` + the Windows display driver, so a compute hang is
  more likely a **TDR reset** than a full lock (safer than native) but not proven
  impossible.

## WSL gotchas surfaced (if that path is ever taken)
- Do NOT set `HSA_OVERRIDE_GFX_VERSION` (that's RDNA3 advice; breaks gfx1201, which
  is natively supported in 7.2).
- `pip install ultralytics` can clobber the ROCm torch — re-verify `torch.__version__`
  says `+rocm` after.
- `.wslconfig` memory cap (~50% host RAM default) can OOM the dataloader at 1280.
- No `rocm-smi`/`amd-smi` in WSL — use `torch.cuda.memory_allocated()`; "can't see it
  in a monitor" != "not using the GPU".
- First-epoch MIOpen JIT looks like a hang (minutes); run 2 is fast (find-db cache).

## Correctness ladder (both, for ANY backend)
1. Fixed-batch loss parity CPU vs GPU (box/cls/dfl each individually nonzero,
   ~1-2% rel) — the DirectML catcher.
2. Assert TaskAlignedAssigner positive-match count matches CPU.
3. Gradient parity (per-layer norm cosine > 0.99).
4. 3-epoch smoke: val mAP50 must climb like CPU.
5. **End-to-end (the one that matters):** train -> ONNX -> local DirectML inference
   on a labeled holdout -> mAP within ~0.02 of the CPU-trained `ball_human` golden.
6. Expect statistical, never bitwise, parity.

## Decision (owner, 2026-07-22)
**Build the Kaggle notebook path.** CPU stays the offline fallback; WSL2+ROCm is a
parked optional side-quest; native-Windows ROCm is dead. Artifacts:
`tools/pack_kaggle_dataset.py`, `kaggle/shotlab_train.ipynb`,
`process/KAGGLE_TRAINING.md`.
