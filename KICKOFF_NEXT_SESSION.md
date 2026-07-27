# ShotLab — KICKOFF (read this first on restart)

Last updated: **2026-07-23** · HEAD `67aebde` · all pushed, tree clean.
Location: `C:\Users\jmaku\Desktop\ShotLab`. Read-order: **this → `PROJECT_NOTES.md`
(the living log) → `process/reviews/2026-07-23_broad_*` → `process/EVAL_HARNESS_RUNBOOK.md`**.

## Where the pipeline is (all measured against a hand-count of 3 clips = 111 attempts)
- **Shot detection: recall 86% / precision 0.99** (greedy tracker ∪ beam-MHT ∪ rim-recovery).
- **Make/miss: 81% LOCO** (learned `make_visual` re-fit on the 89 new labels; geometric was ~51% coin-flip).
- **Run the validated pipeline with ONE flag:** `build_session --validated`
  (= `--detector yolo --weights runs/detect/ball_gpu_kaggle/weights/best.onnx --imgsz 1280
  --stride 1 --beam --make-model auto` + uses the `config/rim_<clip>.json` rims).
  ⚠️ Plain defaults (motion / imgsz 768 / yolo11n) are NOT the validated config.

## What got done in the 2026-07-23 arc (all committed/pushed, 39/39 tests)
1. **Hand-count eval framework** (`tools/hand_count.py`, `verify_rim.py`, `eval_ablations.py`,
   `diagnose_misses.py`, `refit_make_model.py`) — owner hand-counted all 3 clips.
2. **Beam tracker** (`track_beam.py`) + **rim-recovery** (`rim_recovery.py`): recall 55%→86%.
3. **Broad dual review (Codex+Fable)** found make/miss was unmeasured & ~coin-flip → **fixed**:
   make/miss is now a permanent eval gate + wired the learned model into production.
4. **Rim recalibrated** (edge-to-edge + ball-diameter sanity) — this UNLOCKED make_visual AND
   fixed apex-height inflation (now ~1.8 ft, was 5-9 ft).
5. Reviewer items worked in order: max-cardinality matcher + tol sweep; rim-recovery; prod/eval
   calibration parity; cache code-hashing; beam coast fix; **arc-metric honesty pass**
   (camera-aware angle confidence, `--camera`; `process/ARC_METRIC_HONESTY.md`).

## OPEN — the only things left, both need the OWNER
1. **An untouched test session (NEW footage).** The detector was trained on clips 1-2 / val
   clip 3, so the *absolute* 86%/81% won't fully generalize (the tracker *delta* will). Film a
   fresh session (different day/framing/clothing/cadence), hand-count + rim it, and run the eval
   FROZEN. This is the highest-value next step. Workflow in `process/EVAL_HARNESS_RUNBOOK.md`.
2. **Arc-metric VALUE changes (coaching-facing, owner judgment).** Listed in
   `process/ARC_METRIC_HONESTY.md`: apex as a rim..ball ruler range; entry angle constrained to
   the descending arc; release angle anchored to the release frame; deprecate `apex_height_ft`.
   The real fix for trustworthy angles = the (scaffolded, unvalidated) 2-camera rig.

## Gotchas (bit us before)
- **ONNX-DirectML inference runs under SYSTEM python** (`AppData\Local\Microsoft\WindowsApps\
  python.exe`), NOT the `.venv_*` envs. Always `python -X utf8`.
- **verify_rim: click the rim's LEFT then RIGHT edge** (center=midpoint, radius=half-span) — the
  old center+near-center click gave an 8px radius that corrupted make/miss + apex scaling.
- **`--camera side_on` ONLY when filmed perpendicular** — otherwise angles are low-confidence
  image-space diagnostics (the 0720 footage is oblique/behind).
- ⚠️ **HARDWARE:** 6 silent power-losses in 5 days under GPU load → suspect PSU. Watch temps on long runs.

## Full detail
`PROJECT_NOTES.md` = the living log (dated sections for every decision). Memory topic file:
`~/.claude/.../memory/shotlab_shot_analyzer.md`.
