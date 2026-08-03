# ShotLab — KICKOFF (read this first on restart)

Last updated: **2026-07-29** · all pushed, tree clean.
Location: `C:\Users\jmaku\Desktop\ShotLab`. Read-order: **this → `PROJECT_NOTES.md`
(the living log) → `process/reviews/2026-07-23_broad_*` → `process/EVAL_HARNESS_RUNBOOK.md`**.

## ⭐ FROZEN RESULT — held-out 07-29, 3 of 4 clips, 103 hand-counted attempts
- **DETECTION GENERALIZED — settled. recall 88/103 = 85% [CI 77-91], precision 0.98.**
  Per clip 86% / 82% / 92%; the CI straddles the 86% baseline, on footage never trained on at
  a ball scale never seen. The C1→C5 ladder reproduced its shape every time, so the beam and
  rim-recovery passes are not artefacts of the clips they were built on.
- **MAKE/MISS IS AT CHANCE: learned 44/88 = 50% [CI 40-60], geometric 45/88 = 51%.** Both.
  81% is far outside. The per-clip geometric swing (33/52/64) tracks clip make-rate, not
  skill. **Pre-registered in advance**: `make_visual`'s raw-orange-mass features scale with
  rr², and rr moved 36 → ~120. **Task 6 (normalise by rr², re-fit) is the highest-value item
  left** and is unblocked.
- ⭐ **Airballs 4/4, all via the rim-recovery pass, none by C1-C4** — the pass added for
  rim-reaching shots is what makes the "blind by design" airball case visible.
- ⚠️ Rims are MEASURED from the rim's paint, not clicked (clip 1's click was 22% short), with
  ball frames excluded so only truncation remains. Clip 3 takes clip 2's radius: same camera
  position, and clip 3's rim is never fully unoccluded. Clip 1 deliberately left at its filed
  value. Clip 4 uncounted. Detail in PROJECT_NOTES "FROZEN 07-29 RESULT".

## Where the pipeline WAS (baseline, hand-count of 3 clips = 111 attempts, detector trained on them)
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
1. **An untouched test session — THE FOOTAGE IS HERE (2026-07-29), the hand-count is not.**
   Extracted and smoke-tested; details in the PROJECT_NOTES 07-29 section. **FOUR** wide
   clips, **4K30 CFR**, ~26.5 min total, and the ball now reads ~22 px radius in model space
   (vs ~10 px on 0720) — he filmed closer. The detector was trained on the 0720 clips, so
   the *absolute* 86%/81% is what this session tests. ⚠️ **Read the PRE-REGISTRATION in
   PROJECT_NOTES before running it** — pixel-tuned gates and the make/miss features are
   scale-dependent, and there is only ONE untouched session, so no knob may be re-tuned on
   it. Decide which clips are in the eval BEFORE seeing any result. Run FROZEN:
   ```
   python -X utf8 tools/hand_count.py --clip PXL_20260729_155320813
   python -X utf8 tools/verify_rim.py --clip PXL_20260729_155320813
   python -X utf8 tools/eval_ablations.py --clip PXL_20260729_155320813
   ```
   ...then the same three per clip for `PXL_20260729_155914855-001`,
   `PXL_20260729_160743954`, `PXL_20260729_161439291-002`. **`--clip` takes the STEM, not a
   path** (both tools resolve against `data/raw/Camera 1`). ⚠️ **A separate rim per clip —
   the camera moved between clips.** Count FRESH (never seeded from detections; that's what
   makes the eval honest). `verify_rim` starts fresh each run and REPLACES the clip's rim
   file on save, so click every position that clip needs in one session. Workflow:
   `process/EVAL_HARNESS_RUNBOOK.md`.
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
- **Close-cam clips: use `data/raw/Camera 2/upright/`, not `data/raw/Camera 2/`.** The 07-29 S8
  was mounted inverted with a bogus `-90` rotation flag; the `upright/` copies are the fixed
  ones (lossless stream copy). Pass `--close-dir` to the close-cam tools; `flare_report.py` has
  `CLOSE_DIR` hardcoded.
- **Previous session's raw clips live in `data/raw/Camera 1/Old/`** (verified byte-identical to
  the footage behind the frozen baseline). No pipeline glob is recursive, so `Old/` and
  `upright/` never leak into a `Camera 1`/`Camera 2` clip glob — but don't point `--clips` at them
  by accident either. ⚠️ To re-run the 0720 eval you must move those clips back UP into
  `data/raw/Camera 1/`: `hand_count`/`verify_rim`/`eval_ablations` resolve `--clip` as
  `data/raw/Camera 1/<stem>.mp4` and will say "clip not found" while they sit in `Old/`.
- **`--audio` is now OFF under `--validated`** (fixed 2026-08-02) so the profile matches what
  the eval measured — audio fills make/miss *unknowns*, was wrong 13/20 in that role, and none
  of those fills are scored. It stays ON by default otherwise, and an explicit
  `--audio`/`--no-audio` always wins.
- ⚠️ **HARDWARE:** 6 silent power-losses in 5 days under GPU load → suspect PSU. Watch temps on long runs.

## Full detail
`PROJECT_NOTES.md` = the living log (dated sections for every decision). Memory topic file:
`~/.claude/.../memory/shotlab_shot_analyzer.md`.
