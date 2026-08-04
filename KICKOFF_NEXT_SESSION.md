# ShotLab — KICKOFF (read this first on restart)

Last updated: **2026-08-04**.
Location: `C:\Users\jmaku\Desktop\ShotLab`. Read-order: **this → `PROJECT_NOTES.md`
(the living log) → `process/reviews/2026-07-23_broad_*` → `process/EVAL_HARNESS_RUNBOOK.md`**.

## ⛔⭐ 2026-08-04 — "BODY METRICS ARE CAMERA-DEPENDENT" IS RETRACTED
I called opposite Cohen's-d signs across two cameras proof that pose metrics measure the
camera, not the shot, and told the owner to stop pursuing body-form drivers on that basis.
The owner asked for adversarial review; **the claim broke on every count.** Opposite signs are
the MODAL outcome under a null at these n (P(≥2 flips of 4) = 0.69); "same shots" was false
(knee overlap 15); on the shots the cameras DO share, knee agrees in sign; and I omitted the
two metrics that agreed. Full detail + what survives in `PROJECT_NOTES.md`.
- ✅ **The NULL still stands** — nothing separates makes from misses on either camera. And the
  wide camera is still disqualified for form, on independent evidence (22-40% coverage,
  outcome-correlated pose survival). The DECISION was right; the stated reason was wrong.
- ⛔ **Three real code defects came out of it**, all now fixed: the close path was measuring a
  window ending **a full second before the release** (`compute_form` re-finds its own release
  from `shot.frames[0]`, so a `rel_f±45` pseudo-shot mis-anchored everything); `release_conf`
  was never emitted, silently bypassing `correlate.py`'s confidence gate; and the close→wide
  join is not one-to-one (14 shots took 2 releases each) = pseudo-replication.
- ⭐ **The instrument was in the repo, unused.** `form.py` computed joint angles on image
  PIXELS, which cannot be compared across cameras by construction — while MediaPipe's metric
  `world` landmarks were already captured and already used elsewhere. Now emitted as
  `knee_bend_3d_deg` / `elbow_angle_at_release_3d_deg`, beside the 2D values, never replacing.
- ⚠️ **Wording correction, previously repeated by me:** "~47% physically impossible knee reads"
  was overstated. Of 58 raw wide knees, 6 were below 30° but 21 were above 150° (median 138°).
  A 178° knee is straight, not impossible — the wide camera mostly never catches the load.
- 🔬 **The pre-registered sharp test RAN and REFUTED my hypothesis:** 2D-vs-3D agreement is
  *not* worse on the oblique wide camera (knee r=+0.65 / elbow r=+0.68) than on the close
  profile camera (+0.76 / +0.46) — on the elbow the oblique camera agrees BETTER. Projection is
  not what broke the wide camera; **coverage is**, and 3D recovers none of it (58/139, identical
  to 2D). Cross-camera knee in 3D is r=+0.07, which isolates nothing.
- ⚠️ **Trust note:** one reviewer's headline claim (a "77-row, 3-clip" artifact) was FALSE —
  the file has 141 rows across 4 clips. Verify a review's factual claims before acting; it
  cuts both ways.

## ⛔⛔⛔ 2026-08-04 — BODY METRICS ARE CLOSED ON MEASUREMENT GROUNDS. READ BEFORE RE-OPENING.
Re-measuring the SAME 07-20 shots with a different pose model (`--pose-variant heavy`, same
video) — repeated measurements of identical physical events, so this yields a within-shot SD and
a **smallest detectable change**:

| metric | r | within-shot SD | **SDC (95%)** |
|---|---|---|---|
| `knee_bend_3d_deg` | **0.07** | 8.5° | **23.7°** |
| `knee_bend_deg` (2D) | 0.83 | 6.7° | 18.6° |
| `elbow_angle_at_release_3d_deg` | 0.21 | 7.5° | 20.8° |

Smoothing on/off by contrast: knee_3d **r=0.99, SDC 3.0°**. **The filter is not the noise source
— the MODEL is.** Every effect chased on 08-04 was ~5° against an SDC of **23.7°**.
- ⚠️ **The 3D landmarks are WORSE than the 2D ones (r=0.07 vs 0.83.)** They were added that
  morning expecting camera-invariance. Monocular depth is model-INFERRED; the image-plane
  projection is directly OBSERVED. **"More physically principled" ≠ "more reproducible."**
- ✅ What survives: group-level DISTRIBUTION work where random error averages down. Nothing
  per-shot, nothing at the ~5° scale.
- ⭐ **Before any future body-form claim: run this reliability check first and quote the SDC.**
  `tools/flare_report.py --pose-variant heavy` / `--no-smooth` are the two perturbations.

## ⛔⛔ 2026-08-04 (later) — `knee_bend_3d_deg` PROMOTED THEN RETRACTED THE SAME HOUR
Read this before touching any close-cam make/miss number. I promoted it on a "replication"
(07-29 +0.40, 07-20 +0.40, pooled [+0.11,+0.69]) and it was killed **twice, independently**:
1. **The two +0.40s used DIFFERENT wide-side join anchors.** `wide_shot_times` returns the
   MID-FLIGHT time; my 07-20 harness matched on the RELEASE. On a consistent anchor: 07-29
   **−0.19**, 07-20 **+0.40**. Stable core (rows both anchors agree on): **−0.07 / +0.47**.
   ⭐ The labels were fine — the **MEMBERSHIP** was not: ~25% of rows match under only one
   anchor, and that alone swings d by 0.6.
2. **Window truncation** (Codex): `span` began at `frames[0]−20` while the re-derived release
   could land 18f earlier, leaving ~2f of pre-roll, so the knee loop never reached the dip.
   corr(knee, `release_frame_delta`) = **−0.62/−0.72**. ✅ FIXED: `min(frames[0], rel_f) − 20`.
- ✅ **After the fix both symptoms vanish together**: corr → +0.20/−0.12, >150° reads 16→1 and
  14→0, and the anchor swing collapses from 0.24 to **0.01**. One defect drove both.
- ⛔ **No effect survives** (07-20 d≈+0.19, n=82, floor ~0.44). The published BlazePose-World
  knee error is 7.1-17.2° MAE against a 5.4° contrast — it was under the noise floor throughout.
- ⭐⭐ **A replication must run the IDENTICAL pipeline.** Two different pipelines agreeing on one
  number is a coincidence you went looking for.
- ⭐⭐ **Confound checks conditioned on a sample cannot see a defect in SAMPLE SELECTION** —
  coverage, distance, clip-stratification and reactivity all passed and were all irrelevant.
- ⚠️ A pose pass **crashed natively** (`access violation writing 0x…20`), silently losing 42 of
  141 releases. **Check per-clip counts in the log, not just that the file was written.**

## ⭐⭐⭐ RECALL 85% → 96% (2026-08-03), from one scale-correct constant
`detect_shots_to_rim`'s RANSAC `threshold_px` was a raw `8.0` — 0.22 rim radii at 0720 but
0.07 at 4K, starving good arcs of inliers. Now `(8/36)·rr`, which reproduces 8.0 exactly at
the 0720 rim. **137/143 = 96% [CI 91-98], precision 0.986, per clip 93/95/95/100%.**
- ⛔ The backward extension built for this (`shotlab/back_extend.py`) is **MEASURED INERT** —
  eval condition C6 is byte-identical to C5. Kept opt-in (`cloud=`, default None) with C6 as
  the standing ablation. What actually found the bug was the new `reject_log=` argument.
- ⚠️ **96% is NOT held-out** — same 143 attempts that produced the frozen 85%. The constant
  has no free parameter, so this is diagnosis not tuning, but **a fresh session is needed
  before quoting it.** The frozen 85% remains the last clean held-out number.
- ⛔ Make/miss stays **89% LOCO**. The eval briefly printed 94% by scoring a model on its own
  training shots; `models/<model>.trained_on.json` + a RESUBSTITUTION warning now prevent it.

## ⭐⭐ FROZEN RESULT — COMPLETE. Held-out 07-29, all 4 clips, 143 hand-counted attempts
- **DETECTION GENERALIZED — settled. recall 122/143 = 85% [CI 79-90], precision 0.984.**
  Per clip 86% / 82% / 89% / 85%; the CI straddles the 86% baseline, on a session never
  trained on at a ball scale never seen, with 2 false positives in 124 produced shots. The
  C1→C5 ladder reproduced its shape on every clip — the beam and rim-recovery passes are not
  artefacts of the clips they were built on. **This is a clean pass and the harness's payoff.**
- **MAKE/MISS is SESSION-SPECIFIC — diagnosed 08-03, and my pre-registered mechanism was
  WRONG.** The shipped (0720-trained) model scores 55% on 07-29, but **leave-one-clip-out
  WITHIN 07-29 is 89%** — the cues are stronger on this footage than they ever were on 0720
  (84% LOCO there). Normalising the rr²-scaled mass features, the pre-registered fix, moved
  nothing: 58% vs a 62% unnormalised control. **The original "81%" was always a within-session
  number, never a cross-session one.** ✅ `models/make_visual_0729.joblib` fitted on the 122
  labelled 4K shots; `--make-model auto` now prefers the newest model.
  ⛔ **Never quote a make/miss accuracy without naming the session it was fitted on.**
- ⭐ **Airballs 5/5, all via the rim-recovery pass, none by C1-C4** — the pass added for
  rim-reaching shots is what makes the "blind by design" airball case visible. Understand this
  before task 5 rescales any gate.
- ⚠️ Rims are MEASURED from the rim's paint, not clicked (clip 1's click was 22% short), with
  ball-adjacent frames excluded so only occlusion-truncation remains. Clips 2/3/4 share a
  camera position (centres within 3px) so they share one radius, taken from clip 2 — the only
  one whose width distribution plateaus. Clip 1 deliberately left at its filed value.
  Full detail + deviations in PROJECT_NOTES "FROZEN 07-29 RESULT".

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
