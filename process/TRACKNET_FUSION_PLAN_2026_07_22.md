# Plan: better flight-ball tracking (REVISED after Codex + Fable review, 2026-07-22)

> ## ✅ STEP-1 GATE RESULT (2026-07-22) — TrackNet is OFF the table; fix the tracker.
> Experiment 1b/1c (`tools/exp_subthreshold_signal.py`, full output
> `process/step1_gate_results.txt`) run over the 921 labeled ball-present frames of
> the 0720 session (system-python ONNX-DirectML detector):
> - Detector hit-rate: **99%** at conf 0.01 (98% far / 100% mid / 99% close); 95% at
>   the pipeline's conf 0.25. Motion detector 15% (not the lever).
> - **Of the 46 frames the current pipeline (plain@0.25) misses, plain@0.01 recovers
>   85% (67% of the far misses).** Tiling adds nothing (tiled@0.01 == plain@0.01).
>   Motion recovers 0%.
>
> **Decision per the plan's own rule (HIGH recovery ⇒ signal exists sub-threshold):**
> - ❌ **Step 4 (TrackNet/WASB) NOT justified** — the detector already sees the ball;
>   a temporal model would recover shots the detector produces at low confidence.
> - ❌ **Tiling not the lever** on this footage (no gain over plain@0.01).
> - ✅ **Go to Step 2 — fix the tracker.** The greedy `assemble_track` discards the
>   low-conf detections holding 85% of the misses.
> - ⚠️ Caveat: 921 frames from the deliberately-badly-filmed 0720 session; far bucket
>   has only 3 misses. 1a (oracle ceiling on the real tracker, well-filmed sessions)
>   still worth running to size the shots-recovered prize before the full Step-2 build.
>
> **Started:** `assemble_track` velocity + reset bugs FIXED (track.py) — per-frame
> velocity now divided by the gap it spanned; resets now actually start a one-point
> arc instead of bridging velocity across shots. 35/35 test files green.
>
> ## ⭐ STEP-1a ORACLE CEILING (2026-07-22) — DETECTION IS NOT THE LEVER; SEGMENTATION IS.
> `tools/exp_oracle_ceiling.py` (output `process/step1a_oracle_ceiling.txt`): ran the
> REAL `assemble_track` + segmenter over the 1340 labeled 0720 frames, four ways —
> baseline (YOLO@0.25), cloud (YOLO@0.01), **oracle** (baseline + a conf-0.99
> candidate at the GROUND-TRUTH center on every ball-present frame = detector never
> misses), oracle+4px. Shots recovered (TOTAL over 3 clips):
> **baseline 6 · cloud 3 · ORACLE 4 · oracle+4px 3.**
> - **Perfect detection did NOT increase shots — it DECREASED them (4 < 6); the
>   low-conf cloud was worse (3).** More/denser candidates *hurt*.
> - Combined with the 1b gate (detector already sees the ball 99%), the prize from a
>   better detector or a temporal model is **~zero or negative here.**
> - ⚠️ CONFOUND: 1a used `segment_shots` (the gap-split FALLBACK); production
>   `run_phase1` uses rim-anchored `detect_shots_to_rim` when calibrated. Gap-split is
>   known-fragile to dense tracks (memory: "naive gap-splitting FAILS with a dense
>   detector → runs merge → only 1 clean arc; FIX = rim-anchored"). So the exact
>   counts are segmenter-specific — but the DIRECTION (more detection ≠ more shots)
>   is robust and matches 1b.
> - **ROOT CAUSE it exposes:** ball-PRESENCE ≠ ball-in-FLIGHT (clip 1 has 692 present
>   frames — dribble/hold/retrieve, not one shot). Flooding the greedy tracker with
>   true positions can't help because the limiter is telling *flight* from *presence*
>   = **attempt detection / segmentation**, exactly the reviewers' "separate
>   attempt-COUNT from arc-ELIGIBILITY."
>
> ## ⛔ REVISED DECISION (after 1a+1b together)
> - ❌ **Do NOT build the Step-2 recipe as written** (low-conf candidate cloud +
>   parabola-RANSAC over the cloud). 1a shows adding candidates to the current
>   greedy tracker + segmenter REGRESSES shot count. The detector-fusion emphasis of
>   this whole plan is aimed at the wrong bottleneck.
> - ✅ **The lever is SEGMENTATION / attempt-detection** (rim/release/audio-anchored
>   "was this an attempt?"), plus **filming closer** (the free, biggest lever per all
>   prior notes — cleaner flight, bigger ball, fewer presence-vs-flight ambiguities).
> - The `assemble_track` velocity/reset fix stays (correct regardless).
> - Clean follow-up if we still want a hard number: re-run the 1a oracle through
>   rim-anchored `detect_shots_to_rim` (needs a 0720 calibration) to size the
>   attempt-detection prize specifically — but detector work is off the table.

> **Both reviewers: the original "build TrackNet + fusion" draft was OVER-SCOPED.**
> Verdict = **measure-first, then build the CHEAPEST thing that works — which is
> almost certainly a TRACKER fix, not a new model.** TrackNet/WASB is a last resort,
> gated on cheap experiments failing. This doc is the revised plan; the original
> draft is superseded by what's below.

## Reframes the review forced (premises the draft got wrong)
- **The "~28px HARD floor" is NOT established.** It was measured on PLAIN inference;
  the one tiling test that was "a wash" ran on a clip whose ball is median ~80px (no
  tiny-ball problem). `ball_native2` showed tiled +74% on far footage. Nobody has
  scored the fully-trained kaggle model, TILED, on the genuinely-far clip vs the
  1,340 labels. That's a 1-hr test that may move the floor with no new architecture.
- **"Apex frames" is a RED HERRING.** The apex is the vertex of the parabola FIT —
  you need points on the rise+fall, not at the apex, and the ball is SLOWEST there
  (blur peaks at release/rim, not apex). The only prize is DROPPED shots + too-sparse
  fits, not apex gaps. Drop apex-recall as a metric.
- **The bottleneck is likely the TRACKER, not the detector.** `assemble_track`
  (track.py:31-87) keeps ONE greedy candidate/frame, coast 4, resets to highest-conf
  — it THROWS AWAY the sub-threshold detections the detector already produces. The
  07-21 conf-0.05 experiment got +38% ball frames but only +1 shot *for this reason*.
  Also has a velocity bug: displacement is multiplied by the next frame-gap without
  dividing by the previous gap → irregular detections mispredict. Fix regardless.
- **This may be compensating for tripod placement.** The 07-20 session was a filming
  rule #1 violation ("wide cam TOO FAR"). Measure drop-rate on WELL-FILMED sessions
  (0703/0710) too — if those drop ~0 shots, the fix is the filming checklist + at
  most gap-ROI, and TrackNet is gold-plating two weeks of ML to fix camera distance.

## The reordered plan (cheap -> expensive; each gates the next)

### STEP 1 — Two measurement experiments (~1 day total, existing code+labels). THE GATE.
1a. **Oracle ceiling (Codex):** insert the 921 ground-truth centers wherever YOLO
    missed, rerun the REAL tracker/RANSAC/segmentation/metrics. Measure: shots
    recovered, fit-failure rate, and how much survives ±3-5px localization noise.
    = the maximum any temporal model could ever buy. Run on well-filmed sessions too.
1b. **Sub-threshold signal (Fable):** run the current ONNX at conf~=0.01 (plain AND
    tiled) + `MotionBallDetector` over the hard frames; measure how often ANY
    candidate lands within ~1.5r of the true center on frames the conf-0.25 pipeline
    misses. HIGH hit-rate -> signal exists, the tracker is the problem, NO new model.
    NEAR-ZERO -> floor is real at the logit level; the ONLY result that justifies a
    temporal model.
1c. **(1 hr) Tiled kaggle model on the genuinely-far clip vs the 1,340 labels** —
    does tiling help when the ball IS tiny (unlike the median-80px clip we tested)?

**Decision:** if 1a shows few dropped shots on well-filmed footage, or 1b shows the
signal is already there sub-threshold → skip TrackNet entirely, go to Step 2.

### STEP 2 — Fix the TRACKER + cheap fusion (~3-5 days, ZERO training). The likely winner.
Offline two-pass pipeline (NOT a per-frame detector — fusion is a pipeline STAGE):
- Pass 1: YOLO everywhere -> find candidate flight windows.
- Pass 2, inside windows: keep the FULL low-conf candidate cloud + union
  `MotionBallDetector` (scale-free, fires on a 10px blob) + **physics-seeded gap-ROI
  re-detection** (reuse the Mode-1 labeler's predict step at INFERENCE: crop ~160px
  around the predicted position, upscale, re-run YOLO -> a 20px ball becomes 80px).
- Replace greedy assembly with **parabola-hypothesis RANSAC over the cloud** (reuse
  `fit_parabola_ransac` + `ballistic.py` honesty gates). Filled points densify
  existing arcs, NEVER create a shot alone, must pass conf + radius-consistency.
- Fix the assemble_track velocity bug.
This IS "best of both worlds" (YOLO + motion + physics-ROI), cheaply, no new model.

### STEP 3 — Label 2-3 more sessions regardless (cheap; feeds every path incl. single-frame retrains).

### STEP 4 — Temporal model, ONLY if 1b shows signal genuinely absent AND Step 2 fails.
- Use **WASB** (beats TrackNetV2; has an OFFICIAL basketball checkpoint — test
  zero-shot first), not TrackNetV3. 3-frame full-res heatmaps.
- Native-res corridor tiles (full-frame low-res deletes a 20px ball). SAME crop
  coords across the whole stack; build stacks on REAL PTS spacing (VFR: Pixel drifts
  30->24fps, `frame_times_cached`); score the single-frame baseline on tiles too.
- Train on Kaggle, export ONNX/DirectML. Need 3-5 labeled sessions (~3-5k frames),
  1 fully HELD-OUT session before promotion; split val by SESSION never frame.
  Realistic: 1-2+ weeks.

## Fusion decision (settled by review)
- ❌ heatmap-level fusion (most code, unmeasurable benefit at this scale).
- ❌ gated-cascade-behind-`BaseDetector` (needs track context/lookahead a per-frame
  `detect()` can't see).
- ✅ two-pass pipeline stage with source-aware candidate cloud + ballistic-trajectory
  selection. `BaseDetector` stays for DETECTORS; fusion is a pipeline stage.

## Biggest risk (both reviewers agree)
Building/training a temporal model to recover shots that RANSAC already recovers, or
that the current detector already SEES sub-threshold and the greedy tracker discards.
Step 1 adjudicates this for ~1 day before any real work.
