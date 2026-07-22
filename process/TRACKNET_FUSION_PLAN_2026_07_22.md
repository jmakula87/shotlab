# Plan: better flight-ball tracking (REVISED after Codex + Fable review, 2026-07-22)

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
