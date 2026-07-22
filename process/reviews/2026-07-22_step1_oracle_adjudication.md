# Adjudication — Step-1 oracle dual review (Codex + Fable), 2026-07-22 night

Owner request: "have Fable and Codex double-check that this is the best we can do — I feel
like we can push this further." Both ran independent read-only reviews with their own probes.
This file records what I (coordinator) independently VERIFIED, and adjudicates the one place
the reviewers disagreed. Mutual-pushback contract: concede what verifies, refute what doesn't.

## Reviewer convergence (both, independently) — ACCEPTED
1. Retraction of "perfect detection HURTS / cloud regresses" is SOUND. Fable proved the
   mechanism (perfect GT-only track through `segment_shots` drops all 3 certain makes).
2. The labeled-window design is SELECTION-BIASED and cannot size any prize: labels come only
   from previously-detected shots, and every number is an unmatched raw count (no precision/FP).
3. Tracker velocity/reset fix is correct but has ZERO test coverage.
4. Decisive next experiment = full-clip hand-counted attempt eval with staged ablations.
5. "Film closer is the biggest lever" is NOT supported by this evaluation.

## Independent verifications I ran (coordinator)
- **Production path** (`pipeline.py:96-102`): confirmed `detect_shots_to_rim` when calibrated,
  `segment_shots` only as fallback. The rim follow-up does test production. ✅
- **Fabrication** (Fable A-1): confirmed. `make_label_task.py:44-49` builds labels ONLY from each
  detected shot's flight window ±12 frames. Probe: clip1's 692 present frames = ~20 flight-window
  clusters (sizes up to 118), NOT "dribble/hold." The notes' root-cause sentence was fabricated;
  CORRECTED in PROJECT_NOTES. ✅
- **Tracker fix has no coverage** (both): confirmed `grep assemble_track tests/` → nothing. ✅

## The one disagreement — the rim — ADJUDICATED: Codex is right
- Fable A-3(ii): "rim validated, not the problem" — GT flights pass 3-7px from the recorded rim
  center in clip1 windows 2/3/6. TRUE but NECESSARY-NOT-SUFFICIENT.
- Codex §3: "rim is a material confound" — auto-rim vs full-clip cached rim differ 110/225/59px
  (≫ 90px gate); some windows have 0 points in the experiment-rim gate but 18-25 in the cached-rim
  gate.
- **My probe (labels only, clip1, min GT-to-rim per window, gate 90):**
  - EXP rim (1134,470): 9/20 windows within gate — clustered in EARLY frames (147-1296).
  - CACHED rim (1244,461): 10/20 windows within gate — clustered in LATE frames (2699-11224),
    a mostly DISJOINT set.
  - Real shots in ONE clip approach the rim at x≈1130 (early) AND x≈1244 (late): a **110px
    within-clip spread.**
- **Conclusion:** neither auto-rim captures all real shots; which windows "reach the rim" swings by
  ~half depending on an unverified 110px calibration choice. `baseline=2` is calibration-driven, not
  a stable production number. Fable's check confirmed the exp-rim is near SOME shots but didn't test
  that it's near ALL — and it isn't. The 110px within-clip spread further implies the tripod moved /
  multiple shooting positions, so a single per-clip `Calibration` is structurally wrong for this
  footage (a data problem, distinct from "film closer").

## Net decision
Retire the Step-1 oracle family as a sizing instrument. Next = the full-clip hand-counted attempt
evaluation both reviewers specified (Codex's 5-condition ablation is the concrete recipe; owner
must hand-mark attempts FRESH, not seeded from detections, per Codex §Decisive). Owner-time input
(~1hr) is the gate. Open owner choice: build the harness on the existing 3 clips vs re-film with a
LOCKED tripod first (the 110px rim spread argues the current footage is compromised). Cheap segmenter
bug-fixes (walk-back gap stop, gather-poisoned RANSAC, bounce-FP suppression, soft 78° gate,
rim-scaled launch_drop) and cloud@0.01 are all judged AGAINST the eval, not against single-digit counts.
