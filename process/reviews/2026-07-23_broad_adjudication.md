# Broad dual review (Codex + Fable) — adjudication, 2026-07-23

Owner: "are you certain there aren't OTHER areas we can improve? Have the two reviewers check."
Codex verbatim: `2026-07-23_broad_codex.md`. Fable's full review is captured in the session
transcript (agent a71165b7). This file records the coordinator's independent verification and
what was acted on. Mutual-pushback: concede what verifies, refute what doesn't.

## Convergence (both reviewers + coordinator probe) — ACCEPTED + ACTED
1. **Make/miss unmeasured & ~coin-flip.** Coordinator probe: geometric production 51% aggregate
   (44/64/46% per clip, ~40% abstention), miss-biased. Audio fusion anti-signal. → Added make/miss
   scoring to eval (permanent gate); re-fit make_visual on the 89 labels → **81% LOCO**; wired into
   production (`--make-model auto`), audio demoted. ACTED.
2. **Rim radius broken** (8-12px clicked vs ~36-47px true). Corrupts make/miss + apex-ft. → `verify_rim`
   edge-to-edge + ball-diameter sanity; owner re-clicked all 3 (r~36-39). ACTED.
3. **Train/test leakage** — `ingest_labels.py --val-clip 153054`: detector trained on clips 1-2,
   val on clip 3. VERIFIED [code]. → absolute recall optimistic; tracker delta still valid. OPEN.

## Adjudicated corrections
- **Fable corrects the coordinator (×2):** (a) my "hoop-center ~555, entry bias 8.5°" was WRONG —
  555 is the rim's LEFT EDGE; true center ~602-616, real entry bias ~1.5-2°; the RADIUS was the bug.
  ACCEPTED. (b) my "10/clip detection-limited misses" — Fable checked all 22 residual FNs, each has
  near-rim detections → still tracker/segmenter-recoverable, NOT detection-limited. ACCEPTED as a
  strong finding (not yet independently re-verified by coordinator); "film closer is the wall" retracted.
- **Coordinator corrects Fable (×1):** Fable's headline "make_visual → 85% with a corrected ROI" did
  NOT reproduce at face value — coordinator measured make_visual at r=47 → 50%, r=8 → 71%, and only
  the owner's actual edge-to-edge r=36 → 88% (clip1) but 50% (clip2). The shipped model is BRITTLE
  cross-clip; the honest number is the **re-fit 81% LOCO**, not a ROI-tuned 85%. So "just wire it in
  for 85%" is REFUTED; the re-fit + LOCO validation was the correct path.
- **Codex couldn't run python** (managed env) so its make/miss numbers were cache-limited (4/10);
  Fable + coordinator ran the full measurement. Codex's structural claims (leakage, make_visual exists,
  cache-no-code-hash, incompatible rulers, beam-not-true-MHT) all VERIFIED.

## Still OPEN (ranked, not yet acted)
1. Honest held-out test session (leakage) + max-cardinality matcher + tolerance sweep.
2. Rim-anchored backward-recovery pass for the residual ~20% (reviewers: still tracker-recoverable).
3. Production/eval calibration PARITY (process_clip ignores verify_rim rims; build_session defaults
   imgsz768/motion/yolo11n vs eval 1280/best.onnx/beam) + publish the validated flag profile.
4. Cache signatures should hash CODE/schema (stale-serving risk).
5. Arc/form metrics: oblique-camera bias unquantified; report raw+uncertainty or gate confidence.
6. Beam empty-frame coast accounting (same cross-gap class as the walk-back fix); make it a true beam.
7. Pose/form: hardcoded side_on; release=wrist-apex (10-15° elbow bias); release-height uses wrist not ball.

## VERDICT
The tracker work holds, but "80% recall / detection-limited is the wall" was an incomplete story: the
product's headline output (make/miss) shipped at ~51% and unmeasured. Fixed the top item end-to-end
(measure → re-fit 81% LOCO → wire in) + the rim calibration that unlocks it. The leakage, calibration
parity, and residual-recall items remain the highest-value open work.
