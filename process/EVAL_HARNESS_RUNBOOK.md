# Full-clip attempt-evaluation harness — runbook

Built 2026-07-22 after the dual adversarial review (Codex + Fable) retired the
Step-1 oracle family as a sizing tool. See `process/reviews/2026-07-22_step1_oracle_*`.

**Goal:** replace unmatched single-digit output counts (measured inside
detector-selected label windows) with MATCHED recall + precision against a fresh,
full-clip, hand-counted ground truth. That isolates the real bottleneck
(detection vs tracking vs segmentation) with real denominators.

## The one rule the reviewers insisted on
Count attempts **fresh, by eye**. Do NOT seed the attempt list from the pipeline's
detections — that selection bias is exactly what made every prior number
meaningless. Log EVERY attempt, including airballs / bad misses the rim-anchored
detector cannot see; their (near-zero) recall sizes the attempt-detection prize.

## Steps (SYSTEM python for anything that runs ONNX-DirectML)

1. **Hand-count attempts** (owner, ~1hr for the 3 clips):
   ```
   python -X utf8 tools/hand_count.py --clip PXL_20260720_151519220
   ```
   Watch the clip. At each attempt, park near the rim-reaching frame and press:
   `m` make (reached rim) · `n` miss (reached rim) · `b` airball/bad miss.
   `u` undo nearest · `s` save · ESC quit. Autosaves to
   `process/handcount/<clip>_attempts.csv`.

2. **Set the verified rim(s)** (owner, minutes):
   ```
   python -X utf8 tools/verify_rim.py --clip PXL_20260720_151519220
   ```
   Click rim CENTER then EDGE; ENTER adds it from the current frame to clip end.
   For a clip where the camera moved (clip1 has a 110px within-clip shift), add
   position 1 at frame 0, then navigate to where it moved and add position 2 —
   the later one supersedes for later frames. `s` save →
   `config/rim_<clip>.json`. Headless: `--rim X Y --radius R --f0 N`.

3. **Run the ablations**:
   ```
   python -X utf8 tools/eval_ablations.py --clip PXL_20260720_151519220
   ```
   Detects once at conf 0.01 (cached under `data/out/eval_cands/`), derives the
   baseline by filtering to conf≥0.25, segments per frame-ranged rim through the
   PRODUCTION `detect_shots_to_rim`, matches to the hand count, and prints
   precision / recall (split rim-reached vs airball) for:
   - **C1 baseline @0.25** and **C2 cloud @0.01** — runnable now; C1→C2 sizes the
     cheap detection lever with real denominators.
   - **C3–C5** (oracle-assoc / oracle-track / arc-only) — need a dense per-attempt
     GT ball track; stubbed until supplied via `--gt-track`. C2→C4 sizes tracking,
     C4→C5 sizes segmentation/RANSAC.
   Writes `process/handcount/<clip>_eval.json`.

## How to read it
- **C1/C2 recall_rim-reached low** → the segmenter drops shots it should catch →
  work the 5 `court.py:225-291` defects (walk-back gap-crossing, gather-poisoned
  RANSAC, hard 200px launch_drop, 78° gate, bounce re-approach FP).
- **C2 ≫ C1** → the free conf-0.01 cloud recovers real shots → wire it into
  production (watch precision: a junk cloud + greedy max-conf seeding is an FP risk).
- **recall_airball ≈ 0 with many airball attempts** → the rim gate is blind to a
  big chunk of attempts → attempt-detection beyond the rim gate is the prize.
- **precision low** → false positives (bounces/retrieves) → tighten the segmenter,
  don't chase recall.

## Tests
`tests/test_eval_harness.py` (in `run_tests.py`) covers the matcher (TP/FP/FN,
tolerance boundary, one-to-one), frame-ranged rim resolution (camera-move safe),
and the hand-count CSV round-trip. Each tool also has `--selftest`.
