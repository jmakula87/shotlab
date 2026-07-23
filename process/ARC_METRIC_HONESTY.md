# Arc / form metric honesty pass (2026-07-23)

From the broad dual review (items e/f): the single-camera arc & form metrics carried
an unquantified oblique-camera bias and were reported over-confidently. This pass makes
the CONFIDENCE honest without silently changing metric VALUES. Values that would need a
real change are listed at the bottom for the owner to decide (they alter coaching numbers).

## What changed (safe: labeling/confidence only)

1. **Camera geometry is no longer hard-coded `side_on`.** `session.py` hard-coded every
   shot as `camera_angle="side_on"`, which drove HIGH-confidence angle reporting even on
   this behind/oblique footage. Now `build_session --camera {side_on|oblique|behind|unknown}`
   (default `unknown`) threads through to the pose path AND the arc-angle confidence, and is
   stored on each `ShotRecord`. **Pass `--camera side_on` ONLY when you film perpendicular
   to the shot.**

2. **Arc-angle confidence is now camera-aware** (`report.py`). release/entry angle:
   was **"high"** regardless of camera → now **"medium" side-on**, **"low" (image-space
   diagnostic) oblique/unknown**. Monocular angles never rate "high" (no depth). apex_height:
   capped at "medium"/"low" (ruler disagreement, below).

## Quantified issues (evidence)

- **Two rulers disagree ~1.6x.** With the corrected rim (r~36): rim ruler ~48-52 px/ft vs
  ball-bbox ruler ~75-79 px/ft. The rim is foreshortened by the oblique view, so the rim
  ruler under-estimates px/ft. Net: apex-height in feet has ~1.6x uncertainty. (The rim
  RECALIBRATION already fixed the gross inflation: apex-above-rim is now a realistic
  ~1.8 ft median, was 5-9 ft with the old 8px rim.)
- **Entry angle is rim-x sensitive** (~2 deg per the corrected center); it evaluates the
  fitted parabola's tangent AT rim_x, even if rim_x is an extrapolation past the tracked arc.
- **Release angle** is the tangent at the min/max TRACKED x, not a detected release event --
  if tracking starts mid-flight the "release" point isn't the real release.
- **apex_height_ft** (`arc.py`) is height above the LOWEST TRACKED point (varies with where
  tracking began), NOT above rim/floor. The trustworthy one is `apex_above_rim_ft` (rim-scaled).

## Value-change candidates -- OWNER DECISION (these move coaching numbers, can't validate vs GT here)

- Report apex-height as a RANGE (rim-ruler .. ball-ruler) instead of a single number, or
  switch apex-above-rim to the ball ruler (more accurate at the ball's depth).
- Constrain entry angle to the DESCENDING observed arc (reject if rim_x is an extrapolation).
- Anchor release angle to the pose/ball release frame, not min/max tracked x.
- Deprecate/rename `apex_height_ft` (above-lowest-tracked-point) in favor of `apex_above_rim_ft`.

None of these are done -- they change the numbers you coach on, and there's no ground-truth
height/angle to validate them against on this footage. The synthetic harness
(`make_synthetic_clip`) validates side-on recovery to ~0 deg but not oblique bias. The real
fix for trustworthy angles is the 2-camera rig (already scaffolded, unvalidated).
