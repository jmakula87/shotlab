# The 2-minute camera-calibration clip

**Print `charuco_7x5_1p5in.pdf` at 100% scale (no "fit to page") and film it.**
That is the whole ask. It is ~2 minutes of your time and it is NOT the same thing
as filming a new shooting session.

## Why it's worth doing

There are **no camera intrinsics anywhere on disk** — `config/` holds only rim
files, and no ChArUco artifacts exist. Without `K` (focal length + principal
point) several finished, tested pieces of the codebase cannot run on real data:

- **`ballistic.fit_camera_tilt`** recovers the camera's pitch/roll from the
  physics of ≥2 clean arcs alone — no rim, no backboard. It is synth-validated
  (true 18°/4° → recovered 17.1°/4.3°, rmse 0.69px) and **has never been fed real
  arcs.** 137 clean arcs from the 07-29 session are sitting there waiting.
- **The 78° gate becomes camera-aware.** It currently compares an IMAGE-PLANE
  angle against a WORLD constant, which is why it rejects real shots measuring
  "81° release" — a physically impossible number that is really foreshortening
  from a camera nearly in line with the shot. That gate is **2 of the 6 remaining
  misses on 07-29**, a third of the residual 4.2%.
- **True depth (W2)** and the release/entry angles that are currently
  geometry-limited on a single camera.

Retuning the gate without intrinsics would put an assumed focal length underneath
a geometric correction. That is the exact failure pattern that cost this project
a full day on 2026-08-04, so it is being refused deliberately rather than fudged.

## How to shoot it

1. Print at **100% scale**. Measure one square with a ruler and confirm it is
   **1.5 in** — the solver's scale comes from this number, so a "fit to page"
   shrink silently corrupts it. If your print differs, note the true size.
2. Tape it to something **rigid and flat** (clipboard, cardboard). A curled sheet
   is the most common way this fails.
3. **From the wide camera's shooting position and settings** — same phone, same
   4K/30 mode, same lens. Intrinsics are per-configuration; a different zoom or
   camera app setting gives you the wrong `K`.
4. Film ~2 minutes while moving the board around: **near and far, tilted left /
   right / up / down, and rotated**. Variety of ANGLE is what makes the solve
   well-conditioned — a board held flat-on the whole time will not do it.
5. Keep it **in focus** and filling a decent part of the frame.

Drop the clip in `data/raw/` and say it is there.

## Board spec

7×5 squares at 1.5 in → 10.5 × 7.5 in, fits Letter/A4 landscape.
**24 interior corners** — the tool's default board is a coarse 4×3 with only 6,
which exists for the *far-field* case where the camera can barely resolve it.
Intrinsics are shot close-up, so the dense board is both resolvable and far
better conditioned. Regenerate with:

```
python -X utf8 tools/make_charuco.py --squares-x 7 --squares-y 5 \
    --square-in 1.5 --out-dir process/calibration_board
```
