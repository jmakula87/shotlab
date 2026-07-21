# Ball-tracking consult — Codex + Fable (2026-07-21)

Two independent external reviewers (Codex `gpt-5.6-sol`, read-only on the repo; and a
Fable agent with web search) were asked the same question: **how to recover ALL shots +
clean arcs from the far single wide camera**, given the AMD-GPU-on-Windows / CPU
constraint, and told NOT to just repeat what we'd already tried (high-res, TrackNet,
SAM2/CoTracker, ballistic-linking, film-closer).

## The consensus (they converged independently — high signal)

**The fix is not a better per-frame detector. It's an architecture flip to
track-before-detect.** Both reviewers, separately, identified the same root cause:
ShotLab discards weak evidence before it can accumulate —

- `detect_yolo.py` drops candidates below `conf` (~0.20) before tracking;
- `phase1_ball/track.py assemble_track` collapses each frame to ONE candidate and
  coasts only 4 frames;
- `court.py` then expects a near-continuous track.

We own an unusually strong **verifier** (the gravity-pinned `ballistic.py` fit — needs
only ~8 points anywhere on the arc, with a physical residual gate) and very weak
per-frame **evidence** (a 7–13 px orange smear). When the verifier is strong, the
detector's job is to be *exhaustive*, not precise. So: keep ALL weak candidates
(conf 0.005–0.05 + classical orange/motion + frame-diff), search for whole
TRAJECTORIES that accumulate evidence over 20–50 frames and terminate at the rim, and
let physics supply the precision. The orange shirt, leaves, and dribbles coexist as
candidates until temporal physics eliminates them (none traces a rising-then-falling
gravity arc to the rim).

## Merged priority (both reviewers' first-3, reconciled)

1. **Track-before-detect + classical candidate generator** (the flip). Multi-hypothesis
   / beam search or spacetime-RANSAC over a candidate soup, refined by the existing
   ballistic fit + honesty gates. Do trajectory discovery in 2D image/time first; apply
   3D/radius physics AFTER (radius is noisy at 7–15 px). Evaluable on the July-20
   session with NO new labels. Effort ~1–2 wk. Payoff very high. **Primary recovery
   system.**
2. **Native-resolution corridor tiling** ("virtual zoom"): tile only the fixed flight
   corridor (release→arc→rim, upper ~55% of frame) at native res so the ball stays
   ~20 px, event-gated to activity windows. ~1 day, pure ONNX/DirectML. Best *this-week*
   recall move; feeds #1 better candidates.
3. **Retrain for stride, not just image size:** P2/stride-4 head + native-res court
   crops + orange-shirt hard negatives + copy-paste/synthetic small-ball aug + temporal
   (frame-diff) input channels (poor-man's TrackNetV4, stays in YOLO/ONNX). Do it AFTER
   #1, which mines its training labels for free (ballistic-verified auto-labels).
4. **Capture:** 4K30 (ball 20→40 px) + fast shutter ≥1/500 s + tight framing >> 1080p60.
   Largest single multiplier; helps classical CV, YOLO, temporal models, radius, and the
   fit simultaneously. Also: don't wear the orange shirt.

## Genuinely new ideas beyond our list

- **Separate attempt-counting from arc-eligibility** (both). A shot COUNTS via a release
  trigger (pose) + rim event (net-motion 87% model / audio) even if the ball was
  tree-occluded the whole flight; only compute release/apex/entry when observations span
  ascent+apex+descent. Kills BOTH the "0.5 ft apex junk" honesty problem and the
  "8 of an estimated 30–60" denominator problem. Low effort, high payoff.
- **Use the S8 close cam as a pure TIMING sensor** (uncalibrated) to trigger/narrow the
  wide-camera search — reuses footage we already record; wide stays the sole arc camera.
- **Anchored trajectory search** (Fable B): between the pose-release pixel and the rim
  event, a gravity arc has ~2 free params (endpoint depths); grid-search ~400 arcs and
  score each by summing an evidence map (orange-chroma × frame-diff) along the path. The
  human strategy made literal; maximally occlusion-robust.
- **Center/radius refinement as a separate pass** (Codex #7): after the path is known,
  re-estimate subpixel center + elliptical blur + radius at native res and temporally
  regularize radius — the known-diameter reconstruction is very sensitive to radius error
  (1 px on a tiny ball = large depth error).
- **Streak-aware detection** (Fable E.2): a fast far ball is an elongated STREAK; the
  `min_circularity` gate deletes exactly those frames. Fit an ellipse; the major axis is
  a free velocity-direction measurement that must agree with the arc tangent.
- **Per-session color learning** in Lab / normalized-rg sampled from the ball at rest,
  not a fixed HSV box (sun + blur desaturate).
- **Lightweight temporal heatmap model** (Codex #4): a small ONNX-friendly U-Net /
  MobileNet on 5–7 frames → ball-center heatmap for the middle frame; avoids bbox
  regression (ill-conditioned at 10 px); used as one evidence source in the search.
- **Bidirectional track-from-any-hit** (Codex #6): from any confident detection, run
  CSRT/KCF/MOSSE / optical flow forward+backward to bridge 5–10-frame gaps.
- **Honesty gate first** (Fable J): hand-count one session as `shots_truth` (~20 min)
  and report shot-recall + arc-yield against it before tuning anything.

## Constraint correction (Fable, web-sourced)

**The AMD-on-Windows wall has partially fallen in 2026.** AMD now ships PyTorch wheels
for Windows on the RX 9070 XT (ROCm 6.4.4 / 7.x preview). So "CUDA-only, can't run
locally" is no longer strictly true for OFFLINE jobs (CoTracker/SAM2 auto-labeling,
TrackNet experiments, training). It's preview-grade (Py3.12, op gaps, uneven conv perf)
→ keep DirectML-ONNX as the PRODUCTION runtime, but an afternoon of ROCm setup opens the
offline-model door. (VideoCardz + AMD ROCm docs / blog.)

## Target architecture both converge on

`raw YOLO (low-conf) + orange/motion proposals → multi-candidate trajectory search →
rim/physics scoring → native-res center/radius refinement → existing ballistic honesty
gates` — vs today's `detect → one greedy track → fit parabola`.

---
Raw reports archived in the session scratchpad (codex_balltrack.md + the Fable agent
transcript). This doc is the synthesis of record.
