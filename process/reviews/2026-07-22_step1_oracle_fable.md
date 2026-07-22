# ADVERSARIAL REVIEW — ShotLab Step-1 gate & rim-anchored oracle follow-up (Fable)

> Reviewer: Fable (general-purpose agent, read + read-only probe access).
> Filed verbatim. Its GT-only oracle-TRACK probe is at
> `process/reviews/2026-07-22_step1_oracle_fable_probe.py`.

**Method note:** every claim is tagged **[code]** (read from source, file:line) or **[probe]**
(a read-only experiment: a GT-only oracle-TRACK from `data/labels/ball_labels_0720.json` fed
through the real `assemble_track` and an instrumented copy of `detect_shots_to_rim`, using the
exact rim calibrations in `process/step1a_rim_oracle_ceiling.txt`; no tracked files touched).
The probe **reproduces the experiment's ORACLE=4 exactly, per-clip (3/1/0)** — auditing the same
machinery, not a divergent reimplementation.

## Headline findings

1. **The retraction is SOUND — mechanism proven.** The gap-based 1a "perfect detection recovers
   fewer shots" was definitively a `segment_shots` artifact.
2. **The rim-anchored follow-up's "+2 ORACLE" is ALSO not trustworthy:** 1 of the oracle's 4 shots
   is a **bounce-back false positive** [probe]; 2 of its misses are **label-window truncation
   artifacts** [probe]. The experiment cannot size the detection prize in either direction.
3. **The biggest measured defect is the production segmenter `detect_shots_to_rim` dropping shots
   even when detection and tracking are PERFECT.** With a flawless GT track, production recovers
   only 3 real + 1 FP of 12 labeled windows (~7-8 reach the rim gate) [probe].
4. **Decisive next experiment: a full-clip, hand-counted ground-truth run (~1 hour).**

## (A) Point-by-point verification

### A-0. Production path confirmed
`pipeline.py:97-102`: if `calib` given, shots come from `detect_shots_to_rim`; else fallback
`segment_shots`. [code] The rim follow-up tests the production segmenter; the original 1a tested
the fallback. Premise right.

### A-1. Step-1b/1c (99% @0.01, 85% miss-recovery) — arithmetic sound, population biased
- Hit test (`exp_subthreshold_signal.py:36-42`, candidate within `1.5*max(r,6)` of GT) reasonable. [code]
- **Selection bias the docs never state:** the 921 labeled frames exist ONLY inside flight windows
  of shots the OLD pipeline already detected (`make_label_task.py:83-93`, MARGIN=12, line 44). [code]
  A shot the pipeline never saw contributes ZERO labeled frames. "Detector sees the ball 99%" is
  measured where the detector historically worked.
- **Factual error in notes:** PROJECT_NOTES explained clip1's 692 present frames as "dribble/hold."
  FALSE — labels cover only flight windows + 12 frames; the 692 fall in 8 flight windows of ~45-177
  frames each [probe]. The "presence ≠ flight" story as written is a fabricated explanation; the
  conclusion survives for a different reason, but the stated mechanism is wrong.

### A-2. Original 1a retraction — CORRECT, mechanism proven
A perfect GT-only track fed to `segment_shots` (`track.py:113-148`): clip1 yields 4 "shots" (two
airball windows + one 14-frame fragment) while MISSING all three certain makes (GT passes 3-7px
from rim) [probe]. Cause: one global RANSAC per contiguous run (`track.py:135`), `min_inliers_frac=0.5`
(`arc.py:112,153`); a gap-free gather+flight+bounce window can't put 50% on one parabola → rejected.
Gap-based segmentation only worked because detector misses created gaps isolating flight arcs.

### A-3(i). Within-window recovery a valid proxy? NO — invalid BOTH directions
- **Hides the real prize:** windows exist only where the old pipeline found shots. Discovering shots
  the pipeline never found is unmeasurable by design.
- **Deflates the oracle (truncation):** clip1 win3 (a make, min rim 4px) LOST — never ≥200px below
  rim within the window (max 197px, 3px short of `launch_drop=200`, `court.py:226`), so walk-back
  (`court.py:263-265`) marches across the 646-frame unlabeled gap into win2, hits win2's launch, gets
  `seen_launch`-dedup-skipped (`court.py:266-268`). clip1 win1 (min rim 24px) LOST with drop=59px —
  labeled track starts mid-flight (first GT f121), walk-back hits track start, <160px drop (`court.py:269`).

### A-3(ii). Auto-rim: wrong rim producing baseline=2? NO — rim validated for clips 1-2 [Fable's read]
- Rims (1134,470,16),(1306,462,19),(972,469,20), gate 90 (`step1a_rim_oracle_ceiling.txt`). 1920×1080@30fps.
  rim_y≈0.43H inside `detect_rim` y_band (0.08,0.45) (`court.py:57`); half-width 16-20px consistent with
  ball radius ~10px. [code+probe]
- **Decisive check [probe]:** GT trajectories pass within 3/4/7px of the recorded rim center in clip1
  windows 2/3/6 and 4px in clip2 win2. Shots fly through the recorded rim point. Baseline=2 not a rim
  artifact. (Clip3 rim unvalidatable — no GT approach <86px; clip3 contributes 0.)
  > [Coordinator note: this is necessary-not-sufficient — see the adjudication file; the exp-rim is near
  > SOME real shots but MISSES ~half the windows, which cluster near a second rim location 110px away.]
- Design caveat: the experiment detects the rim on labeled FLIGHT frames (`exp_oracle_ceiling_rim.py:104-107`)
  where the orange ball is often in the orange-mask band; production `auto_calibrate` samples mid-clip
  (`court.py:112-113`). Fine here by luck, not design.

### A-3-FP. Unmatched raw counts, and ORACLE=4 contains a false positive
No experiment matches recovered shots to GT windows — conclusions rest on count deltas, which FP
generation can fake. clip1 win7 is one attempt: ball flies past rim (~96px at f10563), lands (y≈824
f10575), bounces, drifts back (86px at f10643). The oracle "recovers" TWO shots (rim events f10564,
f10644); the second (rel=72/ent=78) slips under `min(rel,ent)>78` (`court.py:283-284`) by 0°. So
oracle 4 = 3 real + 1 bounce FP; honest oracle-vs-baseline ≈ +1 with an FP. Original 1a baseline=6
includes airball fragments the rim path rejects — comparing raw counts across differently-composed
mixtures was never valid.

### A-3(iii). Thresholds satisfiable on sparse tracks? Within windows yes; the artifact is BETWEEN windows
854/862 clip1 inter-frame gaps are 1 frame; windows are dense runs of 45-177 frames separated by
218-6388-frame voids [probe]. `min_points=8`/RANSAC fine INSIDE windows. Artifact = walk-back + dedup
crossing voids (win3) + window truncation (win1). The experiment measures a MIXTURE: ~2 non-recoveries
are sparsity artifacts, ~2 are genuine segmenter defects.

### A-3-genuine. Two REAL production-segmenter defects (perfect track, in-window, still lost)
- **clip1 win6 (a make, min rim 7px): RANSAC returns None** on the 68-pt walked-back segment
  (f9155-9225) — gather points + `min_inliers_frac=0.5` (`arc.py:112,153`) → `fit is None` → dropped
  (`court.py:276`). [probe+code]
- **clip3 win1: rejected at exactly 78°** (rel=78.4, ent=82) by `court.py:283-284`. [probe]
- `launch_drop=200` and 0.8·200=160px are hard-coded PIXELS (`court.py:226`), not scaled by rim size/
  px-per-foot — a far/short shooter whose release sits ~170px below rim (win1: max 169px) forces the
  walk-back into the dribble → feeds the win6 failure. Brittle across framings.

### A-4(iv). assemble_track velocity/reset fix — correct, but completely untested
- Old bugs confirmed from `git show 073fe95`: (1) velocity used raw displacement not divided by span,
  ×step → over-extrapolation on irregular detections; (2) reset set `prev_prev=None` then the shared
  tail update clobbered it → resets bridged velocity across dead time. [code]
- New code right: `seed()` (`track.py:51-59`) sets prev_prev=None and `continue`s past the tail update;
  velocity per-frame (`/dt_prev`, `track.py:75-79`) scaled by current gap. [code] Probe over 692
  single-candidate frames: all kept, no spurious drops. [probe]
- **"35/35 test files pass" is vacuous for this fix:** `grep assemble_track tests/` → zero matches;
  only `Shot` is imported by any test. [probe] No test exercises `assemble_track`/`segment_shots`.
  Correct by inspection, not coverage — worth ~30min of unit tests.
- Remaining tracker weakness (matters if cloud@0.01 wired in): greedy single-hypothesis,
  `score = d - 30*conf` (`track.py:87`, mixes px with conf); seed takes global max-conf (`track.py:54`)
  → in a 0.01 cloud a confident wrong-object seed locks the track. Measure FP rate, not just count.

### A-5. Net adjudication
- Retraction: sound, proven. "Inconclusive, not reversed": correct and strengthened (the +2 is ~+1
  with an FP; whole within-window count family can't measure the prize).
- "Primary lever = attempt-detection / film closer": directionally supported but imprecise. Real
  statement: the segmenter loses rim-reaching shots to fixable micro-defects AND the rim-gate can't
  count attempts that miss everything. "Film closer" fixes neither by itself.

## (B) Push further — levers ranked

| Loss bucket | Count (of 12 windows) | Root cause | Fix cost |
|---|---|---|---|
| Never reach 90px rim gate | 4 (+1 junk) | rim-gate excludes airballs | attempt-detection (~3-5d) |
| RANSAC None on walked-back seg | 1 (win6 make) | gather poisons min_inliers_frac=0.5 | hours |
| Walk-back gap-cross + dedup | 1 (win3 make) | no discontinuity stop | 1 line + test |
| Track/window starts mid-flight | 1 (win1) | truncation + hard 200px launch | hours |
| 78° hard gate | 1 (clip3 win1) | threshold on data boundary | trivial |
| Bounce-back FP | +1 fake | no re-approach suppression | hours |

1. **THE EVAL FIX (first, ~1hr, mostly owner time): full-clip hand-counted ground truth.** 3 clips
   ~18.5 min (13004+10240+10163 frames @30fps [probe]). Owner logs every attempt (frame, make/miss,
   rim-reached vs airball). Run full pipeline (baseline @0.25 + cloud @0.01), MATCH to hand count,
   report recall AND precision. Replaces unmatched counts with denominators; sizes airball bucket,
   detection prize, segmenter losses, tracker FP simultaneously. Cheaper than experiments already run.
2. **Segmenter micro-fixes (half-day, ~+2-3 real shots even in labeled windows):** stop walk-back at
   discontinuities; trim segment to last height-min before rim event / two-stage fit; suppress rim
   re-approach after descent; soften 78° to a scored margin; scale `launch_drop` by `rim_radius_px`.
   All `court.py:225-291`, each testable against the probe's itemized failures.
3. **Attempt-detection beyond the rim gate** — biggest bucket IF the 4 non-rim-reaching windows are
   real attempts; the eval fix confirms/kills that for free. Don't build before (1).
4. **Cloud @0.01 through fixed tracker, full clip** — only meaningful measured against (1); greedy
   max-conf seeding over a junk cloud is an FP machine.
5. **Film closer** — helps (bigger ball, fewer 78° rejections, healthier walk-backs) but NOT top
   lever: perfect detection still loses half the rim-reaching shots inside the segmenter.
6. **Tracker tests (30min).**

## VERDICT
The retraction is sound and its mechanism proven, but the rim-anchored follow-up deserves the same
skepticism the gap-based run got: its ORACLE=+2 contains a bounce-back FP and truncation artifacts,
and no experiment matches recovered shots to ground truth — the numbers are evidence for nothing
except "the instrument is too blunt." The auto-rim is not the problem [per Fable] and the tracker fix
is correct (covered by zero tests). "Attempt-detection / film closer" is directionally right but
misnamed: the production segmenter drops ~half the rim-reaching shots with perfect input via five
cheap-fixable defects, and its 90px gate makes badly-missed attempts invisible by design. **The
single most decisive next experiment is the full-clip, hand-counted ground-truth run (~1 hour)** —
it sizes the airball bucket, the detection prize, and the segmenter losses simultaneously with real
denominators; every lever should be judged against it, not single-digit unmatched counts.
