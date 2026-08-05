# ShotLab — Project Notes (living log)

> The canonical "everything" doc. README.md is for usage; this is the decision
> log, filming guide, roadmap, and enhancement backlog. Update it as we go.
> **⭐ ON RESTART READ `KICKOFF_NEXT_SESSION.md` FIRST** — it has the current state
> (recall 86% / precision 0.99 / make-miss 81% via `build_session --validated`, HEAD
> 67aebde) and the two open owner-dependent items. The dated sections below are the log.

Last updated: 2026-07-29 · Location: `C:\Users\jmaku\Desktop\ShotLab`
(GPU is the default for **detection** — DirectML/ONNX, proven, re-verified
5.1 ms/frame after fixing a silent CPU-fallback regression on 07-22 (onnxruntime
conflict). ⛔ **GPU TRAINING IS ON HOLD**: the ROCm path hard-locked the whole
machine on 2026-07-22 and its "verified" speed claim was unsubstantiated;
**torch-directml training was tested and RULED OUT (silent wrong gradients)** —
see the 07-22 evening logs below and `process/GPU_SETUP.md`. **GPU-training path
(Codex+Fable consult): free cloud CUDA (Kaggle) — `process/KAGGLE_TRAINING.md`**;
CPU is the offline fallback; WSL2+ROCm is a parked optional side-quest; native-
Windows ROCm rejected. Far-ball recall fixed via native-scale retrain on human
labels. **Step-1 gate DONE 07-22 night: detection is not the bottleneck, segmentation
is — TrackNet/tiling/candidate-cloud all off the table; lever = film closer. See the
NEXT SESSION PICKUP below.** Session logs current through 07-22; the 07-03 → 07-11 S8/3D/audit arc was
backfilled from git.)

---

## 🎥 FILMING CHECKLIST — read before every session (lessons through 2026-07-03)
1. **Get bigger in frame.** Far shots are where pose fails (skeleton lands on the
   court) — set the camera closer / more side-on so you fill more of the frame.
2. **Clean shoot → reset cycle.** Running back into frame chaotically causes false
   shot detections + confuses the app. Shoot, let the ball go, retrieve calmly,
   RESET at your spot, then shoot. A clear pause between reps = far less junk.
3. **Jumper sessions = jumpers.** Layups/dribbling near the rim trip the
   rim-anchored detector (false "makes") — do them in a separate block or expect
   to curate them out.
4. **Keep others out of frame** when you can (kids/rebounders confuse pose).
5. **Sun behind the camera** (front-lit) so the orange ball reads.
6. **Voice tags need a CLOSE mic** — the tripod phone mic is too far. Use Open
   Camera w/ a BT-headset mic, or a 2nd recorder you carry (synced by sync.py).
7. **In the app: tap 🔍 Scan me first**, stand at your spot whole-body-in-frame
   ~3s, THEN Live — so it locks onto you, not passers-by/objects/you rebounding.
8. Same camera position session-to-session (metrics aren't cross-comparable
   otherwise). The close 2nd camera (S8) is the real form-detail fix.
9. **Mount each phone right-side up and don't move it mid-session.** The 07-29 S8
   was clamped inverted (fixable, but only because it was caught before analysis)
   and the wide cam moved between clip 1 and clip 2 (⇒ a separate rim per clip).

## 🎥 NEW SESSION 2026-07-29 — first untouched test footage (the KICKOFF open item #1)
Four Google-Takeout zips landed in `data/raw` and were extracted. **This is the fresh,
never-trained-on session the frozen eval has been waiting for.**

| Cam | Clips | Format | Length |
|---|---|---|---|
| Camera 1 (wide, Pixel) | 2 | **3840×2160 @ 30 fps, CFR** | 5:08 + 6:29 |
| Camera 2 (close, S8) | 4 | 1080×1920 @ 30 fps, CFR | ~25 min |

- **Constant frame rate on all six clips** — first session needing no VFR correction
  (`is_variable_fps` False everywhere; dt spread 32.9–33.7 ms on the wide cam).
- **Framing improved materially.** Detector smoke probe (validated ONNX weights, imgsz
  1280, 300 sampled frames of wide clip 1): **86% of frames yield a candidate @ conf 0.01,
  53% @ 0.25, median conf 0.52**, median ball radius **66 px at 4K**. Top-confidence crops
  are unmistakably the real ball in flight. ⚠️ **The 4K itself is not the win** — 3840→1280
  letterboxes at ×0.333 vs 1920→1280 at ×0.667, so model-space ball size is unchanged by
  resolution alone. The gain is FRAMING: ~22 px ball radius in model space vs ~10 px on
  0720. The "film closer" lever finally got pulled.
- ⚠️ **The wide camera moved between clip 1 and clip 2** → each needs its own `verify_rim`.
- ⚠️ **The close cam was mounted UPSIDE DOWN** and the files carry a spurious `-90`
  display-rotation flag, so cv2 (auto-orientation on by default) delivered frames rotated
  90°, and the raw stream is 180° off. MediaPipe would have produced garbage.
  **FIXED AS DATA, NOT CODE:** `ffmpeg -display_rotation 180 -i <clip> -c copy` →
  `data/raw/Camera 2/upright/` (stream copy: seconds per clip, zero re-encode, zero quality
  loss). Verified: cv2 now reports 1920×1080 upright with matching frame counts.
  **Why data over code:** ~15 direct `cv2.VideoCapture` call sites would each need the same
  rotation or they'd disagree about coordinates; the ffmpeg/browser paths (`to_h264`,
  `cut_review_clips`, the dashboard player) honor rotation *differently* than cv2 does; and
  cache signatures key on video *content identity* (`video_id` = size:mtime), so a rotation
  applied in code is a new interpretation of the same bytes → same key, different pixels =
  exactly the stale-cache footgun that has bitten this project three times. Fixing the file
  makes every consumer agree for free. **Close-cam tools need `--close-dir
  "data/raw/Camera 2/upright"`; `flare_report.py`'s `CLOSE_DIR` is a module constant.**

**Old session restored** to `data/raw/Camera 1/Old/` (10 clips: 5 wide 0720 + 5 S8) after
being deleted during the re-download. **Verified byte-identical** to the footage that
produced the frozen baseline — `video_id` matches the cached detection params on all three
hand-counted clips (e.g. `1085507173:1784554566`). So the 86%/81% baseline is fully
re-runnable and the **1,340 human ball labels** in `data/labels/ball_labels_0720.json` are
live again. `Old/` and `upright/` are subfolders, and no pipeline glob is recursive, so
neither can contaminate the new session. Note only clip 1's detection cache is at current
weights; clips 2–3 caches carry older weights + pre-recalibration rims and will re-detect.

**GATE (owner):** `hand_count` then `verify_rim` on the wide clips — counted FRESH, not
seeded from detections — then `eval_ablations` frozen. That answers the one question the
86%/81% numbers cannot: do they generalize off the training clips?

**⚠️ CORRECTION (08-02): there are FOUR wide clips, not two.** `PXL_20260729_155914855-001`
(13,306f, 7:24) and `PXL_20260729_161439291-002` (13,423f, 7:27) are also 3840×2160@30 and
also real shooting footage — 47,644 frames / ~26.5 min of wide cam in total. They arrived
after the initial intake listing. **Decide WHICH clips are in the eval before seeing any
result** (see pre-registration below); a subset chosen after the fact is not a frozen test.
A production `--clips "data/raw/Camera 1/*.mp4"` run would sweep all four and fall through
to `auto_calibrate` on any clip lacking a rim file — the auto-rim is known to lock onto the
shooter's shirt, so rim every clip you include and exclude the rest explicitly.

## ⭐⭐⭐ FROZEN 07-29 RESULT — clip 1 (2026-08-02). DETECTION HELD, MAKE/MISS DID NOT.
The pre-registration below was written before any of this existed. Read it first, then this.

**Owner hand-counted `PXL_20260729_155320813` fresh: 28 attempts (16 make / 10 rim-miss /
2 airball), spanning frames 1121-8556.** QA before use: minimum gap between attempts 92f
(3.1s) against a ±30f matcher window — over 3× margin, no double-taps, attempts spread
10/9/9 across thirds. This is a clean ground truth.

**ALL FOUR CLIPS COUNTED AND EVALUATED — 143 attempts. The eval is COMPLETE.** Each count
was QA'd for spacing before use (min inter-attempt gaps 92f / 122f / 58f; only clip 4's 58f
pair sits inside 2× the ±30f window — attempts #39/#40, both makes, 1.9s apart).

| | 0720 (detector TRAINED on it) | **07-29 FROZEN, all 4 clips** |
|---|---|---|
| Detection recall | 96/111 = **86%** [CI 79-92] | 122/143 = **85%** [CI 79-90] |
| Precision | 0.99 (1 FP) | **0.984 (2 FP)** |
| Airball recall | ~0 (blind by design) | **5/5** (all via rim-recovery) |
| Make/miss (learned) | **81%** LOCO (claimed) | 67/122 = **55%** [CI 46-63] |
| Make/miss (geometric) | ~51% | 57/122 = **47%** [CI 38-56] |

Per clip, detection: **86% / 82% / 89% / 85%**, precision 1.00 / 0.97 / 0.97 / 1.00.

**DETECTION GENERALIZED — settled, on a full session.** 85% over 143 held-out attempts with
a CI straddling the baseline, and precision essentially perfect (2 false positives in 124
produced shots). The C1→C5 ladder reproduced its shape on every clip, so the beam tracker and
rim-recovery pass are not artefacts of the three clips they were built on. This is the result
the entire eval harness existed to obtain, and it is a clean pass.

**MAKE/MISS FAILED — 55%, and the honest reading is "no usable signal".** 81% sits far
outside the interval; **chance sits inside it**. The learned model beats the geometric rule
(55% vs 47%) but neither is usable. Per-clip learned scores are 54/48/48/**68** — clip 4's 68%
is the one above-chance clip and, at n=34, is within sampling noise of the rest; nothing about
clip 4 distinguishes it (its make-rate, 48%, is the most balanced of the four, which flatters
any weak classifier). Geometric swings 33/52/64/35 while abstaining on 5-16 shots per clip —
base-rate noise, not skill. The mechanism was **pre-registered before any of these numbers
existed**: three of seven `make_visual` features are logs of RAW orange-pixel counts in
rr-scaled regions, whose areas go as rr², and rr moved 36 → ~120. **Task 6 (normalise by rr²,
re-fit) is now the single highest-value item in the project.**
⭐ **Airballs 5/5, every one recovered by the rim-recovery pass and none by C1-C4** — a pass
added for rim-reaching shots is what makes the "blind by design" airball case visible. Worth
understanding before task 5 rescales any gate.

⭐ **RIM MEASUREMENT — the instrument, and one judgement call.** Rims were measured from the
rim's orange paint, not clicked, after clip 1's click came in 22% short. Two contaminants
push in opposite directions: occlusion (foliage/net/blur) SHORTENS a visible run, a ball
touching the rim EXTENDS it. Ball frames are excluded using the cached candidate cloud, so
what remains can only be truncated and the upper plateau is the truth. Clips 2 and 3 share a
camera position (centres 2px apart) yet measured 120 vs 112 — clip 2's distribution plateaus
(p90 237, p97 240, max 242) while clip 3's is still climbing (p90 217, p97 223, max 228), so
clip 3's rim is never fully unoccluded and **takes clip 2's radius by physical necessity**:
same camera, same hoop, one number. Recorded in each rim file. Clip 1 was deliberately NOT
updated when re-measurement suggested 116 vs its filed 112 — its result is banked, and
re-tuning a frozen input for a 3% refinement is what voids a held-out measurement.

**DETECTION GENERALIZED.** Same recall, better precision, on a session the detector has
never trained on, at a ball scale it has never seen. The C1→C5 ladder also reproduced the
0720 *shape*: greedy 68% → union 82% → +rim-recovery 86%, i.e. the beam and recovery passes
each still earn their keep off their training clips. This is the first honest measurement
of this pipeline and it is a good one.

**MAKE/MISS COLLAPSED, exactly as pre-registered.** 54% against a claimed 81% — 81% sits
outside the 95% CI, so the drop is real, not sampling. Cause was predicted in advance
(item 2 below): three of seven `make_visual` features are logs of RAW orange-pixel counts
in rr-scaled regions, so their areas scale with rr². rr went 36 → 112, moving those masses
~10× outside the scaler's training distribution. **Geometric is worse than a coin flip
here — 33%, abstaining on 12 of 24** — so the learned model is still the better of the two,
just no longer trustworthy. Fix is item 2's: normalise by rr² and re-fit.

⚠️ **Deviations and caveats, recorded because they bound the claim:**
- **I corrected the rim radius after the owner set it, before running the eval.** The click
  gave (654,589) r=88; measuring the rim's orange paint at frames 1121/3000/5000 gave a
  225px span (x496..721, agreeing within 4px) = centre 608, r≈112. The click was
  right-shifted and 22% short. Done for measured reasons and recorded in the rim file, with
  the original kept at `rim_*.json.clicked_backup` — but it is an intervention on a frozen
  input and must be read as one. It also RAISES `shot_gate_px` 177→224, which plausibly
  contributes to why pre-registered item 1 (pixel gates too tight at 4K) did not bite.
- **Pre-registered item 1 did NOT materialise**: recall held, so the scale-dependent gates
  did not cost measurable recall on this clip. That does not clear them — the corrected,
  larger gate may be masking the problem. Item 1 stays open.
- **n = 1 clip, 28 attempts.** Detection's CI is 69-94%; "86%" is not precise to the point.
  Three clips remain uncounted.
- **The stored 0720 eval JSONs' make/miss (75%) is NOT comparable** — those runs predate
  today's fix and scored the superseded `make_visual.joblib`. The 81% is the re-fit's own
  LOCO figure. An apples-to-apples 0720 re-run needs the clips moved back out of `Old/`.
- **Match tolerance is load-bearing**: sweep gives ±5f=14%, ±10f=50%, ±15f=82%, ±30f=86%.
  Half the matches sit 10-30 frames from the hand-marked frame, which is expected when a
  human parks "near" the rim moment by eye, but it means ±30 is doing real work.
- ⛔ **Nothing was tuned before or during this run.** Next knob turned on this session
  invalidates it as a held-out measurement.

## ⛔⛔⛔ THE BODY-METRICS FRONT IS CLOSED ON MEASUREMENT GROUNDS (2026-08-04)
Not on another inconclusive session — on the instrument's own reproducibility, measured from
footage that already existed. The same 07-20 clip-1 shots were re-measured under two
perturbations (`--pose-variant heavy`, `--no-smooth`), which are repeated measurements of
IDENTICAL physical events and therefore give a within-shot SD and a **smallest detectable
change** (SDC = 1.96·√2·SD_within).

**Perturbation A — a different pose MODEL (`heavy` vs `full`), same video, n=22 matched:**

| metric | r | bias | within-shot SD | **SDC (95%)** |
|---|---|---|---|---|
| `knee_bend_3d_deg` | **0.07** | −7.1° | 8.5° | **23.7°** |
| `knee_bend_deg` (2D) | 0.83 | +0.3° | 6.7° | 18.6° |
| `elbow_angle_at_release_3d_deg` | 0.21 | −4.6° | 7.5° | 20.8° |
| `flare_deg` | 0.62 | +10.3° | 5.5° | 15.3° |

**Perturbation B — smoothing on/off, n=28:** knee_3d **r=0.99, SDC 3.0°**; knee 2D r=0.99,
SDC 4.1°; elbow 3D r=0.92, SDC 7.8°; flare r=0.76, SDC 13.8°.

⭐⭐⭐ **The One-Euro filter is NOT the noise source — the MODEL is.** Two versions of the same
estimator, on the same pixels, agree at **r=0.07** on the 3D knee. The per-shot value is close
to arbitrary. **Every effect chased today was ~5°, against an SDC of 23.7°.**
⛔⛔ **This retires single-camera pose body metrics as a per-shot make/miss instrument on this
footage, and this time the reason is measured rather than inferred** — unlike the 08-03
"camera-dependent" claim, which was retracted for being unsupported.
⚠️⚠️ **THE 3D LANDMARKS ARE WORSE THAN THE 2D ONES: r=0.07 vs 0.83.** I added them this morning
expecting camera-invariance. Monocular depth is model-INFERRED and unconstrained; the
image-plane projection is directly OBSERVED. ⭐ **"More physically principled" is not the same
as "more reproducible", and only one of those can be measured.**
⚠️ The heavy model also moved the RELEASE detection — only 22 of 39 releases matched within ±3
frames — so the anchor itself is estimator-dependent, on top of the metric.
⚠️ Scope: one clip, 22-28 matched releases, one shooter. But r=0.07 needs no large n to be
damning, and it agrees with published BlazePose-World error of 7.1-17.2° MAE.
✅ **What survives:** group-level *distribution* work may still be defensible where random error
averages down — but nothing per-shot, and nothing at the ~5° scale. Before any future body-form
claim, **run this reliability check first and quote the SDC.**
✅ **AND IT CLOSES THE OPEN ELBOW QUESTION.** I had flagged the close camera's **27° median
2D-vs-3D elbow gap** (r=+0.46) as the largest disagreement in the table, on the camera that
should be most favourable, and left it unexplained. It needs no special explanation: the 3D
elbow's own reproducibility is **r=0.21, SDC 20.8°**, so a 27° gap is inside the metric's noise.
⭐ The "anomaly" was the 3D estimate being unreliable, not the profile view being strange —
**an unexplained discrepancy between two measurements is usually the noisier one, and it is
cheaper to measure reliability than to theorise about geometry.**

## ⭐ THE 6 REMAINING 07-29 MISSES, DIAGNOSED ONE BY ONE (2026-08-04)
Detection is the part of this system that works (95.8%), so the remaining 4.2% is the real
work-list. Every miss traced to the exact rejection that killed it:

| attempt | rim frame | why the rim event died |
|---|---|---|
| 17 | 5311 | **78° gate** (rel 81, ent 86) |
| 18 | 5403 | never rose above rim height (bounce/roll) |
| 5 | 1984 | **no rim event at all** — ball never tracked to the rim (detection-limited) |
| 30 | 9695 | **78° gate** (rel 78, ent 84) |
| 32 | 9262 | arc curves the wrong way |
| 34 | 9910 | insufficient drop (**−609** px — the "arc" ENDS 609px higher than it starts, i.e. a mis-assembled track running upward) |

⭐ **The 78° gate is 2 of the 6 — a third of the remaining loss**, which quantifies the cost of
having declined to tune it (rightly: it compares an IMAGE-PLANE angle to a WORLD constant, so
retuning on this data is resubstitution).
⛔ **THE PRINCIPLED FIX IS BLOCKED ON CAMERA INTRINSICS, AND NOTHING ELSE.**
`ballistic.fit_camera_tilt` recovers pitch/roll from the physics of ≥2 clean arcs with no rim or
board, is synth-validated (18/4° → 17.1/4.3°, rmse 0.69px), and **has never been fed real
arcs** — 137 are now available. But it needs `K`, and there are **no intrinsics anywhere on
disk** (`config/` holds only rim files; no ChArUco artifacts exist). Assuming a focal length
would put an unmeasured constant under a geometric correction, which is precisely the failure
pattern of this whole day.
⭐⭐ **The unblock is a ~2-MINUTE CALIBRATION CLIP (checkerboard/ChArUco close-ups), NOT a
session.** That is a materially smaller ask than re-filming, and it would also unlock W2 (true
depth) and the release/entry angles that are currently geometry-limited. Worth separating from
the "film a fresh session" item, which is about a held-out detection number.
⚠️ Note miss 34: a drop of **−609px** means `assemble_track` produced a segment travelling
upward across the rim event. That is a tracker defect, not a threshold to loosen.

## ⭐⭐ THE LABELLING RULE: LABEL ~50+ SHOTS OR LABEL NONE — IN BETWEEN IS NEGATIVE RETURN
The learning curve alone was the wrong question. A brand-new session can be pre-labelled for
FREE at ~77-78% by the z-scored transfer model, which needs zero labels from it, so the
operative question is **at what k does labelling this session finally beat the free option**:

| k labels from this session | GBM refit | zLR refit | **free transfer** | worth it? |
|---|---|---|---|---|
| 0 | — | — | **77%** | — |
| 8 | 63% | 62% | 77% | **NO** |
| 16 | 71% | 69% | 77% | **NO** |
| 24 | 74% | 71% | 77% | **NO** |
| 32 | 76% | 73% | 77% | **NO** |
| 48 | **79%** | 75% | 77% | yes |
| 64 | 82% | 76% | 77% | yes |
| 80 | **84%** | 77% | 77% | yes |

⛔ **Labelling fewer than ~48 shots is WORSE THAN LABELLING NONE.** A small hand-labelled refit
loses to the free transfer model, so partial effort has *negative* return — the intuition that
"some labels must help" is exactly wrong here.
⭐ **zLR is the model for TRANSFER only.** Once real labels exist the GBM wins at every k
(84% vs 77% at k=80), so the workflow is: zero labels → z-scored logistic from a prior session;
≥48 labels → refit the GBM on this session. Do not use one where the other belongs.
⭐ This is what the earlier "no plateau" curve was missing: a REFERENCE LINE for doing nothing.
A learning curve without the free alternative on it cannot tell you whether to spend the effort.

## ⚠️ THE 07-29 vs 07-20 RECALL GAP: ball SIZE correlates, rim radius is NOT the cause
07-29 scores **95.8%** and 07-20 **~88%**, so the gap is worth understanding — it is the closest
thing to actionable filming advice the data can give.

| clip | ball r px | filed rim r px | ball/rim | recall |
|---|---|---|---|---|
| 0729 ×4 | 63.5-67.2 | 112-120 | **0.55-0.57** | 92.9-100% |
| 0720 ×3 | 34.5-34.9 | 35.6-39.3 | **0.89-0.97** | 85.7-93.9% |

**corr(ball radius px, recall) = +0.74** across the 7 clips — consistent with the standing
"film closer" lever, though n=7 and confounded with session.
⛔ **A tempting story, MEASURED AND REFUTED.** Regulation ball/rim is 9.43in/18in = **0.524**.
07-29's 0.55-0.57 is physically right; **07-20's 0.89-0.97 is impossible**, implying its rim
radii (filed before the "measure the paint, don't click it" lesson) are ~1.8× too small — and
rim radius feeds `threshold_px = (8/36)·rr`, the exact constant whose scale error cost 11 points
of recall on 4K. So: recompute the rim from physics and recover the gap?
**No.** Substituting the physics-implied radius (~66px) gives **88.3% → 86.5%, −2 tp, +1 fp.**
The correction does not help, so a wrong rim radius is NOT what separates the two sessions and
the filed rims stay untouched.
⭐ The likeliest reading is that the DETECTOR'S ball radius is inflated on a small fast ball at
1080p (motion blur widens the box), so the "implied rim" was itself too large — i.e. the
implausible ratio indicts the ball measurement, not necessarily the rim. **Either way it was a
clean, cheap test of a plausible story, and the story lost.**

## ✅ FROZEN EVAL RE-BANKED AFTER THE 08-04 EDITS — NO REGRESSION
Today changed the default detection path (`back_extend` now fed the cloud), `form.py`'s `span`,
`metric_ranges`, `correlate`, `session.py` and the dashboard. Re-ran the ablation ladder on all
four 07-29 clips to confirm none of it moved detection:

| clip | recall | precision |
|---|---|---|
| 155320813 | 0.93 | 1.00 |
| 155914855-001 | 0.95 | 0.97 |
| 160743954 | 0.95 | 0.97 |
| 161439291-002 | 1.00 | 1.00 |
| **total** | **137/143 = 95.8%** | **~0.986** |

Identical to the pre-existing figure — **no regression**. Full ladder on clip 1: C1 greedy 0.79
→ C2 cloud-greedy 0.79 → C3 beam 0.75 → C4 greedy∪beam 0.89 → C5 +rim-recovery 0.93 → **C6
+back_extend 0.93**. ⭐ **C6 ≡ C5 still holds**, which is the direct confirmation that
`back_extend` is redundant on the full stack and earns its keep only on the default non-beam
path (where it is worth +7 tp on 07-29 and +16 held-out on 07-20).
⚠️ Still NOT held out — same 143 attempts that produced the frozen 85%. The resubstitution
guard fired correctly on every `make_visual` line of the run.
⛔ **07-10 CANNOT serve as a third session for anything pixel-based**: its raw clips are gone,
leaving only annotated overlays (graphics drawn over the rim ROI the make model reads) and
per-shot review clips. Relevant to the zLR transfer check, which still needs a third dataset.

## ⛔⭐ `back_extend` WAS NEVER INERT — I MEASURED IT IN THE ONE CONDITION WHERE IT'S REDUNDANT
It was recorded **MEASURED INERT** (eval C6 byte-identical to C5) and kept opt-in. That test was
correct and the conclusion drawn from it was too broad: C6 sits **on top of the beam union**,
which already recovers the same flights from the same cloud. Re-measured 2026-08-04 against both
hand-counts (recall, precision in brackets):

| config | 07-29 (143 attempts) | 07-20 (111) |
|---|---|---|
| greedy alone — **what shipped by default** | 86.7% [.984] | 54.1% [1.000] |
| greedy + back_extend | **91.6%** [.985] | **57.7%** [1.000] |
| greedy + back_extend (0.01 cloud) | 90.9% [.985] | **68.5%** [1.000] |
| beam-union | **95.8%** [.986] | **86.5%** [.990] |
| beam-union + back_extend | 95.8% [.986] — **identical, hence C6≡C5** | — |

⭐ **07-20 is HELD OUT for back_extend** (it was built on 07-29): +4 tp at the 0.25 cloud, **+16
tp at the 0.01 cloud (54.1%→68.5%), zero false positives in every configuration.**
✅ **PROMOTED**: `pipeline.py` now passes `cands_by_frame` as the cloud on the calibrated path,
so the DEFAULT (no `--beam`) run gets +7 tp / +0 fp on 07-29 and +4 tp / +0 fp on 07-20 for
free. ⚠️ **It is still DOMINATED by `--beam`** (95.8% vs 91.6%) at the *same detection cost* —
conf only FILTERS a YOLO pass that runs regardless, so the low-conf cloud is free. This is the
free win for the default path, **not** the best available one; prefer `--beam`.
⭐ **LESSON: "measured inert" is a statement about a CONDITION, not about a component.** The
component was fine; I tested it only where a better mechanism had already consumed its input.

## ⚠️ THE 78° GATE COSTS 3 REAL SHOTS IN 254 — MEASURED, DELIBERATELY NOT TUNED (2026-08-04)
Task 20 was "make the 78° gate camera-aware". Measured first, across BOTH hand-counted sessions
(7 clips, 254 attempts), using `detect_shots_to_rim(reject_log=)`:

| reason a rim event produced no shot | count |
|---|---|
| too few points | 57 |
| insufficient drop | 11 |
| arc curves the wrong way | 4 |
| **78° gate** | **3** |
| RANSAC failed / never rose above rim | 1 / 1 |

All 3 78°-gate drops sit near a hand-counted attempt that **no other rim event recovered**, so
the gate genuinely loses 3 real shots (2 on 07-29, 1 on 07-20) — 1.2% of attempts. Their angles:
(rel 81, ent 86), (78, 84), (84, 84).

⛔ **I am NOT raising the constant.** An 84° *release* angle is not a steep shot — a real jump
shot releases at 45-60°. It is inflated because this camera sits nearly in line with the shot,
so foreshortening collapses the horizontal motion and inflates every image-plane angle. ⭐ **This
is the SAME defect class as the body metrics and the old raw `8.0` RANSAC threshold: an
IMAGE-PLANE quantity compared against a WORLD-FRAME constant.** Raising 78 to ~85 would recover
3 shots on the only data that could have told me the number — textbook resubstitution, and it
would admit exactly the near-vertical tosses/rebounds the gate exists to reject.
⭐ The principled fix needs the camera's obliquity, which is not calibrated (`ballistic.fit_
camera_tilt` exists and is synth-validated but has never been fed ≥2 real clean arcs). **Pre-
registered:** if a future session supplies a measured tilt, de-project the angles before gating
and re-measure recall AND precision on the frozen harness; until then the gate stays at 78 and
the 1.2% is a known, quantified cost rather than an unexamined one.
⚠️⚠️ **METHOD NOTE — my first run of this measurement was VACUOUS.** I filtered
`r.get("why","")` while `_drop()` writes the field as `"reason"`, so every count came back 0 and
I nearly recorded "the gate never binds, task moot". A stray `55 ?` in an unrelated column is
the only thing that exposed it. **A filter on a key that does not exist returns zero, and zero
looks exactly like a clean result.** Same family as the 08-03 checkguard lesson.

## ⭐⭐ "EVERY SESSION NEEDS ITS OWN LABELS" IS TRUE OF THE MODEL, NOT THE FEATURES (2026-08-04)
Found by Fable during a labelling-design review, then **independently replicated on a
separately-built 137-shot feature matrix** (`tools/transfer_check.py`, both directions):

| transfer | GBM raw | z-scored + logistic | majority |
|---|---|---|---|
| 0720 → 0729 | 64.2% | **78.1%** (AUC .840) | 53.3% |
| 0729 → 0720 | **50.0%** (exactly chance) | **78.4%** (AUC .845) | 51.1% |

⭐ **The session-specificity lives in the GBM's split THRESHOLDS, not in the cues.** A tree
splits on absolute values, and the features are absolute pixel masses that move with exposure,
white balance and rim ROI; a per-session standardised linear model is immune to exactly that
shift. The cue DIRECTIONS transfer fine. Within-session LOCO cost of switching is small
(0729 84.7%→81.0%, 0720 80.7%→79.5%), so the trade is **give up 1-4 points where you have
labels, gain 14-28 where you don't.**
⭐ The z-score is **label-free** — it uses only the new session's feature distribution — so a
brand-new session can be pre-labelled at ~78% with ZERO labels. It is transductive (needs the
session's shots as a batch, and enough of them to estimate mean/sd).
⚠️ **TWO SESSIONS. Not a law.** 78% is still well below a within-session re-fit, so this is a
better STARTING POINT, not a reason to stop labelling. ⭐ **PRE-REGISTERED: re-run
`transfer_check.py` on the next session BEFORE relying on it.** ⛔ Self-training on these
pseudo-labels measured NEGATIVE (−0.9pp) and both reviewers reject it — a confidently-wrong
model under covariate shift would launder its own errors into training truth.
⚠️ My fresh 137-shot matrix gives 0729 LOCO **84.7%**, not the recorded 89-90% (122 shots,
scale-invariant cache). Different matrices, not comparable — do not quote them interchangeably.

## ⛔⭐⭐ BODY METRICS ARE CAMERA-DEPENDENT (2026-08-03) — the deepest result of the arc
`flare_report` now emits the full body-metric set from the CLOSE camera (task 17). The wiring
was small: `compute_form` touches only `shot.frames`/`shot.index`, and `find_release` is
anchored to the WRIST APEX with the ball used solely to raise confidence — so a pseudo-shot
over the release window plus an empty ball track is enough. Every value is gated through
`metric_ranges` at emission, so an implausible read is ABSENT rather than a number.

**Coverage roughly doubled and is now all-plausible**: knee 110/141 vs 58/139 on the wide cam,
balance 111/141 vs 58/139, elbow 103/141 vs 51/139, jump 121/141 vs 44/139.

**Still nothing separates makes from misses.** Joined THROUGH the wide shots (close release
--audio sync--> wide shot --time--> attempt; 125 of 141 paired, because the direct
release→attempt join had residual sd 17-75f and a bad join attenuates real effects toward
zero): flare d=0.08, knee d=0.28, balance d=0.27, elbow d=-0.02, follow-through d=-0.61
(n=13/18, floor 0.71), jump d=-0.00, release height d=-0.06, tempo d=-0.12. Bonferroni
threshold p<0.0063; **nothing survives, nothing is close.**

⛔⛔ **RETRACTED 2026-08-04 — "OPPOSITE SIGNS ⇒ CAMERA-DEPENDENT" WAS WRONG.**
I wrote: *"the same metric from two cameras gives opposite signs — knee -0.29 wide vs +0.28
close, balance -0.39 vs +0.27, same shots, same session, same definitions … not measuring a
property of the shot … retires single-camera pose regardless of sample size … explains why
every apparent effect in this project has evaporated: camera artifacts, not shooting."*
Owner asked for adversarial review; it broke on four independent counts (Fable, all recomputed
from the artifacts, scripts in the session scratchpad):

1. **Opposite signs is the MODAL null outcome.** Sampling sd of d̂ at these n is 0.20-0.38, so
   |d|=0.27-0.39 is 0.8-1.4 SE — ordinary wobble. Among 4 metrics, P(≥2 sign flips | global
   null) = **0.69**; two flips is the single most likely result. Worse, a REAL shared effect of
   d=0.2 still flips 31-40% of the time here, so a flip cannot even distinguish artifact from
   small-real. I treated sign(d̂) as if it were sign(d) at n≈30.
2. **"Same shots" was false.** Subset overlap among the 143 truth-labelled attempts: knee 31
   wide vs 92 close sharing **15**; balance 31; elbow 21; follow-through 6. Largely disjoint —
   which is also why the two estimates are near-independent, i.e. exactly the Q1 regime.
3. **On the shots the cameras ACTUALLY share, knee does not flip**: wide d=-0.71, close
   d=-0.38 (n=6/9), same sign. The headline example is falsified on its own premise.
4. **I cherry-picked.** Two of four metrics flipped; **elbow (-0.28/-0.02) and follow-through
   (-0.22/-0.61) AGREED in sign** and my write-up never mentioned them.

⭐ **What the review found that I had MISSED — and it is a real defect:** wide-camera pose
coverage is **outcome-correlated**. Balance 0.48 of makes vs 0.31 of misses (p=0.057), elbow
0.43/0.27 (p=0.052), follow-through 0.40/0.26 (p=0.075) — three metrics sharing one mechanism
(pose success), so one finding, not three near-misses. The close camera shows no such trend.
So the WIDE analysis rows are selected by something correlated with the outcome.
⚠️ Also: my close-cam d's are **join-variant** — an equally defensible rebuild gives knee +0.15
(not +0.28) and balance +0.40 (not +0.27). Numbers that move that much must not carry doctrine.
⚠️ And the two paths are not the same measurement even in code: the wide window is the
ball-flight span (`form.py:439`), the close one a fixed ±45f around the wrist apex
(`flare_report.py`); `balance_drift` is monotone in window length and projects different world
components on an oblique vs a profile view.

✅ **WHAT SURVIVES.** The NULL is solid — pooled inverse-variance across both cameras: knee
+0.03±0.36, balance +0.09±0.33, elbow -0.16±0.35, follow-through -0.39±0.45; all consistent
with zero, nothing near the d≈0.35 detectability floor. And the practical decision stands, on
**different evidence**: the wide camera is independently disqualified for form (22-40%
coverage, outcome-correlated survival, and 47% of raw knee reads outside 30-150° with a
median of 138° — i.e. it mostly never catches the load; see the wording correction below).
⛔ **STILL OPEN, not refuted:** whether the CLOSE camera measures stable per-shot form. The
cross-camera correlation test was the right instrument but is underpowered (knee r=+0.08 on
n=15, CI [-0.45,+0.57]) — resolving r=0.3 needs ~85 shared pairs and wide-cam coverage caps us
at 15-31. ⭐ The clean next test needs no second camera at all: **within-close-camera split-half
reliability on the 141 releases.** Also unresolved: ungated wide-vs-close knee gives Spearman
ρ=+0.37 (p=0.050, n=29), weak evidence of shared signal that the plausibility gate may be
discarding along with the artifacts.

### ⭐ Second review (Codex, 2026-08-04) — 3 code defects CONFIRMED, its headline claim FALSE
Verified-vs-asserted cuts both ways here, so both directions are on the record.

⚠️ **Its lead claim was false, and checking it first was correct.** Codex reported
`flare_readings_raw.json` as *"77 rows, three clips, no `20260729_120729`"* and called it a hard
reproducibility failure. The file on disk has **141 rows across all four clips** (29/37/33/42).
Its overlap, duplicate-count and close-coverage tables are all derived from that misread, so
those numbers are void — I recomputed the ones that mattered rather than inheriting them.

**Three of its code-level findings are real. I confirmed each in the source before acting:**
1. ⛔⛔ **THE CLOSE PATH NEVER MEASURED THE RELEASE.** `compute_form` accepts no release frame —
   it re-finds one from `shot.frames[0]` (`form.py:186`) and windows every metric off that. My
   `body_metrics()` passed a pseudo-shot spanning `rel_f±45`, so `frames[0] = rel_f−45` and the
   wrist-apex search ran **[rel_f−63, rel_f−30] — it ended a full second BEFORE the release**.
   The knee loop breaks at `f > rel_f`, so it never reached the dip; `span` was 3.7s against the
   wide path's ~1.7s, and `balance_drift` is a max−min range, hence ~4× inflated (close mean
   **1.69** vs wide **0.43**, measured). **"Same function" was never "same estimator."** Fixed:
   the pseudo-shot now starts AT `rel_f`, reproducing the wide window shape exactly.
2. **The release-confidence gate was silently bypassed.** `correlate.py:105` rejects a
   release-anchored metric only when `release_conf` is *known* low — `None` passes through.
   `body_metrics()` emitted per-metric `_conf` but never `release_conf`, while its own docstring
   asserted "release_conf comes back low, which is honest." Now emitted un-laundered, plus
   `release_frame_delta` (flare-detected release − pose-refound release) as the evidence a
   pose-corroborated confidence tier would have to be argued from.
3. **The close→wide join is not one-to-one** (task #24). On the full 141 rows: 141 releases →
   112 distinct wide shots, **14 shots taking 2 releases each**, 15 unmatched. Duplicates then
   enter as independent observations — pseudo-replication, which makes permutation p's
   anticonservative. This is in the product path (`process_pair`), not only the scratch scripts.

⚠️ **WORDING CORRECTION — "physically impossible knee reads" was overstated**, and I had
repeated it. The 58 raw wide knees: **6 below 30°, 21 above 150°**, median **138°**. A 178° knee
is not impossible, it is *straight* — the wide camera mostly fails to catch the load at all.
Arguably worse for the wide camera, but that is a coverage/timing failure, not a physics
violation, and the 30-150° gate is a censoring rule I chose, not a law.

⛔ **This invalidates the cross-camera correlation numbers above.** Recomputed on the full
artifact (nearest release per shot, wide values gated): knee r=+0.08 (n=15), balance −0.15
(n=31), elbow +0.12 (n=21), follow-through +0.29 (n=6) — near-zero throughout. But the close
side was measuring **the wrong second of video**, so near-zero is the expected consequence of
defect 1, not evidence about cameras. The test only becomes interpretable after the re-run.
### 🔬 The 3D-landmark test — PRE-REGISTERED 2026-08-04, before any result
The review's most useful finding was not statistical. `form.py`'s joint angles are computed on
**image-plane pixels** (`pose.py:151`), so an oblique knee and a profile knee are different
projections of one 3D joint — **not comparable across cameras by construction**. But MediaPipe's
`pose_world_landmarks` (metric, meters, hip-origin) were already being captured in
`PoseFrame.world`, already used by `analysis3d.py` and the elbow-flare metric, and **never used
by `form.py`**. So the instrument for the actual question may have been sitting unused all along.
Now emitted as `knee_bend_3d_deg` / `elbow_angle_at_release_3d_deg`, **alongside** the 2D values
(swapping would silently move every historical number). Gated on ANATOMICAL bounds (20-180°) and
deliberately not the 30-150° window, so "did not bend" stays visible as data.

Committing to the readings now, so no result can be reinterpreted after the fact:
- **3D agrees across cameras where 2D does not** → the projection was the defect, camera-
  dependence was self-inflicted, and body-form metrics are salvageable. The strong outcome.
- **3D also fails across cameras** → does NOT isolate a cause. Monocular 3D may simply be
  unreliable at 22% frame height, or the release anchors may still differ. Not a licence to
  retire pose again — that is the exact over-reach being corrected here.
- **2D vs 3D WITHIN the close camera**: expected to AGREE, and agreement proves little. The
  close camera is a profile view, so knee flexion is already in its image plane — projection
  costs almost nothing there. ⭐ **The sharp test is 2D-vs-3D within the WIDE camera**, whose
  oblique view is where a projection should distort. That needs a wide-camera pose rebuild,
  which is queued behind the close pass.
- Standing caveat: monocular 3D is a model ESTIMATE, not a calibrated rig. A clean 3D result
  raises the metric's ceiling; it does not make it ground truth.

### ⛔⭐⭐ RESULT 2026-08-04 — the anchor fix produced a Bonferroni-surviving effect, AND IT IS NOT REAL
Close pass re-ran clean: 141 releases, 126 matched (coverage unchanged, so the fix cost no
reads). With the release anchored correctly and a one-to-one join, `follow_through_hold_s` came
in at **d=+0.52, p=0.0054, n=49/61 — clearing both its power floor (0.38) and the Bonferroni
threshold (0.0071)**. It would have been the first real make/miss signal in the project. Pre-fix
it read d=−0.61 on n=13/18; the sign flip and the 4× n are both explained by the old anchor
having measured a window ending a second before the release.

⛔ **It is the shooter reacting to the outcome, not form.** The metric is measured ENTIRELY
after the ball leaves the hand, over a 1.0s window, while the ball needs ~1s to reach the rim —
so the shooter can read the flight while it is still being measured. The survival curves settle
it, and needed no re-run because `hold >= t` is derivable from the scalar already on disk:

| t (frames) | 1-10 | 11 | 12 | 15 | 18 | 21 | 24 |
|---|---|---|---|---|---|---|---|
| P(hold≥t \| make) | **1.00** | 0.98 | 0.86 | 0.43 | 0.24 | 0.24 | 0.18 |
| P(hold≥t \| miss) | **1.00** | 0.95 | 0.80 | 0.26 | 0.08 | 0.07 | 0.03 |
| Fisher p | 1.000 | 0.63 | 0.61 | 0.072 | **0.032** | **0.013** | **0.011** |

**Identical through t=10, separating only from t≈15 and significant at t=18-24 — 0.6-0.8s after
release, when the ball is at the rim.** The plain mechanism: a miss sends the shooter to
rebound, a make does not. ⭐ It is also **structurally incapable** of being a form metric as
defined — every shot held ≥0.33s, so the metric has NO variance in the window where form lives;
100% of its variance sits where outcome knowledge exists. Dead-lettered in code as
`_OUTCOME_REACTIVE`, wired so it can be neither reported as a driver nor quoted as the largest
effect in the null message. **Any future follow-through driver must separate at t≤6 first.**

⭐ **The genuinely clean candidate is `knee_bend_3d_deg`: d=+0.40, p=0.037, n=49/61**, clearing
its 0.38 floor. Knee bend is measured BEFORE the release, so it cannot be outcome-reactive —
the objection that kills follow-through does not apply. ⚠️ But it does **not** survive
Bonferroni (0.037 vs 0.0071), it is one shooter and one session, and its 2D twin is weaker
(d=0.24, p=0.24) partly on a different subsample. **Not a finding yet — a candidate for the
next session.** Everything else stayed null (balance 0.16, jump −0.15, release height −0.10).
⭐ Note the 2D elbow was correctly **gated out entirely (0/0)** once `release_conf` was emitted —
the bypassed gate doing its job.

**Measured facts about the 3D landmarks** (close cam, n=126/136): 2D-vs-3D knee r=+0.76, median
gap 8°; 2D-vs-3D **elbow r=+0.46, median gap 27°, means 164.5° vs 138.1°** — the image-plane
elbow is systematically ~26° overstated even on a profile view. 11 shots where 3D reports a
knee that the 30-150° window discarded. `release_frame_delta` (flare release vs pose-refound
release): median −3.0f, sd 5.1f, only 28% within 1 frame — the two release estimates disagree by
~0.1-0.17s, which is why `release_conf` staying "low" is the honest reading, not a formality.
⚠️ Cross-camera correlation on 2D went NEGATIVE after the fix (knee −0.38, elbow −0.25, vs +0.08
/ +0.12 before) — but the wide camera is independently disqualified, so this measures the wide
camera's noise, not the close camera's validity. The pre-registered 3D cross-camera test is what
settles it.

### ⛔ SHARP TEST RESULT — MY PROJECTION HYPOTHESIS IS REFUTED (2026-08-04)
Pre-registered reading: much worse 2D-vs-3D agreement on the oblique wide camera than on the
profile close camera would mean the image-plane projection was the defect.

| 2D vs 3D agreement | knee r | knee median gap | elbow r | elbow median gap |
|---|---|---|---|---|
| **Wide (oblique)** | +0.65 | 13.8° | **+0.68** | **7.4°** |
| **Close (profile)** | +0.76 | 8.0° | **+0.46** | **27.2°** |

⛔ Not much worse — and on the ELBOW the oblique camera agrees *better*, the opposite of what
projection geometry predicts. **Image-plane projection is not what broke the wide camera.**
Coverage is, which was already established independently — and 3D recovers **none** of it
(58 of 139 shots, identical to 2D). The unused instrument was worth adding and worth measuring;
it simply did not explain what I expected it to.
⚠️ Cross-camera knee in metric 3D: **r=+0.07, n=35** — 3D did not rescue agreement either. Per
the pre-registration this **isolates nothing**: the wide camera is independently disqualified,
so its values are noise, and near-zero correlation against noise is what you would see whether
or not the close camera is sound. ⛔ Explicitly NOT a second licence to retire pose.
⭐ Open question this raises: the close camera's 27° 2D-vs-3D elbow gap is the *largest*
disagreement in the table, on the camera that should be the most favourable. Worth a look
before trusting any close-cam elbow number; a profile view can put the shooting arm behind the
torso, where monocular depth is least constrained.

### ⭐ `knee_bend_3d_deg` — a CANDIDATE that survived every confound I could test
d=+0.40, p=0.037, n=49/61 makes/misses, clearing its 0.38 power floor. Makes show the HIGHER
knee angle, i.e. **LESS** bend. Ruled out, each measured not assumed:
- **Outcome-reactivity** — measured entirely BEFORE release, so the objection that killed
  follow-through cannot apply.
- **Outcome-correlated coverage** (the defect that disqualified the wide camera) — present on
  **49/49 makes vs 61/62 misses, p=1.000**.
- **Clip artifact / Simpson's paradox** — clip-STRATIFIED permutation (labels shuffled only
  within a clip, the correct null for blocked data) gives **p=0.0372** vs 0.0365 pooled, and all
  four clips agree in sign: +0.41, +0.86, +0.12, +0.24.
- **Shot distance** — the obvious explanation for "makes bend less" is that deeper shots need
  more bend and go in less. Refuted: **corr(knee_3d, rim_dist_px) = +0.01**, and residualising
  knee on distance leaves **d=+0.40, p=0.0398**. Within-zone it also keeps its sign
  (mid-right +0.33 on 36/37, near-right +0.55 on 11/16).
- **Pseudo-replication** — one-to-one join throughout.

⚠️ **Still NOT a finding, and must not be coached on.** It does not survive Bonferroni across
the 7 metrics tested (0.037 vs 0.0071); it is one shooter, one session; and it was found
post-hoc in the same data that produced the follow-through artifact. Expected count of p<0.05
under a global null across 7 tests is 0.35, so a single 0.037 is unsurprising on its own — what
raises it above noise is the consistency across four clips and its survival of four independent
confound tests, neither of which the pooled p captures.
⛔ **THE FIRST PRE-REGISTRATION WAS BROKEN — CORRECTED 2026-08-04 (Fable + Codex, independently
agreeing).** I registered: *"one-sided positive, d ≥ +0.25 with clip-stratified p < 0.05 on ≥ 80
labelled shots; anything less is a failed replication and the candidate is dead-lettered."*
**That rule is self-contradictory and would have destroyed a real effect.** At n=80 balanced,
one-sided α=.05 needs **d ≈ 0.37** just to reach significance — so the "d ≥ 0.25" clause is
NON-BINDING, it can never be the operative constraint. And at a true d=0.25 the test has
**~30% power** (n=122 → ~40%; **~396 shots** are needed for 80%). My rule would have
dead-lettered a genuine d=0.25 roughly **70% of the time**. Registering a threshold is worthless
if you do not check that the design can DETECT it — I checked the direction and forgot the power.
⭐ **CORRECTED REGISTRATION — estimation, not a pass/fail gate.** A single home session cannot
confirm or refute d≈0.25, so it must not pretend to:
- Report `knee_bend_3d_deg` makes-minus-misses as **d with a 95% CI**, clip-stratified, on
  **blind hand-labelled** shots (see below — model labels are disqualified).
- **Non-significance does NOT dead-letter the candidate**, and significance alone does not
  promote it. Both are single-session estimates.
- **Pool across sessions** and re-test the pooled estimate; promote only when the pooled CI
  excludes 0, kill only when it excludes +0.25. Until then the honest state is "unresolved".
- Keep the direction (positive) registered, so the pooled test stays one-sided/confirmatory.
- ⛔ **The permutation test needs a block/circular-shift sensitivity check**: outcomes cluster
  in runs (clip 2 attempts 12-22 are all misses, clip 3 attempts 7-15 all misses — verified in
  the hand-count CSVs), so free within-clip shuffling assumes an exchangeability that the data
  visibly violates.

### ⛔⛔⛔ RETRACTED WITHIN THE HOUR — THE "REPLICATION" COMPARED TWO DIFFERENT JOINS (2026-08-04)
I promoted `knee_bend_3d_deg` below and then broke it myself while looking for a mechanism.
**The two `+0.40`s were computed with DIFFERENT wide-side anchors.** `flare_report.wide_shot_
times` returns the **mid-flight** time (`f[len(f)//2]`), which is what the 07-29 discovery used;
my new 07-20 harness matched on the **release** (`frames[0]`). Same readings, same truth, same
n — only the anchor differs:

| session | release anchor | mid-flight anchor |
|---|---|---|
| 07-29 | **−0.19** | +0.40 |
| 07-20 | +0.40 | **+0.16** |

⭐ **The labels are not wrong — the MEMBERSHIP is.** Of rows labelled under both anchors, **0%
get a different make/miss label**; but 29 rows (07-29) and 19 (07-20) are matched under only one
anchor. So an arbitrary half-flight shift changes *which ~25% of shots enter the analysis*, and
that is enough to swing d by 0.6 and flip its sign.
⛔ **On the STABLE CORE (rows both anchors agree on) there is no effect and no replication:**
**07-29 d=−0.07 (n=95), 07-20 d=+0.47 (n=74)** — opposite signs, discovery session ≈ zero.
Pooled on the core: **d≈+0.17, 95% CI [−0.14, +0.48] — includes 0**, so the pre-registered
promotion criterion is NOT met. **The candidate returns to unresolved.**

⛔⛔ **I IGNORED MY OWN RECORDED WARNING.** This file already said, from the 08-03 work: *"my
close-cam d's are join-variant — an equally defensible rebuild gives knee +0.15 (not +0.28)...
Numbers that move that much must not carry doctrine."* I then built a "replication" without
re-checking join sensitivity, and treated agreement between two different pipelines as
confirmation. ⭐ **A replication must run the IDENTICAL pipeline on new data. Two pipelines
agreeing on one number is not replication — it is a coincidence I went looking for.**
⭐⭐ **THE DURABLE FINDING IS ABOUT THE JOIN, NOT THE KNEE:** the close-cam→wide-shot match is
too fragile to carry ANY make/miss claim. Shots are spaced closely enough that a ~0.5s anchor
shift re-selects a quarter of the sample. **Every close-cam make/miss number in this file
inherits that fragility** and should be read as provisional until the join is anchored on a
single defensible event and that choice is justified rather than inherited.
⭐ Note the coverage/distance/clip/reactivity checks all PASSED and were irrelevant — they test
the rows you kept, and the defect was in which rows you keep. **Confound checks conditioned on a
sample cannot see a defect in sample selection.**

#### ⛔⭐ AND A SECOND, INDEPENDENT KILL — WINDOW TRUNCATION (Codex, same day)
Codex reached the same retraction by a completely different route and supplied the **mechanism**,
which is a defect in the anchor fix I wrote that morning. `compute_form` re-derives its own
release, and `span` began at a hardcoded `frames[0] - 20` — but the internal release can land up
to **18 frames BEFORE** `frames[0]`, leaving as little as **2 frames** of pre-release history.
The knee loop stops at `rel_f`, so it never reaches the dip and reports a nearly straight leg.
- **corr(knee_bend_3d_deg, release_frame_delta) = −0.62 (07-29), −0.72 (07-20)** — the metric is
  substantially measuring *how far the two release detectors disagreed*.
- Rows whose internal release preceded the anchor by ≥15 frames: **156.5° / 159.2°** of "bend"
  versus **103.8° / 107.4°** elsewhere.
- Dropping likely boundary-truncated shots takes 07-29 from **+0.40 → −0.00**, with a residual
  **+0.31–0.36** on 07-20 — **converging with my stable-core split (−0.07 / +0.47) by an
  independent route.** Two decompositions, same verdict: the discovery effect was artifact.
✅ **FIXED**: `span` now starts at `min(frames[0], rel_f) - 20`, so the pre-roll is guaranteed
against whichever release is actually used. Both close passes re-running.
⚠️ **`release_frame_delta` is what made this findable** — I emitted it that morning as an
honesty diagnostic with no idea it would expose a defect in the same change. Cheap diagnostics
pay for themselves.
⚠️⚠️ **Scale reality check, with citations:** published BlazePose-World knee-angle error is
**7.1° MAE (lateral, dynamic)** to **17.2° MAE**, against an observed contrast of **5.4°**. The
effect is *below the per-observation noise floor*. Random error averages down, but the error
here is **systematic with window geometry**, and averaging does not cure that.
⚠️ Codex also measured close-vs-wide 3D knee on 36 shared shots: **r=0.05, CCC=0.029, MAE 30.6°,
only 25% within 10°** — consistent with my r=+0.07 (n=35). **Metric "world" landmarks do NOT
make the camera choice irrelevant on this footage.**
⭐ Knee is NOT a proxy for tempo (tempo's make-direction is opposite) — it is a proxy for **how
much valid pre-release history survived**, which no metric reported directly.

#### ✅ THE FIX VERIFIED — and it explains BOTH defects at once
Re-ran both close passes on the corrected `span`:

| | corr(knee, release_frame_delta) | knee reads >150° |
|---|---|---|
| 07-29 | −0.60 → **+0.20** | 16 → **1** |
| 07-20 | −0.77 → **−0.12** | 14 → **0** |

(My −0.60 / −0.77 independently reproduce Codex's −0.62 / −0.72.) The straight-leg artifact is
essentially gone and the mean knee drops to a physically sensible 98-104°.
⭐⭐ **And the join-anchor sensitivity collapsed with it.** On 07-20 the anchor swing was
+0.40 vs +0.16; it is now **+0.19 vs +0.18** — a 0.24 spread reduced to 0.01. **One defect was
driving both symptoms**: truncated rows are exactly the ones whose release detection is
unstable, which are exactly the marginal join cases. That is why the "membership, not labels"
split found the effect living in the ambiguous 25%.
⛔ **On the fixed metric there is no effect: 07-20 d≈+0.19 on n=82, against a detectable floor
of ~0.44.** The instrument is materially better than it was this morning — correctly anchored,
correctly windowed, honestly gated, and no longer anchor-dependent — and it has produced no
finding. Both statements are worth keeping.
#### ⛔ FINAL VERDICT ON THE FIXED METRIC — the sessions DISAGREE IN SIGN
Clip 4 re-ran clean and merged back (141 rows, per-clip counts identical to the original
29/37/33/42), so both sessions are complete on the corrected metric:

| session | release anchor | mid-flight anchor | anchor swing | n | detectable floor |
|---|---|---|---|---|---|
| 07-29 | **−0.41** | −0.41 | **0.00** | 109 | 0.38 |
| 07-20 | **+0.19** | +0.18 | **0.01** | 82 | 0.44 |
| **POOLED** | **−0.15** | 95% CI **[−0.44, +0.14]** | | | **includes 0** |

⭐ **The anchor swing is now ZERO** (0.00 / 0.01, from 0.60 / 0.24) — the join instability is
fully resolved, which confirms window truncation was the single root cause of both symptoms.
⛔ **And with the metric corrected the two sessions have OPPOSITE SIGNS.** Pooled d=−0.15 with a
CI spanning zero: the pre-registered promotion criterion is **NOT met**. `knee_bend_3d_deg` is
**dead as a make/miss candidate**, not merely unresolved.
⚠️⚠️ **Read this before ever trusting a single session again:** post-fix 07-29 reads **−0.41,
which CLEARS its 0.38 floor** — so a session-of-one analysis today would license "makes bend
MORE", the exact opposite of the claim I made this morning from the same clips. Two
floor-clearing, opposite-signed results from one shooter at one hoop is the strongest possible
statement that **single-session effects in this data are noise wearing a p-value.**

⚠️ **REPRODUCIBILITY HAZARD:** the 07-29 re-run **crashed on clip 4** with a native
`access violation writing 0x0000000000000020` in the pose stack, silently losing 42 of 141
releases (141 → 99). The persist-before-fragile-steps change from earlier that day saved the
other 99 instead of the whole pass. ⚠️ This box already has a hardware question open (6 silent
power-losses in 5 days, PSU suspected) — a native access violation belongs on that list.
**Always check the per-clip counts in the log, not just that the file was written.**

### ⛔ SUPERSEDED — the promotion this retracts (kept for the record)
**It did not need a new session.** 07-20 was untouched for this hypothesis, carries 111
hand-counted outcomes, and its close-cam clips were on disk all along (as S8-named files in
`Camera 1/Old/` — Pixel filenames are UTC, the S8 is UTC−4, which is why they read as unrelated).
That close-cam pose pass had never been run; it was a standing TODO from July.

| session | d | n mk/ms | SE | 95% CI |
|---|---|---|---|---|
| 07-29 (discovery) | +0.40 | 49/61 | 0.194 | [+0.02, +0.78] |
| **07-20 (replication, independent)** | **+0.40** | 38/44 | 0.224 | [−0.04, +0.83] |
| **POOLED (inverse-variance)** | **+0.40** | | 0.146 | **[+0.11, +0.69]** |

✅ **The pre-registered promotion criterion is MET**: the pooled CI excludes 0, and does not
exclude +0.25 (so the kill condition does not fire). The replication's **one-sided
clip-stratified permutation p = 0.0451** in the direction registered *before* the run
(committed in this file at 5070350, hours before 07-20 was touched); two-sided is 0.0756, and
both are reported so the one-sided figure cannot look like a choice made after seeing the sign.
Every confound re-checked independently on 07-20: **coverage 79% of makes vs 88% of misses,
Fisher p=0.282 — clean, and leaning AGAINST the effect**; **corr(knee, rim_dist) = −0.04**
(07-29: +0.01); clip-stratified throughout; one-to-one on both join hops.

⚠️ **WINNER'S CURSE, stated plainly:** the 07-29 estimate is the discovery — it was selected as
the 1 of 7 metrics that cleared its floor, so it is biased UPWARD, and the pooled figure
inherits that bias. The unbiased number is the replication alone: **d=+0.40, CI [−0.04, +0.83]**,
which includes 0. What is genuinely striking is that it did **not shrink** — a winner's-curse
artifact almost always regresses toward 0 on replication, and this did not move at all.
⚠️ **Effect size in real units: 108.8° vs 103.4°, i.e. makes come with ~5.4° LESS knee bend.**
⛔ **This is NOT coaching advice and must not become a drill.** It is a correlation with no
established mechanism, it is counterintuitive (the textbook expectation is deeper legs), and
"bend less" is exactly the kind of knob this project has repeatedly proven to be a bandaid.
⛔ One shooter, two sessions, one hoop, both home footage — external validity is untested.
⭐ It is measured entirely BEFORE release, so the reactivity objection that killed
`follow_through_hold_s` cannot apply. Next: a mechanism, not a third session.

⛔⛔ **MODEL LABELS ARE DISQUALIFIED for this test, and the margin is not close.** At 89%
label accuracy, symmetric error attenuates a true d=0.25 by ~(1−2·0.11)=0.78 → observed d≈0.20,
dropping power from 0.30 to ~0.22; at d=0.40 power falls 0.56→0.40. Worse, make/miss errors are
plausibly **correlated with rim-event kinematics**, which BIASES d rather than merely shrinking
it — it could manufacture or erase the effect while overall accuracy still looks fine. The next
session's knee shots must be **hand-labelled, blind**.

⭐⭐ **LESSONS.** (a) At n≈30 the SIGN of an effect is nearly a coin flip for any |d|≲0.3 —
never build an argument on sign agreement. (b) "Same shots" is a claim to be COMPUTED, not
assumed; the overlap was 15. (c) State the whole table: I reported the two metrics that flipped
and omitted the two that agreed. (d) "Every past effect was a camera artifact" was
fabrication-adjacent — the effects died of noise at d-SE 0.2-0.4, and no artifact mechanism was
ever demonstrated.
⇒ Do not chase body-form make-drivers on the WIDE camera. Remaining routes: the owner's feel
labels (short/long is invisible to every camera here), a close-camera reliability check, and a
genuinely metric 2-camera reconstruction (scaffolded, unvalidated).

## ⭐⭐⭐ FORM-COACH VERDICT (2026-08-03): NOTHING predicts this shooter's makes
The product ran end-to-end on 07-29 for the first time — 135 shots over 28 min from four 4K
clips, with chart, per-shot CSV, zone summary, consistency, fatigue trends and drills. Then
every metric was tested against the **hand-counted** labels (137 of 139 produced shots paired
to truth; per-clip lead ~+40f, residual sd 7-11f).

**All twelve metrics are null.** Ball arc: release d=0.16, entry d=0.08, apex-above-rim
d=0.11, apex height d=0.20. Body: knee d=-0.29, balance drift d=-0.39, elbow d=-0.28,
follow-through d=-0.22. Elbow flare (126 releases from the close cam, audio-synced at
0.83-0.90 confidence): **d=+0.077, p=0.67 — and the SIGN IS FLIPPED from 0710's d=-0.28**.
A result that reverses direction between sessions of the same shooter is noise, and 0710's
labels were later shown to be ~coin-flip anyway. Detectable floor here is d~0.34-0.72.

⛔⭐ **THE KNEE ARTIFACT — my error, worth keeping.** Ungated, knee bend read makes 109° vs
misses 139°, **d=-0.65, p=0.019**, and it survived a within-zone check (pooled within-zone
difference -29.7° vs raw -29.9°, so NOT a position confound). It looked like the first real
find in the project. Then: the knee distribution runs **min 6°, max 178°, with 24 of 58
readings outside a plausible 90-160°** — a 6° knee is a person folded in half. `metric_ranges`
already defines knee_bend_deg as (30,150) and `correlate` already applies it (line 101, added
by the 2026-07-06 audit) — **my ad-hoc analysis bypassed the gate the codebase already had.**
Gated: **d=-0.29, p=0.44**, 26 of 56 readings dropped. The product's own drivers file reports
d=+0.285, p=0.469, i.e. it was right all along. ⭐ Lesson: a significant result on a metric
with impossible values is a data bug wearing a finding's clothes — and the half-split showed
it too (knee absent in H1 d=-0.09, huge in H2 d=-1.23; balance drift the exact opposite,
d=-0.72 in H1 and -0.24 in H2 — two metrics whose "effects" appear in opposite halves).

⛔ **WIDE-CAM POSE IS UNFIT FOR BODY FORM.** Coverage is only 35-42% of shots, and ~46% of
knee readings are physically impossible, because the shooter is ~22% of frame height. Body
metrics must come from the CLOSE camera. `flare_report` already poses the close cam
successfully (141 releases) but emits only flare — **extending it to the full body-metric set
is the concrete next build**, and the only way the form-coach thesis gets a fair test.

✅ **Product changes shipped from this:** the coach no longer prescribes a target off a
foreshortened number (it coaches the SPREAD, which a constant bias does not corrupt); nulls
report as nulls with the power floor instead of the three largest noise results being called
"leans"; the resubstitution guard now lives in `build_session`, not just the eval.

⚠️ **Standing:** the owner's feel-review labels remain the highest-value UNMEASURED axis —
miss depth (short/long) is the one thing no camera here can see, and it is the axis most
likely to be coachable.

## THE LAST 6 MISSES (2026-08-03) — three causes, one of them already built for
137/143 leaves 6, and they are not six unrelated problems:

| cause | n | detail |
|---|---|---|
| **78° near-vertical gate** | 2 | healthy segments (n=32, drop 249-263) rejected at rel 81/ent 86 and rel 78/ent 84 |
| **fragmented track** | 3 | segments of n = 2, 4, 5 points |
| **never reached the rim gate** | 1 | no rim event within 60f of the hand-counted frame |

⭐ **The 78° gate is a hard-coded IMAGE-SPACE angle**, and `ARC_METRIC_HONESTY.md` already
establishes that release/entry angles are only physically valid SIDE-ON — this footage is
oblique/behind, where real shots read steeper than they are. So a fixed 78° threshold is
geometry-dependent in exactly the way the pixel constants were scale-dependent. That is a
principled reason to make it camera-aware, but the *value* is arbitrary either way, so it
must not be dialled against these 143 attempts.
⭐ **The 3 fragmented-track misses are precisely what `back_extend` was built for**, and it
does not catch them: it needs 2-3 coherent points to seed a fit, and after trimming the
teleport these segments are down to ~1. Seeding from the rim event itself (position known,
velocity unknown) is the obvious next attempt — the module is inert today but this is the
case that could earn it back.
⛔ Deliberately NOT acted on yet. Three of my hypotheses about this footage died today;
these are candidates, not fixes, and the tail is worth 4% of recall.

## ⭐⭐⭐ RECALL 85% → 96% (2026-08-03) — and the cause was NOT what I built
Chasing the segmenter produced a large win, but the attribution is the lesson.

| | frozen 08-02 | **now** |
|---|---|---|
| Detection recall | 122/143 = 85% | **137/143 = 96%** [CI 91-98] |
| Precision | 0.984 (2 FP) | **0.986 (2 FP)** |
| Per clip | 86/82/89/85% | **93 / 95 / 95 / 100%** |
| Airballs | 5/5 | 5/5 |

**The whole gain came from one line: RANSAC's `threshold_px`.** It was a raw `8.0`, which is
0.22 rim radii at 0720 (rr≈36) but only 0.07 at 4K (rr≈120) — three times tighter in
physical terms, so a perfectly good arc could not find inliers. Unlike `launch_drop`, this
one genuinely IS a measurement tolerance on pixel positions, and the same physical wobble
covers 3× the pixels at 4K. It is now `(8/36)·rr`, which reproduces 8.0 exactly at the 0720
rim, so that session's behaviour is preserved by construction rather than by hope.

⛔ **The backward extension I built for this is MEASURED INERT.** `shotlab/back_extend.py`
works — it recovers 6/19/40-point runs where the track had 2-6, and its 14 synthetic tests
pass — but once RANSAC stopped starving, condition **C6 (with extension) is byte-identical to
C5 (without)** on every clip. The module stays as an opt-in `cloud=` argument (default None,
so production is untouched) with C6 kept in the eval as the standing ablation that shows it
inert. It may earn its keep on weaker footage; on this footage it does nothing.
⭐ **The diagnosis chain was right and the fix was wrong, twice over:** "every miss is the
segmenter" was correct and led to the win, but both mechanisms I proposed (scaled
`launch_drop`, backward extension) were duds. What actually found it was instrumenting the
rejections — the new `reject_log=` argument on `detect_shots_to_rim`, which turned "RANSAC
failed" from invisible into 7-of-11.

⚠️ **This 96% is NOT a held-out number.** It was obtained by diagnosing failures on the same
143 attempts that produced the frozen 85%. The scaling constant has no free parameter (it is
pinned to reproduce 0720), so this is diagnosis rather than tuning — but it still needs a
fresh session to stand as a claim. **The frozen 85% remains the last clean held-out result.**

⛔ **RESUBSTITUTION TRAP, caught same day.** Preferring the newest model made the eval score
`make_visual_0729.joblib` on the very 122 shots it was fitted on, printing a glorious 94%.
The honest figure is unchanged at **89% LOCO**. Fixed with machinery, not a note:
`models/<model>.trained_on.json` records the training clips, and `eval_ablations` now prints
a loud RESUBSTITUTION warning and tags the model name when the clip under test is in that
list.

## ⛔⭐ RIM-SCALING `launch_drop` — BUILT, MEASURED, REJECTED (2026-08-03)
Having found every miss is the segmenter, the obvious fix was that `launch_drop` is a fixed
200px while `shot_gate_px` scales with the rim — at 4K the gate (240px) had grown PAST the
drop, so a point could be "at the rim" while already below launch height. Using the rim as a
ruler (18in across → px_per_ft = 2·rr/1.5), 200px is 4.2ft at 0720 and only 1.2ft at 4K, so
scaling it gives ~670px.

**Result: recall COLLAPSED 85% → 40%** (46/34/54/33% per clip). Reverted; the banked
122/143 = 85% at precision 0.984 is restored exactly, 41/41 tests green.

⭐ **Why the physics doesn't apply — and this is the useful part.** The walk-back can only
measure what the TRACK contains, and the track does not extend back to the shooter's hands:
the detector picks the ball up partway into the flight. So `launch_drop` is **not** measuring
"how far below the rim the release is" — it measures **"how much of the flight was tracked"**.
That is a frame-count-like quantity, not a physical height, and it must NOT scale with the
rim. Demanding the true physical drop rejects nearly every real shot.

⛔ **Second hypothesis in a row that was right in principle and wrong in fact** (after the
rr² make/miss mechanism). Both were pre-registered, both were tested against a control or a
revert, and both died on measurement rather than on argument — which is the process working,
but it is a standing warning about this footage: *the plausible mechanism has been wrong
twice, so build the check before the fix.*

⛔ **Do not re-derive this.** `launch_drop_px()` is kept in `court.py` as a documented dead
letter with the arithmetic, so the next reader does not rediscover it.

**What the evidence now points at instead.** The failures are "drop 105-153px vs a 160px bar"
and "n = 2-6 points" — i.e. *not enough of the flight is in the track*. The principled fix is
therefore to EXTEND THE TRACK BACKWARD from the rim (backward association from a confirmed
rim event, using the cloud), not to move a threshold. Lowering the bar would admit these
shots by weakening the test that keeps precision at 0.984, and — decisively — it would be
tuning on the 143 attempts that are our only held-out measurement. ⚠️ Any threshold chosen by
sweeping these clips is no longer a held-out number and needs a fresh session to validate.

## ⭐⭐⭐ EVERY REMAINING MISS IS THE SEGMENTER (2026-08-03) — detection is not the wall
`diagnose_misses` over all four 07-29 clips, baseline condition, 34 missed attempts:

| bucket | count |
|---|---|
| DETECTION (detector never saw the ball at the rim) | **0** |
| TRACKER (candidate reached the rim, track didn't follow) | **0** |
| SEGMENTER (track reached the rim, no shot emitted) | **34 / 34** |

**This overturns the 0720-era conclusion that ~10 misses per clip were detection-limited.**
On 4K footage the detector sees the ball at the rim on *every single missed attempt* and the
tracker follows it there *every time*. `detect_shots_to_rim` then throws it away.

Reasons, from the tool's own diagnosis:
- **insufficient launch drop — ~19 of 34**, and *most of the drops are NEGATIVE*
  (−11, −66, −88, −115, −120, −142, −156, −171, −188, −261, −365). A negative drop means the
  segment's start is ABOVE the rim, i.e. the walk-back never reached the release point and
  the segment begins mid-flight.
- **too few points (n = 2,3,5,6,7 < 8) — ~8**. With a detector hitting ~80% of frames at
  stride 1, a ~1s flight should yield ~25 points. n=2 means the segment is a fragment.
- **RANSAC fit failed (gather-poisoned) — 3**; **78° gate — 1**.

⛔ **This reframes task 5.** The pre-registration guessed the pixel gates were too TIGHT at
4K, but the arithmetic runs the other way: `launch_drop=200px` was ~4ft at 0720's ~50 px/ft
and is only ~1.2ft at ~170 px/ft, i.e. *easier* to satisfy. The failure is not a mis-scaled
threshold — it is that **the walk-back is not assembling whole flights on dense 4K tracks**,
so the segment it measures the drop across is a fragment starting above the rim. Rescaling
the gates without fixing that would be tuning around a broken measurement.
⭐ The prize is large and unusually well-localised: 34 baseline misses (21 after C5 recovery)
all in one function, with detection and tracking exonerated by measurement.

## ⛔⭐ THE PRE-REGISTERED MAKE/MISS MECHANISM WAS WRONG (2026-08-03)
Pre-registered item 2 said make/miss failed because three features are raw pixel counts in
rr²-scaled regions, and rr moved 36 → 120. **Tested and refuted.**

Normalising those masses to a reference radius (`REF_RR`, `make_visual.py`) and re-fitting
on 0720, then testing ONCE on 07-29: **58%**. An identical re-fit with normalisation OFF —
the control that exists precisely to catch this — scored **62%**. The fix did nothing; the
control was, if anything, slightly better. Scale was not the binding constraint.

**The real diagnosis, from leave-one-clip-out WITHIN each session:**

| training set | tested on | accuracy |
|---|---|---|
| 07-29, LOCO (3 clips → 4th) | 07-29 | **89-90%** [CI 82-94] |
| 0720, LOCO | 0720 | 83-84% [CI 75-90] |
| 0720 (all 96) | 07-29 | **58-62%** |
| 0720 + 3× 07-29 clips | 07-29 | 84-86% (worse than 4K alone) |

Per clip within 07-29: 92/87/94/82% normalised, 96/84/94/88% control. Base rate is 48%
makes, so majority-guessing scores 52%.

**Make/miss is SESSION-specific, not scale-specific.** The cues are *stronger* on the 4K
footage than they ever were on 0720 (89% vs 84% within-session) — the net whip, the orange
mass below the rim and the white-occlusion dip are all plainly there. What does not survive
is a model fitted to one session's lighting, background and camera geometry. Adding the 0720
clips to the 4K training folds makes it *worse*, so the old session is not merely unhelpful,
it is misleading.

**This reframes the original 81% too: it was never a cross-session number.** It was a
within-session LOCO figure (reproduced here at 83-84%), and the error was in the claim's
scope, not the model. `--validated`'s "make/miss 81%" implied a generality it never had.

**Actions taken:** `models/make_visual_0729.joblib` fitted on all 122 labelled 4K shots
(LOCO 89%); `--make-model auto` now prefers the newest model in both `build_session` and
`eval_ablations`, with a comment that auto is a convenience and not a transfer claim.
The rr-normalisation is **KEPT but corpse-marked MEASURED-INERT** — the geometry argument is
correct and will matter if rim scale ever varies *within* a session, but it moved nothing
here (89 vs 90 LOCO) and must not be cited as a fix.
⛔ **Standing rule this earns: a make/miss accuracy is meaningless without naming the session
it was fitted on.** Quote it as "N% within-session LOCO", never bare.

## ⭐ PRE-REGISTRATION for the 07-29 frozen eval (written 2026-08-02, BEFORE any result)
From the Fable audit. These are alternative explanations recorded in advance so that a poor
07-29 number is not automatically read as "the detector didn't generalize" — and, more
importantly, so **nobody re-tunes a knob on the held-out session.** There is only one
untouched session; spending it on tuning destroys the only honest measurement available.

1. **Pixel-constant gates are scale-dependent and were tuned at ~50 px/ft (0720). The new
   footage is ~170 px/ft** (ball r≈65px vs ≈24px). Measured from the cached clouds:
   in-flight consecutive-frame ball displacement is now median ~30px / p95 60-80px (0720:
   median 14 / p95 29), and 2-4% of steps exceed 90px. Affected, none of which scale with
   frame size or rim radius: `track.assemble_track` `gate_px=120` and `30*conf`;
   `track_beam` `motion_gate=90`, `conf_bonus=25`, `size_penalty=3*|Δr|`, `coast_penalty=40`;
   `court.detect_shots_to_rim` `launch_drop=200px` (was ~4ft of required launch depth, now
   ~1.2ft → bounce FP risk) and RANSAC `threshold_px=8` (was ~0.8 ball radii, now ~0.12 →
   inlier starvation → recall risk); `court.is_real_shot` `below<120` / `apex_y>rim_y+90`;
   `rim_recovery` `x_corridor=650px` (was ~13ft of backward arc, now ~3.8ft — the pass that
   bought 79%→86%). **Prediction: a meaningful share of any recall/precision drop is these
   gates, not the detector.** The fix is to express them in rim-radius / px-per-foot units —
   but do that as a SEPARATE, re-measured change, not as a rescue of a bad number.
2. **`make_visual` features are not scale-normalized.** Three of seven (`o_below`, `o_side`,
   `o_net`) are `log` of raw orange-pixel counts in rr-scaled regions, so their areas scale
   with rr² — rr ≈36 → ≈127 shifts those masses ~×12 (≈ +2.5 in log space), far outside the
   scaler+GBM's training distribution (89 shots at one scale). **Therefore 07-29 make/miss
   accuracy tests "model + representation", not the modelling idea.** Normalising by rr²
   and re-fitting is a legitimate follow-up; doing it before the frozen run would forfeit
   the test.
3. **Already-known and unchanged:** the detector trained on 0720 clips 1-2, and the union
   dedup window, apex-below-rim FP gate, beam `max_coast` 4→6 and the recovery pass were all
   tuned against those same 111 attempts. The 07-29 run is the first measurement free of that.

## ⭐⭐⭐ BROAD DUAL REVIEW → MAKE/MISS FIXED + RIM RECALIBRATED (2026-07-23, later)
Owner asked "are there OTHER areas to improve?" → dual adversarial review (Codex + Fable, broad).
Both converged: we'd over-focused on detection and MISSED bigger issues. Verified + acted:
- **Make/miss (the product's #1 output) was UNMEASURED and ships at ~51% (coin-flip).** Geometric
  `classify_make` is miss-biased + abstains ~40%; audio fusion is anti-signal (only fills unknowns
  as "miss", wrong 13/20). ✅ Added make/miss scoring to `eval_ablations.py` (permanent gate). ✅ A
  learned model (`make_visual`, net/occlusion cues) exists; shipped model brittle cross-session
  (88/50/82% on the 3 clips). ✅ RE-FIT on the 89 new hand-count labels → **81% LOCO** (vs 51%
  geometric); saved `models/make_visual_0720.joblib`. ✅ WIRED into production (`build_session
  --make-model auto`, default-on when present; geometric fallback; audio demoted to last-resort).
- **Rim calibration was broken:** clicked radius 8-12px vs true ~36-47px (center+near-center click).
  Corrupts make/miss thresholds (rim-radius units) + apex-height-ft (~5× inflated). ✅ `verify_rim`
  now clicks LEFT+RIGHT edges (center=midpoint, radius=half-span) + ball-diameter sanity check +
  fresh-start each run. Owner RE-CLICKED all 3 (r~36-39). The corrected rim is what UNLOCKS
  make_visual (r=8→52%, r=36→88% on clip1). ⚠️ my earlier "555 hoop-center / 8.5° entry bias" was
  WRONG (555 is the left edge; true center ~602-616, real entry bias ~1.5-2°). The RADIUS was the bug.
- **REVIEWER FINDINGS — WORKED IN ORDER (2026-07-23 later):** ✅ (1a) max-cardinality matcher +
  tolerance sweep (f515291); ⛔ (1b) untouched test session = OWNER-BLOCKED (needs new footage).
  ✅ (2) rim-anchored RECOVERY pass (09de1e9): VERIFIED 23/23 residual misses have near-rim
  detections → recall 79%→**86%** at prec 0.99 (`rim_recovery.py`, eval C5, in production `_union_beam`;
  "detection-limited is the wall" REFUTED). ✅ (3) calibration PARITY (bdefc8c): build_session now uses
  `verify_rim` rims + `--validated` profile flag. ✅ (4) cache CODE-HASH (9885875): `_code_hash()` in
  both cache sigs. ✅ beam coast frame-gap guard (d4aa787). **STILL OPEN (need owner judgment/footage):**
  (e) arc/form angle oblique-camera bias — metric-honesty labeling (report raw+uncertainty / gate
  confidence); changes coaching outputs, needs owner eye + can't validate vs GT here. (f) pose/form:
  hardcoded side_on, release=wrist-apex (10-15° elbow bias), release-height uses wrist not ball.
  Reviews filed: `process/reviews/2026-07-23_broad_*`.

## ⭐⭐ BEAM TRACKER VALIDATED (3 clips) + WIRED TO PRODUCTION (2026-07-23)
Owner hand-counted all 3 clips (111 attempts) + manual rims. **Aggregate: greedy recall 55%
(61/111), precision 0.98 → greedy∪beam recall 80% (89/111), precision 0.96.** Per clip:
c1 60→76, c2 42→81, c3 64→85 — recall up on EVERY clip. Only blemish: c3 adds 4 FPs (it has
2 airballs + retrieve clutter) → c3 precision 0.88; c1/c2 stay 1.00. **WIRED into production
behind `build_session.py --beam`** (threaded run_phase1→detect_or_load/detect_window→process_clip;
`use_beam` in the record + detection cache sigs so it doesn't collide with greedy caches). With
`--beam` the detector runs at conf 0.01 (the cloud) and `_union_beam` (pipeline.py) unions the
beam shots with greedy (greedy wins ties, merged track for make/miss). Default OFF (extra compute).
**✅ FP REDUCTION DONE 2026-07-23: c3 precision 0.88→0.97 (4 FPs→1), aggregate precision 0.96→0.99,
recall UNCHANGED (80%). Two targeted fixes: (1) apex-below-rim gate in `detect_shots_to_rim` (a
post-miss bounce whose arc never rises above the rim is not a shot; `y_seg.min() >= rim_y` → reject),
(2) union dedup 20→25f merges a miss's bounce-back re-approach (two rim events ~20f apart) while
staying below the 31f min gap between distinct attempts. The remaining 1 c3 FP is a clean arc 9s from
any logged attempt = likely a real shot the hand-count MISSED (so effective precision ~1.00). Both
mutation-tested (`tests/test_segmenter.py`). Precision is now clean enough to flip `--beam` default ON.**
⚠️ the beam benefits from stride 1; build_session's `--stride auto` may thin long
clips to 2+ — pass `--stride 1` for max recall. ⚠️ production `--calib`/auto_calibrate use the
calibrate.py `Calibration` format, NOT `verify_rim`'s `config/rim_<clip>.json` (eval-harness only) —
to run --beam on the 0720 clips in build_session, supply a matching `--calib`. FOLLOW-UPS: (a) reduce
the c3 FPs (tighter beam-shot acceptance, measured vs the 3 evals) then flip `--beam` default ON;
(b) the ~10-per-clip detection-limited misses = film-closer / detector-retrain front.

## ⭐⭐ FIRST EVAL RESULT — clip 1 (2026-07-23)
Owner hand-counted clip 1 (42 attempts) + manual rim (616,232). `eval_ablations` result
(`process/handcount/PXL_20260720_151519220_eval.json`): **C1 baseline@0.25 = precision 1.00,
recall 0.60 (25/42, ZERO false positives); C2 cloud@0.01 = recall 0.29 (WORSE).** So with the
CORRECT rim, precision is perfect — the old FP problem was the bogus rim. Bottleneck = RECALL.
**Root-caused the 17 misses by measurement (`tools/diagnose_misses.py` + cloud-arc RANSAC):**
- Segmenter threshold tuning (min_points 8→3, launch_drop 200→120): **ZERO effect.**
- Tracker scoring tuning (conf_weight 30→0, gate 120→80): **ZERO effect.**
- cloud@0.01: WORSE (floods the greedy tracker).
- The misses split: **~7/17 TRACKER-RECOVERABLE** (a clean launch→rim arc EXISTS in the conf-0.01
  cloud — RANSAC finds it — but the greedy single-pick tracker fragments it by flipping between the
  ball and a stationary distractor) + **~10/17 DETECTION-LIMITED** (no clean arc even in the cloud →
  ball not consistently detected on the fast ascent).
⇒ **Lever = a multi-hypothesis tracker (Viterbi/beam over the candidate cloud, Codex's rec) → ~7 shots
(60%→~77%); the other ~10 need better DETECTION (film closer / retrain on ascent balls).** NOT the
segmenter. ⚠️ ONE clip only — confirm the split on clips 2-3 (owner hand-count pending). Rim is slightly
front-of-center (616 vs hoop-center ~555); doesn't cause FPs but nudging to center may help borderline.

**✅ BEAM TRACKER BUILT + VALIDATED (2026-07-23, owner chose this).** `shotlab/phase1_ball/track_beam.py`
= multi-hypothesis beam MHT over the conf-0.01 cloud (const-velocity motion model, keeps top-K
hypotheses so a momentary distractor doesn't derail the arc; emits coherent track SEGMENTS). Measured
in `eval_ablations` (new conditions C3 beam / C4 greedy∪beam): **C1 greedy 0.60 → C3 beam-alone 0.69 →
C4 greedy∪beam 0.76 recall (32/42), precision 1.00, ZERO FP.** The union recovers exactly the 7
predicted tracker-recoverable shots [1,8,10,12,19,31,33] while keeping all 25 greedy shots (beam alone
loses 3 easy ones greedy holds → union). Remaining 10 misses = the detection-limited bucket
[3,15,18,24,28,32,35,37,40,42]. Covered by `tests/test_track_beam.py` (6 checks; beam follows the ball
through a distractor cloud). Key param: `max_coast=6` (was 4) let it bridge intermittent detections.
NOT yet wired into production `session.py` — do that after clips 2-3 confirm the gain generalizes.
NEXT: (a) owner hand-count clips 2-3 to confirm, then wire beam∪greedy into session.py; (b) the 10
detection-limited shots = film-closer / detector-retrain front.

## NEXT SESSION PICKUP (2026-07-22 night) — ⚠️ SUPERSEDED, see `KICKOFF_NEXT_SESSION.md` + the 07-23 sections above
**Where we are:** Step-1 of the TrackNet plan is DONE, and a DUAL ADVERSARIAL REVIEW
(Codex + Fable, owner-requested "push further") retired the oracle experiment family as a
sizing tool. Settled facts: detector sees the ball 99% inside labeled windows → TrackNet/tiling
stay OFF the table. The gap-based "perfect detection HURTS / cloud regresses" headline was a
`segment_shots` artifact → RETRACTED (mechanism proven). BUT the rim follow-up's `+2` is also
untrustworthy: the labeled-window design is **selection-biased** (labels seeded only from
already-detected shots → blind to the detection prize) and every number is an **unmatched raw
count** (no precision/FP; the `+2` includes a bounce-back false positive). The rim is a
**material confound** — and it turned out WORSE than "110px": on 2026-07-22 the owner set the rim
MANUALLY (`verify_rim`) at **(616,232)** and confirmed the camera is FIXED within a clip (moves only
BETWEEN clips). Rendered frames prove the real rim is the upper-LEFT hoop; the auto-rim (1134,470)
was locked on the shooter's **release zone / yellow shirt** (the ball is held at ~x1170 pre-shot).
So every earlier oracle experiment was anchored to a **bogus rim near where the ball is HELD**, not
the hoop — measuring ball-near-release, not shots-to-rim. ⚠️ **RETRACTED: my "110px within-clip
spread → tripod moved" claim was FALSE** (it was release-position variation near the bogus rim; owner
confirms no intra-clip camera motion). A single manual `Calibration` per clip is correct.
**Biggest EVIDENCED lever = verified-rim + segmenter repair, NOT "film closer"** — with a perfect
track the production `detect_shots_to_rim` still drops ~half the rim-reaching shots via 5 cheap-fix
defects (see the Step-1 section). Tracker velocity/reset fix correct; test coverage since ADDED
(commit e213ca6). All pushed through `6d7503b`.

**THE DECISIVE EXPERIMENT — HARNESS IS BUILT (commit b8ae653), waiting on the ~1hr hand-count.**
Owner chose "build on existing 3 clips." Runbook: `process/EVAL_HARNESS_RUNBOOK.md`. Three steps
(SYSTEM python): (1) `python -X utf8 tools/hand_count.py --clip <clip>` — watch FRESH, log every
attempt (m=make, n=miss, b=airball) → `process/handcount/<clip>_attempts.csv`; (2)
`tools/verify_rim.py --clip <clip>` — click ONE rim, or add frame-ranged rims for clip1's
camera-move (add pos1 @f0, navigate to the move, add pos2) → `config/rim_<clip>.json`; (3)
`tools/eval_ablations.py --clip <clip>` — detects once @0.01 (baseline=filter≥0.25), segments per
rim through the PRODUCTION `detect_shots_to_rim`, MATCHES to the hand count, prints precision +
recall split rim-reached/airball. C1/C2 runnable now; C3-C5 (oracle-track etc.) stubbed pending
dense GT. Reading guide + "what each result implies" in the runbook. AFTER real numbers exist:
the 5 cheap segmenter bug-fixes (`court.py:225-291`) + cloud@0.01, judged against them — NOT
before. Harness validated end-to-end on real GT (5 shots w/ 2 frame-ranged rims incl. the
bounce-back pair); `tests/test_eval_harness.py` covers the matcher/rims/CSV (36/36 suite green).

**⚠️ Gotchas:** (1) ONNX-DirectML inference runs under SYSTEM python, not `.venv_*`.
(2) Machine had 6 silent power-losses in 5 days (suspect PSU transients on the 9070 XT)
— watch PSU/temps before long GPU runs. Detail: session log 2026-07-22 (night) below +
`process/TRACKNET_FUSION_PLAN_2026_07_22.md`.

---

## NEXT SESSION PICKUP (2026-07-02) — HISTORICAL
State (end of a big 07-02 session, all pushed): wrist-apex release in the metric
path; jump height ankle-based + physics-gated; **orange-ball detector retrained**
(hit rate 37→83% on held-out clip, new canonical weights
`runs/detect/ball_orange/weights/best_openvino_model`); **profile-ranking split**
(arc vs pose pools) so form ideals survive; **shooter-height ruler**
(`--shooter-height`, body-scaled jump; release honestly floor-referenced + LOW,
depth-limited → needs 2-cam); **audio make/miss default-on**; **shot map** in
dashboard/report/PDF; app ships **spoken feedback** (`say.js`, TTS coaching) +
**feel-CSV export**; **textbook/universal ideals** (`textbook.py`: entry 45°,
flare 0°-needs-2cam) as a SEPARATE profile block. App SW cache v7, profile
re-exported (personal elbow ideal 117°). Full suite 16/16 + JS green.

**⭐ THE BIG ONE — 2nd camera (Galaxy S8) arrives within a week (~2026-07-08).**
**ALL PRE-ARRIVAL SOFTWARE IS NOW BUILT + SYNTHETIC-VALIDATED (2026-07-02):**
`threed.py` (6/6) + `sync.py` audio sync (6/6, <20ms across mixed rates/gains) +
`stereo.py` checkerboard solver (5/5, triangulation <0.6in vs truth) +
`twocam.py` fusion scaffold (5/5, 12° known flare recovered to 0.5°) +
`tools/make_checkerboard.py` (printable board, regenerate any time) +
`tools/calibrate_rig.py` (mono + stereo CLI). **Day-one S8 runbook:**
(0) BEFORE it arrives — print the checkerboard (verify the 6.000in ruler),
    optionally film it with the Pixel → `calibrate_rig.py --mono` pins the wide
    cam's true focal TODAY; get a 2nd mount; S8 set to 1080p30 + storage +
    transfer path. (1) Both cams rolling, ONE loud clap, wave the board through
    varied tilts visible to BOTH → `calibrate_rig.py A.mp4 B.mp4` →
    `data/calibration/stereo_rig.json` (want rms ≲1px). (2) Film a session
    (same clap ritual) → pose both cams → `twocam.fuse_pose_tracks` →
    `shot_3d_metrics` = ELBOW FLARE + release-point spread, in real feet.
    Remaining to write when real footage exists: the session-level glue
    (build_session-style CLI over fuse_pose_tracks) + LEFT_RIGHT-style flare
    sign pinning. Cam-1 stays wide (arc/rim/makes), Cam-2 = close body-cam.
    (3) **VERIFY COURT DIMS with the metric rig (user asked 2026-07-03).** Court
    ground-truth = **21.5 ft long × 37.375 ft wide** (CORRECTED from a first
    bad conversion of 24.375×42.25). Rig scale comes from the checkerboard, NOT
    the court, so it's an INDEPENDENT check. Easiest: a scale-sanity check —
    triangulate the 5'10" height / 18in rim; if true, all rig distances
    trustworthy. Direct corner triangulation needs both cams to see the corners
    (dedicated framing, not the tight body-cam) and is only ±0.5-1 ft on far
    corners. Single-cam height-ruler estimate BRACKETED the true corner diagonal
    (28.5 ft) between its wide/tele focal estimates (25.7-32.4) — contained but
    not pinned; the rig is what pins it.

**Buildable now (no hardware):**
0. **VOICE-TAG WORKFLOW (chosen 2026-07-02) — record-then-review with spoken tags.**
   User's tagging decision: NO tap buttons; record the workout with the phone
   CAMERA APP (saves video + his voice), say a short phrase per shot, tag
   good/bad + reason POST-session from the audio. Core BUILT: `shotlab/voicetag.py`
   (fixed vocabulary + `parse_phrase` + `assign_to_shots`, tests 10/10) +
   guarded `transcribe_vosk` (offline Vosk, grammar-restricted to vocab). **Word
   list he'll use:** every shot "good"/"bad"; after "bad" optionally "flare" /
   "off hand" (camera-BLIND → the reason voice tagging exists) / "short" / "long"
   / "rushed" / "balance" (optional feels). ⚠️ REMAINING (needs his 1st recorded
   session + `pip install vosk` + model): validate STT on real audio, then a
   `tools/voicetag_session.py` (transcribe → assign → write felt_good + reason
   tags into the session records) + feed into export_profile. NOTE the live app
   does NOT save video (rolling pose buffer only) — that's why this path uses the
   camera app. Guide-hand (his Q): 2-cam CAN check guide-hand POSITION + release
   TIMING (3D wrist vs ball — on-side vs under, lingering); finger-level
   thumb-flick/spin stays the hard stretch (ball occludes fingers).
1. **USER: test the app on the Pixel** — live camera, gold ideal-skeleton overlay,
   spoken feedback (🔊 toggle), Feel log (CSV) button.
2. **Headline finding to chase with filming:** makes come with much deeper knee
   bend (full 0701: 107° vs 137°, d=−0.77, p=0.035; camera-consistent wide subset:
   99° vs 137°, d=−0.97, p=0.017). First make-driver to clear significance.
3. **USER FEATURE REQUEST — situational/context profiles (build once form is
   trusted, post-S8).** Slice the profile by shot CONTEXT because ideal form
   differs per context. Two flavors the user raised:
   (a) **movement** — going LEFT vs RIGHT vs SET;
   (b) **high-arc / shooting-over-a-defender** (2026-07-02 spitball) — a workout
   of deliberately high-arc shots (pretend a tall blocker); the arc rises and
   the mechanics likely shift (deeper legs, higher/softer release, maybe elbow).
   ASSESSMENT: YES the system can learn a per-context ideal — feed it those
   shots, tag the good ones, and export_profile means them into a context
   profile; the arc/knee/elbow/tempo metrics WILL capture the shifts. Caveats:
   needs a SEPARATE context profile (don't blend into the normal average); it's a
   DESKTOP thing (arc/release/entry need the ball+rim; the app is pose-only); and
   there's no universal "ideal high-arc number" beyond entry-angle physics (past
   ~45-50° entry you trade accuracy for clearance) — the value is CONSISTENCY
   within the context + make-correlation per context.
   Pieces (shared by both flavors): (a) per-context ideal profiles (condition
   ideal/correlation engines on a context key — `movement_dir` exists; add an
   arc-band / session-tag key for high-arc); (b) Shot-review filter context ×
   form-grade → one-click "ideal going left" / "ideal high-arc" clips; (c)
   per-context ideals in profile.json eventually. "By movement" dashboard panel
   is the seed.
4. **Audio make/miss PROMOTED to default (A/B 2026-07-02).** On session 0701:
   ZERO contradictions with confident visual calls (16 makes/43 misses all
   agree), resolved 9/12 visual-unknowns (all → miss, consistent with a 27%
   session), classifiable 83%→96%, 37 shots low→medium conf. Corrected make%
   23.5%. Sharpened the drivers: knee bend p=0.0115; NEW second driver — makes
   arc 1.14 ft HIGHER (p=0.045). `--audio` is now DEFAULT-ON in build_session
   (`--no-audio` to disable). Caveat: audio agreeing with visual isn't ground
   truth — a session where the user calls out makes would truly validate it.
5. **Shooter-height ruler ✅ BUILT (2026-07-02, user is 5'10").** `--shooter-height`
   flag (build_session), accepts 5'10" / 5.83 / 70in (`scale.parse_height`). When
   set, release_height_ft + jump_height_ft use a BODY ruler measured per-shot from
   the pose (nose→ankle p90 over the shot window = most-upright stance;
   `scale.px_per_foot_from_body`, NOSE_TO_ANKLE_FRAC 0.875), upgraded to MEDIUM
   conf "body-scaled from your height". Falls back to rim-scale (LOW) when the
   nose isn't visible — never a silent garbage scale. Cache v7 + shooter_height in
   the record sig. tests: scale 7/7 (incl parse + body-not-rim-hot), form 17/17.
   **⚠️ NOT-DONE part:** real-feet SHOT DISTANCE for zones is genuinely ambiguous
   with one camera (release point is at the shooter's depth, rim at its own — the
   depth component needs the pinhole model). Deferred to the calibrated-focal path
   (film the checkerboard → mono intrinsics → true positions). Heights are clean
   because they're vertical distances at the shooter's own plane.
   **VERIFIED ON REAL DATA (2026-07-02):** the per-shot body/rim correction varies
   0.74×–4.5× BY GEOMETRY and that's correct, not noise — on a wide clip the
   shooter stands ~4× closer to the camera than the far rim (rim_radius 15px →
   rim_ppf 20 vs body_ppf ~88), so rim-scaled heights are ~4× hot there; on the
   moved-in clip (rim_radius 47) rim & shooter sit at similar depths so the rulers
   agree (~1×). Aggregate median correction 1.22×; the real win is the DISTRIBUTION
   tightening — release-height p90 10.7→4.6 ft, max 19→10; jump p90 1.9→1.2, max
   4.0→2.4 (the absurd wrong-depth outliers are gone). Ruler measured over the
   planted gather→release window (cache v8) so a long flight's rebound-drift
   doesn't inflate it.
   **RELEASE-HEIGHT FOLLOW-UP RESOLVED (2026-07-02, honestly):** diagnosed by
   rendering the release motion — the footage is REAR/oblique (shooter shoots
   AWAY from the camera), so the release point moves up AND deeper; its image-y
   barely rises through release (depth cancels height in projection). That's
   foreshortening: the release point sits off the body's depth plane, so a
   vertical-image estimate reads low and NO single-camera scaling fixes it.
   Two-part fix: (a) real correctness bug — release height now referenced to the
   FLOOR (`_ground_line`, p80 of the lower-ankle series) not the instantaneous
   (airborne) ankle, so an airborne release no longer loses the jump height;
   (b) release_height is now honestly LOW confidence always (even body-scaled)
   with a note that the true value needs the 2-cam 3D release point
   (`twocam.Shot3D.release_point`). After the fix real shots read 3.6–6.3 ft
   (floor-referenced, internally consistent); JUMP height stays MEDIUM
   body-scaled (feet don't change depth in a vertical jump — that one IS clean).
   Cache v9. tests: form 18/18. Recommended usage: always `--shooter-height 5'10"`.
6. **ORANGE-BALL RETRAIN ✅ DONE + PROMOTED (2026-07-02).** Diagnosis: hit rate
   dropped 35%→20% on 0701 because the old fine-tune only knew the old red/blue
   ball. Retrained yolo11n on 0701 orange-ball frames (old 982 frames + 0629
   raws were purged → orange-only set, fits the personal-use scope):
   make_dataset --ball orange (656 clean labels after a strict sat≥130
   orange≥0.05 skin/leaf sweep — junk/ball separate cleanly at 0.04/0.05),
   52 epochs (early-stop, best @42), val on held-out clip 182946: mAP50 0.995.
   **Head-to-head on the held-out clip: hit rate/sampled flight frame 37%→83%
   median (every shot better; worst 5%→65%) — >2× the old model and above its
   35% on its own old-ball footage.**
   **NEW CANONICAL WEIGHTS: `runs/detect/ball_orange/weights/best_openvino_model`
   (imgsz 640)** — use in all build commands. Cache-sig footgun fixed in the
   same change: weights identity now includes the run name (both models' export
   dirs are literally named best_openvino_model; basename-keyed caches would
   have silently reused the old model's detections). Old model kept at
   runs/detect/ball_finetune for provenance.
   **Re-detect+rebuild VALIDATED + ADOPTED (same day): 88 shots (was 71),
   100% make-classifiable, n_points/shot 14→25 median, make% 33%.** Eyeball
   verification: ALL 18 shots in 183225 (was 4) and ALL 21 in 183742 (was 9)
   are clean dense arcs into the rim; several old-only "shots" were dribbles /
   ball-in-hand junk from sparse tracks. Release-angle median 43→56° = arcs now
   include the true early flight (the late-lock fix, visible in data). Splits +
   reports regenerated. ⚠️ Session metrics NOT comparable to the old-build
   numbers (different detector = different shots); make-driver signals persist
   (knee bend still #1).
   **✅ PROFILE-EXPORT RANKING FIXED (2026-07-02):** export_profile now splits
   the pool — ARC ideals (release/entry angle) from the outcome/arc-good pool
   (select_good, clean arcs on all 88 shots), FORM ideals + skeletons from a
   pose-anchored pool (select_form_good: knee-bend present, feel>made>all
   ladder, release-conf sorted). On the 88-shot data: form ideals now from 18
   pose-reliable made shots (was 2/10 with elbow → dropped); elbow back at
   117.0° (cross-validates the prior 118.9°), all 5 form metrics present,
   skeletons 1→2. Deployed app/profile.json + SW cache v4→v5. Note arc angles
   are steeper now (release 54→64, entry 44→55) — the late-lock fix capturing
   true early flight, not a regression. tests/test_profile 7/7.
7. Smaller ideas left: goal-progress tracking, report emailing, ingest the app's
   feel-CSV into the desktop records (join on session/shot time).

---

## Session log 2026-07-22 (night) — Step-1 GATE run: TrackNet OFF, detection is NOT the lever + machine-crash diagnosis
Resumed after an unexpected machine restart. Two threads:

**🖥️ Restart diagnosis (Windows event log).** 6 unclean shutdowns in 5 days
(Event 6008/41), escalating: 3 on 7/18, 2 on 7/22 (10:10 AM + 4:42 PM). EVERY one:
`BugcheckCode=0` (no BSOD), no MEMORY.DMP/minidump/LiveKernelReport, no WHEA, no
GPU-TDR, `PowerButtonTimestamp=0` = silent instantaneous power-loss / hard hang,
NOT a software crash. Tight correlation with the RX 9070 XT work that ramped up
7/17–7/22. **Leading suspect: PSU tripping OCP on the 9070 XT's power TRANSIENTS**
(secondary: thermal, or unstable EXPO/PBO). Two multi-minute sustained GPU/DirectML
sweeps ran CLEAN today → not "any GPU load"; transient spikes (idle→load) are the
likely trigger. TODO for user: check PSU wattage/age + GPU temps under load.

**✅ Step-1 GATE experiments run (the whole point of the revised TrackNet plan).**
- **1b/1c (`tools/exp_subthreshold_signal.py` → `process/step1_gate_results.txt`):**
  over the 921 labeled ball-present frames, the detector hits **99% @conf0.01**
  (98% far), 95% @0.25; of the 46 frames the pipeline (plain@0.25) MISSES, plain@0.01
  recovers **85%** — tiling adds nothing (tiled@0.01==plain@0.01), motion 0%.
  ⇒ signal is present sub-threshold; the TRACKER, not the detector, is the limiter.
  **❌ TrackNet/WASB and tiling are OFF the table.**
- **1a oracle ceiling (`tools/exp_oracle_ceiling.py` → `process/step1a_oracle_ceiling.txt`):**
  ran the REAL `assemble_track`+segmenter four ways — baseline(YOLO@0.25)=**6** shots,
  cloud(@0.01)=**3**, ORACLE(GT center injected, detector never misses)=**4**,
  oracle+4px=**3**. Through THIS segmenter perfect detection DECREASED the count.
  ⚠️ **FABRICATION CORRECTED (dual-review 07-22 night):** an earlier note explained clip1's
  692 present frames as "dribble/hold" — FALSE. `make_label_task.py:44-49` builds labels ONLY
  from each already-detected shot's flight window ±12 frames, so those 692 frames are ~20
  flight-window clusters (sizes up to 118), NOT dribble/hold. The real reason perfect detection
  didn't help was the segmenter artifact below, not "presence≠flight."
  ⚠️ confound flagged at the time: used `segment_shots` (gap-split fallback); production
  uses rim-anchored `detect_shots_to_rim`. I claimed the DIRECTION was robust — **it is not
  (see the 07-22 night follow-up below), so that claim is RETRACTED.**
- **1a-rim follow-up (`tools/exp_oracle_ceiling_rim.py` → `process/step1a_rim_oracle_ceiling.txt`,
  07-22 night):** same four candidate streams, but segmented with the PRODUCTION rim-anchored
  `detect_shots_to_rim` (rim auto-detected per clip, median over the decoded labeled frames).
  baseline=**2**, cloud(@0.01)=**3 (+1)**, ORACLE=**4 (+2)**, oracle+4px=**3 (+1)**. The oracle
  delta **flips sign** vs the gap-based run (−2 → +2): the "perfect detection HURTS / cloud
  regresses" result was an **artifact of `segment_shots` merging flights**, not a system
  property. ⚠️ BUT the counts are single-digit, 3 clips, within labeled footage only, over
  sparse labeled-only tracks with an auto-rim — **too thin to decide either way.**

**⭐⭐ DUAL ADVERSARIAL REVIEW (Codex + Fable, 07-22 night, owner-requested "push further";
verbatim in `process/reviews/2026-07-22_step1_oracle_*.md`). Both converged independently:**
1. **Retraction SOUND** (Fable proved the mechanism: a perfect GT-only track through `segment_shots`
   drops all 3 certain makes — the gap-based path only ever segmented BECAUSE the detector missed
   frames that isolated the arcs).
2. **The whole labeled-window design is SELECTION-BIASED and cannot size any prize.** Labels come
   only from previously-DETECTED shots (`make_label_task.py:44-49`), so a missed shot yields no
   window — "99% recall" is conditional recall inside already-found regions, structurally blind to
   the detection prize. And every number is an UNMATCHED raw count (no attempt IDs, no precision/FP),
   so counts can't tell a recovered shot from a false positive. The rim `+2` includes a **bounce-back
   FP** (clip1 win7 2nd rim event) and its losses include **truncation artifacts** → honest delta ≈ +1-with-an-FP.
3. **Rim is a MATERIAL CONFOUND — worse than first thought.** Auto-rim vs cached differ 110/225/59px.
   ⚠️ **CORRECTED 2026-07-22 (owner manual rim + rendered frames):** the real rim is the upper-LEFT
   hoop at **(616,232)**; the auto-rim (1134,470) was locked on the shooter's **release zone / yellow
   shirt** (ball held ~x1170 pre-shot). So the earlier oracle experiments were anchored to a bogus rim
   near where the ball is HELD, not the hoop → `baseline=2` etc. were garbage, not just calibration-
   nudged. My "110px within-clip spread → tripod moved" was FALSE (release-position variation near the
   bogus rim; owner confirms the camera is FIXED within a clip, moves only BETWEEN clips). A single
   manual `Calibration` per camera position is correct; the fresh-detection eval sidesteps it entirely.
4. **Biggest EVIDENCED lever = verified-rim + segmenter repair, NOT "film closer."** With a *perfect*
   track the production `detect_shots_to_rim` still drops ~half the rim-reaching shots via 5 cheap-fix
   defects: walk-back crossing detection gaps + `seen_launch` dedup, gather-poisoned RANSAC
   (`min_inliers_frac=0.5`), hard-coded 200px `launch_drop` (not rim-scaled), the 78° hard gate on the
   data boundary, and no bounce re-approach suppression (all `court.py:225-291`). Separately the 90px
   rim gate makes airballs invisible **by design** (attempt-detection can't count them).
   → **✅ #1 FIXED (commit e213ca6): walk-back now stops at dead-ball voids (`max_launch_gap=45f`),
   a correctness bug (not tuning). The other 4 are TUNING knobs → deferred until the eval gives real
   numbers (judge against them, not before). Production shot-count effect of the walk-back fix also
   awaits the eval.**
5. **Tracker fix is CORRECT but had ZERO test coverage** → **✅ FIXED: `tests/test_segmenter.py`
   (9 checks, mutation-verified) covers assemble_track + detect_shots_to_rim; `test_eval_harness.py`
   covers the harness. 37/37 suite.**
6. **"Film closer is the biggest lever" is NOT SUPPORTED** by this evaluation (both reviewers). It helps
   tiny wholly-missed balls + pose, but perfect detection still loses half the shots inside the segmenter.

**⛔ DECISION (07-22 night, post-dual-review): the Step-1 oracle experiment family is RETIRED as a
sizing tool — it measures a selection-biased, unmatched mixture.** The single decisive next experiment
(both reviewers, independently): a **full-clip, hand-counted attempt evaluation** — owner watches the
3 clips fresh (NOT seeded from detections), logs every attempt (frame, make/miss, rim-reached vs airball)
+ ONE manually-verified rim; then run 5 staged ablations (baseline→tracker→segmenter, GT-oracle-TRACK,
GT-windows→arc-fit) and report **matched recall + precision** per stage. That isolates detection vs
tracking vs segmentation with real denominators (~1hr owner time). The cheap segmenter bug-fixes (item 4)
and cloud@0.01 are judged AGAINST that, not against single-digit counts. Fixed the reviewer-flagged
`assemble_track` velocity bug (per-frame velocity now divided by the gap it spanned)
+ a dead-code reset bug (resets were bridging velocity across shots); 35/35 test files
green — correct regardless of the plan pivot. **⚠️ RUNTIME: ONNX-DirectML inference
runs under the SYSTEM python (`AppData\Local\Microsoft\WindowsApps\python.exe`,
onnxruntime 1.24.4 + DmlExecutionProvider), NOT the `.venv_*` envs (those are
torch/ROCm training envs, no onnxruntime).** Full detail in
`process/TRACKNET_FUSION_PLAN_2026_07_22.md`.

---

## Session log 2026-07-22 (evening) — GPU inference regression FIXED + torch-directml training ruled out
Picked up the "use the GPU for ShotLab" thread. Two concrete outcomes:

**⭐ Fixed a silent GPU-inference regression (the real win).** The main env had
**both** `onnxruntime` (CPU) and `onnxruntime-directml` installed — a later
`torch` pulled in the plain CPU package, which shadows the DirectML build and
drops `DmlExecutionProvider`. Detection had **silently fallen back to CPU (~20×
slower) with no error.** Uninstalled both, reinstalled only `onnxruntime-directml`
→ **re-verified 5.1 ms/inference at imgsz 1280 on the RX 9070 XT.** Guard rule +
verify snippet added to `process/GPU_SETUP.md` §1. (Rule: never `pip install
onnxruntime` plain into the detection env.)

**⛔ torch-directml training — tested end-to-end, RULED OUT (silent wrongness).**
Built isolated `.venv_directml` (py3.12, torch 2.4.1 + torch-directml 0.2.5),
monkeypatched ultralytics `select_device` to accept the `privateuseone` device,
added CPU-fallback shims for unimplemented ops (`unique(return_counts)`,
`bincount`, `scatter_add_`). A full epoch DOES run on the GPU (~4.2 it/s) — **but
the gradients are corrupt.** Same batch, CPU vs DirectML: CPU `box=3.26 cls=5.15
dfl=3.01` vs DirectML `box=0.0 cls=0.20 dfl=0.0`. The TaskAlignedAssigner produces
**zero positive matches** on DirectML → the model would "train" and learn nothing
about localization. Not a crash — silent numerical corruption, which is worse.
Ruled out; details + reproducers (`scratchpad/dml_*.py`) in `GPU_SETUP.md` §2.

**Standing conclusion for GPU training (UPDATED — see evening-2 log below):** after
a Codex+Fable consult, the chosen path is **free cloud CUDA (Kaggle)**, not local
ROCm. CPU is the offline fallback; WSL2+ROCm is a parked optional side-quest;
native-Windows ROCm is REJECTED (AMD: "No ML training support" on Windows). GPU
*detection* via `.onnx`/DirectML remains the proven, safe local win.

---

## Session log 2026-07-22 (evening-2) — GPU-training consult → CLOUD (Kaggle) chosen
Owner pushed back on WSL2+ROCm ("is there no better option?") and asked for a
Codex+Fable dual review. **Both converged: for OCCASIONAL nano-model fine-tuning,
free cloud CUDA beats WSL2+ROCm** on reliability-to-effort (most-tested path, ~30-60
min, zero freeze risk, no monkeypatches). Native-Windows ROCm REJECTED (AMD's own
7.2 docs: "No ML training support"; and our notes already show the MIOpen bug hits
7.2.1 too). WSL2 downgraded to optional side-quest. Consult record + adjudication:
`process/GPU_TRAINING_CONSULT_2026_07_22.md`.

**Corrections the review forced:** (1) the freeze post-mortem's "NOT hardware" was
over-confident — a PSU power-transient under full CPU+GPU load was never ruled out
(if so, WSL wouldn't help; cloud sidesteps it). (2) WSL runbook version fix: ROCDXG
is ROCm **7.2.1 + Adrenalin 26.2.2**, not base 7.2.

**Built the Kaggle path:** `tools/pack_kaggle_dataset.py` (→ 884 MB zip: both real
dirs + base weights, verified), `kaggle/shotlab_train.ipynb` (yolo11n, imgsz 1280,
freeze-10, 40 ep, mAP50>0.1 sanity gate, ONNX export), `process/KAGGLE_TRAINING.md`
(the per-session loop: label → pack → upload → Run All → download → verify vs the
CPU-trained `ball_human` golden). Inference stays local on the AMD GPU.

**First real run (2026-07-22, verified end-to-end).** Kaggle T4, 40 epochs,
`ball_gpu_kaggle`. Loop proven: trained in cloud → downloaded → runs locally on the
RX 9070 XT via DirectML at **4.6 ms/frame**. Notebook needed two live fixes now
committed: recursive dataset glob (Kaggle nests at `/kaggle/input/datasets/<user>/
<slug>/`) + the phone-verify / internet-toggle gates are Kaggle-account steps.
⚠️ **Honest head-to-head vs `ball_human` (12-ep interrupted CPU) on the 227-img
ground-truth val set: basically a WASH.** Kaggle: mAP50 0.936 / mAP50-95 0.742 /
P 0.904 / R 0.861. ball_human: mAP50 0.940 / mAP50-95 0.720 / P 0.827 / R 0.875.
Kaggle wins precision + box localization; ball_human a hair more recall; mAP50 tied
within noise (120 instances). **Takeaway: more epochs did NOT lift recall → far-ball
recall is a DATA (more labels) problem, not a compute problem — consistent with the
prior finding. The cloud pipeline's value is a safe/repeatable retrain, not a recall
jump.** `ball_gpu_kaggle` is a reasonable new canonical (better precision, fully
trained, no freeze baggage).

**Far-ball test done (raw 07-20 wide clip, 1920x1080, 200 frames paired,
`scratchpad/farball_compare.py`).** ⚠️ First finding: the 146 human-labeled val
tiles are CROPS where the ball is LARGE (median 131px, all >50px) — they do NOT
represent the small-far-ball challenge, so ~0.94 mAP on them is misleading. On the
RAW wide frames the two models are **essentially identical**: both 66% frames-with-
ball, mean conf 0.58; paired both=126 / kaggle-only=6 / human-only=7 / neither=61.
Kaggle caught a marginally smaller ball (32px vs 48px min). **Confirms: more epochs
≠ better far-ball recall.** The binding constraint is PIXELS ON TARGET — a hard
detection floor of ~28-32px; balls smaller vanish.

⚠️ **Two over-claims CORRECTED empirically (2026-07-22):**
- **"4K = biggest free win" was WRONG on its own.** Inference resizes every frame to
  imgsz 1280, so a far ball is the same FRACTION of frame either way: 1080p 20px and
  4K 40px BOTH shrink to ~13px at 1280. 4K only helps IF paired with tiling (or a
  higher imgsz) so the native pixels survive.
- **Tiling is NOT a guaranteed win.** Ran plain vs `tiles="auto"` on the raw 07-20
  clip via the real `YoloBallDetector`+Kaggle ONNX (`scratchpad/tile_compare.py`,
  DirectML): **net +0** (both 62% detect, tiled-only=10 / plain-only=10, floor
  31px→28px only). Reason: that clip's ball is MEDIUM (median ~80px), no tiny-ball
  problem to solve; tiling costs ~2x compute for nothing when the ball isn't tiny.

**Honest levers (ranked): (1) label the frames the model MISSES during flight (the
sub-30px balls) — recall is data-bound; (2) more pixels on the ball at CAPTURE — get
the camera as close as the arc allows; 4K ONLY WITH tiling; (3) tiling situationally,
per-session, only for genuinely-far/tiny-ball footage; (4) motion-based tracking
(TrackNet) as the real architectural step to break the ~28px single-frame floor.**
Nothing at inference (tiling / more epochs / bigger GPU) breaks that floor.
Tiling/resolution/framing are ALL local — none touch the Kaggle training step.

**✅ `ball_gpu_kaggle` PROMOTED to canonical detector (2026-07-22).** Weights
`runs/detect/ball_gpu_kaggle/weights/best.{pt,onnx}` (onnx = the DirectML GPU path,
4.6 ms/frame). Updated the weights references in README examples + `dashboard/app.py`
(analyze3d help) from the old `ball_orange`/`ball_finetune` to
`ball_gpu_kaggle/weights/best.onnx --imgsz 1280`. (runs/ is gitignored, so the weight
files live locally + in the Kaggle output, not the repo — same as all prior models.)

---

## Session log 2026-07-22 (later) — ⛔ GPU-training FREEZE post-mortem
The ROCm GPU-training run **hard-locked the entire machine.** Diagnosed the next
session from the Windows event log + run artifacts; hardware is fine.

**What happened (timeline):**
- 07:53 — `ball_human` CPU retrain started (30 epochs, batch 8) — the valuable work.
- 09:59 — `gpu_yolo_smoke` GPU test launched (batch 16, imgsz 1280, device 0). No
  `results.csv`, empty `weights/` → never finished an epoch.
- 10:06 — CPU retrain reached epoch 12, **still running**.
- 10:08 — full `ball_human_gpu` ROCm run launched (batch **48**, imgsz 1280, 40 ep).
- **10:10:08 — system hard-locked** (last training write was `labels.jpg` at 10:09).
- 10:36:50 — manual power cycle (frozen ~27 min).

**Root cause = ROCm-Windows GPU compute deadlock, not hardware.** Event log:
- **Kernel-Power 41** + Event 6008 "previous shutdown at 10:10:08 was unexpected"
  → hard lock exactly as GPU epoch 1 started.
- **No WHEA-Logger events** → no *logged* machine-check (argues against thermal/
  PCIe). GPU reports Status OK after reboot; no damage.
  ⚠️ **CORRECTION (Fable consult 2026-07-22):** "NOT hardware" was over-confident.
  A **PSU power-transient** under simultaneous full CPU+GPU load (RDNA4 spikes +
  7900X full-load) would ALSO produce a WHEA-less hard lock and was never isolated.
  If the cause is power, WSL2 wouldn't protect the box (same silicon/watts) — which
  is a further reason the training path went to **cloud**, not local GPU.
- **No TDR (Event 4101 "display driver stopped responding") and no BSOD/minidump**
  → the GPU wedged so completely Windows couldn't reset the driver or write a dump.
  Classic full ROCm-Windows compute hang (RX 9070 XT / gfx1201, 7.13 *preview*
  wheels — the immature path GPU_SETUP.md already warned about).
- **Aggravating factor:** the batch-48 GPU job was launched *while the CPU retrain
  was still running* → heavy CPU+RAM pressure stacked under a fresh ROCm job.

**Honesty correction:** PROJECT_NOTES + GPU_SETUP claimed GPU training was verified
at "~3 min/epoch." **No GPU run ever wrote a `results.csv`** — both attempts
produced nothing and the second froze the box. That claim was an extrapolation
from the first-epoch iteration rate; corrected in both docs.

**What survived:** `ball_human` (CPU) has `best.pt` + `last.pt` on disk (reached
epoch 12 of 30 before the freeze — interrupted but usable). The 1340 human labels
+ the labeling pipeline are committed and safe. Nothing lost, no hardware damage.

**Standing rule going forward:** GPU *inference* (DirectML/ONNX) is safe and stays
default. GPU *training* is ON HOLD — do not re-run the ROCm batch-48 config, and
never concurrent with a CPU job. CPU training is the safe default. Mitigations
(TDR watchdog / batch / isolation / WSL2-ROCm) under external consult →
`process/GPU_FREEZE_CONSULT_2026_07_22.md`.

---

## Session log 2026-07-22 — native-scale retrain, human-in-the-loop labeling, GPU training
Continuation of the far-ball recall work. The chain: cheap knobs are exhausted
(1280 is the detection sweet spot) → the real fix is teaching the detector the
small far-ball SCALE it never trained on (existing balls median 46px from close
footage; the 0720 far ball is ~20px).

**Retrain recipe — one failure, then the fix.**
- ⛔ First attempt (`ball_native`): full fine-tune + 800 copy-paste-aug images
  (43% of data, paste artifacts) + 300 negatives → washed out the orange model's
  appearance → detection got WORSE at every threshold. High val mAP was
  misleading (val was easy aug/close images, not real far balls).
- ✅ Fix = **freeze the backbone (`--freeze 10`, preserve appearance features,
  adapt only the head) + REAL labels + NO synthetic aug.** `ball_native2` (frozen,
  real auto-track crops): **+49% ball-frames plain, +74% tiled, higher conf, and
  `--tile` finally HELPS** (native-scale training unlocked corridor tiling).
- Added `--freeze`/`--mosaic` to `tools/train_ball.py`; `tools/make_dataset_native.py`
  builds native 1280×1080 crops (ball keeps ~20px).

**⭐ Human-in-the-loop labeling — the breakthrough (owner's idea).** Owner's
insight: the far ball MUST be small (you can't film-closer out of it — the whole
flight + rim have to stay in frame), so the detector can't auto-label the frames
it misses. Built `tools/make_label_task.py` (**Mode 1** — physics-assisted: per
detected shot, fit linear-x/quadratic-y, PREDICT the ball on every flight frame
incl. gaps + margin, emit a self-contained HTML; user confirms Enter / fixes
click / rejects N; ~80% auto). **Owner labeled all 1340 frames → 921 ball + 419
no-ball.** `tools/ingest_labels.py` → native tiles + YOLO boxes (present) +
empty-label hard negatives (absent). Labels saved to `data/labels/ball_labels_0720.json`
(force-added — irreplaceable). `ball_human` = frozen retrain on 575 close + 1194
human (incl. 312 real negatives). This is the repeatable loop: ~15-20 min of
labeling per session → a measurably better model on YOUR court + ball.

**⭐ GPU training UNLOCKED (see `process/GPU_SETUP.md`).** The box has an RX 9070
XT but PyTorch training on AMD/Windows was blocked. Two Codex+Fable consults
(`process/BALLTRACK_CONSULT_2026_07_21.md` is the earlier one) traced it to a
KNOWN bug: MIOpen can't JIT-compile BatchNorm for gfx1201 on ROCm-Windows (HIPRTC
`'type_traits' file not found`, ROCm/ROCm #6150) — present in both 7.2.1 AND the
7.13.0 preview. Everything else works incl. CONVOLUTION. **Fix: route BatchNorm
through pure-torch primitives** (`tools/rocm_bn_patch.py`, auto-applied by
`train_ball.py` on a ROCm GPU device) — conv stays GPU-accelerated, MIOpen BN
never called. Setup: isolated Py3.12 `.venv_rocm713` (main env is 3.13, too new
for ROCm) + AMD ROCm 7.13 wheels. GPU inference (DirectML) was already the ~20×
win. ⚠️ **CORRECTION (see the freeze post-mortem below):** the GPU-*training*
claim in this paragraph — "yolo11n @1280 trains ~3 min/epoch on the GPU" — was an
extrapolation from the first-epoch iteration rate, **NOT a completed epoch**. No
GPU training run ever finished an epoch, and the batch-48 attempt **hard-locked
the machine**. GPU training is UNVERIFIED and on hold; detection via the `.onnx`
stays the default, training falls back to CPU.

---

## Session log 2026-07-21 — first 07-20 two-cam session + GPU (DirectML) detection
First run of a NEW session's footage (filmed 2026-07-20): two cameras, Pixel wide
(`data/raw/Camera 1`, arc/makes) + S8 close (`data/raw/Camera 2`, body/flare), 5
clips each, no calibration paper (scale from the ball + 5'10" height + court dims,
as before). Excluded the first clip of each cam (a "moved a camera a smidge" setup
clip). Ingested from 4 zips + 1 loose mp4.

**Finding #1 — the wide camera was TOO FAR this session (filming rule #1).** The
default pipeline (`--imgsz 640`, the OpenVINO CPU detector) found only **8 shots
over ~18 min** with garbage arcs — apex heights of 0.5 ft and `apex_above_rim`
down to **−1.57 ft** (below the rim; physically impossible). Proof it was missed
shots, not a quiet court: a frame pulled from a 3-min "no-shot" gap shows the ball
in the air by the rim (a real shot the 640 detector dropped). Root cause: at 640
the ball (~20 px in the 1080p frame) shrinks to ~7 px after the model's downscale
— below detection. The 0/8 make% is the weak geometric detector on junk arcs; not
meaningful. **Lesson stands: get the wide cam closer/lower so the ball is bigger.**

**Finding #2 — higher-res detection recovers shots AND fixes arc quality.** A/B on
one window: 640 → 3 shots, **1280 → 8**. Full session at 1280: **8 → 11 shots**,
and — bigger deal — apex heights went from impossible (0.5 ft) to **sane (2.5–4.0
ft, matching real jumpers ~3.8 ft)**. So 1280 is a real lever, but 11 shots is
still short of what he actually took — resolution helps, doesn't SOLVE far framing.
Durable fixes: film closer, and/or a temporal tiny-ball tracker (TrackNet family).

**Finding #3 — we are NOT CPU-bound; built the GPU path.** "CPU-only" was a
software state, not the hardware: this box has a **Radeon RX 9070 XT (16 GB)** +
Ryzen 9 7900X. The catch is AMD-on-Windows in a CUDA-first ecosystem (OpenVINO's
GPU plugin is Intel-only; torch on Windows has no CUDA/ROCm). Reached the GPU via
**DirectML**: exported the orange model to ONNX (`imgsz 1280`) and added an
onnxruntime `DmlExecutionProvider` backend to `detect_yolo.py` (commit 060d9b7).
**6.2 ms/frame on the GPU vs 121 ms on CPU = 19.6×.** Validated it matches the
ultralytics `.pt` output on every clean detection; the two differ only on ONE
sub-threshold dribble false-positive (a torch-vs-ONNX numeric tie-break, verified
by eye — he's dribbling near the hoop in that frame — not a decode bug). Drops in
with zero pipeline changes: just point `--weights` at the `.onnx`. This is what
makes high-res / SAHI / TrackNet all cheap to run going forward.

**Finding #4 — external consult (Codex + Fable) + recall experiments.** Both
reviewers independently converged: the fix is a **track-before-detect
architecture flip** (keep weak candidates at low conf, search whole rim-bound
trajectories, let the `ballistic.py` verifier supply precision), not a better
per-frame detector. Full synthesis in `process/BALLTRACK_CONSULT_2026_07_21.md`.
Prototyped + MEASURED the two cheap levers on 0720:
- **Corridor tiling (`--tile`, commit 98f327f):** native-res tiles so a far ball
  keeps ~20px. REGRESSES — 11 → 2 shots. The orange model was fine-tuned on
  DOWNSCALED balls, so native-scale balls detect in fewer frames at lower conf
  (30 vs 47/120 sampled, conf 0.62 vs 0.72). Tiling is COUPLED to a native-scale
  retrain; kept as opt-in infra, do NOT use on current weights.
- **Conf floor (`--conf`):** 0.25 → 0.05 recovers +38% ball FRAMES but only
  11 → 12 SHOTS — because the pipeline keeps one candidate/frame + coasts 4, so
  extra low-conf blips just densify existing arcs (n_points 15 → 38), can't
  stitch new shots across gaps. Confirms the flip is what's needed.

**Consolidated: 640→8 (junk arcs) · 1280→11 (sane arcs, the sweet spot) ·
tiled→2 · conf05→12+denser. Cheap knobs exhausted.** Real recall gains need
EITHER (a) retrain the detector on native-scale court crops (~days, unlocks
tiling + small-ball recall), or (b) the track-before-detect flip (~1-2 wk,
scale-independent, reviewers' #1), or (c) film the wide cam CLOSER (free, biggest
lever). AMD-on-Windows update: ROCm PyTorch preview now supports the RX 9070 XT
for offline training/labeling jobs (keep DirectML for production).

TODO next session: reshoot with the wide cam closer; run the S8 close-cam flare
pass (`flare_report.py`/`analyze3d.py` still hardcoded to 0710 — parametrize for
0720); pick the retrain-vs-track-before-detect direction. Full suite 35/35.

---

## Session log 2026-07-15 — external-audit fixes (4 commits, all verified)
An independent read-only audit flagged 3 correctness risks + hygiene; all
claims verified against the code before fixing (severity re-ranked: VFR was
the real data-corruptor; cache/chunk were real but narrower). Fixes:
1. **VFR timing (the big one):** abs_time + the audio make/miss window now use
   real container PTS (`session._real_time`, `video_io.frame_times_cached` —
   cached one-grab()-pass per clip, keyed on video content). Frame/nominal-fps
   drifted by whole seconds on long VFR clips → the swish/clank window could
   miss the rim event entirely; audio is default-on and feeds make-drivers.
   Slow-mo scaled playback→capture time; frame/fps fallback preserved.
   ⚠️ **Every audio-fused make/miss label built before v20 is suspect on
   long clips — rebuild sessions before trusting make-driver stats.**
   (Local frame-diff metrics — tempo, follow-through — still use nominal fps;
   bounded ~10% local error, accepted.)
2. **Cache identity:** record sig now carries video content (size:mtime),
   effective calibration ('auto' or values), and weights CONTENT (a retrain
   re-exported to the same best_openvino_model path was invisible). Track
   cache params carry video id too. 3rd instance of this footgun class —
   content identity, not naming. All pre-v21 caches invalidate (intended).
3. **Chunk seams:** detection windows now read `_CHUNK_OVERLAP`=600 frames
   past their end + frame-range seam dedup (fuller arc wins). Disjoint windows
   silently lost ~0.5-1% of shots (launch in one window, rim event in the
   next) and could emit truncated straddlers with wrong release metrics.
4. **Hygiene:** scikit-learn/joblib into requirements (+ pyyaml into lock);
   `run_tests.py` now runs the 6 node suites too (33 files total);
   `.github/workflows/tests.yml` runs the suite on every push (Pages was
   deploying review-free); README test docs de-staled; explicit note that
   publishing app/profile.json to Pages is intentional.
_CACHE_VERSION 19→21. Suite 33/33 (27 py + 6 mjs). CI green on first run
(only annotation: bump checkout/setup-* action versions sometime).

**v21 REBUILD EXECUTED same day (both sessions, ~37 min total, log
`data/out/rebuild_v21.log`):** 0703 114→120 raw shots, 0710 91→92. New
detections are borderline candidates flipping in near the conf threshold
(env drift vs July inference, NOT the seam fix — none sat at frame 7000) —
mix of junk (auto-flagged) + 2 plausible real jumpers. **Labels carried over
via NEW `tools/remap_shot_keys.py`** (frame-range overlap matching; snapshot
first: `shot_map_pre_v21.json` + `session_shots_pre_v21.csv` per session):
0710 make_truth 91/91 remapped (5 renumbered in warmup clip); 0703 exclude
remapped (28 renumbered, 1 old borderline shot vanished). Downstream regen:
make_pred (92 preds, 100% in-sample agreement w/ truth), flare_by_shot (80
shots, same sync offsets), 0710 recap PDF (truth), 0703 report.html.
**Corrected 0703 numbers (VFR-fixed audio + right excludes): make% 36→40.2%
(37/92), fatigue trend now 32%→49% second half — the old fade story
REVERSED; treat with the usual heuristic-label skepticism.** TO LABEL in the
audit view: 0710 warmup-clip shot 20 + 0703's 7 new (mostly junk-flagged).

---

## Session log 2026-07-11 — the make-label reckoning + visual make/miss + form-labeling views
The day the make-driver story got falsified and rebuilt honestly.
- **User audited all 91 shots of 0710 → the make labels were garbage.** Ground
  truth: **17 non-shots (19% tracker false positives), 74 real shots, real make%
  49%** (tracker had said 24%), and the old geometric make/miss classifier was
  **49% accurate = a coin flip.** The 07-10 recap's "higher arc + deeper legs =
  makes" story was a **pure artifact**: the tracker labeled cleanly-tracked high
  arcs as makes (easy to follow through the rim), manufacturing a fake arc→make
  correlation. On verified labels every arc/knee driver **collapses or flips**
  (release +0.64→−0.19, entry +0.57→−0.24, apex +0.54→−0.35, knee −0.53→+0.19);
  only **BALANCE survives** (steadier base = makes, d=−0.51 p=0.10) and nothing
  is significant. `session_recap_pdf --truth` drops non-shots + uses your labels
  ("VERIFIED BY YOU"); dashboard recap auto-uses `make_truth.json` ("✓verified").
  ⚠️ **Lesson burned in: never report a make-driver off tracker labels — verify
  the outcomes first.**
- **Visual make/miss detector — 87%, up from 49%** (`shotlab/make_visual.py`,
  `models/make_visual.joblib`). User's insight ("a human reads it instantly, the
  signal IS there") was right; a Fable pass found the tracked ball is the wrong
  thing to watch — the human cues are the **net whipping** and the ball dropping
  **through/below** the rim vs caroming to the **side**. HSV + frame-differencing
  in a rim-anchored ROI, CPU-only, no ball tracking at the rim → 7 features →
  gradient boosting on the audited labels. **5-fold AUC 0.967 / acc 90.7%;
  leave-one-clip-out AUC 0.942 / acc 86.5%** (vs 0.49 geometric, 0.61 audio).
  Wired into the audit view (auto-label + uncertain-first ordering).
- **Shot Explorer + viewing tools** (dashboard): filter every shot by any
  measurable with date-labeled dropdowns; **elbow flare joined onto wide shots**
  (per-shot flare in the explorer); **form-vs-ideal skeleton overlay** in the
  film room (your rep vs your own ideal); **shot scatter** (plot any measurable
  vs another, click a dot to watch); shot chart, makes-vs-misses, closest-to-ideal.
- **Flare review view + auto-reject bent-arm frames.** Root cause of bogus high
  flare: `refine_release_frame` now snaps each wrist-apex candidate to the nearby
  most-extended-elbow frame and **drops candidates whose elbow never reaches ~145°**
  (gathers/pumps read artificially flared — 0710 shot 18's −22° was a bent
  gather). Each reading carries `elbow_deg`. Plus a **Flare review** audit view
  (flag bad readings, recompute) and a **Form-notes view** (watch the full-clip
  overlay, rate each shot good/ok/bad + note → `form_notes.json`, eye-rated form
  as future labels).

---

## Session log 2026-07-10 — ⭐ the Galaxy S8 arrives; monocular 3D + 2-camera flare land
**The second camera showed up and the whole 3D stack came online.** An adversarial
Fable review overturned the earlier "wide camera is too far, must re-shoot"
conclusion — **3 of 4 blockers were misdiagnosed** and fixable on existing footage.
- **3D foundation** (`arc3d.py`, `ballistic.py`, `charuco.py`): VFR time-base fix
  (Pixel clips are variable frame rate — nominal 27fps was really 30→24 mid-clip;
  assuming constant 30 inflated reconstructed gravity 1.56× and auto-rejected good
  arcs). Single-camera **metric arc from the ball's known diameter** (X,Y in feet,
  focal-free, built-in gravity self-check) + a **gravity-constrained ballistic
  fit** (fits P0,V0 of a projectile with accel pinned to g through sparse/gappy
  pixel points; reproj + radius-consistency = independent honesty gates). Coarse
  **ChArUco** board the far camera can actually resolve (fine checkerboard got 0
  detections).
- **W7 monocular 3D elbow flare:** exposed MediaPipe's metric `world` landmarks
  (33×3, meters, hip-origin) — a 3D estimate we were already discarding. On the
  close S8 clip: **flare median −9.5° (sd 3.5, n=21)**, agreeing with the 2D
  image-plane estimate (−10.4°). Session-relative + model-biased → LOW-MED conf.
- **W4 camera-tilt self-cal:** recover the wide camera's pitch/roll from the
  physics of ≥2 arcs alone (no rim/board) — synthetic true 18/4° → 17.1/4.3°,
  rmse 0.69px. Unlocks absolute depth + true release angle.
- **3D analysis pipeline + dashboard view** (`analysis3d.py`, `tools/analyze3d.py`
  → `analysis3d.json`): the focal-free per-radius gravity check was too noisy at
  the far ball's size (0/10 passed), so the "clean arc" gate switched to the
  **ballistic reprojection** (<5px), stable at small ball size. Then a dense
  orange detector made dribble+shot+retrieve one continuous run → gap-splitting
  merged shots; fixed by reusing the standard **rim-anchored** shots from the
  detection cache. Result on 0710's 4 clips: **91 shots → 27 clean arcs**, apex
  above release median **3.8 ft** (2.9–4.8), reproj 1.3–4.3px. Apex = the
  unambiguous trustworthy metric; horizontal channel still mixes L/R with
  toward-rim depth until W4 tilt is applied.
- **2-camera flare-vs-make** (`tools/flare_report.py`): audio-sync each (wide,
  close) pair, measure flare at each release + render an annotated still, time-map
  to the wide clip's outcome. 159 releases, 94 matched. **FINDING: flare does NOT
  track make/miss** — makes −10.9 vs misses −9.8 (d −0.28, perm p 0.31). Flare is
  a **consistent habit** (~−10° either way); the **arc is the stronger driver.**
- **Rich session recap PDF** (`tools/session_recap_pdf.py`): 5-page recap (stats,
  make/miss drivers with Cohen's d + permutation p, cross-metric relationships,
  flare + takeaways). ⚠️ Its 0710 findings were computed on **tracker labels** and
  were **overturned the next day** (see 07-11) — kept here only as the cautionary
  example.
- **Make/miss audit view** (dashboard): verify the tracker's calls against ground
  truth, "NOT a shot" option (detector fires on dribbles/retrieves), auto-flag
  suspected non-shots, real-time per-shot clips extended past the rim with the rim
  drawn. This is the tool that produced the 07-11 reckoning.

---

## Session log 2026-07-07 — test-suite hardening + scaled-print calibration
- **Hollow-guard hardening:** made the test guards actually **mutation-die** —
  several were passing against broken code (asserting nothing that would fail if
  the logic regressed). The suite now bites.
- **`calibrate_rig --square-in`:** support a scaled checkerboard print (measure the
  real printed square, pass it in) so the mono/stereo focal solve stays honest
  when the board didn't print at exactly 6.000in.

---

## Session log 2026-07-05/06 — two audit passes + the v17→v19 rebuild ladder
A concentrated correctness campaign (independent audit → fixes → rebuild →
redeploy, repeated). Batches:
- **Release detection & honest labeling** (07-05): fixed release-frame detection,
  plugged curation leaks, made low-confidence readings label themselves.
- **Second audit pass** (07-06): gate **false-confident releases** + shared-artifact
  gating (stop trusting a cached artifact that a later stage invalidated).
- **Data-correctness batch:** fixed **RANSAC-inlier corruption** + gated physically
  **impossible values** so junk can't reach the stats.
- **Pipeline batch:** auto-flag **phantom shots**, fix zone-release, catch **config
  staleness** (stale config silently changing results).
- **Medium/low batch:** more robust make/miss + audio timing + **2-cam roll** +
  **layup reclassification** (layups were polluting the jumper pool).
- **App + coaching batches:** stop the live app **nagging on the wrong things**;
  point coaching at his **real make-drivers**.
- **Final sweep:** caught **4 self-inflicted regressions** + real misses before
  redeploy.
- **Rebuild ladder:** profile redeployed from the **v17** (phantom-cleaned pool),
  **v18** (make/miss + layup reclass), and **v19** (make/miss window fix) rebuilds
  in turn — each rebuild re-derives the profile from the cleaned shot pool, not by
  hand-editing the profile.

---

## Session log 2026-07-03 — release target, calibration gating, "Scan me" enrollment
- **Textbook ~52° release target** added; **arc angles gated on calibration** (don't
  report an arc angle in degrees unless the camera geometry supports it).
- **App "Scan me" enrollment:** stand whole-body-in-frame ~3s and the live app
  **locks onto YOU**, not passers-by / objects / you rebounding — the fix for the
  live app chasing the wrong person. (Now step 7 of the filming checklist.)
- **Filming checklist** written up (lessons through 07-03) — see the top of this
  doc.

---

## Session log 2026-07-02 — release-sync + jump fixes land in the pipeline; shot map; feel CSV
- **Wrist-apex release in the metric path** (`form.py find_release`; cache v3):
  pose-only wrist-apex estimate alongside ball divergence; when the ball
  "release" lags the wrist snap >0.12s (far/small-ball late detection) the apex
  wins (medium conf, noted). Clean footage keeps the sharper ball estimate.
  **Real-footage audit (71 shots): 24% of elbow readings were biased ~−30° each
  (mean −7.3°); 9 shots upgraded from low release-conf.**
- **Jump height rebuilt honest** (3 commits, cache v4→v6): ankle-based (squat no
  longer counts as jump) → + both-ankles gate + median-3 (v5; real footage showed
  the naive ankle version was WORSE than hip: one-ankle-occluded frames + 1-frame
  glitches faked flight) → + physics gate >4 ft = None (v6). Final: median
  2.18→1.56 ft, max 13.8→3.9, 11/65 honestly nulled. Lesson: a synthetic-clean
  estimator can still lose to noise on real footage — audit BEFORE/AFTER on real
  data every time.
- **Shot map**: `rim_dx_px`/`rim_dy_px` now in ShotRecord (zone_for_release
  already computed them); `viz.draw_shot_map` (dot=make, X=miss, shape carries
  identity) in dashboard Session view + report.html + PDF page 1.
- **App: feel-log CSV export** (`app/js/feelcsv.js`, node-tested 11/11): live
  feel tags now persist per-shot METRICS with the label; ⬇️ button downloads all
  stored sessions as one mergeable CSV. SW cache v3→v4.
- Session 0701 + `_wide`/`_moved` rebuilt (3× full pose passes for the cache
  bumps; detection always from `_track.json` cache); reports regenerated;
  profile re-exported (ideal elbow 118.9°, skeletons from 2 clean shots).
- Tests: 12/12 py files (15 form tests) + 4 node suites green.

---

## Vision
Upload phone videos of shooting workouts → per-shot arc metrics, pose/form
feedback, and a local dashboard (overlay video next to stats). Honest about what
a single 2D camera can and can't measure. Targets are tunable (e.g. ~45° entry).

## Status
- **Phase 1 — ball tracking + arc:** ✅ DONE & validated (release angle exact to
  0.0° on the synthetic ground-truth clip).
- **Phase 2 — pose/form:** ✅ BUILT. MediaPipe **Tasks API** PoseLandmarker
  (legacy `mp.solutions` is gone in 0.10.35; model `.task` auto-downloads to
  `models/`). One-Euro smoothing. Form math unit-tested (5/5). Real-keypoint
  accuracy pending a real clip with a person.
- **Phase 3 — spin (stretch):** ✅ BUILT. fps-gated (skips <110fps with a clear
  message). Log-polar phase-correlation rotation; validated on synthetic spin to
  **~0-2% at 240fps**, ~1-10% at 120fps (bias grows with rpm — hence shoot 240).
  Real-ball accuracy pending a slow-mo clip with a marked ball.

**Validation gap remaining:** both Phase 2 & 3 need ONE real clip each to confirm
on real footage. Everything else (logic, integration, gating) is tested.

## Environment
Python 3.13.9 · **CPU-only (no NVIDIA GPU)** · ffmpeg 8.1.1 · Windows 11.
All model choices are the CPU-friendly best-in-class for that reason.

---

## Model decisions (from a 2026 best-in-class web survey, not defaults)

### Ball detection
- **Architecture: per-frame detection → RANSAC + degree-2 polyfit. NOT a
  multi-object tracker.** For a single ball, ByteTrack/BoT-SORT *hurt* (Kalman
  constant-velocity breaks on small/fast/blurred balls). RANSAC treats blur &
  occlusion as outliers and interpolates gaps via the parabola. → `shotlab/arc.py`.
- **Backends (swappable behind `BaseDetector`):**
  - `ColorBallDetector` — HSV-orange + circularity. No ML deps, fast on CPU,
    best on clean/well-lit footage. **Current default.**
  - `YoloBallDetector` — default `yolo11n` (survey's safe fallback). Stock COCO
    "sports ball" is unreliable → for real footage pass a **fine-tuned basketball
    model** (`--weights basketball.pt --ball-class 0`).
  - License: `ultralytics` = AGPL-3.0 (fine for personal use; swap to RT-DETR /
    D-FINE / RF-DETR if ever distributed closed-source).
- **Scale:** apex height in feet uses the ball's pixel diameter as a ruler
  (no court calibration) → MEDIUM confidence. Angles need no scale → HIGH.

### Pose estimation (Phase 2)
- **#1 MediaPipe BlazePose-33** (Apache-2.0, CPU real-time, 33 keypoints + feet).
- Upgrade path: RTMPose/RTMW (133 kp incl. fingers) via `rtmlib` (GPU).
- **Smoothing is mandatory** (jitter worst on the fast release frames) → add a
  One-Euro filter; the new MediaPipe Tasks API dropped built-in smoothing.
- **Confidence rule (baked into output):**
  | Metric | Confidence | Why |
  |---|---|---|
  | Knee bend depth (side-on) | HIGH | in-plane sagittal |
  | Release vs jump apex | HIGH | vertical image tracking |
  | Elbow angle at release (side-on) | MED-HIGH | in-plane if camera square |
  | Follow-through hold | MED | timing fine; finger state noisy |
  | Balance / squareness | LOW-MED | partly out-of-plane |
  | **Elbow flare** | **LOW** | pure out-of-plane; 14–27% perspective error, model-independent |

---

## Filming guide (how to get the best results)

**General (both angles):**
- Stationary tripod, **mark the spot** so sessions are comparable.
- **Square the camera:** optical axis exactly perpendicular (side-on) or parallel
  (front-on) to the shot direction — this is the #1 driver of angle accuracy.
- Camera height ≈ **release height (~chest/shoulder, ~5–6 ft)**, not low looking
  up (low angles distort joint angles and the arc).
- **Frame the whole arc**: release point through the rim, ball never leaves frame;
  shooter fully visible head-to-toe.
- **≥120fps slow-mo + fast shutter (~1/500–1/1000s) + extra light.** This both
  tightens apex/entry estimates and is *required* for Phase 3 spin.
- Plain, contrasting background behind the ball path; avoid windows/backlight.
- Standard orange ball; wear clothing that contrasts with the ball.
- **Put the rim in frame** and we can mark `rim_x` for entry angle *at the rim*.
- **Scale calibration (big accuracy win):** include the rim (known 10 ft) or place
  a marker of known size in the shot plane → real-feet apex/release/jump height.

**Side-on session** (perpendicular to the shot line, ~10–15 ft to the side):
best for arc, **knee bend, release-vs-apex, follow-through timing, release angle**.

**Front-on session** (directly facing the shooter, aligned with the shot line):
the only way to see **elbow alignment/flare and squareness** — and even then it's
low-confidence on one camera (see two-camera idea below).

---

## Release-frame sync upgrade (2026-06-30) — backlog #4 ✅
Replaced the coarse release detector (last frame within 2.5× ball-radius of the
wrist, integer frames) with a **divergence-onset detector + sub-frame
interpolation** in `shotlab/phase2_pose/form.py`:
- `find_release()` returns a `ReleaseEstimate` (frame, sub-frame `t`, `confidence`
  high/medium/low, `diverging` flag). Release = END of the in-hand minimum-distance
  cluster (onset of divergence) — sharper and EARLIER than the old threshold, which
  only tripped after the ball had already travelled a few radii. On the synthetic
  ground-truth clip the old detector landed 2 frames late (20 vs 18); the new one
  hits the true release exactly.
- Sub-frame `t`: interpolates where ball-wrist separation crosses ½ a ball radius
  between the onset frame and the next. Matters because at 30 fps the ball moves
  ~a foot/frame, so whole-frame release quantizes timing + the release angle.
- Fed into metrics: `elbow_angle_at_release_deg` now interpolated to the sub-frame
  release (`_elbow_angle_at_t`); `release_vs_apex_s` uses sub-frame release AND a
  **sub-frame apex** (`_apex_subframe`, parabolic vertex of hip-y). Confidence
  falls back to LOW with a min-distance frame when the hand-off never cleanly
  diverges (e.g. a non-shot).
- `ShotForm` gained `release_t` + `release_conf` (in `as_row`); report adds a
  `release_conf` column. `find_release_frame()` kept as an int shim (overlay/
  compare/pipeline still use the integer frame). Tests: test_form 7/7 (added
  sub-frame + no-divergence cases), test_arc 6/6.

## Make-correlation engine (2026-06-30) — backlog #3 ✅ (the "holy grail" framework)
`shotlab/correlate.py`: correlates YOUR form/arc metrics against YOUR make/miss
to surface which mechanics track with the ball going in (vs comparing to a
textbook ideal). **Honesty-gated** because make/miss is a LOW-confidence heuristic
(`make.classify_make`) on small samples:
- Per metric: mean(made) vs mean(miss), `diff`, **Cohen's d** (effect size),
  point-biserial r, and a **permutation p-value** (numpy, assumption-free, robust
  at small n, seeded/deterministic). Gates on min_n=8 of BOTH made and miss;
  confidence **capped at "medium"** on purpose (the label itself is low-conf);
  most real findings land "low". Depth-dependent metrics (elbow flare) carry an
  extra caveat. `summarize_make_drivers()` = plain-English review.
- Bug caught + fixed mid-build: NaN metric values (missing pose/spin) leaked through
  `float()` and produced a spurious tiny p (nan comparisons) — now dropped via
  `np.isfinite`; regression test added.
- Records now carry the extra form metrics for correlation: `ShotRecord` gained
  `elbow_angle_at_release_deg`, `follow_through_hold_s`, `balance_drift_px_per_ht`,
  `release_conf` (populated in `session._records_from_shots`).
- Wired into `build_session.py` (prints + `make_drivers.csv`), the dashboard
  Session view (🎯 panel + expandable table), and `report.html`.
- **Real Hoops session (27 made/58 miss) says:** strongest lean = KNEE BEND
  (makes ~14° more bend, d=-0.47, p=0.15), then lower release angle, then later
  release vs apex — all "low" conf, none significant. Sensible + honest. Will
  firm up with calibration footage + volume. Tests: test_correlate 5/5.

## Cross-session consistency tracking (2026-06-30) — backlog #6 ✅
Within-session consistency (`consistency_stats`, within-zone std) + fatigue
trends already existed; the gap was tracking consistency ACROSS sessions.
- `aggregate_sessions` now emits `std_<metric>` columns (within-zone std per
  session) alongside the existing `avg_<metric>` — so progress tracks BOTH level
  and repeatability.
- New `consistency_progress(agg)`: per metric, first vs latest std, delta, slope
  per session, and an `improving` flag (negative slope = tighter = better). Needs
  ≥2 built sessions.
- Dashboard Progress view: "📈 Consistency over time" table (✅ tighter / ⚠️ wider),
  std_ metrics flagged "lower is better", and a ⚠️ caveat that cross-session
  comparison only holds when the CAMERA SETUP is consistent (foreshortening
  changes the absolute spread) — relevant because the two real sessions on disk
  used different cameras. Tests: test_consistency 3/3.

## Auto shot-type tagging (2026-06-30) — backlog #11 ✅
`shotlab/shottype.py` tags each shot on two axes a box score would carry:
- **form**: jumper | layup | floater. Mid/far range ⇒ confident jumper (MEDIUM);
  near-the-rim is ambiguous, so layup (flat + low arc) / floater (lobbed) / close
  jumper stay LOW until calibration. `classify_form(depth, apex_ft, release_deg)`.
- **setup**: catch_and_shoot | on_the_move | off_dribble. Reuses `movement_dir`
  (set/left/right) and adds `detect_dribble()` — a prominent bounce (image-y local
  bottom with real prominence) in the ~1.5 s of ball track before release ⇒ ball
  was put on the floor first; overrides 'set'. Sparse pre-shot track ⇒ unknown.
- `ShotRecord` gained `shot_form` + `shot_setup`, populated in
  `session._records_from_shots` (release frame from the pose sync). Dashboard
  Session view: "By shot type" panel (form/setup × shots × make%). Report: cols
  added to the all-shots table.
- Real Hoops rows: 88 confident jumpers + 10 close jumpers + 9 floaters (sensible
  for a driveway). Setup all 'unknown' there only because that session predates the
  movement_dir widening — populates on fresh builds. Tests: test_shottype 8/8.

**Session total (2026-06-30): 4 backlog features (#4, #3, #6, #11). 29 tests pass
(arc 6, form 7, correlate 5, consistency 3, shottype 8). Dashboard AppTest clean.
No new footage needed; all validated on existing data + synthetic ground truth.**

## Phone app MVP — PWA (2026-06-30) — user greenlit "build the app now"
Architecture (settled w/ user): **desktop builds the rich profile → ships small
`profile.json` to the phone → phone app is lightweight (reads profile, on-device
pose, overlay + feedback).** The profile (data) is trivially portable; only the
heavy ML *processing* had to be rebuilt for mobile. Chose **PWA** (installable web
app) over native Kotlin: fastest to on-phone + testable + $0 + cross-platform,
wrap native later. Target = Android/Pixel.
- `app/` — `index.html`, `styles.css`, `js/{pose,analyze,overlay,main}.js`,
  `profile.json`, `manifest.json`, `sw.js`, `icon.svg`, `README.md`.
- **On-device pose** via MediaPipe Tasks-Vision (CDN ESM, WASM/WebGL, GPU→CPU
  fallback) — same BlazePose-33 as desktop. Pick a clip → per-frame pose → **live
  green skeleton overlay** → detect load/release/follow → elbow-at-release +
  knee-bend → compare to `profile.json` ideal → deltas + plain feedback. "Jump to
  release" freezes + overlays the ideal skeleton (gold) ON yours when the profile
  has one. Angles computed in PIXEL space (normalized coords are aspect-distorted).
- **v0 scope:** POSE ONLY (ball/arc = heavy on-device model, deferred to v2;
  elbow flare needs 2-cam 3D). Ideal targets are PLACEHOLDER until
  `tools/export_profile.py` generates a real profile (ideal metrics + ideal
  skeletons) from the user's feel-good shots — NEXT STEP, lights up the overlay.
- **User's feature vision:** ideal per-phase poses in the profile + app overlays
  actual-vs-ideal skeleton at load/rise/release/follow. Also: "feels good/off"
  self-labeling = the personalization signal (beats weak make-detection).
- **Test:** `python -m http.server 8080 --directory app`; open
  `http://192.168.4.52:8080` on the Pixel (same Wi-Fi). File-upload works over
  HTTP; live camera + full install need HTTPS (host on GitHub Pages later). JS
  syntax-checked via node; can't browser-test here → user eyeballs on phone.
- **Sellable-as-app note (user floated):** single-camera baseline IS viable for
  the core (in-plane metrics + consistency + deviation + feel-labeling); 2-cam 3D
  = premium "pro mode". Hard 80% = detector generalization across courts/phones +
  mobile polish + competition (HomeCourt/DribbleUp exist). Server cost: ~$0
  self-host+tunnel to ~$5-12/mo tiny VPS.

## Backlog BUILT (2026-07-01) — 9 features, 58 tests
User: "let's start working on all of that stuff." Built the whole improvement
backlog, each tested (run_tests.py; JS via node). All green.
1. **Rim-based real feet** (`scale.py`): px/ft from the 18in rim → apex-above-rim,
   release-height, jump-height. `apex_above_rim_ft` is the trustworthy one (ball ~
   at rim depth); release/jump are LOW-conf (shooter off the rim plane).
2. **Shot tempo** (`tempo_dip_to_release_s`): deepest load → release (quickness),
   tracked in consistency + fatigue.
3. **Fatigue breakdown + drift alerts** (`fatigue_breakdown`, `mean_drift`): which
   part of the shot fades most as you tire (SD-normalized); cross-session level creep.
4. **Auto-handedness** (`detect_handedness`, `--handedness auto`): shooting wrist
   rises highest through the shot.
5. **Feel-correlation** (`correlate_feel`): generalized correlate.py to any binary
   label → correlate on your "felt good/off" tags; `felt_good` on ShotRecord.
6. **Drill-effectiveness** (`prescribe_target`, `drill_effectiveness`): the one
   least-repeatable metric to work on + did it improve next session.
7. **Audio make/miss** (`audio.py`, `--audio`): rim/backboard loudness → make/miss
   hint fused with the visual call (loud clang=miss, soft swish=make).
8. **Live camera + auto-shot-detection** (`app/js/live.js`): getUserMedia + rolling
   buffer + release-motion trigger (`releaseIndex`/`ShotDetector`) → instant per-shot
   feedback card. "🔴 Live" button in the PWA. Unit-tested in node (test_live.mjs).
   NEEDS HTTPS to use the camera on the phone (GitHub Pages, or localhost).
9. **Hygiene:** `run_tests.py` (one-command runner), `test_regression.py` (locks the
   analytics layer on a fixed fixture), `requirements-lock.txt` (pinned versions).
**Surfacing (2026-07-01, DONE for dashboard):** dashboard now shows real-feet +
tempo KPIs, a **shot chart** (half-court 9-zone make% map), a **"what fades as you
tire"** panel, **feel-tagging** (edit good/off → writes felt_good → live
feel-drivers), and in Progress: **level-drift** + **"did your homework pay off?"**.
New metrics also added to the metric-over-time picker. AppTest clean.
More dashboard UI (2026-07-01): **metric relationship explorer** (scatter any two
metrics, color by make/feel/zone), **shot inspector** (click a row → full metrics +
its rendered clip via st.dataframe on_select), **Compare sessions** view (A/B means
+ consistency deltas), and **report.html parity** (new real-feet/tempo cols + KPIs).
AppTest clean across Session/Compare-sessions/Progress.
Even more dashboard UI (2026-07-01): **PDF session report** (`tools/export_pdf.py`,
3 pages via matplotlib PdfPages — no new deps; shared `shotlab/viz.draw_court`;
"⬇️ PDF report" download + rebuild-HTML buttons), **data-health panel** (pose%,
make-classifiable%), **personal-bests board** (Progress), **goal lines** (target +
band from targets.yaml on the metric chart). AppTest clean; PDF builds during test.
**Volleyball note (user asked):** the concept exists commercially — VolleyVision,
TechniqueView (pose + skeleton + per-skill scores) + academic 3D-spike/IMU work;
same recipe maps over if we ever point ShotLab at volleyball.
**STILL TODO:** build_session printout for new metrics (minor). Live app needs
HTTPS hosting to test camera on phone. Ideas left: goal-progress tracking, richer
shot-map (per-shot scatter needs rim_dx_px in records), report emailing.

## Scope + roadmap (2026-07-01)
**SCOPE DECISION (user, 2026-07-01): PERSONAL USE ONLY — not a public product.**
→ Drop detector generalization / cross-court robustness / competition worries.
We can **hard-tune everything to the user's court, phone, and (orange) ball** — a
big simplification (the "hard 80%" of productizing is off the table).

**Live-camera app vision (user, 2026-07-01):** the sideloaded app uses the phone
camera **in-app (live)** and gives **instant feedback after each shot** — no file
picking. What it takes:
- **Live camera:** `getUserMedia` needs **HTTPS**. Options: host the PWA on
  **GitHub Pages (free HTTPS, still 100% on-device)**, or wrap it as a native /
  **TWA sideloaded APK** (cleanest for "sideloaded + camera", no HTTPS hoop).
- **Auto shot-detection in the live stream (NEW core piece):** run pose
  continuously + keep a **rolling frame buffer**; detect the **release motion**
  (shooting wrist rises above head + arm extends) as a "shot event" → analyze the
  buffered shot → show feedback → reset for the next. Continuous pose is fine
  on-device; continuous ball detection is heavier (v2).

**Improvement backlog (curated 2026-07-01) — effort/value:**
- **Quick wins:** rim-based REAL FEET (release/jump/apex height — we already detect
  the 10ft rim); shot **tempo/rhythm** + its consistency; **which form breaks
  first when tired**; multi-session **drift alerts**; auto-handedness.
- **Coaching loop:** **drill-effectiveness tracking** (did the prescribed metric
  improve next session?); **feel-correlation** personalization (felt good/off →
  your ideal); confidence **calibration** (needs 2-cam ground truth).
- **Reliability (lighter now, personal scope):** **audio make/miss** (swish/rim
  sound fused with visual — fixes our weakest signal); pick-the-shooter + rim-ROI.
- **Bigger/research:** warmup→miss-tendency prediction; reference-form overlay;
  voice / hands-free ("how'd that look?").
- **Hygiene:** one-command test runner + CI; real-data regression fixtures; pinned env.
- **Top picks:** rim-based feet · drill-effectiveness loop · audio make/miss ·
  the live-camera app.

## Two-camera 3D core BUILT (footage-independent, 2026-06-30) — backlog #1 foundation
Priority (user): **elbow flare + release consistency first.** Built the math
foundation now, synthetic-validated, so real S8 footage plugs straight in later.
`shotlab/threed.py`:
- `Camera` (pinhole K[R|t], `look_at`, `project`), `triangulate` (DLT) +
  `triangulate_joints` (matched 2D in both views → 3D).
- `elbow_flare(shoulder, elbow, rim)` → angle (deg) the upper arm swings out of
  the shoulder→rim vertical plane + signed lateral `offset`. ~0 = tucked; sign is
  setup-dependent (pin on real footage like LEFT_RIGHT_FLIP), magnitude is the
  signal. Needs rim 3D (comes from Cam 1, which sees the rim) + joints (both cams).
- `release_point_spread(points)` → `rms_spread` (headline: tight cluster =
  repeatable release) + per-axis (lateral/vertical/depth) std; points should be
  shoulder-relative so they're comparable across spots.
- **Synthetic ground-truth test (`test_threed.py`, 6/6):** project known 3D joints
  (with KNOWN flare 0/10/20/−15°) into 2 virtual cams → triangulate back →
  recovers 3D exactly (sub-cm at 1px noise) and flare within 0.5°. The stereo
  analog of the Phase-1 arc test.
- **Still needed for REAL 3D (have the math, need the inputs):** (1) temporal SYNC
  (clap/bounce frame), (2) stereo CALIBRATION from the measured-marker clip →
  each Camera's K,R,t, (3) wire triangulated joints into the form pipeline. These
  are the footage-dependent steps for when the S8 arrives.

## Orange ball (user, 2026-06-30)
User will use an ORANGE ball → on the CLOSE Cam-2 the cheap `ColorBallDetector`
(HSV orange + circularity) becomes viable again (big, front-lit ball), no YOLO
needed there; Cam-1 wide stays on the fine-tuned YOLO. Helps detection + the
future hand/ball work.

## Full real-footage validation of all 4 features (2026-06-30)
Re-downloaded the 0629 session (6 long clips, 1080p/30fps, 13–30 min each, ~115
min / 208k frames) into `data/raw/Hoops/` and ran the whole pipeline end-to-end.
- **Run config:** `build_session.py --detector yolo --weights
  runs/.../best_openvino_model --imgsz 640 --stride 2 --chunk-frames 7000 --pose
  --no-spin`, auto rim-detect per clip (verified visually on clip 152555).
  Chunked + resumable; ~2 hrs CPU.
- **Result: 107 shots / 34 min.** shot_form 99 jumper/7 floater/1 layup;
  shot_setup 54 on_the_move/17 off_dribble/12 catch_and_shoot/24 unknown;
  release_conf 15 high/11 med/81 low; elbow@release on 67/107. **Make-correlation
  cleared the n-gate and produced coherent (all "low" conf, none significant)
  findings:** makes hold follow-through LONGER (0.63 vs 0.43s), more bent/controlled
  elbow (116° vs 128°), LESS balance drift (0.67 vs 2.37), deeper knee bend — all
  textbook-sensible. Make% 21% (rose 16%→26% 2nd half). Report:
  `data/out/session_0629_full/report.html`.
- **Honest caveats:** all form metrics LOW-confidence (far/small shooter on one wide
  cam, foreshortened, pose resolves ~63%); make/miss heuristic. These are HINTS —
  the 2nd/closer camera + a calibration clip are what make them trustworthy.

### Two bugs found + fixed while testing
1. **OpenVINO model frozen at 640×640** → must run `--imgsz 640` (768 crashes;
   PyTorch `best.pt` is dynamic if a bigger size is wanted).
2. **Stale-cache footgun (FIXED):** `process_clip`'s per-clip record cache was keyed
   only on the filename, so after any code change it silently returned old-schema
   records. Added `_record_cache_sig()` (folds in the ShotRecord field set +
   detector/pose params + `_CACHE_VERSION`); cache now stored as
   `{"sig":…, "records":[…]}` and recomputed on mismatch — whole-clip AND chunk
   caches. Old bare-list caches auto-invalidate. Tests: test_session_integration 3/3.

### Two-camera filming plan (settled with user 2026-06-30)
Can't have "big body everywhere" AND "full ball arc" with 2 cams (a tight body-cam
loses the arc). So: **Cam 1 stays WIDE** (court+rim+arc, owns ball metrics);
**Cam 2 = body-cam** on the shooting-hand side, perpendicular, chest-height, framed
head-to-feet on the shooting AREA (rim NOT needed) — owns form/pose, ~2× bigger body
fixes the low-confidence form. Sync = one ball-bounce/clap in both at the start. For
future 3D, ensure the calibration marker is visible in BOTH. **Cheap 2nd cam:** a
used name-brand phone with a good REAR cam at 1080p/60 (Pixel 3a/4a or Galaxy S8,
~$50–75); NOT no-name junk (e.g. the Kchsji U8 — only a 2MP front cam, disqualified).
30fps is fine (matches Cam 1); 60fps a nice-to-have. True 2-cam 3D FUSION is still
backlog #1 (not built) — getting the camera unlocks building it.

## "Make it the best program we can" — enhancement backlog
Ranked by impact. (✅ done, ⏳ planned, 💡 idea)

1. 💡 **Two-camera capture for true 3D** — the *real* fix for elbow flare &
   squareness (the depth-limited metrics). Sync side+front clips → triangulate.
   Biggest accuracy unlock that no single-cam model can match.
2. 💡 **Scale calibration from the 10 ft rim** (or a known marker) → convert all
   heights to real feet (apex, release height, jump height) with confidence.
3. ✅ **Rim/hoop detection + make/miss classification** → shooting % per session
   and *correlate form to makes* (the holy grail: which mechanics → makes).
   Engine built 2026-06-30 (`shotlab/correlate.py`); reliability grows with
   calibration footage + volume. See section above.
4. ✅ **Release-frame sync between ball and pose** (ball leaves the hand) → precise
   elbow-at-release and release-vs-apex timing. (2026-06-30, see section above.)
5. 💡 **Fine-tune a basketball YOLO on your own footage** (a few hundred labeled
   frames) → robust detection in your gym/lighting; removes the color-tuning step.
6. ✅ **Session history + trend charts** — track the *consistency* (variance) of
   release/entry/elbow across shots and across sessions over time. Consistency
   matters as much as the mean. (Cross-session piece added 2026-06-30;
   `consistency_progress`. See section above.)
7. ⏳ **One-Euro keypoint smoothing** (mandatory per research) before angle calc.
8. 💡 **Calibration wizard in the dashboard** — click the rim, click a known
   distance; set handedness; saved per session.
9. 💡 **Per-shot clip export** + side-by-side vs a "reference clean shot."
10. 💡 **Audio-assisted make detection** (rim/swish sound) on makes.
11. ✅ **Auto shot-zone / shot-type tagging.** (2026-06-30, `shotlab/shottype.py`:
    jumper/layup/floater + catch-and-shoot/on-the-move/off-dribble. See above.)

---

## How to add your footage
Drop a clip into `C:\Users\jmaku\Desktop\ShotLab\data\raw\`, name it clearly
(e.g. `2026-06-28_sideon.mp4`). Then:
```
python analyze.py data/raw/2026-06-28_sideon.mp4 --detector color
streamlit run dashboard/app.py
```
Tell me the filename and angle (side-on / front-on) and I'll tune detection +
validate the metrics against it.

## Validation harness
`scripts/make_synthetic_clip.py` makes a clip with KNOWN release angles — our
ground-truth regression test. Phase 1 recovers them to 0.0°. (No human in it, so
it can't validate pose — that needs your real clip.)

## Real-clip findings (2026-06-28) — PXL_20260514 (first real upload)
First real footage: 1920x1080 **30fps**, 23s, outdoor driveway hoop.
- **Detectors color & stock-YOLO both fail here:** ball is small (~20px, zoomed
  out), **backlit = dark silhouette not orange** (color → 536/696 = false
  positives), YOLO COCO sports-ball → 1/120 in the flight window. So added a
  **MotionBallDetector** (MOG2 background subtraction): isolates the fast ball
  from static clutter → 206/696, cleaner. NEW 3rd backend `--detector motion`
  (best for cluttered outdoor footage). Residual movers = wind-swayed leaves.
- **Tracked a real shot end-to-end** (shot 1, 14 pts) — pipeline works on real
  video — BUT numbers unreliable because:
  1. **Only the RISING limb is captured**: ball exits top-left toward the hoop
     before we see the descent → entry angle is bogus, parabola fit on a partial
     arc. Must frame the WHOLE arc (release → rim).
  2. **Camera very low (ground level) + oblique** → angle foreshortening.
  3. **Zoomed out** → ball/shooter too few pixels for reliable detect + pose.
  4. **30fps** → motion blur + no spin.
- Lesson → the capture recipe in the filming guide above is the fix. Resolution
  (fill frame with shooter) + full-arc framing + square/raised camera + clean
  background matter more than any model.

## Session analytics layer (2026-06-28) — "fatigue / zones / make%" build
User asked: can it tell metrics by court zone/direction, and track fatigue over a
session (make% + knee bend + arch declining as I tire, using timestamps)? YES —
built the layer. Key idea (user's): **a shot = a ball flight that reaches near the
rim** (dribbles never do).

NEW modules:
- `shotlab/court.py` — **Calibration** (rim x/y, radius, shot-gate). `detect_rim`
  (orange-rim HSV) + `auto_calibrate` (median rim over ~9 sampled frames, **PER
  CLIP** because the tripod moved between clips). `filter_shots_by_rim` +
  `is_real_shot` (rim-anchored + launched-below-rim + apex-reaches-rim +
  not-near-vertical gates → cut 14 raw flights to ~3 real shots). `zone_for_release`
  (left/center/right × near/mid/far, image-space proxy until full court homography).
- `shotlab/make.py` — make/miss heuristic from post-rim trajectory. **LOW
  confidence** (ball often lost at rim on consumer footage); reported, not trusted.
- `shotlab/session.py` — `parse_clip_time` (filename PXL_YYYYMMDD_HHMMSS →
  datetime), `process_clip` (per-clip rim-anchored shots → ShotRecords, **cached**
  to `<clip>_shots_session.json`), `build_session` (stitch clips → one timeline w/
  elapsed_min), `fatigue_trends` (linear slope of each metric vs elapsed time).
- `build_session.py` CLI — process many clips → session_shots.csv +
  fatigue_trends.csv + zone_summary.csv + session_chart.png + make%.
- Dashboard: added **Session analytics** view (timeline chart, trends, zones, make%).

NEW detectors: `MotionBallDetector` (`--detector motion`, MOG2) and
`MotionColorBallDetector` (`--detector motion+color`, moving∩orange). For the
**red/blue ball** use plain `motion` (color gate is orange-only).

**Slow-mo fps fix:** `video_io.probe` now reads `com.android.capture.fps` — Pixel
saves 120/240fps slow-mo as a 30fps-PLAYBACK file; using 30 would make spin 4× low
and time metrics 4× long. VideoInfo.fps now = true capture fps.

**2nd-session footage (2026-06-28, Hoops/, 11 clips):** 19:00–19:06 regular 30fps,
19:10+ slow-mo 120fps. Child clips to skip: 190656, 191516, 191606. Camera was
**repositioned between clips** (rim x≈1100 early → x≈620 later) → per-clip calib.
Framing is diagonal (shot travels toward hoop/into frame) → absolute release/entry
angles foreshortened (entry reads ~23-50° vs real ~45); CONSISTENT distortion so
relative fatigue trends still valid. User will try dead-side-on next time (pole may
partly block). Ball = red/blue → `motion` detector.

## Ball-detector fine-tune (2026-06-28) — better detection on own footage
Detection completeness (motion finds only ~2-3 of ~10+ shots/clip vs leafy bg) is
the limiter on trustworthy session analytics. Chose to **fine-tune on the user's
OWN footage** over a ready-made model, because:
- **Security:** ready-made `.pt` = pickle → arbitrary code on load. The verified
  best community model (`avishah3/AI-Basketball-Shot-Detection-Tracker/best.pt`,
  YOLOv8n, ball=cls0, hoop=cls1, no license) got auto-blocked (untrusted source).
  Did NOT work around it.
- **Fit:** that model is INDOOR-ORANGE-trained; user's ball is red/blue outdoor
  (out-of-distribution). Roboflow path dead too: `inference` pkg needs Py3.10-3.12,
  user is on 3.13.
**Approach = weak supervision, no manual boxing:** `tools/make_dataset.py` uses the
motion detector's in-flight, rim-anchored ball positions (the reliable subset that
passed RANSAC + rim gate) as YOLO labels → distills motion cues into an
appearance-based detector that fires where motion fails. `tools/train_ball.py`
fine-tunes the TRUSTED base yolo11n (not an untrusted download) → local best.pt →
plug into existing `--detector yolo --weights ... --ball-class 0`. Roboflow backend
also added (`detect_roboflow.py`, `--detector roboflow`) but parked (Py3.13).
Contact sheet (`label_contact_sheet.jpg`) for label QA before training.

## Ball detector TRAINED (2026-06-28)
Dataset: `dataset_ball/` — 982 train / 19 val clean labels (auto-labeled from
motion-tracked shots across 7 adult clips, red/blue color-filtered; val = held-out
19:21). `tools/clean_dataset.py` color filter dropped ~55% of raw labels (heads/
blur/foliage) — contact sheet then pristine. Trained yolo11n 768px (`tools/train_ball.py`):
**6 epochs → val mAP50 0.995, recall 1.0, mAP50-95 0.825** (killed at epoch 5/40 by a
cap but already converged — ball is distinctive single-class). Weights:
`runs/detect/ball_finetune/weights/best.pt` (LOCAL, trusted — trained from trusted
base). Use: `--detector yolo --weights runs/detect/ball_finetune/weights/best.pt --ball-class 0 --imgsz 768`.
Head-to-head vs motion (coverage + rim-shot count) RUNNING. If it wins → reprocess
session for real fatigue curves. mAP is only on motion-labeled frames; the real
test is whether it finds the ball where motion FAILED (full-clip coverage).

## Detector breakthrough + continuous-track shots (2026-06-28)
Head-to-head on 19:21: fine-tuned YOLO tracks the ball in **~30% of frames / 24
activity clusters** vs motion's sparse handful. BUT exposed a NEW problem: YOLO
tracks the ball CONTINUOUSLY (dribble+shoot, no gaps) → the gap-based
`segment_shots` can't isolate shot arcs (motion "worked" only because its gaps
accidentally segmented). FIX = `court.detect_shots_to_rim(track, calib)`: anchor
on the rim — each time the ball path reaches the rim, walk back to the launch
(ball well below rim), treat that ascending arc as a shot. Dribbling never reaches
rim → auto-ignored, no gaps needed. Validated: **3 clean shots / 3000-frame window**
(vs motion's 3 in the whole 11k clip), realistic entries (45-46°). Wired:
`run_phase1(..., calib=, stride=)` uses rim-anchored detection when calib given;
`session.process_clip` supports `--detector yolo --weights ... --stride`.
**STRIDE** added (detect every Nth frame; 120fps clips → stride 3 = 40fps eff, 3×
faster) to fit CPU + the ~15min background-job cap (long jobs get killed; per-clip
cache makes build_session RESUMABLE — just re-run). **Spin breaks under stride**
(needs consecutive full-fps frames) → None for now; needs a dedicated full-fps pass
on shot windows (deferred, it's the stretch feature). Reprocessing slow-mo clips
(19:17/21/22/25, stride 3, pose) RUNNING/resumable. Per-clip rim still via
constrained `detect_rim`.

## Spin tested on REAL footage — doesn't work here (2026-06-28)
Tested estimate_spin on 3 real shots (dense full-rate track, 120fps): only 1/3
gave a value (60rpm, low-conf, consistency 0.73); other 2 "rotation inconsistent
(blur/plain ball)". Root: ~25px ball + 120fps + motion blur → crop too low-res to
track seam rotation. **Did NOT build the full-rate spin pipeline** (would produce
garbage). Needs 240fps + ball bigger in frame. Filming fix, not code. `--no-spin`
in session runs.

## Dead-side-on explainer + full-session extension (2026-06-28)
Generated `data/out/dead_side_on_guide.png` (tools/dead_side_on_guide.py): bird's-eye
camera placement + in-frame arc contrast (diagonal=foreshortened vs side-on=true
parabola). Added **auto-stride** in process_clip (≈40 eff fps + long-clip thinning
to fit job cap). Extended reprocess (all adult clips incl early 19:00-19:10, YOLO,
auto-stride, pose, no-spin) RUNNING/resumable → full ~25-min fatigue timeline.
Prior 4-clip (19:17-25) YOLO result: **45 shots/10min**, make% 53%→16% half-to-half,
release/entry angles declining; angle scatter still wide (camera foreshortening →
dead-side-on is the fix).

## Complete session + UI/features (2026-06-28)
Fine-tuned YOLO + rim-anchored detection reprocessed ALL 8 adult clips →
**107 shots over 26.6 min** (vs motion's 6). Full session: release angle −0.46°/min
(59→46° across session, clearest fatigue signal), entry −0.23°/min, knee +0.19°/min
(less bend = mild fatigue), make% ~mild decline. Angle scatter still wide (camera
foreshortening). Spin garbage in 4 cached records (pre-no-spin) — ignore.

**4 features built (all live in dashboard):**
1. **Consistency** — `session.consistency_stats` (within-zone std = true repeatability,
   removes position confound; first/2nd-half = more erratic when tired). On this
   footage within-zone std ~16° = mostly measurement noise (needs calibration).
2. **Per-shot review** — `overlay.render_shot_clip` + `tools/render_shots.py` →
   per-shot overlay clips (ball trail+arc+metrics) + index.json; dashboard "Shot
   review" view plays them. Demoed 19:21's 3 shots from cached track.
3. **Multi-session progress** — `session.aggregate_sessions` (one row/session, dated
   from shots) + dashboard "Progress" view (only 1 session so far = baseline).
4. **Report export** — `tools/export_report.py` → self-contained `report.html`
   (embedded chart + all tables).

**Dashboard now 4 views** (`dashboard/app.py`): Per-clip · Session analytics
(interactive Altair: KPIs w/ 1st-vs-2nd-half make delta, zone-colored fatigue chart
+ trend, zone filter, zone bars, consistency) · Shot review · Progress. AppTest all
green; 11/11 unit tests pass.

**STILL PENDING (needs user's marked footage next session):** court-calibration
correction (true angles when roaming the arc) + reliable layup detection (by true
distance). User will place a measured-rectangle marker + film a calib clip next time;
build+validate the homography correction on that (NOT blind). Diagrams:
`data/out/dead_side_on_guide.png`, `moving_shooter_geometry.png`.

## Coaching layer (2026-06-29)
`shotlab/coach.py`: `generate_review` (plain-English what-you-did-well/work-on/
focus from fatigue+consistency+zone signals, honest re: foreshortening),
`grade_shots` (per-shot good/'off' vs YOUR-OWN zone norm = reliable despite
foreshortening; ties misses to the deviating metric), `recommend_drills` (concrete
drills from weaknesses: form-shooting weak zone, conditioning ladder if fatigued,
arc drill if flat, star drill if one-spot-heavy, beat-your-spread), `arc_from_angles`
+ `IDEAL_ARC` (52→45) for the reference-arc overlay. `session.volume_stats` (makes/
attempts/longest make-streak). Dashboard Session view now shows: Coach review +
drills, "Your arc vs ideal" Altair chart, volume/streak KPIs, per-shot grades table
(filter to 'off'). report.html includes the review. Tests 11/11.
**IDEAL metrics stance:** only ~45° entry is textbook; the TRUE personalized ideal =
correlate YOUR makes vs YOUR form — needs calibration + reliable make + volume (the
"holy grail", deferred). Vision given to user: tool's superpower = measurement at
volume (consistency/fatigue/zone patterns across 100s of shots), not replacing a
coach's eye. Next build after calibration footage = make-correlation engine.

## Best-shots reel + court scale (2026-06-29)
`coach.rank_shots` scores shots by ideal form/arc (made + clean-vs-own-norm grade +
soft arc) → top-10 `best_shots.csv` + dashboard "⭐ Best shots" table. Rendered review
clips for 19:21 (9) + 19:22 (15) via render_shots → Shot review tab plays them. Top
made/clean: 19:21 #4/#7, 19:22 #8/#12, 19:10 #26/#14.
**Court scale attempt:** shooter pose height ~1026px full / 5.833ft → you ~8ft from
camera (7–9.5 across FOV 60–75°); hoop ~3-4× farther → camera-to-hoop ~25–35ft ROUGH
(±large: unknown lens FOV, diagonal angle, rim ground-ref unreliable, driveway not
regulation). Precise size NEEDS the marker calib clip. Camera is to shooter's right
(foreground), hoop far-left background.

## Data management policy (2026-06-29)
Project hit 11GB. Policy: **metrics are tiny+precious, raw clips+intermediate are
huge+disposable.** `tools/curate.py`:
- `--session <out> --name <date_name>` → archives KEEPERS to `data/sessions/<name>/`
  (metrics/ CSVs+review+report+chart, caches/ per-clip *_shots_session.json to rebuild,
  clips/ h264 best+worst review clips). Whole session ≈ **15MB**.
- `--purge-zips` (extracted zips), `--purge-dataset` (training imgs; model kept in
  runs/), `--purge-intermediate` (mp4v/overlays/_frames/logs/yolo_track; keeps h264),
  `--purge-raw <glob>` (raw clips — ONLY after archiving + rendering wanted clips).
**Workflow per session:** process → render best+worst shot clips → curate archive →
purge intermediate + raw. `render_shots.py` now `--skip-done` + auto-stride (resumable
batch). Done 2026-06-29: purged zips(4.4G)+dataset(547M). Rendering best/worst clips
(19:00/19:10/19:18/19:25) then will purge raw+intermediate. KEEP: data/sessions/,
models/, runs/weights, code. The trained best.pt is the one irreplaceable artifact.

## Session 2026-06-29: movement, OpenVINO, comparison stills
- **OpenVINO export** (`best_openvino_model/`, FP16) = **6.6× faster** (53fps vs 8fps,
  same accuracy 175 vs 172 balls) — THE fix for long 120fps clips. Use
  `--weights runs/detect/ball_finetune/weights/best_openvino_model`. Now a KEY
  artifact (keep alongside best.pt). `pip install openvino` works on Py3.13.
- **run_phase2 decode bug FIXED:** it iterated the WHOLE clip to pose-extract shot
  windows; now stops at last needed frame (big speedup for long clips).
- **Movement direction** (`form.movement_direction`, `LEFT_RIGHT_FLIP` anchor):
  left/right/set into the shot, from hip trajectory ~0.3s pre-release vs facing rim.
  In ShotRecord.movement_dir + dashboard "By movement" table. **Per-camera label**
  (front/back flips it) → confirm each setup. ~half come back 'unknown' (pre-window
  reaches before extracted frames; widen _needed_frames pre to fix).
- **Today's footage:** 6 clips, 2 setups → split into `2026-06-29_shooting` (Set A,
  36 shots) + `2026-06-29_form` (Set B close-up, 10 shots, user EXHAUSTED). 15:39
  found 0 shots (unstable corner rim — needs manual calibrate, deferred). **Widened
  detect_rim band** to (y 0.08-0.45, x 0.08-0.95) for corner rims. **Don't compare
  absolute metrics across the 2 sessions (different cameras).**
- **Shot-comparison stills** (`shotlab/compare.py`, `tools/compare_shots.py`,
  dashboard "Compare shots"): two shots × 4 phases (load/rise/release/follow-through),
  skeleton + RED dots on elbow/knee + angle labels, cropped to shooter, side-by-side.
  Demo (form 15:30 shot1 made vs shot4 miss) clearly showed **made=deep load (knee 81)
  vs miss=stood tall (knee 104)**. Needs RAW present → render before curating.
- **Op note:** long-clip processing must fit the ~11min bg-job cap → OpenVINO +
  `--max-frames 8000` (first ~67s/clip) does ~2-3 clips/cycle. Bg jobs occasionally
  die instantly (transient) → relaunch (resumable via per-clip cache).

## Pipeline-improvement pass (2026-06-29)
- **Crop-to-shooter pose: REFUTED.** Tested full-frame vs cropped+upscaled pose on
  distant shooting footage — cropping made vis WORSE 3/4 (0.84→0.58-0.97). MediaPipe
  already self-crops; upscaling a small blurry person adds no detail. Distant-pose
  'unknown' movement is a FOOTAGE limit (close framing), not software-fixable.
- **Non-shot gates added to `detect_shots_to_rim`** (the continuous-track detector
  lacked the gap-path's gates): reject `fit.n_used < 7` (noisy fit) and
  `min(release, entry) > 78` (near-vertical toss/rebound). Verified on synthetic
  (normal kept, realistic near-vertical/few-point rejected). Cleans false-positive
  'shots' (e.g. today's shooting #2 82/77, form #3 83/85). Applies to FUTURE
  processing; existing today-sessions have 1-2 non-shots (reprocess to clean).

## Detection caching (2026-06-29) — fixes the re-detect timeouts
Root cause of today's render/compare timeouts: detection (~80% of work) was re-run
from scratch every time (session build, render_shots, compare, label clips) on
27k-53k-frame clips. **`shotlab/detect_cache.py`**: `detect_or_load()` caches the
ball track + shots per clip to `<clip>_track.json` (keyed by weights/imgsz/stride/
max_frames/rim); reuses across session build + `render_shots` + `compare.py`.
Verified: detect 13.5s → load **0.02s (876×)**, save/load roundtrip metrics-identical.
Track cache (~100KB) also archived by curate → enables re-analysis (new gates/metrics)
WITHOUT raw or re-detection. NOTE: this fixes REPEATED ops; the FIRST detect of a
long clip still must fit the job cap (→ shorter clips / 60fps / OpenVINO).
**FPS clarification given to user:** processing cost = FRAME COUNT, not playback speed;
120fps = 2× frames of 60fps regardless of slow/normal playback. Use 60fps unless spin.

## Auto-chunking long clips (2026-06-29) — fixes the FIRST-detect timeout
The detection cache above fixes REPEATED ops but the first detect of a long clip
still had to fit the job cap. **Auto-chunking** removes that ceiling: pass
`--chunk-frames N` (yolo only) and any clip longer than N frames is processed in
absolute frame WINDOWS of N. Each window's cache (`<clip>_chunk_<start>.json`) holds
BOTH its detection (track+shots) and its records, so a job kill **resumes at the next
window** — re-detecting only the unfinished window, not from frame 0. Windows merge
with shots renumbered 1..N across the whole clip; a full-clip `_track.json` is written
(keyed max_frames=None) so render/compare load it without re-detecting.
- `run_phase1(start_frame=...)` bounds decode to [start, max) with ABSOLUTE indices.
- `detect_cache.serialize/deserialize_detection` shared by whole-clip + window caches.
- `session._process_chunked` does the window loop + merge + renumber.
- Verified on synthetic (3 windows): chunked output **metric-identical** to whole-clip
  (5 shots, same release/entry/zone/timestamps/renumbered 1-5); resume after dropping
  1 window re-detected exactly 1 (not 3); chunked track cache reused by detect_or_load
  with 0 fresh detections. curate `--purge-intermediate` drops the resume-only `_chunk_`
  caches (redundant once `_shots_session.json` + `_track.json` exist).
- Suggested size: `--chunk-frames 7000` (≈ one job cap's worth after auto-stride).

## Spin status (answered to user 2026-06-29)
Spin (Phase 3) is **OFF in every session build** (`--no-spin` / not passed). It was
tested on the real 120fps footage and is **unreliable**: ball ~25px + motion blur →
seam rotation unreadable (1 of 3 shots gave a low-conf value). `estimate_spin` gates
≥110fps. To actually get spin: ball BIGGER in frame (closer camera) + ideally 240fps.
Code exists and re-enables by dropping `--no-spin`, but won't deliver on current footage.

## Open judgment calls / decisions log
- 2026-06-27: default detector = `color` (more accurate on clean footage, no GPU
  needed) over YOLO. YOLO is one flag away for messy real footage.
- 2026-06-27: chose detection+RANSAC over MOT trackers for the ball (survey).
- 2026-06-27: MediaPipe over RTMPose for pose (CPU-only machine).
