# ShotLab — Project Notes (living log)

> The canonical "everything" doc. README.md is for usage; this is the decision
> log, filming guide, roadmap, and enhancement backlog. Update it as we go.

Last updated: 2026-07-22 · Location: `C:\Users\jmaku\Desktop\ShotLab`
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

## ⭐ NEXT SESSION PICKUP (2026-07-22 night)
**Where we are:** Step-1 of the TrackNet plan is DONE, and a DUAL ADVERSARIAL REVIEW
(Codex + Fable, owner-requested "push further") retired the oracle experiment family as a
sizing tool. Settled facts: detector sees the ball 99% inside labeled windows → TrackNet/tiling
stay OFF the table. The gap-based "perfect detection HURTS / cloud regresses" headline was a
`segment_shots` artifact → RETRACTED (mechanism proven). BUT the rim follow-up's `+2` is also
untrustworthy: the labeled-window design is **selection-biased** (labels seeded only from
already-detected shots → blind to the detection prize) and every number is an **unmatched raw
count** (no precision/FP; the `+2` includes a bounce-back false positive). The rim is a
**material confound** (auto-rim vs cached differ 110/225/59px; real shots span a 110px x-spread
within one clip → tripod moved / a single per-clip `Calibration` is wrong). **Biggest EVIDENCED
lever = verified-rim + segmenter repair, NOT "film closer"** — with a perfect track the production
`detect_shots_to_rim` still drops ~half the rim-reaching shots via 5 cheap-fix defects (see the
Step-1 section). Tracker velocity/reset fix is correct but has ZERO test coverage. Committed
through `68cb7d3` + this session (`b7bdf2d` + docs/reviews); **NOT pushed.**

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
3. **Rim is a MATERIAL CONFOUND (adjudicated: Codex right, Fable's validation was necessary-not-sufficient).**
   Auto-rim vs full-clip cached rim differ 110/225/59px (≫ 90px gate). My own label probe: within clip1
   real shots approach x≈1130 (early windows) AND x≈1244 (late) — a **110px within-clip spread** → the
   tripod moved / multiple positions; a single per-clip `Calibration` is wrong for this footage, and
   `baseline=2` is calibration-driven, not a stable production number.
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
