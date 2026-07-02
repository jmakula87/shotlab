# ShotLab — Project Notes (living log)

> The canonical "everything" doc. README.md is for usage; this is the decision
> log, filming guide, roadmap, and enhancement backlog. Update it as we go.

Last updated: 2026-07-02 · Location: `C:\Users\jmaku\Desktop\ShotLab`

---

## NEXT SESSION PICKUP (2026-07-02)
State: pickup items 1/2/4 from 07-01 are DONE + pushed (see the 2026-07-02 session
log below): wrist-apex release is in the METRIC path (elbow bias fixed, −7.3°
mean), jump height is ankle-based + physics-gated (median 2.18→1.56 ft, impossible
tails now None), per-shot **shot map** ships in dashboard/report/PDF, and the app
exports **feel logs as CSV**. Session 0701 (+`_wide`/`_moved`) rebuilt on the fixed
pipeline; app profile re-exported (elbow ideal now 118.9°); SW cache v4.

**⭐ THE BIG ONE — 2nd camera (Galaxy S8) arrives within a week (~2026-07-08).**
That's the real form/flare accuracy unlock. The footage-independent 3D core
(`shotlab/threed.py`: stereo triangulate + elbow_flare + release-point spread) is
already built + synthetic-validated (6/6). When it lands, per the settled 2-cam
plan: (1) S8 as close body-cam side → existing form metrics reliable; (2) marker
clip → stereo calibration + temporal sync; (3) **build ELBOW FLARE + release
consistency FIRST**. Cam-1 stays wide (arc/rim/makes), Cam-2 = body-cam.

**Buildable now (no hardware):**
1. **USER: test the app on the Pixel** — live camera, gold ideal-skeleton overlay,
   new Feel log (CSV) button (visible once any live shots are tagged).
2. **Headline finding to chase with filming:** makes come with much deeper knee
   bend (full 0701: 107° vs 137°, d=−0.77, p=0.035; camera-consistent wide subset:
   99° vs 137°, d=−0.97, p=0.017). First make-driver to clear significance.
3. **USER FEATURE REQUEST (2026-07-02) — movement-context form breakdowns, build
   once form is trusted (post-S8):** he wants to slice by shot context — all
   clips where form was "ideal" going LEFT vs going RIGHT vs SET — because the
   ideal form differs per context. Pieces: (a) per-context ideal profiles
   (condition the ideal/correlation engines on `movement_dir`, which is already
   in records, min-n gated); (b) Shot-review filter = movement_dir × form-grade
   so the rendered clips of "ideal going left" are one click; (c) same split in
   the app profile eventually (profile.json per-context ideals). The "By
   movement" dashboard panel is the seed; this is the full version.
4. **Audio make/miss PROMOTED to default (A/B 2026-07-02).** On session 0701:
   ZERO contradictions with confident visual calls (16 makes/43 misses all
   agree), resolved 9/12 visual-unknowns (all → miss, consistent with a 27%
   session), classifiable 83%→96%, 37 shots low→medium conf. Corrected make%
   23.5%. Sharpened the drivers: knee bend p=0.0115; NEW second driver — makes
   arc 1.14 ft HIGHER (p=0.045). `--audio` is now DEFAULT-ON in build_session
   (`--no-audio` to disable). Caveat: audio agreeing with visual isn't ground
   truth — a session where the user calls out makes would truly validate it.
5. **Shooter-height ruler (user is 5'10" in shoes) — promote into the pipeline.**
   Diagnosed 2026-07-02: nose-to-ankle px per shot = a depth-plane ruler at the
   SHOOTER's depth (median 51 px/ft vs the rim plane's 20 — shooter ~2.5× closer
   to camera). Measured his court with it: camera ~33-42 ft from hoop, shots
   median ~13-15 ft / max ~26-32 ft from rim, ~26 ft of width used. Build a
   `--shooter-height` flag: use body-scaled px/ft for release_height_ft /
   jump_height_ft (upgrades them from rim-scaled LOW conf ~2.5× hot to honest),
   and real-feet shot distances for zones. Script: scratchpad court_from_height
   (rewrite properly into shotlab/scale.py + a tools/ diag).
6. **ORANGE-BALL RETRAIN (diagnosed 2026-07-02) — the top tracking lever.** The
   orange ball did NOT help 0701: wide-cam YOLO hit rate DROPPED 35%→20% per
   sampled flight frame vs 0629, because the fine-tune dataset was auto-labeled
   on the OLD ball (red/blue filters) — orange is out-of-distribution. Likely
   also the cause of the 0.4s late-lock. Meanwhile the HSV color detector (dead
   on the old dark ball) now locks the orange ball in-hand at 30+ ft (verified
   on rendered frames; false positives in grass/trees → needs the existing
   motion gate). FIX: make_dataset on 0701 clips with ORANGE HSV auto-label,
   merge with the old 982 frames, retrain yolo11n, re-measure hit rate +
   late-lock. Orange ball stays right for the S8 close-cam color path.
7. Smaller ideas left: goal-progress tracking, report emailing, ingest the app's
   feel-CSV into the desktop records (join on session/shot time).

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
