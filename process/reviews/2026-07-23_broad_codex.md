I do not accept the “80% recall / 0.99 precision, and detection is now the wall” conclusion as established. The tracker gain is real on these artifacts, but the evaluation has leakage and semantic blind spots, make/miss is not production-validated, and several coaching metrics are systematically mis-scaled or misidentified.

## What the committed evaluation actually supports

[probe] Parsing the three committed hand-count CSVs and corresponding `*_eval.json` files produced 111 attempts: 55 makes, 56 misses, 108 rim-reaching attempts, and 3 airballs.

[probe] Aggregate committed results are:

| Condition | Produced | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| C1 greedy | 61 | 61 | 0 | 50 | 1.000 | 0.550 |
| C2 cloud | 29 | 29 | 0 | 82 | 1.000 | 0.261 |
| C3 beam | 80 | 77 | 3 | 34 | 0.963 | 0.694 |
| C4 union | 90 | 89 | 1 | 22 | 0.989 | 0.802 |

[probe] C4’s 22 misses are distributed 10, 7, and 5 by clip—7.3 per clip, not approximately 10 per clip.

[code] The harness matches all attempts, including airballs, solely by temporal proximity to a rim event; its self-test explicitly counts an unrelated rim event near an airball as a true positive (`tools/eval_ablations.py:47-98`, `tools/eval_ablations.py:247-274`; `tests/test_eval_harness.py:34-67`).

[probe] As an adversarial sensitivity bound—not a definitive correction—reclassifying the three airball matches as unmatched attempts plus false rim events changes C4 to 86 TP, 4 FP, 25 FN: 0.956 precision and 0.775 recall. This demonstrates that the reported 0.99/0.80 can materially depend on unverifiable airball matches.

[code] Matching greedily sorts candidate pairs by distance and consumes them one at a time; it is one-to-one but not a maximum-cardinality assignment (`tools/eval_ablations.py:47-70`).

[probe] Reimplementing that exact matcher on attempts `[100,130]` and shots `[70,105]` at ±30 frames returns one TP, although the assignment `70→100, 105→130` returns two. The matcher can therefore undercount recall as well as obscure which shot was associated.

[probe] The smallest hand-count attempt gap is 31 frames, while union dedup is 25 frames and matching tolerance is ±30; only one observed gap is within 60 frames. Those thresholds are effectively tuned just below this dataset’s closest pair and are not stress-tested on rapid-fire or rebound sequences.

## Make/miss: presently not a trustworthy production output

[code] The committed eval artifacts contain only aggregate counts; they do not retain match pairs, source trajectories, make/miss predictions, or confusion matrices (`tools/eval_ablations.py:216-241`).

[probe] The managed environment rejected every Python invocation, so I could not rerun the three videos. Because the missing pair/source data cannot be reconstructed from the aggregate JSON, a defensible all-three-clip `classify_make` accuracy is not measurable from the committed artifacts.

[probe] I could perform a limited probe on the current clip-1 production cache covering its first 3,500 frames. Ten detected shots matched hand-count attempts within 0–6 frames:

| Ground truth | Predicted make | Predicted miss | Unknown |
|---|---:|---:|---:|
| Make | 1 | 2 | 4 |
| Miss | 0 | 3 | 0 |

[probe] That is 6/10 classification coverage, 4/6 accuracy conditional on returning an answer, and 4/10 correct if “unknown” is treated as failure. The sample is too small and cache-limited to estimate global accuracy, but it is plainly inconsistent with treating make/miss as validated.

[code] `classify_make` is a geometric cylinder-crossing heuristic, frequently returns `None`, and labels every decisive result as low confidence (`shotlab/make.py:30-91`).

[code] Session FG% excludes `None` outcomes, then labels the remaining classified subset as “attempts”; this can create selection-biased make percentages (`shotlab/session.py:493-511`).

[code] Beam shots are classified against the merged global track, where greedy coordinates take precedence over beam coordinates; `classify_make` can consequently inspect a distractor instead of the beam hypothesis that generated the shot (`shotlab/phase1_ball/pipeline.py:127-152`; `shotlab/session.py:240-252`).

[code] Cache serialization does not preserve each shot’s original coordinates, and deserialization reconstructs shot points from that same merged global track (`shotlab/detect_cache.py:74-97`, `shotlab/detect_cache.py:117-137`).

[code] A learned rim/net visual classifier already exists and its module documents an earlier 74-shot result of 86% accuracy versus 49% for geometry, but it is only exposed through an offline application script—not wired into production—and that historical result is not a current three-clip validation (`shotlab/make_visual.py:1-15`, `shotlab/make_visual.py:152-183`; `tools/apply_make_model.py:1-10`, `tools/apply_make_model.py:46-59`).

## Ranked opportunities by expected value per effort

### 1. Fix calibration and build an outcome-aware held-out evaluation

**Expected value: very high; effort: low–medium.**

[code] Manual rim verification stores a clicked center and a second point whose distance becomes the radius (`tools/verify_rim.py:61-64`, `tools/verify_rim.py:86-100`).

[probe] The saved clip-1 rim is `(616,232,r=12.21)`, while the suggested hoop center is approximately `x=555`. Re-evaluating the first ten cached arc fits at `x=555` instead of `616` raises entry angle by 4.61°–8.96°, averaging 6.84°. That is large enough to change coaching interpretation and threshold-gate behavior.

[code] Production and evaluation use incompatible calibration formats: `build_session.py` loads a single `Calibration` object, while verified eval rims use a separate per-segment/list schema (`build_session.py:67-68`, `build_session.py:114`; `tools/rim_segments.py:9-14`, `tools/rim_segments.py:61-94`).

[code] When no production calibration is supplied, the session can fall back to automatic rim calibration, which chooses an orange blob and has previously been vulnerable to shirt/ball confusion (`shotlab/court.py:57-98`, `shotlab/session.py:401-405`).

Recommendation: define one calibration schema containing actual hoop center, visible half-width, valid frame interval, and provenance; consume it identically in production, evaluation, scaling, and visual make classification. Then persist exact attempt↔shot pairs, shot-local trajectories, classifier outputs, confidence, and GT outcome.

### 2. Establish an honest test set before further tuning

**Expected value: very high; effort: low–medium.**

[code] Project notes state that all 1,340 labeled frames came from these three clips, and label ingestion assigns the third clip to validation while the other clips feed training (`PROJECT_NOTES.md:572-583`; `tools/ingest_labels.py:52-74`).

[probe] The evaluated model is `runs/detect/ball_gpu_kaggle/weights/best.onnx`, the output of that labeled-dataset training workflow. Consequently, clips 1–2 are training-source footage and clip 3 is validation/model-selection footage; none is an untouched test session.

[code] The final apex gate and union dedup changes are evaluated on the same clips used to choose them (`shotlab/court.py:291-296`; `shotlab/phase1_ball/pipeline.py:127-152`).

Recommendation: record one untouched session with different framing, player, light, clothing, and shot cadence. Freeze detector, tracker, gates, and thresholds before scoring it. Replace greedy matching with maximum-cardinality/minimum-cost assignment and publish results across ±5/10/15/30-frame tolerances.

### 3. Repair or replace the “beam” association and preserve provenance

**Expected value: high; effort: medium.**

[code] Each active hypothesis extends to only its single best candidate; it does not branch among multiple plausible candidates, so this is not a conventional multi-hypothesis beam search (`shotlab/phase1_ball/track_beam.py:61-80`).

[code] Frames with no candidates execute `continue`, so `max_coast=6` counts candidate-bearing frames with failed extensions, not six elapsed missing-detection frames (`shotlab/phase1_ball/track_beam.py:47-58`, `shotlab/phase1_ball/track_beam.py:81-86`).

[code] The `.01` cloud is filtered to `.05`, and a trajectory cannot begin unless a candidate reaches `.30`; therefore failure of this beam does not demonstrate that the detector supplied no usable evidence (`shotlab/phase1_ball/track_beam.py:41-56`, `shotlab/phase1_ball/track_beam.py:87-90`).

[code] Multiple hypotheses can consume the same candidate, pruned hypotheses are discarded without segment emission, and final dedup ignores vertical position (`shotlab/phase1_ball/track_beam.py:60-93`, `shotlab/phase1_ball/track_beam.py:98-109`).

[code] Velocity is a blended constant-velocity estimate rather than a ballistic/accelerating motion model, and copying the entire path dictionary at each extension produces avoidable growth on long tracks (`shotlab/phase1_ball/track_beam.py:72-78`).

[code] Tests cover a clean synthetic arc plus stationary distractor but do not exercise genuine branching, empty-frame coasting, pruning, low-confidence starts, shared candidates, or vertical-only duplicates (`tests/test_track_beam.py:33-73`).

Recommendation: use a rim-seeded backward DAG/Viterbi or a real top-k ballistic beam. Score complete paths, count actual frame gaps, allow low-confidence starts inside a plausible rim-to-shooter corridor, and attach immutable source coordinates to every emitted shot.

### 4. Stop presenting current arc/form numbers as physically precise

**Expected value: high; effort: medium.**

[code] Arc geometry assumes a side-on camera and fits height as a parabola in image x; foreshortening or camera yaw violates the stated angle assumptions (`shotlab/arc.py:12-23`).

[code] “Release angle” is evaluated at the minimum or maximum tracked x—not at a detected release event—and “entry angle” evaluates an unconstrained polynomial tangent at `rim_x`, takes its absolute value, and does not require that the point be on the descending observed arc (`shotlab/arc.py:84-103`).

[code] Phase-1 “apex height” is fitted vertex height above the lowest tracked point, not apex above floor or rim (`shotlab/arc.py:73-86`).

[code] Phase 1 derives feet from median detector bounding-box diameter even though the project’s scaling module explicitly says ball diameter jitters and changes with depth and that rim scaling is preferable (`shotlab/phase1_ball/pipeline.py:33-54`; `shotlab/scale.py:1-10`).

[probe] In the clip-1 cache, median detected ball radii range from 25.47–37.30 px while the saved rim half-width is 12.21 px; closest-to-rim ball radii reach 34.11–56.88 px. Those incompatible rulers show that at least one of bbox size or the clicked rim radius is not measuring physical object width reliably.

[code] Session-level release location is the lowest fitted inlier, not the pose-derived or ball-detachment release frame; that point is also used for shot-zone classification (`shotlab/session.py:176-207`).

[code] Zone classification is explicitly an image-space proxy based on percentages and radial pixel bins rather than court calibration (`shotlab/court.py:309-335`).

Recommendation: report raw image-space metrics plus uncertainty until calibrated. Use time-indexed ball release, validate that the apex and rim evaluation lie within supported trajectory ranges, retain signed slopes, and separate “height above tracked low point” from “height above rim.”

### 5. Run a targeted detector-ceiling experiment

**Expected value: high; effort: medium.**

[code] `diagnose_misses.py` diagnoses misses from baseline confidence-.25 candidates, not the residual greedy∪beam misses; absence of a candidate near the rim does not prove the fast ascent was absent from the full detector cloud (`tools/diagnose_misses.py:28-43`, `tools/diagnose_misses.py:84-123`).

[code] Project notes document detailed residual diagnosis for clip 1 while marking clips 2–3 as pending, so the “roughly half detection-limited” conclusion was not established across all three clips (`PROJECT_NOTES.md:69-93`).

[code] The intended dense-GT/oracle ceiling condition remains stubbed out of the ablation harness (`tools/eval_ablations.py:237-238`).

[code] Earlier oracle/ROI results used auto-detected rims, and project notes later retract them because of bogus rim locks, selection bias, and unmatched output counts (`tools/exp_oracle_ceiling_rim.py:104-120`; `PROJECT_NOTES.md:95-111`).

[code] Current ONNX inference reads a fixed input shape from the exported model, so changing the CLI `imgsz` cannot meaningfully test another native resolution without re-exporting the model (`shotlab/phase1_ball/detect_yolo.py:128-143`, `shotlab/phase1_ball/detect_yolo.py:199-228`).

[code] Detector comments say tiling hurt one downscale-trained model, while later project notes say native-scale retraining made tiling beneficial; neither settles performance for the currently promoted weight on an untouched test set (`shotlab/phase1_ball/detect_yolo.py:56-71`; `PROJECT_NOTES.md:554-568`).

Recommendation: densely label only the residual missed-attempt ascent windows, then compare current full-frame inference, correctly exported larger input, native-scale tiling, and a calibrated rim-to-shooter corridor crop on an untouched session. “Film closer” remains useful, but it is not yet proven to be the only lever.

### 6. Validate pose/form against real annotations

**Expected value: high; effort: medium–high.**

[code] Production hardcodes every pose analysis as `side_on`, affecting confidence and interpretation of elbow and knee metrics (`shotlab/session.py:165-170`; `shotlab/phase2_pose/form.py:448-491`).

[code] Release is defined as wrist apex, with the module acknowledging that this is often 1–2 frames late and can bias elbow angle by 10–15° (`shotlab/phase2_pose/form.py:127-176`).

[code] “Release height” uses wrist position rather than ball position despite the record field describing ball height at release (`shotlab/phase2_pose/form.py:593-628`).

[code] Pose inference requests one person and returns the first pose without shooter identity association (`shotlab/phase2_pose/pose.py:125-162`).

Recommendation: annotate a modest real set for ball detachment frame, shooter identity, elbow/knee angle, wrist/ball height, and jump. Until then, label these outputs experimental and rename wrist-derived measurements accurately.

### 7. Make cache invalidation and shot-local data reliable

**Expected value: medium–high; effort: low.**

[code] Session cache version is manually fixed at 21, while cache signatures include runtime parameters but no source-code or algorithm hash (`shotlab/session.py:99-133`).

[code] Detection-cache identity likewise includes model, parameters, rim, and video but no code/schema version (`shotlab/detect_cache.py:62-71`).

[probe] The inspected clip-1 session and track caches were written around 09:00, while beam production landed at 09:01 and the apex/dedup change at 09:28. A parameter-compatible stale cache can therefore survive consequential code changes.

Recommendation: include code/schema hashes, calibration-file content hash, and tracker implementation version in cache identity. Serialize each shot’s exact source trajectory rather than rebuilding it from a global track.

### 8. Treat two-camera metrics as a research path, not production truth

**Expected value: potentially high; effort: high.**

[code] `twocam.py` describes itself as orchestration scaffolding, and its synchronization uses nearest mapped frames (`shotlab/twocam.py:1-15`, `shotlab/twocam.py:35-78`).

[code] The 3-D module explicitly states that it was validated on synthetic data before real footage (`shotlab/threed.py:7-10`).

Recommendation: conduct one calibrated real-rig pilot with reprojection error, synchronization error, and triangulation consistency reported before investing further in 3-D coaching metrics.

## Test and artifact gaps worth fixing immediately

[code] Make tests use synthetic geometric sequences, while visual-make tests exercise synthetic feature signals and model round-tripping—not real labeled shots (`tests/test_make.py:31-75`; `tests/test_make_visual.py:14-62`).

[code] The evaluation test suite enshrines the airball temporal-match issue rather than detecting it (`tests/test_eval_harness.py:34-67`).

[code] Production `_union_beam` and evaluation `_union` are separate implementations: evaluation reduces shots to rim frames, while production retains greedy-first shot objects and merged tracks. Equal detection counts therefore do not guarantee equal downstream make/arc inputs (`tools/eval_ablations.py:162-171`; `shotlab/phase1_ball/pipeline.py:127-152`).

[probe] Current `.01` candidate caches are only about 1.4–1.9 MB per clip, so storage is not yet the principal scaling concern. The larger risks are detector runtime and the beam’s repeated path copying, neither of which has a committed long-video benchmark.

[probe] Final `git status --short` was empty; this review made no repository changes.

**VERDICT:** The highest-value next task is a calibration-correct, outcome-aware held-out evaluation: re-click actual hoop center and half-width, unify the calibration consumed by production and eval, persist exact shot-local tracks and match pairs, and compare geometric, audio, existing visual-model, and fused make/miss predictions with coverage and confusion matrices. [code] The current artifacts support a substantial tracker improvement on these three clips, but the model was trained/validated on the same source footage, airballs can be credited by timing alone, the “beam” ignores sub-.05 detections and requires a .30 seed, and the dense detector ceiling is still stubbed (`tools/ingest_labels.py:52-74`; `tools/eval_ablations.py:47-98`, `tools/eval_ablations.py:237-238`; `shotlab/phase1_ball/track_beam.py:41-90`). [probe] Thus “80% recall / 0.99 precision” is a dataset-specific harness result—not a generalization result—and “detection-limited is the wall” is missing make/miss validation, calibration correctness, honest holdout testing, tracker-provenance fixes, and a real detector-ceiling experiment.