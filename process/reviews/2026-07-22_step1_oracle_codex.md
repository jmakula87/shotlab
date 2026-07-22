## Audit result

The retraction is sound, but the rim-anchored follow-up is not a trustworthy detector-ceiling experiment. It confirms only that the original negative headline does not transfer to production segmentation. It does not establish that detection, tracking, or filming distance is—or is not—the dominant full-clip bottleneck.

The strongest evidenced lever currently left on the table is verified rim calibration plus better rim-event segmentation. “Film closer is biggest” is not supported by this evaluation.

### A. Point-by-point verification

1. Production path and retraction

Production does call `assemble_track`, then `detect_shots_to_rim` when calibration exists; `segment_shots` is only the fallback: [pipeline.py:94](C:/Users/jmaku/Desktop/ShotLab/shotlab/phase1_ball/pipeline.py:94), [pipeline.py:96](C:/Users/jmaku/Desktop/ShotLab/shotlab/phase1_ball/pipeline.py:96), [pipeline.py:99](C:/Users/jmaku/Desktop/ShotLab/shotlab/phase1_ball/pipeline.py:99), [pipeline.py:102](C:/Users/jmaku/Desktop/ShotLab/shotlab/phase1_ball/pipeline.py:102).

The fallback forms gap-delimited runs and emits at most one shot per run: [track.py:101](C:/Users/jmaku/Desktop/ShotLab/shotlab/phase1_ball/track.py:101), [track.py:126](C:/Users/jmaku/Desktop/ShotLab/shotlab/phase1_ball/track.py:126), [track.py:141](C:/Users/jmaku/Desktop/ShotLab/shotlab/phase1_ball/track.py:141). Adding oracle detections can therefore join data that baseline gaps separated, changing or destroying the single RANSAC fit.

Consequently:

- Retracting “perfect detection hurts” and “the cloud regresses” is correct.
- Attributing the exact mechanism specifically to merged flights is plausible, but the original experiment did not log enough run-level diagnostics to prove that was the only cause.
- The sign flip from −2 to +2 is not evidence of a real +2 production improvement because the follow-up has separate instrumentation problems.

2. The evaluation windows invalidate the word “ceiling”

The labels were not sampled independently from the footage. The label task begins “for each detected shot,” loads the pipeline’s existing shots, and labels only their predicted intervals plus a 12-frame margin: [make_label_task.py:9](C:/Users/jmaku/Desktop/ShotLab/tools/make_label_task.py:9), [make_label_task.py:43](C:/Users/jmaku/Desktop/ShotLab/tools/make_label_task.py:43), [make_label_task.py:54](C:/Users/jmaku/Desktop/ShotLab/tools/make_label_task.py:54), [make_label_task.py:86](C:/Users/jmaku/Desktop/ShotLab/tools/make_label_task.py:86).

Both experiments then evaluate only those labeled frames: [exp_subthreshold_signal.py:53](C:/Users/jmaku/Desktop/ShotLab/tools/exp_subthreshold_signal.py:53), [exp_oracle_ceiling.py:74](C:/Users/jmaku/Desktop/ShotLab/tools/exp_oracle_ceiling.py:74), [exp_oracle_ceiling_rim.py:75](C:/Users/jmaku/Desktop/ShotLab/tools/exp_oracle_ceiling_rim.py:75).

My read-only label probe found:

- 1,340 labeled frames, 921 ball-present.
- Fourteen disconnected labeled windows: 9/3/2 by clip.
- Median labeled radius 22.4 px; only 51/921 present frames had radius ≤12 px.

This creates severe selection bias:

- A completely missed shot generated no label window and can never be recovered by this “oracle.”
- The 99% number is conditional recall inside regions proposed by an earlier successful pipeline.
- Raw output count is called “shot recovery,” but there are no human attempt IDs or output-to-attempt matches. False shots and duplicate shots are not distinguished from recovered attempts.
- The 99% result remains useful narrowly: inside already-selected flight windows, the detector usually emits a nearby low-confidence candidate. It cannot prove detection is not a full-clip bottleneck.

The claim “therefore the tracker discards subthreshold signal” also overreaches. `assemble_track` does not discard a nonempty candidate frame: if association fails, it immediately seeds the highest-confidence candidate [track.py:54](C:/Users/jmaku/Desktop/ShotLab/shotlab/phase1_ball/track.py:54), [track.py:91](C:/Users/jmaku/Desktop/ShotLab/shotlab/phase1_ball/track.py:91). Candidate-level recall does not measure whether the selected track follows the real ball.

3. Rim calibration is a major instrument confound

`detect_rim` chooses the largest orange blob in a broad upper-frame band, optionally merges nearby orange blobs, and applies no rim-shape, stationarity, dispersion, or multimodal-consensus validation: [court.py:57](C:/Users/jmaku/Desktop/ShotLab/shotlab/court.py:57), [court.py:89](C:/Users/jmaku/Desktop/ShotLab/shotlab/court.py:89).

The follow-up runs that detector precisely on ball-flight frames [exp_oracle_ceiling_rim.py:105](C:/Users/jmaku/Desktop/ShotLab/tools/exp_oracle_ceiling_rim.py:105), then takes coordinate-wise medians [exp_oracle_ceiling_rim.py:114](C:/Users/jmaku/Desktop/ShotLab/tools/exp_oracle_ceiling_rim.py:114). An orange basketball inside the search band directly violates the detector’s assumption that the ball is outside it.

A read-only comparison against the existing full-clip cache produced:

| Clip | Follow-up rim | Full-clip cached rim | Offset | Gate |
|---|---:|---:|---:|---:|
| 151519220 | (1134,470) | (1244,461) | 110 px | 90 px |
| 152319112 | (1306,462) | (1082,443.8) | 225 px | 90 px |
| 153054813 | (972,469) | (914.5,480) | 59 px | 90 px |

That does not prove the cached rim is correct, but it proves the auto-calibration is context-sensitive by more than the shot gate. The experiment contains no manual verification to decide which calibration is real.

The effect is material. Against the GT centers:

- Clip 1 windows 2618–2750, 5874–5956, and 11180–11224 had 0 points inside the experiment rim gate but 25, 21, and 18 inside the cached-rim gate.
- Clip 2 window 6808–6848 had 0 versus 17.

Thus the low baseline count of 2 can absolutely be calibration-driven. A prior full-clip run reported 7 and 4 shots for the first two clips, versus 2 and 0 in this follow-up: [session_0720_gpu.log:11](C:/Users/jmaku/Desktop/ShotLab/data/out/session_0720_gpu.log:11), [session_0720_gpu.log:12](C:/Users/jmaku/Desktop/ShotLab/data/out/session_0720_gpu.log:12). Detector/model details differ, so that is not a controlled comparison, but it reinforces that “2” is not a stable production baseline.

4. Sparse-window threshold audit

The segmenter requires:

- A tracked point within `shot_gate_px` [court.py:247](C:/Users/jmaku/Desktop/ShotLab/shotlab/court.py:247).
- At least eight selected points and approximately 160 px of launch rise with the defaults [court.py:225](C:/Users/jmaku/Desktop/ShotLab/shotlab/court.py:225), [court.py:269](C:/Users/jmaku/Desktop/ShotLab/shotlab/court.py:269).
- Seven RANSAC inliers [court.py:280](C:/Users/jmaku/Desktop/ShotLab/shotlab/court.py:280).
- Not both release and entry above 78° [court.py:284](C:/Users/jmaku/Desktop/ShotLab/shotlab/court.py:284).

These thresholds are satisfiable on some labeled windows; `min_points=8` is not universally impossible. My GT-only gate replay reproduced the reported oracle split—3/1/0—without needing detector misses. The failures were dominated by rim proximity, launch-drop, RANSAC consensus, and the vertical gate.

There are nevertheless two sparse-input artifacts:

- The 12-frame labeling margin is not tied to frame rate and can omit the below-rim launch needed by a fixed 200 px launch threshold.
- The backward launch walk has no maximum duration or detection-gap boundary; it simply walks through the sorted track until the 200 px condition is met [court.py:264](C:/Users/jmaku/Desktop/ShotLab/shotlab/court.py:264). On the labeled-only track it walked across a large unlabeled interval into an earlier window. Production can suffer the same behavior after long detection gaps.

The experiment is therefore measuring the production segmenter’s response to cropped, discontinuous tracks—not the segmenter’s full-clip ceiling.

5. Tracker fix

The recent fix itself is correct:

- Velocity is now normalized by the frame interval and rescaled to the next interval [track.py:75](C:/Users/jmaku/Desktop/ShotLab/shotlab/phase1_ball/track.py:75).
- Reset seeding leaves `prev_prev=None`, and the immediate `continue` prevents the shared tail update from restoring stale velocity [track.py:54](C:/Users/jmaku/Desktop/ShotLab/shotlab/phase1_ball/track.py:54), [track.py:67](C:/Users/jmaku/Desktop/ShotLab/shotlab/phase1_ball/track.py:67).

An arithmetic probe gives the expected behavior: positions 0 at frame 0 and 20 at frame 2 predict 40 at frame 4, not the old erroneous 60.

But the broader tracker remains fragile:

- It is a single greedy hypothesis.
- A rejected gate force-locks to the highest-confidence blob rather than coasting.
- Confidence changes the effective distance gate because the score is `distance − 30×confidence` [track.py:87](C:/Users/jmaku/Desktop/ShotLab/shotlab/phase1_ball/track.py:87).
- `max_coast=4` is measured in source-frame indices, while production changes stride with FPS and clip length [session.py:428](C:/Users/jmaku/Desktop/ShotLab/shotlab/session.py:428). At stride 3, one missed sampled detection creates a six-frame gap and forces a reset.
- The repository contains no focused `assemble_track` tests; the reported green suite does not directly protect this logic.

Also, the experiment’s “oracle” is not an oracle track. It appends a high-confidence GT candidate to baseline candidates [exp_oracle_ceiling_rim.py:100](C:/Users/jmaku/Desktop/ShotLab/tools/exp_oracle_ceiling_rim.py:100), then still lets the greedy tracker choose. A nearby false candidate can win. A real oracle-track condition must replace the track with GT centers.

### B. Biggest levers

Ranked by expected shot recall per engineering effort:

1. **Verified rim calibration plus segmenter repair — high prize, low-to-medium effort.** Persist a manually checked rim, add an overlay/sanity check, reject dispersed or multimodal auto-rim detections, stop launch walks at temporal gaps, and search a local ascending suffix around each rim event. The observed calibration movement already exceeds the entire shot gate.

2. **Time-aware local arc fitting — medium/high prize, medium effort.** Current RANSAC fits `h(x)` and requires half of the entire walked-back segment to agree [arc.py:112](C:/Users/jmaku/Desktop/ShotLab/shotlab/arc.py:112), [arc.py:153](C:/Users/jmaku/Desktop/ShotLab/shotlab/arc.py:153). That breaks on near-vertical/foreshortened shots and when gather/dribble points outnumber flight points. Fit `x(t)` and `y(t)`, or search for the best temporal suffix before applying RANSAC.

3. **Low-confidence candidates with global single-ball association — medium but unresolved prize, medium effort.** Hungarian assignment is not the right abstraction for one desired ball. Use a Viterbi/beam search over the candidate DAG with velocity, acceleration, size continuity, gap cost, and optional rim-terminal reward. First measure selected-track accuracy; candidate recall alone is insufficient.

4. **Film closer — potentially useful and operationally free, but prize unknown.** It will help genuinely tiny, wholly missed balls and pose quality. This audit contains no unbiased full-clip evidence that those misses dominate lost shots.

5. **TrackNet/retraining/full fusion — high effort, currently weak evidence.** The selected-window 99% result argues against starting here, but selection bias means it cannot take temporal detection permanently “off the table.”

6. **Make/miss changes — important for outcome accuracy, approximately zero shot-recall gain.** The code itself describes make/miss as low-confidence when post-rim tracking is sparse [make.py:11](C:/Users/jmaku/Desktop/ShotLab/shotlab/make.py:11).

### Single decisive next experiment

Use the full `PXL_20260720_151519220` clip. Independently hand-mark every actual attempt with release/rim timestamps and one manually verified rim; do not seed this list from current detections. Densely label the ball from gather through rim for those attempts. Replay five conditions and match outputs to attempt IDs:

1. Baseline candidates → current tracker → current segmenter.
2. Conf-0.01 candidates → current tracker → current segmenter.
3. Baseline candidates → GT-nearest/oracle association → current segmenter.
4. GT-only track → current segmenter.
5. GT attempt windows → arc fitting.

Report attempt recall and false positives after each stage. Conditions 2–3 size tracking; 3–4 size detection; 4–5 size segmentation/RANSAC. This is far more decisive than comparing raw single-digit output counts.

**VERDICT:** The fallback-result retraction is correct, and the project is right that the rim result is inconclusive; however, “film closer is the biggest lever” is not supported. The biggest evidenced production lever is verified rim calibration plus local, gap-aware shot segmentation, with greedy association next. The single most decisive next experiment is a full-clip, hand-counted attempt evaluation containing a GT-only oracle track and stage-by-stage ablations—not another injected-candidate count inside detector-selected label windows.