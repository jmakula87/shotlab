## Verdict

Do not build active learning first. For the next preregistered test, manually label every detected shot through a fast, blind, keyboard-driven clip viewer.

You are conflating two jobs:

- Full-session hand counting is needed to validate detection recall.
- Make/miss labels for the knee test only require reviewing the already-cut detected shots.

The one-hour workflow is therefore not the unavoidable price of the knee test. At 122–137 detected shots, a purpose-built viewer should turn outcome labelling into minutes, not an hour. More importantly, predicted outcomes are not acceptable substitutes for human labels in this confirmatory test.

There is a second, more serious problem: the preregistered replication is badly underpowered. With 80 balanced observations, one-sided α=.05 requires approximately \(d=0.37\) merely to achieve significance. Thus the `d ≥ .25` condition is nonbinding. If the true effect is exactly \(d=.25\):

- n=80: about 30% power.
- n=122: about 40% power.
- Roughly n=396 is needed for 80% power.

Calling a non-significant result at n=80 a “failed replication” or dead-lettering the candidate would kill a genuine \(d=.25\) effect roughly 70% of the time. Fix that decision rule before collecting the next session.

## 1. The priority

The immediate priority should be:

1. Build a blind, one-key-per-shot human labeller for detected review clips.
2. Separate detection validation from outcome labelling.
3. Repair the preregistered power/failure rule and add a temporal-dependence sensitivity test.
4. Only then investigate label reduction and session-invariant features.

The existing dashboard is close, but unsafe for confirmatory labels. It presents predictions, says the model is “~87% accurate,” preselects an answer, and offers “Save & next” ([dashboard/app.py](C:/Users/jmaku/Desktop/ShotLab/dashboard/app.py:1232), [dashboard/app.py](C:/Users/jmaku/Desktop/ShotLab/dashboard/app.py:1273), [dashboard/app.py](C:/Users/jmaku/Desktop/ShotLab/dashboard/app.py:1281)). That is a rubber-stamping machine, not independent annotation.

For the knee test:

- Hide every model prediction.
- Autoplay the rim outcome.
- Use `m`, `n`, `b/not-shot`, `u/unsure`.
- Save immediately and advance.
- Randomize or interleave clips if practical.
- Record label source, timestamp, and whether the answer was changed.
- Reveal model disagreements only after the blind human pass.

The repo already has 3-second-pre/1.5-second-post windows ([feelreview.py](C:/Users/jmaku/Desktop/ShotLab/shotlab/feelreview.py:56)). You do not need a better classifier to exploit them.

## 2. Correct learning-curve protocol

There are two distinct estimands. Do not blur them.

### Deployment-matched curve

If deployment means “all new-session clips are already available; label k shots distributed across them; classify the remainder of those same clips,” then:

- Draw k across all clips.
- Enforce a quota proportional to clip size, with every clip represented.
- Spread selections across time within each clip.
- Do not stratify on make/miss; those labels are unknown at acquisition time.
- Evaluate on the unlabelled complement.

This is the operationally relevant curve. It benefits legitimately from seeing each clip’s lighting and geometry.

### Cross-clip curve

If deployment means “label earlier clips and classify a completely unseen later clip,” then LOCO controls the curve:

- Hold one clip entirely untouched.
- Draw k only from the other three clips, again clip-balanced and time-spread.
- Train and score on the held clip.
- Repeat subset draws for each held clip.
- Pool only out-of-fold predictions and report each clip separately.

You cannot first draw k across all four clips and then call the result LOCO.

### The larger limitation

Neither curve establishes that the chosen k works on a brand-new session. The outer unit for that claim is session:

```text
hold out session S
    expose k labelled shots from S for adaptation
    fit using those k
    test on untouched shots from S
repeat over sessions
```

With only two labelled sessions, “k≈30 is enough in future sessions” is not estimable. A curve on 2026-07-29 is descriptive of 2026-07-29.

Also, if you inspect the curve and select the smallest successful k on the same held-out rows, that k is optimistically selected. Use the existing session exploratorily to choose k, then freeze k and evaluate it once on the next session—or reserve a final untouched audit set.

Report more than accuracy:

- Sensitivity and specificity.
- Balanced accuracy.
- Worst-clip accuracy.
- Calibration/Brier score.
- Confusion by clip and time block.
- Effect sensitivity after adversarially flipping the plausible number of erroneous outcomes.

The committed full-data LOCO result is 89–90%, with per-clip accuracy 92/87/94/82 and a binomial interval around 82–94%. Four clips are too few for a dependable cluster-level interval.

I could not compute a new feature-level curve: this managed read-only environment allowed repository inspection but rejected Python execution, including the existing evaluation script. No learning-curve numbers beyond the committed results are newly computed here.

## 3. Active learning

I would not use it at k≈30.

The classifier has seven inputs but is an 80-tree, depth-2 GradientBoosting model with `subsample=0.8` ([make_visual.py](C:/Users/jmaku/Desktop/ShotLab/shotlab/make_visual.py:178)). At k=30, every tree is fitted on about 24 rows. Its probability output is not calibrated anywhere: training reports only AUC and accuracy ([train_make_model.py](C:/Users/jmaku/Desktop/ShotLab/tools/train_make_model.py:64)). Therefore distance from 0.5 is not yet a trustworthy measure of epistemic uncertainty.

Likely small-k failure:

- The initial model places the wrong boundary.
- Uncertainty sampling concentrates labels near that wrong boundary.
- Stable, confidently wrong regions remain unseen.
- ROI failures and unusual misses are overrepresented.
- The resulting training distribution no longer resembles deployment.

One correction to your trap statement: an uncertainty sample is not inherently optimistic. Predictions on queried points before refitting will generally look pessimistic because those points were selected as difficult. Scoring the refitted model on those same labelled points is optimistic because it is resubstitution. Both are invalid estimates of deployment accuracy.

To compare active against random sampling honestly:

- Reserve a fixed 20–25% audit set within every clip before acquisition.
- Never query or train on it.
- Give active and random methods the identical initial seed.
- Compare them on exactly the same audit rows over many repeated splits.
- Report \(P(\text{active}<\text{random})\), not merely mean accuracy.

At n=122, a 20% audit contains only about 24 shots. At 89% accuracy its nominal 95% margin is about ±13 percentage points. That is barely capable of distinguishing active learning from random sampling. You would need aggregation across several sessions.

My recommendation:

- k<40: systematic random sampling across clips/time.
- k≈40–80: perhaps compare active learning experimentally, but do not deploy it for the confirmatory labels.
- k≈80–120: the possible saving is too small to justify the bias and validation machinery; direct human labelling wins.

## 4. Cheaper framings

### Direct human labels

This is the missing cheap framing. Outcome recognition from a short rim clip is a much easier human task than scanning full footage for attempts. Use the model only as a second-pass disagreement detector after blind annotation.

### Session-invariant features

Plausible, and not disproved.

The repo normalized only three orange pixel-count features by rim area ([make_visual.py](C:/Users/jmaku/Desktop/ShotLab/shotlab/make_visual.py:126)). It did not normalize:

- Fixed HSV orange/white thresholds.
- Exposure and white balance.
- Motion magnitude.
- Net/background texture.
- Camera timing or FPS.
- ROI-placement errors.
- Background changes.

Thus cross-session failure is unsurprising. Two sessions do not justify the universal conclusion that every session must always be relabelled.

Promising tests include:

- Per-shot changes relative to the pre-rim baseline rather than absolute counts.
- Orange/white fractions instead of masses.
- Adaptive color thresholds from the session or ROI.
- Motion normalized by local background noise and frame interval.
- Unlabelled-session feature quantile alignment.
- A session-standardized model evaluated with session as the outer holdout.

This is the only research direction that could eliminate recurring labelling, but it requires more sessions. It should not block the next knee test.

### Self-training

Reject it now. A 55–62% cross-session model can be confidently wrong under covariate shift. Pseudo-labelling would turn that systematic error into training truth.

### Streaks and sequence structure

The CSVs contain conspicuous runs: clip 2 has attempts 12–22 all recorded as misses; clip 3 has attempts 7–15 all misses. That verifies local clustering in this dataset, though I did not calculate a formal autocorrelation test.

Do not exploit this by copying neighboring outcomes. Knee bend, fatigue, location, and time may also change in blocks; smoothing labels would manufacture precisely the outcome–mechanics association you want to test.

Instead, temporal dependence attacks the preregistered permutation test. Shuffling labels freely within a clip assumes within-clip exchangeability. Pre-register a block-permutation or circular-shift sensitivity analysis, or model time/block explicitly.

### Audio

Low priority. The repo documents roughly 0.6 AUC, and the validated profile disabled audio after it was wrong on 13/20 fills. Audio may help a human reviewer, but it is not a credible primary automation path yet.

## 5. The most dangerous silent failure

It already exists:

- `--make-model auto` knowingly chooses the newest previous-session model even though the code comments say cross-session performance is only 58–62% ([build_session.py](C:/Users/jmaku/Desktop/ShotLab/build_session.py:171)).
- The provenance guard warns only when the target clips overlap the training clips. A brand-new session does not overlap, so the known bad-transfer case produces no warning.
- The dashboard then advertises “~87% accurate,” shows the model’s call, preselects an outcome, and lets the owner save it.
- `make_truth.json` does not distinguish a genuinely observed label from a default that was rubber-stamped.
- `apply_make_model.py` similarly prints a hard-coded “honest held-out was ~87%” statement ([apply_make_model.py](C:/Users/jmaku/Desktop/ShotLab/tools/apply_make_model.py:75)).

That can silently convert a known 55–62% cross-session classifier into apparently “verified” ground truth.

Fail closed:

- Never auto-load a model from another session.
- Never preselect a human ground-truth answer.
- Never display model accuracy without naming training session, evaluation session, and protocol.
- Make provenance mandatory, not optional.
- Keep predicted, human-confirmed, and blindly human-labelled outcomes as distinct fields.
- For the knee replication, accept only blind human labels.

Even 89% overall accuracy is unsafe for the candidate test. Under approximately symmetric 11% label error, a true \(d=.25\) is attenuated roughly by \(1-2(0.11)=0.78\), giving observed \(d≈.20\). Worse, if the 11% errors concentrate among extreme knee values, they can create or erase the effect while overall accuracy still looks excellent.

## Verified vs hypothesized

Verified in the repo:

- Seven raw visual features and the exact 80-tree model configuration.
- Only orange masses received rim-area normalization.
- 89–90% within-session LOCO, 82–94% interval, 92/87/94/82 per clip.
- 58–62% cross-session after normalization; adding the old session hurt.
- No probability-calibration evaluation.
- Existing auto-cut review clips.
- Existing confirm UI exposes predictions, preselects outcomes, and hard-codes ~87%.
- Auto model resolution selects the previous-session model on a new session.
- Resubstitution protection does not guard cross-session transfer.
- Long miss runs in two hand-count CSVs.
- Audio’s documented weak performance.

Hypotheses requiring new data:

- Direct clip labelling will take only several minutes.
- Relative/adaptive features will transfer across sessions.
- Active learning will underperform random sampling at small k here.
- The outcome sequence has statistically meaningful autocorrelation.
- Make/miss errors are differential with respect to knee bend.

Bottom line: ship the blind fast human labeller, not confirm-only active learning. And do not call n=80 a decisive replication of \(d=.25\); the statistical gate is currently a bigger threat to the result than labelling cost.