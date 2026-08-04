# Adversarial review: per-session make/miss labelling cost

Reviewer: Fable, 2026-08-04. Repo read-only; all numbers below computed fresh from
`feat_cache_scaleinv.json` (this scratchpad — the exact 122-shot/4-clip feature+label
matrix the 89% LOCO and `models/make_visual_0729.joblib` were computed from; provenance
verified against `loco_0729.py`, `fit_0729_model.py`, `models/make_visual_0729.trained_on.json`).
Probe script: `probe_labelling.py` (same directory).

## Verdict in one paragraph

The priority is **half right**. Cheap labelling is worth building, but two of the four
proposals (learning-curve-to-find-small-k, active learning) are answered by measurement
and the answers are "no cheap plateau" and "not worth it". The learning curve I actually
computed never plateaus below the 89% ceiling — **k=30 buys 82%, and you need all ~120
labels to reach 89%** — so "ask for fewer labels" does not solve the problem; it just
buys a worse label set. Meanwhile the pre-registered knee test **cannot run on model
labels anyway** (power 0.30 → 0.22 at the pre-registered minimum effect; and model
errors are non-random, which biases rather than merely attenuates d). The two things
worth doing instead: (1) a **confirm-every-shot UI** over the already-cut review clips —
~120 one-keystroke decisions ≈ 15–20 min, hand-label quality, no ML risk; (2) the
genuinely new result from this review: **the session-specificity mostly lives in the
GBM, not the features** — a per-session z-score + logistic regression transfers
0720→0729 at **81%** (AUC 0.87) with **zero** new-session labels, vs 58% for the
shipped GBM. That gives every future session free ~80% pre-labels on day zero and is
the one lever that could actually shrink the problem — pending confirmation on a second
session pair.

## What I VERIFIED in the repo (vs hypothesised)

- `shotlab/make_visual.py`: 7 features, GBM(80 trees, depth 2, lr .08, subsample .8)
  behind a StandardScaler; REF_RR normalisation present, corpse-marked measured-inert
  in PROJECT_NOTES.
- `tools/train_make_model.py`: 5-fold + LOCO reported; reads `make_truth.json`.
  **`data/out/session_0729/make_truth.json` DOES NOT EXIST** — only `session_0710` has
  one. The 07-29 labels live in `process/handcount/PXL_20260729*_attempts.csv` and are
  joined to detections by rim_frame within tol=30 (greedy nearest, `refit_make_model.py`
  pattern). The brief's stated path is wrong; see failure mode 4.
- `tools/hand_count.py` docstring: attempts must be counted **fresh by eye, never seeded
  from detections** — a 2026-07-22 adversarial ruling. This constrains any new tool.
- 122 labelled 0729 shots (58 makes, 48%), 96 labelled 0720 shots — matches
  `trained_on.json` and the PROJECT_NOTES table. Reproduced LOCO = 88.5% from the cache.
- `session_shots.csv` for 0729 already carries `knee_bend_3d_deg`, `made`, `make_conf`
  columns — the knee-test join is one CSV, no new plumbing.
- Hand-count inter-attempt gaps (all 4 clips): min 58 frames (once, same outcome),
  otherwise ≥92 — at tol=30 an outcome-flipping mis-join is currently near-impossible,
  but the margin is one bad rim_frame away (see failure mode 3).

Everything in the numbered sections below is computed, not argued, except where marked
HYPOTHESIS.

## 1. Learning curve (computed, 200 reps/point, random k spread across the 4 clips)

Accuracy of the model on the (122−k) unlabelled remainder; "eff" = whole-session label
accuracy if k are hand-labelled and the model fills the rest.

| k | GBM acc | p10 | LR acc | eff. session labels |
|---|---|---|---|---|
| 10 | .691 | .562 | .721 | .716 |
| 15 | .738 | .617 | .753 | .770 |
| 20 | .774 | .686 | .766 | .811 |
| 30 | .821 | .761 | .793 | .865 |
| 40 | .844 | .780 | .799 | .895 |
| 60 | .869 | .823 | .812 | .933 |
| 80 | .884 | .833 | .817 | .960 |
| 100 | .893 | .818 | .830 | .981 |

**The k≈30 hypothesis is dead.** k=30 leaves 18% error on filled shots and the curve is
still climbing at k=100. There is no elbow because the ceiling itself (89%) is only 11%
error — the model never gets good enough for "few labels + fill" to approach hand
quality. Also note the p10 column: at k=30 one run in ten lands ≤76%.

Clip-blocked variant (labels concentrated in whole clips, tested on unlabelled clips):
1 clip (k=24–34) → .846 [.775–.901]; 2 clips → .880; 3 clips (=LOCO) → .888.
Within-session clip effects are mild — concentrating labels in the first clip costs only
~2–4 pp vs spreading. Cross-SESSION is where transfer dies, not cross-clip.

### Protocol errors you would otherwise make (question 2)

1. **Random-k across clips is the deployment-matched estimand** for this tool, because
   in deployment the model predicts other shots of the SAME session/clips it was
   labelled on. LOCO is the right honesty bar for a model shipped to a clip with zero
   labels — and I computed both; they differ by ~2–4 pp within-session. Do not build the
   curve LOCO-style and then claim it as the deployment number, or vice versa; label
   both curves with their estimand (your own standing rule: name the scope).
2. **Do not stratify the k draw by label** — at labelling time you don't know the
   labels. My curves use unstratified draws; ~0.3% of k=10 draws are single-class
   (I skipped 1/200; a real tool must handle it, not crash or silently emit a
   constant classifier).
3. **The curve is model-dependent, not a property of the problem**: LR beats GBM at
   k≤15, GBM wins from k≥20. If you sweep k with one hyperparameter set you are
   measuring that model's sample-efficiency, and mv.train's 80×depth-2 trees at k=15 is
   a different beast than at k=120.
4. **The denominator is detected shots only.** 143 attempts → 122 labelled = the join
   drops ~15% (undetected + unmatched + airballs). Model-fill covers only detected
   rim-reaching attempts; any session FG% derived from it is biased vs the hand count.
   Fine for the knee test (3D pose exists only for detected shots anyway), wrong for
   anything that quotes a make rate.

## 2. Active learning (computed, uncertainty vs random, batch 5, fixed 30% eval split, 60 reps)

| budget | random | uncertainty | delta |
|---|---|---|---|
| 20 | .794 | .795 | +.001 |
| 30 | .822 | .842 | +.020 |
| 40 | .841 | .861 | +.020 |
| 60 | .886 | .890 | +.004 |

Peak gain **+2.0 pp**, at budgets 30–40 only, gone by 60. The known trap measured: CV
accuracy computed ON the acquired set was **−9.2 pp ± 9.2 biased (pessimistic)** here —
uncertainty sampling hoards hard shots, so the acquired set under-reports. The sign is
the opposite of the classic optimism fear, but the disqualification is the same: never
measure on the acquired pool. **Verdict: not worth building.** The entire achievable
gain is ≤2 pp on a task whose total budget is ~120 labels ≈ 20 minutes of keystrokes,
and it adds a fit-predict loop plus a biased-measurement footgun to a labelling tool.
The one salvageable piece: SORT the confirm queue by |p−0.5| so disagreement-prone
shots get the owner's freshest attention — ordering is free and has no protocol cost
as long as every shot is still confirmed.

## 3. Confirm-only UI risk (computed from LOCO probabilities)

The model's errors are **confident**: mean |p−0.5| = 0.336 for the 14 LOCO errors vs
0.444 for correct calls. Auto-accept bands:

| threshold | coverage | errors auto-accepted |
|---|---|---|
| .60 | 98% | 13 of 14 (10.9% of band) |
| .80 | 92% | 10 (8.9%) |
| .90 | 80% | 4 (4.1%) |
| .95 | 66% | 2 (2.5%) |

A "review only the unsure" UI at any sane threshold silently keeps most of the errors.
At 0.95 you still ship 2 wrong labels while re-reviewing a third of the session — at
which point you have spent most of the full-review cost anyway. **Confirm-only is fine
ONLY as confirm-EVERY-shot** (model call as a sort key and a default keystroke, human
eyes on 100% of clips). Anything that skips human viewing of a subset is the silent-bad-
labels machine of question 5.

## 4. Cheaper framings (question 4) — one is real

- **Session-invariant features: PARTIALLY YES, and it's the finding of this review.**
  Train on 0720, test on 0729 (zero 0729 labels):
  | variant | acc | AUC |
  |---|---|---|
  | shipped-style GBM, raw | .582 | .680 |
  | plain LR, raw | .713 | .846 |
  | LR + per-session z-score | **.811** | .867 |
  | LR + per-clip z-score | **.828** | .852 |
  | LR + per-session rank-normal | .762 | .842 |
  | CORAL + LR | .770 | .860 |
  | GBM + per-session z-score | .598 | .706 |
  Reverse direction (0729→0720): zLR .781 (.792 rate-matched), AUC .856 — it holds both
  ways. The per-session z-score uses only the session's own UNLABELLED feature
  distribution, so it is label-free at deployment. Read: **the feature DIRECTIONS
  survive the session change; the GBM's non-linear split thresholds are what's
  session-specific.** "Make/miss is session-specific" is a true statement about the
  shipped model, not about the cues. Caveats, stated plainly: this is ONE session pair;
  81% ≠ 89%; and it does NOT remove the need for hand labels on pre-registered tests.
  What it does do: gives a brand-new session ~80%-accurate pre-labels on day zero
  (worth ~30 hand labels for free, per the learning curve), i.e. it breaks the
  chicken-and-egg of a confirm UI that needs a session model that needs labels.
  It must be re-verified on the next fresh session before being trusted as a pattern.
- **Self-training: dead.** k=20: −0.9 pp; k=30: −0.1 pp (100 reps). The confident
  pseudo-labels are the shots the model already gets right; the wrong-but-confident
  ones (see §3) get amplified.
- **Make/miss dependence across a session**: the only label-free session statistic is
  the make rate, and a rate-matched threshold on the raw model bought .582→.623 —
  noise-level. Nothing there.
- **Mixing old-session rows into a k-label fit**: measured negative at every k ≥ 15
  (k=30: −4.2 pp GBM, ±0 LR). Confirms the repo's own "0720+4K is worse" finding at
  small k too. Don't.

## 5. The knee test does not tolerate model labels (computed)

One-sided alpha .05, n=80, 48/52 split, symmetric-error attenuation (1−2e):

| label error | power @ true d=.25 | @ d=.40 | @ d=.60 |
|---|---|---|---|
| 0% (hand) | .30 | .56 | .85 |
| 11% (89% model) | .22 | .40 | .67 |
| 13.5% (k=30+fill) | .20 | .37 | .62 |

Two things. First, **even with perfect labels the pre-registered test is underpowered
at its own minimum effect** (power .30 at d=.25) — it only reliably detects effects
well above the registered floor; know that before reading a null. Second, model labels
turn an already-marginal test into a coin flip at d=.40. And the attenuation row is the
BEST case: it assumes label errors independent of kinematics. HYPOTHESIS (mechanism
argued, not measured): make_visual errors concentrate in visually atypical rim events
(long rebounds, rattle-outs), which plausibly correlate with entry angle and hence with
form features — correlated label error does not attenuate d, it BIASES it, in an
unknown direction, inside a pre-registered one-shot test. **Hand-confirm 100% of the
labels for the knee-test session. This is non-negotiable and it costs ~20 minutes.**

## 6. Failure modes that ship silent bad labels (question 5, concrete)

1. **Auto-accept above a confidence threshold** — quantified in §3. 10.9% of the
   auto-accepted band is wrong at t=.6; errors are confident by construction.
2. **Anchoring**: showing the model's call BEFORE the human watches the clip converts
   "label" into "agree". The 89% model makes the human's effective error rate converge
   toward the model's on exactly the shots where the model is wrong-and-confident.
   Mitigation that costs nothing: show the clip first, take the keystroke, THEN reveal
   agreement; or at minimum randomise 10% of shots to blind re-label and measure the
   owner-vs-owner and owner-vs-model disagreement rates every session.
3. **The rim_frame join**: labels attach to detections by nearest rim_frame within 30
   frames. Today's data is safe (min gap 58f, same outcome), but the `union_tol_sweep`
   in the eval jsons shows pipeline-vs-hand rim_frames disagree by 10–30 frames
   routinely (tol=10 matches only 54%). A future session with a quick putback or a
   double-pump 40 frames after a miss WILL cross-join. Add an assertion: nearest match
   must beat second-nearest by ≥2× tol, else flag for human.
4. **Two truth stores already diverge**: `make_truth.json` (clip|shot_idx keyed,
   session_0710 only) vs `process/handcount/*_attempts.csv` (rim_frame keyed, 0720/0729).
   The brief itself cited a `session_0729/make_truth.json` that does not exist. A new
   tool that writes a third store, or reads the wrong one, produces labels that
   silently drift from what training reads. Pick ONE canonical store and make
   `train_make_model.py` and the new tool read the same file.
5. **Resubstitution creep in the UI**: pre-labels for session S generated by the model
   trained on session S look ~95% right and teach the owner to rubber-stamp. The
   `trained_on.json` guard covers `eval_ablations` only; the labelling UI must also
   read it and refuse (or at least banner) pre-labels from a model whose training set
   includes the clips being labelled.
6. **Seeding attempt COUNTS from detections** — forbidden by the 07-22 ruling recorded
   in `hand_count.py`. Keep the two jobs separate forever: fresh-eye attempt counting
   (rare; only when a detection number is being validated) vs make/miss labelling of
   detected shots (routine; review clips fine). A merged "labelling tool" that shows
   detections while counting attempts quietly voids every future recall number.
   Corollary the brief gets wrong: **a make/miss labelling tool does nothing for
   validating the 96% detection figure** — that gate needs a fresh hand count of
   attempts, which is the expensive scrubbing part of the hour, and no ML shortcut is
   admissible for it by the project's own rules.

## 7. What to actually do (priority-ordered)

1. **Build the confirm-every-shot UI** over the existing review clips: play clip,
   one keystroke (m/n), model call revealed after the keystroke, queue sorted by
   |p−0.5|, writes to the ONE canonical truth store, refuses resubstitution pre-labels
   without a banner. ~120 shots ≈ 15–20 min/session, hand-grade labels. This alone
   removes most of the hour, because the hour was mostly scrubbing raw video — which
   only detection-eval sessions need.
2. **Label the knee-test session 100% by hand** through that UI. Do not model-fill a
   pre-registered test (§5).
3. **Re-verify the zLR transfer on the next session** (one command, zero extra
   labelling: fit zLR on 0729, predict the new session's confirmed labels). If it holds
   near 80%, adopt zLR as the day-zero pre-labeller for all future sessions and record
   it; if it collapses, the 81% was one lucky pair — record that instead.
4. **Skip**: active-learning machinery (≤2 pp, §2), self-training (negative, §4),
   further learning-curve tooling (§1 is the answer: there is no cheap k).

Files: probes in `probe_labelling.py` (this scratchpad), data provenance
`feat_cache_scaleinv.json` = features behind `models/make_visual_0729.joblib` per
`fit_0729_model.py`; repo files read: `shotlab/make_visual.py`,
`tools/train_make_model.py`, `tools/apply_make_model.py`, `tools/refit_make_model.py`,
`tools/hand_count.py`, `tools/eval_ablations.py` (excerpts), `PROJECT_NOTES.md`
(lines ~560–668), `process/handcount/*_attempts.csv`, `models/make_visual_0729.trained_on.json`.
