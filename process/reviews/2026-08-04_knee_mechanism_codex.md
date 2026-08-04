## Verdict

The published pooled `d=+0.40` does not survive the window-truncation attack intact.

The metric is strongly contaminated by release-window truncation. On 07-29—the discovery session—the entire positive effect is concentrated in shots whose inferred dip bottom is at the left edge of the window. Removing those shots changes `d≈+0.40` to approximately zero. On 07-20, truncation explains part, but not all, of the effect: a residual `d≈+0.31–0.36` remains.

After excluding likely truncated shots, the approximate pooled estimate is `d≈+0.15`, 95% CI roughly `[-0.19,+0.48]`. Thus:

> The replicated pooled finding is not currently defensible as “less knee bend causes makes.” A weaker residual signal survives in the replication only, with no established mechanism.

This is not merely “gather duration” in the ordinary sense. The sharper failure is disagreement between two release detectors, which can leave as little as 2–5 frames before the internally detected release.

## 1. Window truncation: the main result

The pseudo-shot contains only the externally detected release and post-release frames ([flare_report.py:238](/C:/Users/jmaku/Desktop/ShotLab/tools/flare_report.py:238)). `compute_form` then independently redetects release and starts its metric window at the external release minus 20 frames ([form.py:439](/C:/Users/jmaku/Desktop/ShotLab/shotlab/phase2_pose/form.py:439)).

Therefore, if the internal release is 18 frames earlier than the external release:

```text
window start = external release − 20
internal release = external release − 18
usable pre-release history = 2 frames
```

The true dip is almost guaranteed to be absent.

The JSON fortunately emits that disagreement as `release_frame_delta` ([flare_report.py:269](/C:/Users/jmaku/Desktop/ShotLab/tools/flare_report.py:269)). It also emits `tempo_dip_to_release_s`, whose 2D bottom frame is calculated from the same clipped span ([form.py:548](/C:/Users/jmaku/Desktop/ShotLab/shotlab/phase2_pose/form.py:548)). I used these to estimate distance from the dip bottom to the left boundary, with about one-frame precision.

### Verified without outcome joins

Across every stored row with knee and tempo:

| Session | Likely boundary minima | Mean 3D knee | Safe-window mean |
|---|---:|---:|---:|
| 07-29 | 20/130 | 130.5° | 101.6° |
| 07-20 | 24/95 | 124.1° | 107.5° |

More damningly, for rows where the internal release was at least 15 frames earlier than the external release:

| Session | n | Mean 3D knee | Other rows |
|---|---:|---:|---:|
| 07-29 | 6 | 156.5° | 103.8° |
| 07-20 | 8 | 159.2° | 107.4° |

Correlation between knee angle and `release_frame_delta` was:

- 07-29: `r = −0.62`
- 07-20: `r = −0.72`

The negative sign means that the earlier the internally detected release lies relative to the pseudo-shot anchor, the shallower the reported bend becomes. This is exactly the predicted software artifact.

### Outcome-stratified sensitivity analysis

Because process execution was blocked in this read-only environment, I could not rerun the Python harness or MediaPipe. I reconstructed the hand-count joins directly from close-frame times, registered synchronization offsets, and hand-counted rim frames. These sensitivity joins reproduce the reported effect closely:

- 07-29: 49/56 rather than the exact 49/61, `d=+0.404`
- 07-20: 37/44 rather than 38/44, `d=+0.422`

So these outcome-stratified values are strong reconstructions, not exact reruns.

| Exclusion | 07-29 d | 07-20 d | Approx. pooled d |
|---|---:|---:|---:|
| None | +0.40 | +0.42 | +0.41 |
| Exclude bottom ≤0.75 frame from boundary | −0.00 | +0.36 | +0.16 |
| Exclude bottom ≤3.25 frames from boundary | +0.01 | +0.31 | +0.15 |

The corresponding pooled CIs are approximately `[-0.16,+0.48]` and `[-0.19,+0.48]`.

For 07-29 specifically:

- Likely-boundary group: 12 makes / 6 misses.
- Non-boundary group: 35 makes / 50 misses.
- Non-boundary knee effect: `d=−0.003`.

That is an artifact signature, not a robust bend-depth result.

The 07-20 residual prevents me from saying “the whole phenomenon is certainly fake.” But the discovery evidence—the evidence that made pooling decisive—is effectively gone.

## 2. Is knee a proxy?

For the near-harness sensitivity joins:

| Metric | r with 3D knee, 07-29 | r, 07-20 | Its own d, 07-29 / 07-20 |
|---|---:|---:|---:|
| Tempo | −0.64 | −0.51 | −0.16 / −0.19 |
| Jump height | −0.15 | −0.26 | −0.10 / −0.08 |
| Release height | −0.17 | −0.04 | −0.19 / +0.16 |
| Balance drift | −0.13 | −0.40 | +0.32 / −0.03 |
| 3D elbow | +0.15 | +0.12 | +0.21 / +0.23 |
| 2D knee | +0.79 | +0.80 | +0.23 / +0.27 |

No ordinary measured metric carries both the same make effect and a consistently strong relationship with 3D knee.

Tempo is the closest candidate, but it has the opposite outcome direction: makes were slightly quicker, while higher knee angles indicate less bend. Thus knee is not merely tempo under another name.

What it is proxying is more specific:

> the amount of valid pre-release history remaining after the two release detectors disagree.

That variable is not currently reported as a metric except indirectly through `release_frame_delta`, tempo, and boundary position.

I did not obtain an exact shot-form/setup-adjusted result. My reconstructed 07-29 context join was incomplete and was especially likely to omit the pathological release candidates, so its within-type estimates are not trustworthy enough to promote. Fatigue looked weak in the usable reconstructed context (`corr(time,knee)=−0.13`, `corr(time,made)=−0.07`), giving no evidence for a shared fatigue trend.

## 3. Is monocular 3D accurate enough for 5.4°?

No—not as an absolute per-shot difference.

Recent direct validation found BlazePose World’s mean absolute 3D knee-angle error around `17.2°`, with roughly `146 mm` mean 3D joint-position error, even though it was the best direct monocular estimator tested. [Scientific Reports validation study](https://www.nature.com/articles/s41598-025-22626-7)

A 2026 lateral-view validation preprint reported dynamic knee-angle MAE `7.07°`, a systematic bias of `−5.12°`, and static RMSE `15.49°`; bias was angle-dependent and unstable across its very small retest sample. [JMIR preprint](https://preprints.jmir.org/preprint/102399)

Your 5.4° contrast is below those per-observation error scales. Shooter size at 30–35% of frame does not improve that conclusion.

That does not mathematically prevent detecting a 5.4° group mean: independent, nondifferential noise averages down and ordinarily attenuates `d`. But here the error is demonstrably systematic with release-window geometry. Averaging does not cure systematic truncation.

MediaPipe does indeed define world landmarks in meters around the hip midpoint, as your description says. That definition does not make them calibrated motion-capture coordinates. [MediaPipe pose documentation](https://mediapipe.readthedocs.io/en/latest/solutions/pose.html)

## 4. Reliability substitutes from disk

Two empirical checks were possible.

First, 2D and world-3D knee minima correlate `r≈0.79–0.80` in both sessions. That is encouraging internal consistency, but it is not independent reliability: both use the same MediaPipe landmark inference, visibility gates, smoothing, window, and release anchor.

Second, I matched simultaneous close- and wide-camera 3D knee values for 36 shots on 07-29:

- Pearson `r = 0.05`
- concordance correlation `CCC = 0.029`
- MAE `30.6°`
- RMSE `37.9°`
- only 25% agreed within 10°

This demolishes any practical claim of camera invariance across these two views. It does not prove the close camera is unreliable by itself—the wide camera is known to be badly under-resolved—but it shows that “world landmarks” do not make camera choice irrelevant.

The strongest defensible reliability study available with existing video would be deterministic remeasurement of the same shots under perturbations:

- original versus slightly shifted/cropped frames;
- smoothed versus unsmoothed landmarks;
- model complexity variants;
- 20/30/40-frame windows;
- external versus internal release anchoring;
- left versus right knee where both are visible.

Those are repeated measurements of identical physical events. They permit ICC, within-shot SD, and a smallest-detectable-change calculation without filming anything new. The current JSON cannot support them because it stores summary minima, not frame-level landmarks.

## 5. What physical mechanism survives?

A physically plausible story exists, but it is not verified:

- In a solo workout, a shallow, compact load may indicate a clean catch and rhythm shot requiring little corrective force.
- Deeper bends may occur on low catches, recovery steps, balance corrections, or self-created/off-rhythm attempts.
- Excess bend can add unnecessary vertical impulse and additional timing degrees of freedom; “load the legs” does not imply “maximize knee flexion.”
- A shallower load could keep the head and release platform more stable on comfortable-range shots.

None of that is evidence for the mechanism. Distance has been ruled out, but gather quality, low-catch height, foot-set state, vertical impulse, and trunk motion have not been measured cleanly.

There is no biomechanical reason less bend must be impossible. There is also no evidence here that less bend itself improves accuracy.

## Bottom line

What I verified:

- The estimator is severely window-sensitive.
- Large negative release-detector disagreement produces extremely shallow reported bends: roughly 156–159° versus 104–107°.
- The discovery-session make/miss effect disappears when likely boundary-truncated shots are removed.
- The replication retains a smaller positive effect after the same exclusion.
- Ordinary measured metrics do not consistently explain knee.
- Monocular world-landmark error is larger than the observed 5.4° per-shot contrast.
- Cross-camera agreement is terrible in this footage.

What I am hypothesising:

- The residual 07-20 effect may reflect compact, rhythmically clean solo-workout shots.
- It could equally be residual release/window bias, shot-type composition, or chance.

The honest surviving claim is:

> `knee_bend_3d_deg` as currently implemented is partly a release-anchor/window-coverage metric, not a clean bend-depth metric. The promoted pooled finding should be withdrawn. A weaker “less bend on makes” residual remains in 07-20, but it is unresolved and has no verified physical mechanism.