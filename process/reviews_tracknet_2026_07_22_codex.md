## Verdict

**Measure first, then build a reduced version only if the oracle ceiling is meaningful.**

The draft is directionally sound but over-scoped. Do not begin with a TrackNet/WASB architecture bake-off, tiling system, or three fusion modes. First establish whether perfect recovery of the missing apex frames actually restores dropped shots or materially changes metrics. Then test the existing pretrained **WASB basketball checkpoint** before training anything.

A better order is:

1. Oracle gap-fill evaluation.
2. Zero-shot pretrained WASB-basketball evaluation.
3. If both succeed, integrate candidate-level temporal supplementation.
4. Only then collect more sessions and fine-tune.

Cut TrackNetV3, temporal-YOLO, and heatmap fusion from the initial project.

### 1. Technique

**TrackNet-class temporal heatmaps are appropriate, but TrackNetV3 is not the first model I would choose.** TrackNetV3’s background estimation and trajectory-repair stages target shuttlecock occlusion and add machinery ShotLab’s ballistic RANSAC already partly supplies.

Start with **WASB**, because:

- It outperformed TrackNetV2-family methods across the paper’s sports datasets.
- It uses three-frame, full-resolution heatmap prediction.
- Most importantly, the official model zoo already provides a **basketball-trained checkpoint**—the plan overlooks this. [WASB paper](https://arxiv.org/abs/2311.05237), [official basketball weights/instructions](https://github.com/nttcom/WASB-SBDT/blob/main/GET_STARTED.md)

The alternatives:

- **Smarter Kalman/interpolation:** cheapest operational improvement, but cannot break a visibility floor. It can only bridge gaps between existing anchors. Also, constant velocity is a poor vertical-flight model; use a ballistic/parabolic state if pursuing this.
- **Optical flow or frame differencing:** the simplest method that could genuinely expose a sub-threshold moving ball. Use it inside a YOLO/physics-constrained corridor, not over the whole image. This is worth a small baseline alongside zero-shot WASB.
- **Temporal YOLO:** poor effort-to-value here. You would be inventing/training a temporal detector with little data and no clear advantage over established heatmap models.
- **Track-before-detect:** probably the best non-neural engineering approach: generate weak motion/appearance peaks at very low threshold, then choose an entire ballistic trajectory globally. That aligns better with ShotLab than asking the current forward tracker to make local decisions.

One code-level warning: `assemble_track` is not as fusion-ready as the plan claims. Its velocity is displacement between observations but is subsequently multiplied by the next frame gap without dividing by the previous gap. Irregular temporal detections can therefore produce a bad prediction. It also resets to the highest-confidence candidate, making it vulnerable to differently calibrated temporal confidences.

### 2. Fusion

Choose **candidate-level late fusion**, but not naïve union into the unchanged tracker.

Recommended behavior:

- Run YOLO and the temporal model as complete passes.
- Preserve good YOLO detections.
- Add temporal candidates primarily inside YOLO gaps or plausible flight corridors.
- Score/select trajectories using whole-flight ballistic consistency, source-aware confidence, and ideally rim/release constraints.

Why not the others:

- **Gated inference on YOLO-miss frames:** wrong initial design. A temporal model needs surrounding frames, and YOLO may miss an entire flight—leaving no window to trigger it. Offline inference cost is not important enough to justify this failure mode.
- **Naïve union:** temporal false positives can seed or hijack the current greedy tracker, especially because confidence scores from two models are not calibrated.
- **Heatmap fusion:** excessive coupling and maintenance for a solo project. It is unlikely to beat source-aware candidate fusion enough to matter.

Keep the downstream `BallCandidate` contract, but do **not** force temporal inference through the current single-frame `BaseDetector.detect()` abstraction. Add a sequence API such as `detect_video()` or a temporal prepass/cache. A buffering detector would introduce output latency and awkward handling of earlier-frame predictions.

### 3. Data sufficiency

**1,340 correlated frames from one session are enough for a prototype or very light adaptation, not evidence of generalization.**

They may represent only a few dozen statistically distinct flights; adjacent frames are not independent examples. The danger is learning that court, camera placement, lighting, clothing, and background.

Practical targets:

- Zero-shot checkpoint test: no additional labels.
- Proof-of-concept fine-tune: current data may suffice if most layers are frozen.
- Credible personal-use model: roughly **3,000–5,000 hard flight frames across 4–6 sessions**, preferably 50–100 flights.
- More robust across lighting/framing: nearer **10,000 frames**.

“Label more sessions first” is therefore **not a hard prerequisite to test the idea**, but it is a prerequisite before promoting a fine-tuned model as reliable. Split validation by entire session, never randomly by adjacent frame.

Prioritize misses, blur, apexes, negatives, and occlusions rather than accumulating easy large-ball frames.

### 4. Resolution and tiling

Do not begin with independently selected native-resolution tiles. Use a **fixed shot/court ROI across the entire temporal window** so the 20 px ball retains useful scale while temporal motion remains spatially coherent.

If tiling is necessary:

- Use exactly the same tile coordinates for all frames in a stack.
- Include generous overlap.
- Stitch candidates after inference.
- Never recenter or choose tiles independently per frame.
- Train and infer with the same crop/scale distribution.

WASB’s published setup resized frames to `288×512`, despite retaining high-resolution feature maps internally. A 20 px ball from 1920-wide footage becomes about 5 px at that input, so first test whether the basketball checkpoint already handles it; its training domain contains broadcast basketball, but ShotLab is a substantial domain shift. [WASB implementation details](https://arxiv.org/abs/2311.05237)

Temporal-specific pitfalls:

- Apply geometric augmentation identically across every frame and heatmap in a stack.
- Include no-ball/occluded targets; otherwise the model may hallucinate along learned motion.
- Phone tripod shake creates global motion. If measurable, stabilize the stack with a background homography/ECC transform or train with coherent synthetic shake.
- Motion blur is signal as well as damage; include real blurred examples rather than sharpening every frame.
- Three frames are the sensible first choice. Do not expand to long recurrent windows until three-frame performance is known.

### 5. Phase 0

**Phase 0 is the correct gate, not a stall—but sharpen it into an oracle experiment.**

Do not merely count YOLO gaps. Use the confirmed per-frame labels as perfect temporal-model outputs, insert them where YOLO missed, and rerun the exact tracker, segmentation, RANSAC, and metrics.

Measure:

- Additional real shots recovered.
- Existing shots whose metric error/availability improves.
- Whether apex-only additions change anything downstream.
- How much improvement survives when oracle points receive realistic localization noise, e.g. ±3–5 px.

If perfect gap filling barely changes shot recovery or metric quality, temporal ML is gold-plating. It is not worth building “regardless.”

### 6. Biggest risk and cheapest experiment

The biggest risk is **optimizing tiny-ball frame recall that produces no user-visible improvement because RANSAC already reconstructs the useful arc—or because the greedy tracker cannot exploit the added candidates safely.**

The single cheapest de-risking experiment is:

> **Oracle gap-fill ablation:** substitute ground-truth centers for every YOLO-missed labeled flight frame, then rerun the existing end-to-end pipeline and compare recovered shots and final metrics.

That directly estimates the maximum possible return from any temporal detector. It is more decision-useful than training a model.

If the oracle result is strong, the next experiment should be zero-shot inference with the official pretrained WASB basketball checkpoint on the same clips. TrackNet originally demonstrated the value of consecutive-frame heatmaps for tiny, blurred balls, but also showed a substantial cross-video recall drop—exactly why session-held-out testing matters. [TrackNet paper](https://arxiv.org/abs/1907.03698)

## Recommended revised plan

- Keep Phase 0, replacing it with the oracle ceiling test.
- Add Phase 0.5: zero-shot WASB-basketball benchmark, including ONNX/DirectML export feasibility.
- Add a small trajectory-constrained motion/frame-difference baseline.
- Implement source-aware candidate fusion plus ballistic trajectory selection only if those tests succeed.
- Collect several sessions before serious fine-tuning.
- Defer tiling, heatmap fusion, TrackNetV3 rectification, and temporal-YOLO indefinitely.

So the answer is: **measure-first, then build a reduced WASB-based version if—and only if—the oracle and zero-shot tests both show a meaningful downstream win.**