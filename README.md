# 🏀 ShotLab — basketball shot-analysis tool

Upload phone videos of your shooting workouts → get per-shot arc metrics, form
feedback, and a local dashboard with a tracked-overlay video next to the stats.

Built in phases:

| Phase | What | Status |
|---|---|---|
| **1** | Ball tracking + arc (release angle, apex, entry angle) + overlay | ✅ working |
| **2** | Form analysis via pose (elbow, knee, release-vs-apex, follow-through, balance) | ✅ working* |
| **3** | Spin rate (slow-mo only) | ✅ working* |

\* Phases 2 & 3 are validated on synthetic data + unit tests; final accuracy
needs a real clip (Phase 2: any clip with a person; Phase 3: a ≥120fps slow-mo
clip with a marked ball). Drop a clip in `data/raw/` to validate on your footage.

## Filming assumptions
Stationary tripod, consistent angle, decent light, one angle per session:
**side-on** for arc + knee/release, **front-on** for elbow alignment.
Shoot **1080p at 120–240 fps** with a fast shutter if you can — it materially
tightens apex/entry-angle estimates and is *required* for Phase 3 spin.

---

## Model choices (and why)

You asked me to check current best-in-class rather than default to the familiar.
I ran a 2026 survey of both problems. Summary:

### Ball detection — fine-tuned detector + RANSAC, **not** a tracker
- **Key finding:** for a *single* ball, multi-object trackers (ByteTrack/BoT-SORT)
  actively hurt — their constant-velocity Kalman assumption breaks on a small,
  fast, blurred ball. The right tool is per-frame detection → **RANSAC + degree-2
  polynomial fit**. Blur/occlusion become outliers RANSAC discards; brief
  disappearances are interpolated by the parabola itself. That's exactly how
  `shotlab/arc.py` works.
- **Detector:** stock COCO "sports ball" is documented as unreliable (and indeed
  whiffs on most frames even in our clean test clip). So:
  - `--detector color` — classical HSV-orange + circularity. Zero ML deps, fast
    on CPU, excellent on clean/well-lit footage. Good for quick iteration.
  - `--detector motion` — MOG2 background subtraction. For cluttered OUTDOOR
    footage where the ball is small/dark against trees+houses; isolates the fast
    ball from the static scene. (Residual movers = wind-swayed leaves.)
  - `--detector motion+color` — **recommended for outdoor leafy backgrounds**:
    requires a blob to be BOTH moving AND orange. Kills swaying-leaf and
    tan-ground false positives at once. Needs the ball FRONT-LIT (sun behind the
    camera) so it reads orange, not silhouette.
  - `--detector yolo` — Ultralytics YOLO (default `yolo11n`, the survey's safe,
    battle-tested fallback). For real accuracy on messy footage, pass a
    **fine-tuned basketball model** (e.g. a Roboflow Universe basketball model,
    ~96% mAP@50) via `--weights basketball.pt --ball-class 0`.
  - *License:* `ultralytics` is **AGPL-3.0** — fine for personal/local use. If you
    ever distribute this closed-source, swap to an Apache-2.0 detector (RT-DETR /
    D-FINE / RF-DETR) behind the same `BaseDetector` interface.

### Pose estimation (Phase 2) — MediaPipe default, honest about depth
- **#1 MediaPipe Pose (BlazePose-33):** Apache-2.0, real-time on CPU (you have no
  GPU), 33 keypoints covering every joint we need + feet, actively maintained.
- **Upgrade path:** RTMPose/RTMW (133 kp incl. fingers) via `rtmlib` if you want
  follow-through finger detail and have a GPU.
- **Honesty rule baked in:** single-camera video is 2D. In-plane angles on a
  square camera (knee bend side-on, release-vs-apex) are **high confidence**;
  anything depth-dependent — **elbow flare, squareness** — is **low confidence**
  (14–27% perspective error, independent of model). The output labels these.

Hardware detected here: **CPU-only** (no NVIDIA GPU), Python 3.13, ffmpeg present.
All choices above are the CPU-friendly options for that reason.

---

## Install
```bash
pip install -r requirements.txt          # core + YOLO
# the classical color detector needs only the core (no ultralytics)
```

## Use
```bash
# generate a synthetic test clip (no real footage needed)
python scripts/make_synthetic_clip.py --fps 60 --shots 5

# analyze it (classical detector)
python analyze.py data/raw/synthetic_side.mp4 --detector color

# add form analysis (Phase 2) and spin (Phase 3)
python analyze.py data/raw/sideon.mp4 --detector color --pose --spin \
    --camera-angle side_on --handedness right

# real footage with a fine-tuned basketball model
python analyze.py data/raw/myworkout.mp4 --detector yolo \
    --weights basketball.pt --ball-class 0 --imgsz 960 --pose

# dashboard: overlay video (ball + skeleton) next to the stats, live-tunable targets
streamlit run dashboard/app.py
```

Outputs land in `data/out/<clip>/`:
`*_overlay_h264.mp4` (tracked arc), `*_shots.csv`, `*_shots.json`.

## Configurable targets
Edit `config/targets.yaml` to tune what counts as "clean" — e.g. set
`arc.entry_angle_deg.target: 45` and its band. Metrics outside their band raise a
flag in the table. The dashboard also lets you retune entry/release targets live.

## Project layout
```
analyze.py                 CLI entry point
config/targets.yaml        shot targets + deviation bands (yours to tune)
shotlab/
  arc.py                   parabola fit + angle geometry (pure math, tested)
  video_io.py              frame access, fps/slow-mo probe, H.264 transcode
  config.py  report.py     targets loading, flagging, session table
  phase1_ball/
    detect.py              ColorBallDetector
    detect_yolo.py         YoloBallDetector (production)
    track.py               gap-aware single-ball association + shot segmentation
    overlay.py             trajectory/arc/metrics + skeleton overlay renderer
    pipeline.py            Phase-1 orchestration
  phase2_pose/
    pose.py                MediaPipe PoseLandmarker wrapper + One-Euro smoothing
    form.py                elbow/knee/release-vs-apex/follow-through/balance metrics
    pipeline.py            Phase-2 orchestration
  phase3_spin/
    spin.py                fps-gated backspin via log-polar phase correlation
    pipeline.py            Phase-3 orchestration
models/                    auto-downloaded pose .task model
dashboard/app.py           Streamlit dashboard (Arc/Form/All views)
scripts/make_synthetic_clip.py
tests/test_arc.py          geometry unit tests (6)
tests/test_form.py         form-metric unit tests (5)
```

## Tests
```bash
python tests/test_arc.py        # geometry validated to <1° on clean arcs
```
