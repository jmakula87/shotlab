# 🏀 ShotLab — basketball shot-analysis tool

Film your shooting workouts → get per-shot arc + form metrics, make/miss, a
local dashboard, and (on the phone) **spoken coaching after every shot**.
Personal-use tool, hard-tuned to one court / phone / orange ball. Honest about
what a single camera can and can't measure.

There are two front doors:

- **📱 Phone app** (live, on-device): point the camera at yourself, shoot, and
  *hear* what to fix — no upload, no rim needed. → [`app/README.md`](app/README.md)
- **💻 Desktop pipeline**: full arc + form + session analytics from recorded
  clips, a Streamlit dashboard, and shareable HTML/PDF reports.

---

## Phone app (live form feedback)

**https://jmakula87.github.io/shotlab/** — open in Chrome on the phone, **Add to
Home Screen** to install. Set the camera **close, side-on, no rim needed**, tap
**🔴 Live**, and shoot. After each shot it speaks a short cue ("bend your knees a
bit more", "hold your follow-through"). Everything runs on-device.

It compares each shot to **your own profile** (see below), plus a small set of
**universal targets** (`shotlab/textbook.py`) kept separate from it. Pose-only
today (elbow bend, knee, tempo, follow-through, balance); elbow flare needs the
2-camera rig.

## Desktop pipeline (recorded clips)

```bash
pip install -r requirements.txt

# one clip, quick look
python analyze.py data/raw/clip.mp4 --detector yolo \
    --weights runs/detect/ball_orange/weights/best_openvino_model --imgsz 640 --pose

# a whole session (many clips -> one timeline: fatigue, zones, make%, shot map)
python build_session.py --clips "data/raw/Hoops/PXL_*.mp4" \
    --detector yolo --weights runs/detect/ball_orange/weights/best_openvino_model \
    --imgsz 640 --stride 2 --chunk-frames 7000 --pose \
    --shooter-height 5'10" --out data/out/session

# dashboard (overlay video + analytics) and a shareable report
streamlit run dashboard/app.py
python tools/export_report.py data/out/session      # report.html
python tools/export_pdf.py    data/out/session      # report.pdf
```

Key flags: `--shooter-height` gives honest body-scaled jump/release heights;
`--audio` (default on) fuses rim sound into make/miss; `--chunk-frames` makes
long clips resumable within a job time cap.

---

## Your shooting profile — what it's built on

`tools/export_profile.py` builds `app/profile.json` from **your own shots**: the
`ideal` is the mean of your *good* ones (your **felt-good** tags first, then
best-ranked, then **made** shots), split so arc ideals come from clean-arc shots
and form ideals from pose-reliable shots. It is NOT textbook/pro form — it's
"match your own best", which is what a single camera can measure reliably and
what correlates with your makes.

Separately, a `textbook` block carries the few **universal** ideals that are the
same for everyone (`shotlab/textbook.py`): ~45° **entry angle** (rim geometry)
and ~0° **elbow flare** (a tucked elbow stays on line — but flare is
out-of-plane, so it only comes online with the 2nd camera). Body-form angles
stay personal on purpose — copying a pro's numbers can hurt a shot that works.

---

## Model choices (and why)

- **Ball:** per-frame detection → **RANSAC + degree-2 polyfit** (not a
  multi-object tracker — those break on a small, fast, blurred ball). Detectors
  behind one `BaseDetector` interface: `color` (HSV orange, clean footage),
  `motion` / `motion+color` (cluttered outdoor), and `yolo` (a **yolo11n
  fine-tuned on this court's orange ball**, exported to OpenVINO for a ~6.6×
  CPU speedup — the production path). `ultralytics` is AGPL-3.0 (fine for
  personal use).
- **Pose:** MediaPipe **BlazePose-33** (Apache-2.0, real-time on CPU), One-Euro
  smoothed. **Honesty rule:** in-plane angles (knee side-on, release-vs-apex) are
  high-confidence; depth-dependent ones (elbow flare, squareness, release height)
  are low-confidence on one camera and labeled so.
- **Environment:** CPU-only, Python 3.13, ffmpeg. Every choice is the
  CPU-friendly option.

## The 2-camera unlock (in progress)

The real form/flare accuracy needs a 2nd camera. The footage-independent 3D
core is built + synthetic-validated: `shotlab/threed.py` (triangulation,
elbow flare, release-point spread), `shotlab/sync.py` (audio sync),
`shotlab/stereo.py` (checkerboard calibration → metric rig),
`shotlab/twocam.py` (fusion), plus `tools/make_checkerboard.py` and
`tools/calibrate_rig.py`. Day-one plan lives in `PROJECT_NOTES.md`.

---

## Component map

```
analyze.py                 single-clip CLI
build_session.py           multi-clip session -> timeline + analytics
dashboard/app.py           Streamlit dashboard (per-clip, session, review, progress)
shotlab/
  arc.py                   parabola fit + angle geometry
  court.py  scale.py       rim detect / zones / shooter-height ruler
  session.py               per-clip records, caching, session stitch
  make.py  audio.py        make/miss (visual + rim-sound fusion)
  correlate.py             what-tracks-with-your-makes engine
  coach.py  textbook.py    written review + universal ideals
  viz.py                   shared plots (court + shot map)
  skeleton.py              ideal per-phase skeletons for the app overlay
  phase1_ball/             detection (color/motion/yolo) + track + overlay
  phase2_pose/             pose + form metrics (form.py)
  phase3_spin/             fps-gated backspin (off by default on this footage)
  threed/sync/stereo/twocam  the 2-camera 3D core (synthetic-validated)
app/                       the on-device PWA (pose, analyze, spoken feedback)
tools/                     dataset build/train, profile export, reports, calibration
```

## Tests
```bash
python run_tests.py                 # 16 Python test files
node tests/test_say.mjs             # (+ analyze / voice / live / feelcsv JS suites)
```
