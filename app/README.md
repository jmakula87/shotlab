# ShotLab Form Check — on-device PWA

## 🔴 Live app (HTTPS — camera + mic work here)
**https://jmakula87.github.io/shotlab/**  — open on your phone (Chrome), tap
**Add to Home Screen** to install. Auto-deploys from `app/` on every push to
`master` (GitHub Pages, `.github/workflows/pages.yml`).

At the court: tap **🔴 Live**, allow camera + mic, and shoot — you get feedback
after each shot, and you can just **say "good" / "off"** to tag how it felt
(hands-free), which trains your personal ideal. All processing is on-device.



A phone web-app that runs **MediaPipe pose on your device** (no upload, no
server), overlays your skeleton on a shot clip, detects the shot phases, and
compares your form to a `profile.json` — the first slice of the "instant
feedback vs your normal" vision.

## What v0 does
- Pick a shot clip (record on the phone, then choose it).
- Runs pose per frame **on-device**, draws your skeleton live over the video.
- Finds the **release** frame (+ load/follow) and measures elbow-at-release and
  knee-bend.
- Compares to `profile.json`'s ideal targets → per-metric deltas + plain feedback.
- "Jump to release" freezes the release frame; when the profile has ideal
  **skeletons**, it overlays them (gold) on yours (green).

## Run it (test on your Pixel, same Wi-Fi as the PC)
From `Desktop/ShotLab`:
```
python -m http.server 8080 --directory app
```
On the Pixel's browser open:  `http://<PC-LAN-IP>:8080`   (e.g. http://192.168.4.52:8080)

- File-upload works over plain HTTP (what we use here).
- **Live camera** and full "Install to Home Screen" need HTTPS — that comes when
  we host it (e.g. GitHub Pages, free HTTPS). For now, record then pick the clip.

## Notes / limits (honest)
- **Pose only** in v0 — the ball/arc (release angle) is the heavy on-device model,
  deferred to v2. Elbow flare needs the 2-camera 3D and isn't here yet.
- The ideal targets are a **placeholder**. `tools/export_profile.py` will generate
  a real `profile.json` (ideal metrics + ideal skeletons for the overlay) from
  YOUR feel-good shots.
- If pose looks jumpy or misses you: film closer / brighter, keep your whole body
  in frame. Check the phone browser's console for errors and report them.

## Files
- `index.html`, `styles.css` — shell/UI
- `js/pose.js` — MediaPipe on-device pose (GPU→CPU fallback)
- `js/analyze.js` — phases, angles, compare-to-profile
- `js/overlay.js` — skeleton + ideal-skeleton overlay
- `js/main.js` — wiring
- `profile.json` — your ideal targets (+ skeletons when generated)
- `manifest.json`, `sw.js`, `icon.svg` — PWA install/offline
