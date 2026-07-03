"""Per-shot form CLOSEUPS for the film room.

The wide-camera thumbnails are useless for studying form (you're tiny in them).
This crops TIGHT to your body at each phase -- load (deepest crouch), release
(peak wrist extension), follow-through -- draws the skeleton, and highlights the
joint that phase is about (knee on the load, elbow on the release/follow). The
result is a zoomable per-phase closeup you can flip through shot by shot to see
what your good reps look like and copy them.

build_shot_closeups() writes JPGs to <session>/closeups/ and returns an index
(shot -> phase -> path + the shot's key metrics), reused by the film-room HTML
gallery and the dashboard.
"""

from __future__ import annotations

import json
import os

import numpy as np

# BlazePose bones to draw (shoulder/arm/torso/legs)
_BONES = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12),
          (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
          (24, 26), (26, 28)]
_DRAWN = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
# joints to emphasize per phase (the thing that phase is about)
_HILITE = {"load": (24, 26, 28), "release": (12, 14, 16), "follow": (12, 14, 16)}
_PHASE_ORDER = ["load", "release", "follow"]
_PHASE_LABEL = {"load": "Load (legs)", "release": "Release (elbow)",
                "follow": "Follow-through"}


def _bbox(fp, w, h, pad=0.5, aspect=0.8):
    """Tight box around the visible body, padded, nudged to a portrait aspect,
    clamped to the frame. Returns (x0, y0, x1, y1) or None."""
    xs = [float(fp.xy[i][0]) for i in _DRAWN if fp.vis[i] >= 0.3]
    ys = [float(fp.xy[i][1]) for i in _DRAWN if fp.vis[i] >= 0.3]
    if len(xs) < 4:
        return None
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
    x0 -= bw * pad; x1 += bw * pad
    y0 -= bh * pad * 0.6; y1 += bh * pad * 0.6
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw, bh = x1 - x0, y1 - y0
    if bw / bh > aspect:                    # too wide -> grow height
        bh = bw / aspect
    else:                                   # too tall -> grow width
        bw = bh * aspect
    x0, x1 = cx - bw / 2, cx + bw / 2
    y0, y1 = cy - bh / 2, cy + bh / 2
    x0 = max(0, int(x0)); y0 = max(0, int(y0))
    x1 = min(w, int(x1)); y1 = min(h, int(y1))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    return x0, y0, x1, y1


def _draw_closeup(frame, fp, phase, out_h=520):
    """Crop to the body and draw the skeleton, emphasizing this phase's joints."""
    import cv2
    h, w = frame.shape[:2]
    box = _bbox(fp, w, h)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    crop = frame[y0:y1, x0:x1].copy()
    hot = _HILITE.get(phase, ())
    for a, b in _BONES:
        if fp.vis[a] >= 0.3 and fp.vis[b] >= 0.3:
            pa = (int(fp.xy[a][0] - x0), int(fp.xy[a][1] - y0))
            pb = (int(fp.xy[b][0] - x0), int(fp.xy[b][1] - y0))
            hotline = a in hot and b in hot
            cv2.line(crop, pa, pb, (0, 200, 255) if hotline else (0, 255, 0),
                     4 if hotline else 2)
    for i in _DRAWN:
        if fp.vis[i] >= 0.3:
            p = (int(fp.xy[i][0] - x0), int(fp.xy[i][1] - y0))
            cv2.circle(crop, p, 5 if i in hot else 3,
                       (0, 0, 255) if i in hot else (0, 255, 0), -1)
    scale = out_h / crop.shape[0]
    return cv2.resize(crop, (int(crop.shape[1] * scale), out_h))


def build_shot_closeups(session_dir, only_made=None, raw_dirs=None,
                        handedness="auto", limit=None, pose_stride=2):
    """Generate phase closeups for a session's shots. `only_made`: True = makes
    only, False = misses only, None = all. Returns a list of shot dicts (in
    session order) with per-phase image paths + metrics. Cached on disk."""
    import cv2
    import pandas as pd
    from .detect_cache import _load as load_track
    from .video_io import probe, iter_frames
    from .phase2_pose.pose import PoseExtractor
    from .skeleton import _phase_frames, _PRE, _POST

    raw_dirs = raw_dirs or [os.path.join("data", "raw", "Hoops"),
                            os.path.join("data", "raw")]
    cdir = os.path.join(session_dir, "closeups")
    os.makedirs(cdir, exist_ok=True)
    df = pd.read_csv(os.path.join(session_dir, "session_shots.csv"))

    from .curate import apply_excludes
    df = apply_excludes(df, session_dir)          # drop junk + layups

    def _nm(v):
        return True if v in (True, "True") else (False if v in (False, "False") else None)
    if only_made is not None and "made" in df.columns:
        df = df[df["made"].map(_nm) == only_made]
    if limit:
        df = df.head(limit)

    out = []
    for clip, group in df.groupby("clip"):
        loaded = load_track(clip)
        if not loaded:
            continue
        _, track, shots = loaded
        by_idx = {s.index: s for s in shots}
        raw = next((os.path.join(dd, clip) for dd in raw_dirs
                    if os.path.exists(os.path.join(dd, clip))), None)
        if raw is None:
            continue
        info = probe(raw)
        fps = info.fps
        for _, row in group.iterrows():
            sn = int(row["shot_num"])
            s = by_idx.get(int(row["shot_in_clip"]))
            if s is None:
                continue
            paths = {p: os.path.join(cdir, f"shot_{sn}_{p}.jpg") for p in _PHASE_ORDER}
            need = [p for p in _PHASE_ORDER if not os.path.exists(paths[p])]
            if need:
                lo = max(0, int(s.frames[0]) - _PRE)
                hi = int(s.frames[-1]) + _POST
                ext = PoseExtractor(fps=fps, variant="full", smooth=True)
                poses = {}
                try:
                    for idx, frame in iter_frames(raw, start=lo, stop=hi + 1):
                        if pose_stride > 1 and (idx - lo) % pose_stride:
                            continue     # phase frames don't need every frame
                        fp = ext.process_frame(idx, frame)
                        if fp is not None:
                            poses[idx] = fp
                finally:
                    ext.close()
                phases, _keys = _phase_frames(s, track, poses, fps, handedness)
                cap = cv2.VideoCapture(raw)
                for p in _PHASE_ORDER:
                    f = phases.get(p)
                    if f is None or poses.get(f) is None:
                        continue
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    img = _draw_closeup(frame, poses[f], p)
                    if img is not None:
                        cv2.imwrite(paths[p], img, [cv2.IMWRITE_JPEG_QUALITY, 88])
                cap.release()
            have = {p: paths[p] for p in _PHASE_ORDER if os.path.exists(paths[p])}
            if have:
                out.append({
                    "shot": sn, "made": _nm(row.get("made")),
                    "zone": row.get("zone", ""),
                    "release_angle_deg": row.get("release_angle_deg"),
                    "knee_bend_deg": row.get("knee_bend_deg"),
                    "elbow_angle_at_release_deg": row.get("elbow_angle_at_release_deg"),
                    "follow_through_hold_s": row.get("follow_through_hold_s"),
                    "paths": have})
    out.sort(key=lambda r: r["shot"])
    with open(os.path.join(cdir, "index.json"), "w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in r.items() if k != "paths"} for r in out], f)
    return out


def _b64(path):
    import base64
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _num(v, unit="", dp=0):
    try:
        if v is None or str(v) in ("nan", "None", ""):
            return "—"
        return f"{float(v):.{dp}f}{unit}"
    except (ValueError, TypeError):
        return "—"


def film_room_html(closeups, title="Film room — study your reps"):
    """A self-contained, arrow-key-navigable film room: one shot per screen with
    its load / release / follow closeups + the metric each phase is about. ←/→
    (or the buttons) step through the shots."""
    shots = []
    for r in closeups:
        imgs = {p: _b64(path) for p, path in r["paths"].items()}
        shots.append({
            "shot": r["shot"], "made": bool(r["made"]) if r["made"] is not None else None,
            "zone": r.get("zone", ""),
            "imgs": imgs,
            "load": _num(r.get("knee_bend_deg"), "°"),
            "release": _num(r.get("elbow_angle_at_release_deg"), "°"),
            "arc": _num(r.get("release_angle_deg"), "°"),
            "follow": _num(r.get("follow_through_hold_s"), "s", 2),
        })
    outs = {s["made"] for s in shots}
    has_make, has_miss = True in outs, False in outs
    default = "make" if has_make else ("miss" if has_miss else "all")
    if has_make and has_miss:
        fbar = ("<div class='filters'>"
                "<button data-f='make' class='on'>Makes</button>"
                "<button data-f='miss'>Misses</button>"
                "<button data-f='all'>All</button></div>")
    else:
        fbar = ""
    data = json.dumps(shots)
    html = """<!doctype html><html><head><meta charset='utf-8'><title>""" + title + """</title>
<style>
:root{color-scheme:dark}
body{margin:0;background:#0f141a;color:#e8edf2;font-family:system-ui,Arial,sans-serif}
header{padding:14px 20px;display:flex;align-items:center;gap:16px;flex-wrap:wrap}
h1{font-size:18px;margin:0}
.counter{color:#9fb0c0;font-size:14px}
.badge{padding:3px 10px;border-radius:12px;font-size:13px;font-weight:600}
.make{background:#173d1e;color:#7ee08a} .miss{background:#3d1717;color:#ff9b9b}
.phases{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;padding:0 20px 16px;max-width:1200px;margin:0 auto}
.phase{background:#161d26;border-radius:12px;overflow:hidden;text-align:center}
.phase img{width:100%;display:block;background:#000}
.phase .t{padding:8px 6px 2px;font-weight:600}
.phase .m{padding:0 6px 10px;color:#9fb0c0;font-size:14px}
.hot{color:#ffd27a}
.nav{display:flex;gap:10px;align-items:center}
button{background:#1f2b38;color:#e8edf2;border:0;border-radius:8px;padding:8px 16px;font-size:15px;cursor:pointer}
button:hover{background:#2a3a4c}
.hint{color:#7d8ea0;font-size:13px;padding:0 20px 14px;max-width:1200px;margin:0 auto}
.filters{padding:0 20px 10px} .filters button.on{background:#2e6fd8}
</style></head><body>
<header>
  <h1>🎬 Film room</h1>
  <div class='nav'><button id='prev'>← Prev</button>
    <span class='counter' id='counter'></span><button id='next'>Next →</button></div>
  <span id='badge' class='badge'></span>
  <span class='counter' id='meta'></span>
</header>
__FBAR__
<div class='phases' id='phases'></div>
<div class='hint'>← / → arrow keys (or the buttons) to move between shots. Gold =
the joint this phase is about: knees on the load, elbow on the release. Study
the makes and copy the shape.</div>
<script>
const SHOTS = """ + data + """;
let filt='""" + default + """', i=0, view=[];
function apply(){view=SHOTS.filter(s=>filt==='all'||(filt==='make')===s.made); if(i>=view.length)i=0; render();}
function render(){
  if(!view.length){document.getElementById('phases').innerHTML='<p>No shots.</p>';return;}
  const s=view[i];
  document.getElementById('counter').textContent=`${i+1} / ${view.length}`;
  const b=document.getElementById('badge');
  b.textContent = s.made===null?'?':(s.made?'MADE':'MISS'); b.className='badge '+(s.made?'make':'miss');
  document.getElementById('meta').textContent=`shot #${s.shot} · ${s.zone} · arc ${s.arc}`;
  const P=[['load','Load (legs)','knee bend','load'],
           ['release','Release (elbow)','elbow','release'],
           ['follow','Follow-through','hold','follow']];
  document.getElementById('phases').innerHTML=P.map(([k,title,mlabel,mkey])=>{
    const img=s.imgs[k]; if(!img) return `<div class='phase'><div class='t'>${title}</div><div class='m'>(not captured)</div></div>`;
    return `<div class='phase'><img src='data:image/jpeg;base64,${img}'>
      <div class='t'>${title}</div><div class='m'><span class='hot'>${mlabel}: ${s[mkey]}</span></div></div>`;
  }).join('');
}
document.getElementById('prev').onclick=()=>{i=(i-1+view.length)%view.length;render();};
document.getElementById('next').onclick=()=>{i=(i+1)%view.length;render();};
document.addEventListener('keydown',e=>{
  if(e.key==='ArrowLeft'){i=(i-1+view.length)%view.length;render();}
  if(e.key==='ArrowRight'){i=(i+1)%view.length;render();}});
document.querySelectorAll('.filters button').forEach(btn=>btn.onclick=()=>{
  document.querySelectorAll('.filters button').forEach(x=>x.classList.remove('on'));
  btn.classList.add('on'); filt=btn.dataset.f; i=0; apply();});
apply();
</script></body></html>"""
    return html.replace("__FBAR__", fbar)

