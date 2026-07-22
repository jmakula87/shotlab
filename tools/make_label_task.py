#!/usr/bin/env python
"""Physics-assisted ball labeler (Mode 1) -- generate real small-ball labels.

The far ball is unavoidably small (you must keep the whole flight + rim in frame),
so the detector misses the hardest frames -- and those are exactly the ones the
retrain needs. It can't auto-label what it misses (chicken-and-egg), but a HUMAN
sees the ball instantly. This tool breaks that loop:

  for each detected shot -> fit x(frame) linear + y(frame) quadratic (gravity) ->
  predict the ball on EVERY frame in the flight window (including the gaps + a
  margin past the detected ends where the small ascent/near-rim balls live) ->
  crop native-res around each prediction -> emit a self-contained HTML page.

You then just confirm: Enter accepts the predicted marker, a click fixes it, N
says "no ball here" (occluded / out of frame). Physics does ~80%; you tap through
the rest. Every confirmed frame becomes a native-scale training label (feed
tools/ingest_labels.py -> make_dataset_native -> retrain). Fully local; the page
saves to your browser + downloads a labels.json.

Usage:
  python tools/make_label_task.py --clips "data/raw/Camera 1/PXL_20260720_*.mp4" \
     --exclude 150124 --out data/out/label_task_0720.html
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.detect_cache import _path as track_path, deserialize_detection
from shotlab.video_io import iter_frames

CROP = 400          # native-res crop half-window is CROP/2 around the ball
MARGIN = 12         # predict this many frames past each detected shot end
JPEG_Q = 72


def _predict_shot(frames, xs, ys, radii):
    """(detected frames) -> per-frame (x,y,r,source) over the flight window, with
    the gaps + a margin filled by a linear-x / quadratic-y (gravity) fit."""
    frames = np.asarray(frames); xs = np.asarray(xs, float)
    ys = np.asarray(ys, float); radii = np.asarray(radii, float)
    det = {int(f): (float(x), float(y), float(r))
           for f, x, y, r in zip(frames, xs, ys, radii)}
    f0, f1 = int(frames.min()) - MARGIN, int(frames.max()) + MARGIN
    if len(frames) >= 3:
        px = np.polyfit(frames, xs, 1)
        py = np.polyfit(frames, ys, 2)
        xf = lambda f: np.polyval(px, f)
        yf = lambda f: np.polyval(py, f)
    else:                                        # too few points: linear both
        px = np.polyfit(frames, xs, 1); py = np.polyfit(frames, ys, 1)
        xf = lambda f: np.polyval(px, f); yf = lambda f: np.polyval(py, f)
    rmed = float(np.median(radii))
    out = {}
    for f in range(f0, f1 + 1):
        if f in det:
            x, y, r = det[f]; out[f] = (x, y, r, "det")
        else:
            out[f] = (float(xf(f)), float(yf(f)), rmed, "pred")
    return out


def build_items(clips):
    """One item per (clip, frame): the native crop + the predicted ball in it."""
    items = []
    for clip in clips:
        stem = os.path.splitext(os.path.basename(clip))[0]
        tj = track_path(clip)
        if not os.path.exists(tj):
            print(f"  {stem}: no track cache -> run build_session first, skipped")
            continue
        with open(tj, encoding="utf-8") as f:
            _, shots = deserialize_detection(json.load(f))
        # frame -> (x,y,r,source, shot#) across all shots in the clip
        want = {}
        for si, s in enumerate(shots, 1):
            pred = _predict_shot(s.frames, s.xs, s.ys, s.radii)
            for fr, (x, y, r, src) in pred.items():
                want[int(fr)] = (x, y, r, src, si)
        if not want:
            continue
        need = set(want)
        got = {}
        for idx, frame in iter_frames(clip):
            if idx not in need:
                continue
            H, W = frame.shape[:2]
            x, y, r, src, si = want[idx]
            cx0 = int(min(max(x - CROP / 2, 0), max(0, W - CROP)))
            cy0 = int(min(max(y - CROP / 2, 0), max(0, H - CROP)))
            crop = frame[cy0:cy0 + CROP, cx0:cx0 + CROP]
            if crop.shape[0] < 10 or crop.shape[1] < 10:
                continue
            ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
            if not ok:
                continue
            got[idx] = {
                "clip": stem, "frame": int(idx),
                "cx0": cx0, "cy0": cy0, "cw": int(crop.shape[1]), "ch": int(crop.shape[0]),
                "px": round(x - cx0, 1), "py": round(y - cy0, 1),  # marker in crop
                "r": round(r, 1), "src": src, "shot": si,
                "img": base64.b64encode(buf).decode("ascii"),
            }
        for idx in sorted(got):
            items.append(got[idx])
        print(f"  {stem}: {sum(1 for i in got.values() if i['src']=='det')} detected "
              f"+ {sum(1 for i in got.values() if i['src']=='pred')} predicted frames")
    return items


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>ShotLab ball labeler</title>
<style>
 body{margin:0;background:#111;color:#ddd;font:14px system-ui;-webkit-user-select:none;user-select:none}
 #bar{padding:8px 12px;background:#1b1b1b;position:sticky;top:0;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
 #stage{display:flex;justify-content:center;padding:16px}
 #wrap{position:relative;cursor:crosshair}
 #img{image-rendering:pixelated;display:block}
 #mk{position:absolute;border:2px solid #2ee66a;border-radius:50%;pointer-events:none;box-shadow:0 0 0 1px #000}
 #mk.pred{border-color:#ffca3a}
 .tag{padding:2px 8px;border-radius:10px;font-weight:600}
 .det{background:#14432a;color:#6ee7a0}.pred{background:#4a3a10;color:#ffca3a}
 button{background:#2a2a2a;color:#ddd;border:1px solid #444;padding:6px 12px;border-radius:6px;cursor:pointer}
 #prog{height:6px;background:#333;border-radius:3px;flex:1;min-width:120px;overflow:hidden}
 #progf{height:100%;background:#2ee66a;width:0}
 kbd{background:#333;border:1px solid #555;border-radius:4px;padding:1px 6px;font-size:12px}
 #done{color:#6ee7a0;font-weight:700}
</style></head><body>
<div id="bar">
 <b>Ball labeler</b>
 <span id="pos"></span><span class="tag" id="srctag"></span>
 <span>shot <b id="shot"></b></span>
 <div id="prog"><div id="progf"></div></div>
 <span id="done"></span>
 <button onclick="save()">⬇ Save labels</button>
 <span style="color:#888"><kbd>Enter</kbd> confirm · <kbd>click</kbd> fix · <kbd>N</kbd> no ball · <kbd>←</kbd> back · <kbd>→</kbd> skip</span>
</div>
<div id="stage"><div id="wrap"><img id="img"><div id="mk"></div></div></div>
<script>
const ITEMS = __ITEMS__;
const KEY = "shotlab_labels_" + (ITEMS[0]?ITEMS[0].clip:"x") + "_" + ITEMS.length;
const SCALE = 1.9;
let labels = JSON.parse(localStorage.getItem(KEY) || "{}");
let i = 0;
const img=document.getElementById('img'), mk=document.getElementById('mk'), wrap=document.getElementById('wrap');
function firstUnlabeled(){ for(let k=0;k<ITEMS.length;k++){ if(!(k in labels)) return k;} return ITEMS.length-1; }
function show(){
 const it=ITEMS[i];
 img.src="data:image/jpeg;base64,"+it.img;
 img.width=it.cw*SCALE; img.height=it.ch*SCALE;
 const lab=labels[i];
 const mx=(lab?lab.px:it.px), my=(lab?lab.py:it.py), present=lab?lab.present:true;
 const d=14;
 mk.style.display=present?'block':'none';
 mk.style.width=mk.style.height=d+'px';
 mk.style.left=(mx*SCALE-d/2)+'px'; mk.style.top=(my*SCALE-d/2)+'px';
 mk.className=(it.src==='pred'?'pred':'');
 document.getElementById('pos').textContent=`frame ${it.frame}  (${i+1}/${ITEMS.length})`;
 const st=document.getElementById('srctag'); st.textContent=it.src==='pred'?'PREDICTED — check':'detected';
 st.className='tag '+(it.src==='pred'?'pred':'det');
 document.getElementById('shot').textContent=it.shot;
 const n=Object.keys(labels).length;
 document.getElementById('progf').style.width=(100*n/ITEMS.length)+'%';
 document.getElementById('done').textContent=n===ITEMS.length?'ALL DONE ✔ (Save)':`${n} labeled`;
}
function setLabel(px,py,present){
 const it=ITEMS[i];
 labels[i]={clip:it.clip,frame:it.frame,cx0:it.cx0,cy0:it.cy0,px:px,py:py,r:it.r,present:present};
 localStorage.setItem(KEY,JSON.stringify(labels));
}
function confirm(){ const it=ITEMS[i]; const lab=labels[i]; setLabel(lab?lab.px:it.px, lab?lab.py:it.py, true); next(); }
function noball(){ setLabel(0,0,false); next(); }
function next(){ if(i<ITEMS.length-1){i++;} show(); }
function prev(){ if(i>0){i--;} show(); }
wrap.onclick=(e)=>{ const rc=img.getBoundingClientRect(); const px=(e.clientX-rc.left)/SCALE, py=(e.clientY-rc.top)/SCALE; setLabel(px,py,true); next(); };
document.onkeydown=(e)=>{
 if(e.key==='Enter'||e.key===' '){confirm();e.preventDefault();}
 else if(e.key.toLowerCase()==='n'){noball();}
 else if(e.key==='ArrowLeft'){prev();}
 else if(e.key==='ArrowRight'){next();}
 else if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='s'){save();e.preventDefault();}
};
function save(){
 const out=Object.values(labels);
 const blob=new Blob([JSON.stringify(out,null,1)],{type:'application/json'});
 const a=document.createElement('a'); a.href=URL.createObjectURL(blob);
 a.download='ball_labels.json'; a.click();
}
i=firstUnlabeled(); show();
</script></body></html>"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--out", default="data/out/label_task.html")
    args = ap.parse_args(argv)
    clips = sorted(glob.glob(args.clips))
    clips = [c for c in clips if not any(x in os.path.basename(c) for x in args.exclude)]
    if not clips:
        print("no clips matched"); return 1
    items = build_items(clips)
    if not items:
        print("no labelable frames (need cached tracks from build_session)"); return 1
    html = HTML.replace("__ITEMS__", json.dumps(items))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    mb = os.path.getsize(args.out) / 1e6
    print(f"\n{len(items)} frames -> {args.out}  ({mb:.1f} MB)")
    print("Open it in a browser: Enter=confirm, click=fix, N=no ball, then Save.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
