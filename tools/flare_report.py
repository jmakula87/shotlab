#!/usr/bin/env python
"""Elbow-flare stills + flare-vs-make correlation for a 2-camera session.

For each (wide, close) clip pair:
  * audio-sync them,
  * pose the CLOSE clip, find each release (wrist apex), measure elbow flare from
    the metric world landmarks (W7) AND render an annotated still (arm drawn,
    flare offset marked, cropped to the upper body),
  * time-map each release to the WIDE clip's rim-anchored shot -> its make/miss.

Then correlate flare_deg with made (Cohen's d + permutation p, the same engine as
the make-drivers). Writes stills + a flare_makes block into analysis3d.json.

Honesty: flare is monocular / session-relative (LOW-MED) and make/miss is LOW
confidence; the cross-camera time-map adds a little noise. Treat as exploratory.
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.video_io import iter_frames, frame_times, probe
from shotlab.phase2_pose.pose import PoseExtractor
from shotlab.threed import elbow_flare
from shotlab.sync import sync_clips
from shotlab.detect_cache import _path as track_path, deserialize_detection
from shotlab.correlate import correlate_label
from shotlab.analysis3d import Analysis3D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UP = (0.0, -1.0, 0.0)
PAIRS = [("PXL_20260710_175751234", "20260710_135805"),
         ("PXL_20260710_180449842", "20260710_140431"),
         ("PXL_20260710_181146426", "20260710_141132"),
         ("PXL_20260710_181811930", "20260710_141758")]
WIDE_DIR = os.path.join(ROOT, "data", "raw", "Camera 1")
CLOSE_DIR = os.path.join(ROOT, "data", "raw", "Camera 2")
OUT = os.path.join(ROOT, "data", "out", "session_0710_3d")
STILLS = os.path.join(OUT, "flare_stills")


def wide_shot_times(wide_stem):
    """[(mid_pixel_time_s, made)] for each rim-anchored wide shot, by joining the
    cached track (frames) with the per-clip records (made), in shot order."""
    tj = track_path(os.path.join(WIDE_DIR, wide_stem + ".mp4"))
    with open(tj, encoding="utf-8") as f:
        _, shots = deserialize_detection(json.load(f))
    rec_path = os.path.join(ROOT, "data", "out", wide_stem, wide_stem + "_shots_session.json")
    with open(rec_path, encoding="utf-8") as f:
        rd = json.load(f)
    recs = rd["records"] if isinstance(rd, dict) and "records" in rd else rd
    made_by_shot = {int(r["shot_in_clip"]): r.get("made") for r in recs}
    ts = frame_times(os.path.join(WIDE_DIR, wide_stem + ".mp4"))
    out = []
    for i, s in enumerate(shots):
        f = np.asarray(s.frames)
        mid = int(f[len(f) // 2])
        t = ts.get(mid, mid / 30.0)
        out.append((t, made_by_shot.get(i + 1)))
    return out


def draw_flare_still(frame, fp, flare_deg, made):
    sh = fp.pt("r_shoulder"); el = fp.pt("r_elbow"); wr = fp.pt("r_wrist")
    img = frame.copy()
    v = wr - sh
    foot = sh + (np.dot(el - sh, v) / max(np.dot(v, v), 1e-6)) * v
    P = lambda p: (int(round(p[0])), int(round(p[1])))
    cv2.line(img, P(sh), P(wr), (210, 210, 210), 2)           # shooting-plane edge
    cv2.line(img, P(el), P(foot), (60, 60, 255), 2)           # the flare offset
    cv2.line(img, P(sh), P(el), (0, 180, 255), 3)             # upper arm
    cv2.line(img, P(el), P(wr), (0, 180, 255), 3)             # forearm
    for p in (sh, wr):
        cv2.circle(img, P(p), 5, (0, 180, 255), -1)
    cv2.circle(img, P(el), 7, (40, 40, 255), -1)              # elbow
    pts = np.array([sh, el, wr, foot])
    x0, y0 = pts.min(0) - [90, 130]; x1, y1 = pts.max(0) + [90, 70]
    H, W = img.shape[:2]
    x0, y0 = max(0, int(x0)), max(0, int(y0)); x1, y1 = min(W, int(x1)), min(H, int(y1))
    crop = img[y0:y1, x0:x1].copy()
    tag = "MAKE" if made else ("miss" if made is not None else "?")
    col = (80, 200, 80) if made else ((120, 120, 255) if made is not None else (200, 200, 200))
    cv2.rectangle(crop, (0, 0), (crop.shape[1], 34), (30, 30, 30), -1)
    cv2.putText(crop, f"flare {flare_deg:+.0f}deg   {tag}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    return crop


def process_pair(wide_stem, close_stem):
    wide = os.path.join(WIDE_DIR, wide_stem + ".mp4")
    close = os.path.join(CLOSE_DIR, close_stem + ".mp4")
    offset, conf = sync_clips(wide, close)     # close behind wide by `offset`
    wtimes = wide_shot_times(wide_stem)
    fps = probe(close).fps
    print(f"  {wide_stem} <-> {close_stem}: sync {offset:.2f}s conf {conf:.2f}, "
          f"{len(wtimes)} wide shots", flush=True)

    ext = PoseExtractor(fps=fps, variant="full", smooth=True)
    series, frames = {}, {}
    for idx, frame in iter_frames(close, start=0, stop=None):
        fp = ext.process_frame(idx, frame)
        if fp is None or fp.world is None:
            continue
        if all(fp.v(n) >= 0.5 for n in ("r_shoulder", "r_elbow", "r_wrist", "nose")):
            series[idx] = fp; frames[idx] = frame
    ext.close()
    idxs = sorted(series)
    cts = frame_times(close)
    rel = []
    for k in range(1, len(idxs) - 1):
        f = idxs[k]; wy = series[f].pt("r_wrist")[1]
        if (wy < series[f].pt("nose")[1] and wy <= series[idxs[k-1]].pt("r_wrist")[1]
                and wy < series[idxs[k+1]].pt("r_wrist")[1]):
            if not rel or f - rel[-1] > 15:
                rel.append(f)

    os.makedirs(STILLS, exist_ok=True)
    rows = []
    for f in rel:
        fp = series[f]
        fl = elbow_flare(fp.w("r_shoulder"), fp.w("r_elbow"), fp.w("r_wrist"), up=UP)
        # map this release to the wide shot by time
        ptime = cts.get(f, f / fps) + offset
        made = None
        if wtimes:
            j = int(np.argmin([abs(t - ptime) for t, _ in wtimes]))
            if abs(wtimes[j][0] - ptime) < 1.5:
                made = wtimes[j][1]
        still = draw_flare_still(frames[f], fp, fl.angle_deg, made)
        name = f"{close_stem}_{f}_{'make' if made else ('miss' if made is not None else 'unk')}.jpg"
        cv2.imwrite(os.path.join(STILLS, name), still)
        rows.append({"clip": close_stem, "frame": int(f), "flare_deg": fl.angle_deg,
                     "made": made, "still": os.path.join("flare_stills", name)})
    return rows


def main():
    all_rows = []
    for w, c in PAIRS:
        print(f"pair {w} <-> {c} ...", flush=True)
        try:
            all_rows.extend(process_pair(w, c))
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
    labeled = [r for r in all_rows if r["made"] is not None]
    print(f"\n{len(all_rows)} releases, {len(labeled)} matched to a make/miss")

    corr = None
    if len(labeled) >= 8:
        res = correlate_label([{"flare_deg": r["flare_deg"], "made": r["made"]}
                               for r in labeled], min_n=6)
        for c in (res or []):
            if getattr(c, "metric", None) == "flare_deg":
                corr = c
    fl = np.array([r["flare_deg"] for r in all_rows])
    summary = {"n": int(len(fl)), "median_deg": round(float(np.median(fl)), 1),
               "sd_deg": round(float(fl.std()), 1)} if len(fl) else None

    a = Analysis3D.load(os.path.join(OUT, "analysis3d.json"))
    a.flare = dict(a.flare or {}, shots=all_rows, summary=summary,
                   confidence="low-med",
                   note="monocular world-landmark flare; session-relative. "
                        "Make/miss cross-mapped from the wide camera by audio sync.")
    # store the correlation in a JSON-friendly way
    made_fl = [r["flare_deg"] for r in labeled if r["made"]]
    miss_fl = [r["flare_deg"] for r in labeled if not r["made"]]
    a.flare["make_vs_miss"] = {
        "n_make": len(made_fl), "n_miss": len(miss_fl),
        "flare_make_median": round(float(np.median(made_fl)), 1) if made_fl else None,
        "flare_miss_median": round(float(np.median(miss_fl)), 1) if miss_fl else None,
        "cohens_d": round(getattr(corr, "cohen_d", None), 3) if corr and getattr(corr, "cohen_d", None) is not None else None,
        "p_perm": round(getattr(corr, "p_perm", None), 4) if corr and getattr(corr, "p_perm", None) is not None else None,
        "confidence": getattr(corr, "confidence", None) if corr else None,
    }
    a.save(os.path.join(OUT, "analysis3d.json"))
    mm = a.flare["make_vs_miss"]
    print(f"flare make vs miss: make {mm['flare_make_median']} ({mm['n_make']}) "
          f"vs miss {mm['flare_miss_median']} ({mm['n_miss']}), "
          f"d={mm['cohens_d']} p={mm['p_perm']}")
    print(f"stills -> {STILLS}")


if __name__ == "__main__":
    main()
