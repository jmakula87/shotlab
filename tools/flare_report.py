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
from shotlab.analysis3d import Analysis3D, refine_release_frame
from shotlab import paths

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UP = (0.0, -1.0, 0.0)
# The 0710 session's pairs, kept as the default so the original invocation still
# works. Any other session supplies its own with --pair WIDE:CLOSE (repeatable).
DEFAULT_PAIRS = [("PXL_20260710_175751234", "20260710_135805"),
                 ("PXL_20260710_180449842", "20260710_140431"),
                 ("PXL_20260710_181146426", "20260710_141132"),
                 ("PXL_20260710_181811930", "20260710_141758")]
PAIRS = DEFAULT_PAIRS
SHOOTER_FT = 5.83          # 5'10" -- the body-height ruler for height metrics
# Pose estimator configuration. Overridable so the SAME shots can be re-measured
# under a different estimator (--pose-variant heavy / --no-smooth), which turns
# existing footage into repeated measurements of identical events -- the only
# route to a within-shot SD and a smallest-detectable-change without filming.
POSE_VARIANT = "full"
POSE_SMOOTH = True
WIDE_DIR = paths.wide_cam_dir(ROOT)
CLOSE_DIR = paths.close_cam_dir(ROOT)     # prefers the rotation-corrected upright/
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


def assign_one_to_one(event_times, wide_times, max_dist=1.5):
    """Match close-cam releases to wide-cam shots so NEITHER side is reused.

    Returns {event_index: (wide_index, distance_seconds)}, omitting events with no
    match inside `max_dist`.

    Greedy nearest-first: repeatedly take the globally closest surviving pair and
    spend both sides. This is not guaranteed optimal (that would be Hungarian),
    but with well-separated shots it is the same answer, and it is monotone --
    adding a distant event can never steal an existing close match.

    Why this exists: each release used to pick its own nearest wide shot
    independently, which is not injective. Measured 2026-08-04, 141 releases
    claimed only 112 distinct wide shots, 14 of them twice. Duplicates then get
    analysed as independent observations = pseudo-replication, which makes
    permutation p-values anticonservative.
    """
    if not event_times or not wide_times:
        return {}
    cand = sorted((abs(float(t) - float(e)), i, j)
                  for i, e in enumerate(event_times)
                  for j, t in enumerate(wide_times))
    out, used_e, used_w = {}, set(), set()
    for dist, i, j in cand:
        if dist >= max_dist or i in used_e or j in used_w:
            continue
        used_e.add(i); used_w.add(j)
        out[i] = (j, dist)
    return out


def process_pair(wide_stem, close_stem):
    wide = os.path.join(WIDE_DIR, wide_stem + ".mp4")
    close = os.path.join(CLOSE_DIR, close_stem + ".mp4")
    offset, conf = sync_clips(wide, close)     # close behind wide by `offset`
    wtimes = wide_shot_times(wide_stem)
    fps = probe(close).fps
    print(f"  {wide_stem} <-> {close_stem}: sync {offset:.2f}s conf {conf:.2f}, "
          f"{len(wtimes)} wide shots", flush=True)

    # POSE_VARIANT / POSE_SMOOTH are module-level so a reliability run can
    # re-measure the SAME shots under a different estimator without touching the
    # rest of the pipeline. Two passes over one clip are repeated measurements of
    # identical physical events, which is the only way to get a within-shot SD --
    # and therefore a smallest-detectable-change -- out of footage that already
    # exists. Defaults are the shipping configuration.
    ext = PoseExtractor(fps=fps, variant=POSE_VARIANT, smooth=POSE_SMOOTH)
    # `series` = arm-visible frames, used to find the wrist apex (the release).
    # `allposes` = EVERY valid pose, because the body metrics are not all measured
    # at release: knee bend peaks at the gather and follow-through/balance run past
    # it, so restricting to arm-visible frames would silently truncate them.
    series, frames, allposes = {}, {}, {}
    for idx, frame in iter_frames(close, start=0, stop=None):
        fp = ext.process_frame(idx, frame)
        if fp is None or fp.world is None:
            continue
        allposes[idx] = fp
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

    # Refine every candidate release FIRST, then assign wide shots ONE-TO-ONE.
    # Letting each release pick its own nearest wide shot is not injective:
    # measured 2026-08-04 on this session, 141 releases claimed only 112 distinct
    # wide shots, with 14 shots taking TWO releases each. Those duplicates then
    # enter analyses as independent observations -- pseudo-replication, which makes
    # permutation p-values anticonservative. Global nearest-first assignment
    # instead: the closest pair wins and spends BOTH the release and the shot.
    events = []                       # (frame, elbow_deg, flare, pose-clock time)
    for f0 in rel:
        # snap to the extended-arm release; skip gathers/pumps (bent elbow)
        f, elb = refine_release_frame(series, f0)
        if f is None:
            continue
        fp = series[f]
        fl = elbow_flare(fp.w("r_shoulder"), fp.w("r_elbow"), fp.w("r_wrist"), up=UP)
        events.append((f, elb, fl, cts.get(f, f / fps) + offset))

    assign = assign_one_to_one([ev[3] for ev in events], [t for t, _m in wtimes])
    if wtimes:
        print(f"    {len(assign)}/{len(events)} releases matched one-to-one to "
              f"{len(wtimes)} wide shots; {len(events) - len(assign)} unmatched",
              flush=True)

    rows = []
    for i, (f, elb, fl, _pt) in enumerate(events):
        j, dist = assign.get(i, (None, None))
        made = wtimes[j][1] if j is not None else None
        still = draw_flare_still(frames[f], series[f], fl.angle_deg, made)
        name = f"{close_stem}_{f}_{'make' if made else ('miss' if made is not None else 'unk')}.jpg"
        cv2.imwrite(os.path.join(STILLS, name), still)
        row = {"clip": close_stem, "frame": int(f), "flare_deg": fl.angle_deg,
               "elbow_deg": round(float(elb), 0), "made": made,
               # WHICH wide shot this release owns (1-based, matching the CSV's
               # shot_in_clip) and how far off the match was. Emitting these means
               # downstream code can audit or re-do the join instead of silently
               # re-deriving a different one, which is how the duplicates hid.
               "wide_shot": None if j is None else j + 1,
               "join_dist_s": None if dist is None else round(float(dist), 3),
               "still": os.path.join("flare_stills", name)}
        row.update(body_metrics(allposes, f, fps, shooter_height_ft=SHOOTER_FT))
        rows.append(row)
    return rows


def body_metrics(poses, rel_f, fps, *, shooter_height_ft=None, post=30):
    """Knee bend, follow-through, balance drift etc. from the CLOSE camera.

    The wide camera struggles here: the shooter is ~22% of frame height there,
    which yielded 35-42% coverage and a raw knee median of 138 deg -- i.e. it
    mostly fails to catch the load at all. The close camera frames him ~35% and
    in clean profile.

    THE PSEUDO-SHOT MUST START AT THE RELEASE. compute_form does not accept a
    release frame; it re-finds one from `shot.frames[0]` (form.py:186) and windows
    every metric off that. The first version here spanned rel_f+-45, so:
      - the wrist-apex search ran [rel_f-63, rel_f-30] -- it ended a full second
        BEFORE the release it was meant to measure;
      - the knee load search breaks at `f > rel_f`, so it never reached the dip;
      - `span` was 3.7s vs the wide path's ~1.7s, and balance drift is a max-min
        range, so it came out ~4x inflated (close 1.69 vs wide 0.43).
    Starting at rel_f reproduces the wide path's own window shape -- span is
    frames[0]-20 .. frames[-1], i.e. 0.67s of load before the release and `post`
    frames of follow-through after it. Same code was never the same estimator
    until this held. (Found by adversarial review, 2026-08-04.)

    Every value is gated through metric_ranges, so an implausible read is dropped
    here rather than reaching the CSV -- an ungated knee bend produced a p=0.019
    "finding" that collapsed to p=0.44 once gated.
    """
    from shotlab.phase2_pose.form import compute_form
    from shotlab.metric_ranges import in_range

    win = sorted(f for f in poses if rel_f <= f <= rel_f + post)
    if len(win) < 8 or rel_f not in poses:
        return {}

    class _Shot:                       # compute_form needs only these two
        index = 0
        frames = win

    try:
        form = compute_form(_Shot(), {}, poses, fps, handedness="right",
                            camera_angle="side_on",      # the close cam IS profile
                            shooter_height_ft=shooter_height_ft)
    except Exception as e:
        return {"body_error": f"{type(e).__name__}: {e}"}

    out = {}
    for m in form.metrics:
        if m.value is None or not in_range(m.name, m.value):
            continue                    # implausible -> absent, never a number
        out[m.name] = m.value
        out[m.name + "_conf"] = m.confidence
    # Emit the release confidence, un-laundered. The close camera cannot see the
    # ball, so no hand-off can confirm the release and this comes back "low" --
    # which makes correlate.py drop the release-anchored metrics. That gate was
    # silently BYPASSED before, because a missing release_conf reads as None and
    # correlate only rejects a KNOWN-low one (correlate.py:105).
    # release_frame_delta is the independent check: the flare detector's release
    # vs the one compute_form re-finds from pose alone. If that stays tight, a
    # pose-corroborated confidence tier can be argued for with evidence.
    out["release_conf"] = form.release_conf
    out["release_frame"] = int(form.release_frame)
    out["release_frame_delta"] = int(form.release_frame) - int(rel_f)
    return out


def main(argv=None):
    global WIDE_DIR, CLOSE_DIR, OUT, STILLS, PAIRS, POSE_VARIANT, POSE_SMOOTH
    import argparse
    ap = argparse.ArgumentParser(
        description="Elbow-flare stills + flare-vs-make for a 2-camera session.")
    ap.add_argument("--pair", action="append", metavar="WIDE:CLOSE",
                    help="clip stems to pair, repeatable. Default: the 0710 session.")
    ap.add_argument("--wide-dir", default=WIDE_DIR)
    ap.add_argument("--close-dir", default=CLOSE_DIR,
                    help="close-cam clips; defaults to data/raw/Camera 2/upright "
                         "when it exists (the rotation-corrected copies)")
    ap.add_argument("--out", default=OUT, help="session output dir for analysis3d.json")
    ap.add_argument("--pose-variant", default="full", choices=("lite", "full", "heavy"),
                    help="MediaPipe model size. Re-running a session with a different "
                         "variant re-measures the SAME shots with a different estimator, "
                         "which is how within-shot SD / smallest-detectable-change are "
                         "obtained without new footage.")
    ap.add_argument("--no-smooth", action="store_true",
                    help="disable the One-Euro filter (the other reliability perturbation)")
    args = ap.parse_args(argv)

    POSE_VARIANT, POSE_SMOOTH = args.pose_variant, not args.no_smooth
    WIDE_DIR, CLOSE_DIR = args.wide_dir, args.close_dir
    OUT = args.out
    STILLS = os.path.join(OUT, "flare_stills")
    if args.pair:
        PAIRS = []
        for spec in args.pair:
            if ":" not in spec:
                ap.error(f"--pair needs WIDE:CLOSE, got {spec!r}")
            w, c = spec.split(":", 1)
            PAIRS.append((w.strip(), c.strip()))

    print(f"wide  {WIDE_DIR}\nclose {CLOSE_DIR}\nout   {OUT}")
    if not paths.close_cam_is_corrected(CLOSE_DIR):
        alt = os.path.join(CLOSE_DIR, "upright")
        if os.path.isdir(alt):
            print("  ⚠️ using the RAW close-cam dir while a rotation-corrected "
                  "upright/ exists -- pose on sideways frames is garbage")

    all_rows = []
    for w, c in PAIRS:
        print(f"pair {w} <-> {c} ...", flush=True)
        try:
            all_rows.extend(process_pair(w, c))
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
    labeled = [r for r in all_rows if r["made"] is not None]
    print(f"\n{len(all_rows)} releases, {len(labeled)} matched to a make/miss")

    # PERSIST FIRST. Posing four clips costs ~20 minutes, and on 2026-08-03 all of
    # it was lost to a missing-file error in the write step below. Expensive work
    # goes to disk the moment it exists, before anything that can fail.
    if all_rows:
        os.makedirs(OUT, exist_ok=True)
        raw_path = os.path.join(OUT, "flare_readings_raw.json")
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, indent=2)
        print(f"raw readings -> {raw_path}")

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

    # load_or_new: a session that never ran analyze3d.py has no analysis3d.json,
    # and we only want to ADD the flare block to it
    a = Analysis3D.load_or_new(os.path.join(OUT, "analysis3d.json"))
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
