#!/usr/bin/env python
"""Data hygiene: archive a session's KEEPERS (tiny metrics + a few review clips)
into a permanent per-session folder, then optionally purge the gigabytes.

Philosophy: the metrics are tiny and precious (your progress record); raw clips
and intermediate renders are huge and disposable once a session is processed and
you've pulled the clips you want to study.

Archive layout (small, keep forever):
    data/sessions/<name>/
        metrics/   session_shots.csv, review.md, best_shots.csv, *_trends.csv,
                   consistency.csv, zone_summary.csv, session_chart.png, report.html
        caches/    per-clip *_shots_session.json (lets you rebuild the session)
        clips/     h264 review clips for your best + worst shots

Usage:
    # archive keepers (safe; copies only)
    python tools/curate.py --session data/out/session_Hoops --name 2026-06-28_Hoops

    # then reclaim space (each is opt-in; prints what it removes)
    python tools/curate.py --name 2026-06-28_Hoops --purge-intermediate   # mp4v + overlays + _frames
    python tools/curate.py --purge-zips                                    # extracted zips
    python tools/curate.py --purge-raw "data/raw/Hoops/*.mp4"              # raw clips (after archiving!)
    python tools/curate.py --purge-dataset                                 # training images (model kept)
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil

METRIC_FILES = ["session_shots.csv", "review.md", "best_shots.csv",
                "shot_grades.csv", "consistency.csv", "fatigue_trends.csv",
                "zone_summary.csv", "session_chart.png", "report.html"]


def _size(paths):
    return sum(os.path.getsize(p) for p in paths if os.path.isfile(p))


def _fmt(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.1f}{u}"
        n /= 1024


def archive(session_dir, name, keep_best, keep_worst):
    dst = os.path.join("data", "sessions", name)
    os.makedirs(os.path.join(dst, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(dst, "caches"), exist_ok=True)
    os.makedirs(os.path.join(dst, "clips"), exist_ok=True)

    for fn in METRIC_FILES:
        src = os.path.join(session_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst, "metrics", fn))

    # per-clip metric + detection caches (small; let you rebuild the session and
    # re-analyze the tracks without re-detecting, even after raw is deleted)
    for pat in ("*_shots_session.json", "*_track.json"):
        for c in glob.glob(os.path.join("data", "out", "*", pat)):
            shutil.copy2(c, os.path.join(dst, "caches", os.path.basename(c)))

    # review clips: best + worst shots (h264 only)
    import pandas as pd
    picks = []
    bp = os.path.join(session_dir, "best_shots.csv")
    if os.path.exists(bp):
        b = pd.read_csv(bp).head(keep_best)
        picks += [(r["clip"], int(r["shot_in_clip"]), "best") for _, r in b.iterrows()
                  if "shot_in_clip" in r and r["shot_in_clip"] == r["shot_in_clip"]]
    # worst = 'off'-graded + missed shots, by how far entry is from ~45
    sp = os.path.join(session_dir, "session_shots.csv")
    gp = os.path.join(session_dir, "shot_grades.csv")
    if os.path.exists(sp) and os.path.exists(gp):
        df = pd.read_csv(sp)
        g = pd.read_csv(gp)[["shot_num", "grade"]]
        m = df.merge(g, on="shot_num")
        w = m[(m["grade"] == "off") & (m["made"] == False)].copy()
        if "entry_angle_deg" in w:
            w["bad"] = (w["entry_angle_deg"] - 45).abs()
            w = w.sort_values("bad", ascending=False).head(keep_worst)
        picks += [(r["clip"], int(r["shot_in_clip"]), "worst") for _, r in w.iterrows()
                  if "shot_in_clip" in r and r["shot_in_clip"] == r["shot_in_clip"]]
    copied = 0
    for clip, sic, tag in picks:
        stem = os.path.splitext(clip)[0]
        src = os.path.join("data", "out", stem, "shots", f"shot_{sic}_h264.mp4")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst, "clips", f"{tag}_{stem}_shot{sic}.mp4"))
            copied += 1
    # comparison stills (study material; need raw to regenerate, so preserve them)
    comps = glob.glob(os.path.join("data", "out", "comparisons", "*.png"))
    if comps:
        os.makedirs(os.path.join(dst, "comparisons"), exist_ok=True)
        for c in comps:
            shutil.copy2(c, os.path.join(dst, "comparisons", os.path.basename(c)))

    arc_sz = _size(glob.glob(os.path.join(dst, "**", "*"), recursive=True))
    print(f"archived -> {dst}  ({copied} review clips, {_fmt(arc_sz)} total)")
    if copied == 0:
        print("  note: no rendered review clips found — run tools/render_shots.py "
              "on the clips you want, then re-curate.")
    return dst


def purge_intermediate():
    targets = []
    targets += glob.glob(os.path.join("data", "out", "*", "shots", "shot_*.mp4"))
    targets = [t for t in targets if "_h264" not in t]          # keep h264
    targets += glob.glob(os.path.join("data", "out", "*", "*_overlay.mp4"))  # keep h264
    targets += glob.glob(os.path.join("data", "out", "*overlay.mp4"))
    targets += glob.glob(os.path.join("data", "out", "_frames", "*"))
    targets += glob.glob(os.path.join("data", "out", "*.log"))
    targets += glob.glob(os.path.join("data", "out", "yolo_track_*.json"))
    # long-clip chunker's per-window resume caches: redundant once the clip's
    # _shots_session.json + _track.json are written (those hold the merged result)
    targets += glob.glob(os.path.join("data", "out", "*", "*_chunk_*.json"))
    _remove(targets, "intermediate renders/logs")


def purge_zips():
    _remove(glob.glob(os.path.join("data", "raw", "*.zip")), "extracted zip files")


def purge_raw(pattern):
    _remove(glob.glob(pattern), f"raw clips ({pattern})")


def purge_dataset():
    targets = glob.glob(os.path.join("dataset_ball", "images", "*", "*")) + \
              glob.glob(os.path.join("dataset_ball", "labels", "*", "*"))
    _remove(targets, "training images/labels (model in runs/ is kept)")


def _remove(paths, label):
    paths = [p for p in paths if os.path.isfile(p)]
    sz = _size(paths)
    for p in paths:
        os.remove(p)
    print(f"purged {len(paths)} {label} files, reclaimed {_fmt(sz)}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", help="session out dir to archive (e.g. data/out/session_Hoops)")
    ap.add_argument("--name", help="archive name (e.g. 2026-06-28_Hoops)")
    ap.add_argument("--keep-best", type=int, default=5)
    ap.add_argument("--keep-worst", type=int, default=5)
    ap.add_argument("--purge-intermediate", action="store_true")
    ap.add_argument("--purge-zips", action="store_true")
    ap.add_argument("--purge-raw", metavar="GLOB")
    ap.add_argument("--purge-dataset", action="store_true")
    args = ap.parse_args(argv)

    if args.session and args.name:
        archive(args.session, args.name, args.keep_best, args.keep_worst)
    if args.purge_intermediate:
        purge_intermediate()
    if args.purge_zips:
        purge_zips()
    if args.purge_raw:
        purge_raw(args.purge_raw)
    if args.purge_dataset:
        purge_dataset()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
