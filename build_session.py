#!/usr/bin/env python
"""Build a whole-session analysis from many clips.

Stitches every clip's rim-anchored shots into one timeline (using the filename
timestamps), then reports fatigue trends, per-zone breakdowns, and make%.

Usage:
  python build_session.py --calib config/calibration_Hoops_20260628.json \
      --clips "data/raw/Hoops/PXL_20260628_19*.mp4" \
      --exclude 190656 191516 191606 \
      --pose --out data/out/session_Hoops

--exclude takes substrings (e.g. the child clips). --pose adds knee bend (slow).
Per-clip detection is cached, so re-runs are fast.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from shotlab.court import Calibration
from shotlab.scale import parse_height
from shotlab.session import (process_clip, build_session, fatigue_trends,
                             consistency_stats)


def chart(df: pd.DataFrame, out_path: str):
    metrics = [m for m in ["release_angle_deg", "entry_angle_deg",
                           "apex_height_ft", "knee_bend_deg", "backspin_rpm"]
               if m in df.columns and df[m].notna().sum() >= 3]
    if not metrics:
        return None
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.4 * n), sharex=True)
    if n == 1:
        axes = [axes]
    import numpy as np
    for ax, m in zip(axes, metrics):
        sub = df[["elapsed_min", m]].dropna()
        ax.scatter(sub["elapsed_min"], sub[m], s=30, alpha=0.7)
        if len(sub) >= 3:
            sl, ic = np.polyfit(sub["elapsed_min"], sub[m], 1)
            xs = np.array([sub["elapsed_min"].min(), sub["elapsed_min"].max()])
            ax.plot(xs, sl * xs + ic, "r--",
                    label=f"slope {sl:+.2f}/min")
            ax.legend(loc="best", fontsize=8)
        ax.set_ylabel(m.replace("_deg", "°").replace("_", " "))
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("minutes into session")
    fig.suptitle("ShotLab — metrics over the session (fatigue view)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default=None,
                    help="calibration JSON; omit to auto-detect the rim per clip")
    ap.add_argument("--clips", required=True, help="glob of clip paths")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="filename substrings to skip (e.g. child clips)")
    ap.add_argument("--detector", default="motion")
    ap.add_argument("--weights", default=None, help="fine-tuned YOLO .pt (for --detector yolo)")
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--stride", default="auto",
                    help="detect every Nth frame; 'auto' = ~40fps + long-clip thinning")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="cap frames processed per clip (fits the job time cap on long slow-mo clips)")
    ap.add_argument("--chunk-frames", type=int, default=None,
                    help="auto-chunk: process clips longer than this in frame windows "
                         "(yolo only) so long exercise clips finish + resume within the job cap")
    ap.add_argument("--pose", action="store_true")
    # default ON since the 2026-07-02 A/B: zero contradictions with the visual
    # call, resolved 9/12 unknowns, classifiable 83%->96% on session 0701
    ap.add_argument("--audio", action=argparse.BooleanOptionalAction, default=True,
                    help="fuse rim/backboard SOUND with the visual make/miss call "
                         "(--no-audio to disable)")
    ap.add_argument("--no-spin", action="store_true")
    ap.add_argument("--handedness", default="right")
    ap.add_argument("--out", default="data/out/session")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--shooter-height", default=None,
                    help="your height (e.g. 5'10\" or 5.83) -> body-scaled, "
                         "honest release/jump heights instead of ~2.5x-hot "
                         "rim-scaled ones")
    ap.add_argument("--tile", action="store_true",
                    help="native-resolution corridor tiling. ⚠️ REGRESSES recall "
                         "on the current downscale-trained model (measured 11->2 "
                         "shots); only useful after a native-scale retrain.")
    ap.add_argument("--validated", action="store_true",
                    help="THE hand-count-validated profile (recall 86%%/precision 0.99, "
                         "make/miss 81%% LOCO): sets --detector yolo --weights "
                         "runs/detect/ball_gpu_kaggle/weights/best.onnx --imgsz 1280 "
                         "--stride 1 --beam --make-model auto, and uses the verify_rim "
                         "config/rim_<clip>.json rims. This is the eval config; plain "
                         "defaults (motion / imgsz 768 / yolo11n) are NOT validated.")
    ap.add_argument("--beam", action="store_true",
                    help="union the multi-hypothesis beam tracker (over the conf-0.01 "
                         "cloud) with the greedy tracker -- recovers fragmented-arc "
                         "shots (validated: recall 55%%->80%% at precision 0.96 across "
                         "3 hand-counted clips). Slower (detects the full cloud).")
    ap.add_argument("--make-model", default="auto",
                    help="learned make/miss model (joblib). 'auto' uses "
                         "models/make_visual_0720.joblib if present, else geometric. "
                         "Geometric classify_make is ~coin-flip on real footage "
                         "(measured); the re-fit visual model is 81%% LOCO. Pass "
                         "'none' to force geometric.")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="ball-detection confidence floor (default 0.25). Lower "
                         "(e.g. 0.05) recovers ~38%% more ball frames for the "
                         "physics/rim gates to filter -- the track-before-detect "
                         "lever.")
    args = ap.parse_args(argv)

    if args.validated:                 # THE eval-validated profile (opt-in)
        args.detector = "yolo"
        args.weights = args.weights or "runs/detect/ball_gpu_kaggle/weights/best.onnx"
        args.imgsz = 1280
        args.stride = "1"
        args.beam = True
        print("VALIDATED profile: yolo/best.onnx, imgsz 1280, stride 1, beam, "
              "make-model auto, verify_rim rims")

    # resolve the learned make/miss model: 'auto' -> the re-fit model if present
    if args.make_model == "auto":
        _cand = os.path.join("models", "make_visual_0720.joblib")
        make_model = _cand if os.path.exists(_cand) else None
    elif args.make_model in ("none", "", None):
        make_model = None
    else:
        make_model = args.make_model
    print(f"make/miss classifier: {'visual ' + os.path.basename(make_model) if make_model else 'geometric (coin-flip -- no model)'}")

    shooter_ft = parse_height(args.shooter_height)

    fixed_calib = Calibration.load(args.calib) if args.calib else None

    # optional map of clip-substring -> calibration name (from calibrate.py)
    cal_map = {}
    map_path = os.path.join("config", "calibration_map.json")
    if fixed_calib is None and os.path.exists(map_path):
        with open(map_path, encoding="utf-8") as f:
            raw = json.load(f)
        for sub, name in raw.items():
            p = os.path.join("config", f"calibration_{name}.json")
            if os.path.exists(p):
                cal_map[sub] = Calibration.load(p)

    def calib_for(clip_path):
        if fixed_calib is not None:
            return fixed_calib
        for sub, cal in cal_map.items():
            if sub in os.path.basename(clip_path):
                return cal
        # verify_rim manual rim (config/rim_<stem>.json) -- the SAME rim the eval
        # uses (auto_calibrate is unreliable on cluttered outdoor footage: it locked
        # onto a shirt on these clips). Single-rim clips only; a mid-clip camera
        # move would need frame-ranged support in process_clip (deferred).
        stem = os.path.splitext(os.path.basename(clip_path))[0]
        if os.path.exists(os.path.join("config", f"rim_{stem}.json")):
            from tools import rim_segments as rs
            doc = rs.load_rims(stem)
            if doc and doc.get("rims"):
                if len(doc["rims"]) > 1:
                    print(f"  ⚠️ {stem}: {len(doc['rims'])} rim segments; using the first "
                          f"(process_clip is single-rim). Re-verify if the camera moved.")
                return rs.calib_at(doc, doc["rims"][0]["f0"])
        return None   # -> auto-detect per clip

    clips = sorted(glob.glob(args.clips))
    clips = [c for c in clips if not any(x in os.path.basename(c) for x in args.exclude)]
    if not clips:
        print("No clips matched.")
        return 1
    print(f"Processing {len(clips)} clips ...")

    all_records = []
    for c in clips:
        recs = process_clip(c, calib_for(c), detector_name=args.detector,
                            weights=args.weights, imgsz=args.imgsz, stride=args.stride,
                            max_frames=args.max_frames, chunk_frames=args.chunk_frames,
                            with_pose=args.pose,
                            with_spin=(False if args.no_spin else "auto"),
                            handedness=args.handedness,
                            use_cache=not args.no_cache, with_audio=args.audio,
                            shooter_height_ft=shooter_ft,
                            tiles="auto" if args.tile else None,
                            conf=args.conf, use_beam=args.beam,
                            make_model=make_model)
        print(f"  {os.path.basename(c)}: {len(recs)} shots")
        all_records.extend(recs)

    df = build_session(all_records)
    os.makedirs(args.out, exist_ok=True)
    if df.empty:
        print("No shots with timestamps found.")
        return 1

    # Preserve USER-entered columns (feel tags + feel-review answers) across
    # rebuilds -- the records are rebuilt with felt_good=None, so without this a
    # rebuild silently wipes the labels the dashboard / voicetag / feel-review
    # persisted into the CSV (they're tier-1 of the profile's good-shot ladder;
    # 2026-07-06 final sweep). NOTE a re-DETECTION can renumber shot_in_clip --
    # remap sidecars first (tools/remap_shot_keys.py), then re-apply the review.
    from shotlab.feelreview import USER_REVIEW_COLS
    _csv = os.path.join(args.out, "session_shots.csv")
    if os.path.exists(_csv):
        old = pd.read_csv(_csv)
        keys = [c for c in ("clip", "shot_in_clip") if c in old.columns and c in df.columns]
        user_cols = [c for c in ("felt_good", "felt_reasons", *USER_REVIEW_COLS)
                     if c in old.columns]
        if keys and user_cols:
            df = df.drop(columns=[c for c in user_cols if c in df.columns], errors="ignore")
            df = df.merge(old[keys + user_cols], on=keys, how="left")

    # The RAW per-shot table is written whole (so a later film-room review can
    # curate it without a rebuild). Every DERIVED table below, though, runs on
    # the CURATED set -- one exclude.json review then cleans the fatigue trends,
    # best-shots ranking, zone/consistency tables, make% and make-drivers alike,
    # instead of leaking human-flagged junk into half the report (2026-07-05
    # audit). On a first build there's no exclude.json yet, so this is a no-op.
    df.to_csv(_csv, index=False)

    from shotlab.curate import apply_excludes, load_excludes
    cdf = apply_excludes(df, args.out)
    ex, lay = load_excludes(args.out)
    if ex or lay:
        print(f"\nCuration: {len(df)} shots -> {len(cdf)} after dropping "
              f"{len(ex)} flagged + {len(lay)} layups (derived tables use the curated set)")

    print(f"\n=== SESSION: {len(cdf)} shots over "
          f"{cdf['elapsed_min'].max():.1f} min ===")

    # fatigue trends
    trends = fatigue_trends(cdf)
    trends.to_csv(os.path.join(args.out, "fatigue_trends.csv"), index=False)
    print("\nFatigue trends (slope per minute; negative = declines as you tire):")
    print(trends.to_string(index=False))

    # coach review (written feedback) + drills + per-shot grades
    from shotlab.coach import (generate_review, review_markdown, grade_shots,
                               recommend_drills)
    from shotlab.session import volume_stats
    review = generate_review(cdf)
    drills = recommend_drills(cdf)
    md = review_markdown(review)
    if drills:
        md += "\n\n### 🏀 Prescribed drills\n" + "\n".join(f"- {x}" for x in drills)
    with open(os.path.join(args.out, "review.md"), "w", encoding="utf-8") as f:
        f.write(md)
    grades = grade_shots(cdf)
    if not grades.empty:
        grades.to_csv(os.path.join(args.out, "shot_grades.csv"), index=False)
    from shotlab.coach import rank_shots
    best = rank_shots(cdf, top=10)
    if not best.empty:
        best.to_csv(os.path.join(args.out, "best_shots.csv"), index=False)
    vs = volume_stats(cdf)
    print("\n" + "=" * 60 + "\nCOACH REVIEW\n" + "=" * 60)
    print(md.replace("**", "").replace("###", ""))
    print(f"\nVolume: {vs['shots']} shots · {vs['makes']}/{vs['attempts']} makes "
          f"({vs['make_pct']}%) · longest make-streak {vs['longest_make_streak']}")

    # consistency (spread + does it worsen with fatigue)
    cons = consistency_stats(cdf)
    if not cons.empty:
        cons.to_csv(os.path.join(args.out, "consistency.csv"), index=False)
        print("\nConsistency (std dev; within-zone removes the position confound):")
        print(cons.to_string(index=False))

    # zone breakdown
    if "zone" in cdf.columns:
        agg = {"shot_num": "count", "release_angle_deg": "mean",
               "entry_angle_deg": "mean", "apex_height_ft": "mean"}
        agg = {k: v for k, v in agg.items() if k in cdf.columns}
        zone = cdf.groupby("zone").agg(agg).round(1)
        zone = zone.rename(columns={"shot_num": "shots"})
        # make% per zone -- WHERE the ball actually goes in. Without it a
        # high-volume low-make spot looked fine (audit D14a: 74% of volume from a
        # 22% spot). Low-confidence like all make/miss here.
        if "made" in cdf.columns:
            m = cdf[cdf["made"].isin([True, False])]
            if len(m):
                mk = m.groupby("zone")["made"]
                zone["attempts"] = mk.count().astype("Int64")
                zone["makes"] = mk.apply(lambda s: int((s == True).sum())).astype("Int64")
                zone["make_pct"] = (100.0 * zone["makes"] / zone["attempts"]).round(0)
        zone.to_csv(os.path.join(args.out, "zone_summary.csv"))
        print("\nBy zone:")
        print(zone.to_string())

    # make%
    if "made" in cdf.columns:
        m = cdf[cdf["made"].isin([True, False])]
        if len(m):
            pct = 100.0 * (m["made"] == True).mean()
            print(f"\nMake% (low confidence): {pct:.0f}% on {len(m)} classifiable shots")
            # make% over time (first vs second half)
            half = cdf["elapsed_min"].median()
            for label, part in [("first half", m[m["elapsed_min"] <= half]),
                                ("second half", m[m["elapsed_min"] > half])]:
                if len(part):
                    print(f"   {label}: {100*(part['made']==True).mean():.0f}% "
                          f"({len(part)} shots)")

    # make-correlation engine: which mechanics track with makes (advisory)
    from shotlab.correlate import correlate_makes, summarize_make_drivers
    assocs = correlate_makes(cdf.to_dict("records"))
    pd.DataFrame([a.as_row() for a in assocs]).to_csv(
        os.path.join(args.out, "make_drivers.csv"), index=False)
    print("\n" + "=" * 60 + "\nMAKE DRIVERS (advisory)\n" + "=" * 60)
    print(summarize_make_drivers(assocs).replace("**", "").replace("_", ""))

    c = chart(cdf, os.path.join(args.out, "session_chart.png"))
    if c:
        print(f"\nchart: {c}")
    print(f"table: {os.path.join(args.out, 'session_shots.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
