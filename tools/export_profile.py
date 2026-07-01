#!/usr/bin/env python
"""Build the phone app's profile.json = YOUR ideal form, learned from your best
shots. The app ships this to the phone and compares each live shot against it.

v1 (this file): ideal metric targets + tolerances, computed straight from a
session's CSV -- no video needed. "Good" shots are chosen by priority:
  1. shots you tagged felt_good = True   (your own feel labels win)
  2. best_shots.csv                       (coach-ranked clean form + makes)
  3. made shots
  4. all shots                            (fallback -- flagged in the note)

The ideal is the MEAN of the good shots; the tolerance is their spread (with a
floor so a lucky-tight session doesn't set an impossibly narrow band).

Usage:  python tools/export_profile.py data/out/session_0629_full [--out app/profile.json]

(Ideal per-phase skeletons for the overlay come in v2 via a pose re-run.)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (metric column, minimum tolerance so the band never collapses)
IDEAL_METRICS = [
    ("elbow_angle_at_release_deg", 6.0),
    ("knee_bend_deg", 8.0),
    ("tempo_dip_to_release_s", 0.05),
    ("follow_through_hold_s", 0.10),
    ("balance_drift_px_per_ht", 0.05),
    ("release_angle_deg", 5.0),
    ("entry_angle_deg", 5.0),
]


def _truthy(s):
    return s.isin([True, "True"])


def select_good(df: pd.DataFrame, session_dir: str, min_good: int = 5):
    """Return (good_df, method) using the priority above."""
    if "felt_good" in df.columns:
        good = df[_truthy(df["felt_good"])]
        if len(good) >= min_good:
            return good, "your felt-good shots"
    best_csv = os.path.join(session_dir, "best_shots.csv")
    if os.path.exists(best_csv):
        best = pd.read_csv(best_csv)
        key = "shot_num" if "shot_num" in best.columns and "shot_num" in df.columns else None
        if key:
            good = df[df[key].isin(best[key])]
            if len(good) >= min(min_good, len(best)):
                return good, "your best-ranked shots"
    if "made" in df.columns:
        good = df[_truthy(df["made"])]
        if len(good) >= min_good:
            return good, "your made shots"
    return df, "all shots (not enough good ones tagged/made — tag shots by feel!)"


def build_profile(df: pd.DataFrame, session_dir: str, *, name="me",
                  handedness="right", min_good=5, with_skeletons=True,
                  raw_dirs=None) -> dict:
    good, method = select_good(df, session_dir, min_good)
    ideal, tol = {}, {}
    for col, floor in IDEAL_METRICS:
        if col not in good.columns:
            continue
        vals = good[col].dropna()
        if len(vals) < 3:
            continue
        ideal[col] = round(float(vals.mean()), 2)
        tol[col] = round(float(max(vals.std(ddof=0) if len(vals) > 1 else floor, floor)), 2)

    skeletons = {"load": None, "release": None, "follow": None}
    skel_note = ""
    if with_skeletons and {"clip", "shot_in_clip"}.issubset(good.columns):
        skeletons, skel_note = build_ideal_skeletons(good, handedness, raw_dirs)

    profile = {
        "name": name,
        "handedness": handedness,
        "note": f"Ideal learned from {len(good)} {method}." + (
            f" {skel_note}" if skel_note else ""),
        "n_good": int(len(good)),
        "source_method": method,
        "ideal": ideal,
        "tolerance": tol,
        "skeletons": skeletons,
    }
    return profile


def build_ideal_skeletons(good: pd.DataFrame, handedness, raw_dirs):
    """Build per-phase ideal skeletons from the good shots' raw clips (v2). Falls
    back to null skeletons + a note if pose can't run (missing raw clips)."""
    from shotlab.skeleton import build_skeletons
    if raw_dirs is None:
        raw_dirs = [os.path.join("data", "raw", "Hoops"), os.path.join("data", "raw")]
    pairs = list(zip(good["clip"].astype(str), good["shot_in_clip"]))
    try:
        skeletons, stats = build_skeletons(pairs, raw_dirs, handedness=handedness)
    except Exception as e:                       # raw clip gone / pose unavailable
        return {"load": None, "release": None, "follow": None}, \
               f"(skeletons skipped: {e})"
    got = [p for p, v in skeletons.items() if v]
    if not got:
        return skeletons, "(no clean poses for skeletons -- raw clips missing?)"
    n = stats.get("release", 0)
    return skeletons, f"Ideal skeletons from {n} clean-pose shots ({', '.join(got)})."


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--out", default=os.path.join("app", "profile.json"))
    ap.add_argument("--name", default="me")
    ap.add_argument("--handedness", default="right")
    ap.add_argument("--no-skeletons", action="store_true",
                    help="skip the v2 ideal-skeleton pose re-run (metrics only)")
    ap.add_argument("--raw-dir", action="append", default=None,
                    help="where the raw clips live (repeatable); "
                         "default data/raw/Hoops then data/raw")
    args = ap.parse_args(argv)

    csv = os.path.join(args.session_dir, "session_shots.csv")
    if not os.path.exists(csv):
        print(f"no session_shots.csv in {args.session_dir}")
        return 1
    df = pd.read_csv(csv)
    profile = build_profile(df, args.session_dir, name=args.name,
                            handedness=args.handedness,
                            with_skeletons=not args.no_skeletons,
                            raw_dirs=args.raw_dir)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    print(f"wrote {args.out}")
    print(f"  {profile['note']}")
    print(f"  ideal: {profile['ideal']}")
    got = [p for p, v in profile["skeletons"].items() if v]
    print(f"  skeletons: {got or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
