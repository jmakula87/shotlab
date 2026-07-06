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

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Arc/ball ideals: measured off the ball flight, present on every tracked shot.
# Rank the pool for these by ARC quality (clean-arc / made shots).
ARC_METRICS = [
    ("release_angle_deg", 5.0),
    ("entry_angle_deg", 5.0),
]
# Form/pose ideals: only present when pose ran cleanly. Rank the pool for these
# by POSE quality -- ranking them by arc quality (as the arc pool does) lands on
# far, clean-arc shots with no usable pose, which is the 2026-07-02 regression
# where the ideal lost its elbow/follow-through targets.
FORM_METRICS = [
    ("elbow_angle_at_release_deg", 6.0),
    ("knee_bend_deg", 8.0),
    ("tempo_dip_to_release_s", 0.10),        # 0.05 was below 30fps resolution (audit D6)
    ("follow_through_hold_s", 0.10),
    ("balance_drift_px_per_ht", 0.05),
    # release timing vs jump apex: his CLEANEST make-driver (d=0.58 on trustworthy
    # shots) -- now scored so the app coaches toward releasing at the top of his
    # jump (2026-07-06 audit product rec).
    ("release_vs_apex_s", 0.06),
]
# The most-available reliable form metric (knee side-on is in-plane, so
# high-confidence); its presence means pose ran on the shot.
_POSE_ANCHOR = "knee_bend_deg"
# (kept for back-compat: some callers import the combined list)
IDEAL_METRICS = FORM_METRICS + ARC_METRICS


def _truthy(s):
    return s.isin([True, "True"])


def select_good(df: pd.DataFrame, session_dir: str, min_good: int = 5):
    """Outcome/arc 'good shots' (feel > best-ranked > made > all). These have
    clean arcs, so they anchor the ARC ideals (release/entry angle)."""
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


def select_form_good(df: pd.DataFrame, min_good: int = 5):
    """Pose-reliable good shots for the FORM ideals + skeletons.

    Anchors on a present pose metric (knee bend), then walks the same
    feel > made > all ladder -- so the elbow/knee/follow-through ideals come
    from shots where pose actually ran, not from clean-arc shots the pose model
    couldn't read. Within each tier, higher release-confidence sorts first
    (elbow-at-release trusts the release frame)."""
    pose = df[df[_POSE_ANCHOR].notna()] if _POSE_ANCHOR in df.columns else df.iloc[0:0]
    if len(pose) == 0:
        return df, "all shots (no pose-reliable shots)"
    if "release_conf" in pose.columns:
        order = {"high": 0, "medium": 1, "low": 2}
        pose = (pose.assign(_c=pose["release_conf"].map(order).fillna(3))
                    .sort_values("_c", kind="stable").drop(columns="_c"))
    if "felt_good" in pose.columns:
        g = pose[_truthy(pose["felt_good"])]
        if len(g) >= min_good:
            return g, "your felt-good shots (pose-reliable)"
    if "made" in pose.columns:
        g = pose[_truthy(pose["made"])]
        if len(g) >= min_good:
            return g, "your made shots (pose-reliable)"
    return pose, "pose-reliable shots"


from shotlab.metric_ranges import VALID_RANGE as _VALID_RANGE  # shared artifact gate


def _add_ideal(ideal: dict, tol: dict, pool: pd.DataFrame, col: str, floor: float):
    """Set ideal[col] = MEDIAN and tol[col] = robust spread (floored) over the
    pool's in-range values, if >=3 remain; otherwise leave it out.

    Median + MAD (not mean + std) so one splayed-skeleton outlier can't drag the
    target or blow the tolerance so wide it flags nothing (2026-07-05 audit: a
    5.6 balance-drift outlier and 172 deg knee artifacts were doing exactly
    that). Out-of-range artifacts are dropped before aggregating."""
    if col not in pool.columns:
        return
    vals = pool[col].dropna()
    lo, hi = _VALID_RANGE.get(col, (None, None))
    if lo is not None:
        vals = vals[vals >= lo]
    if hi is not None:
        vals = vals[vals <= hi]
    if len(vals) < 3:
        return
    v = vals.to_numpy(dtype=float)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    robust_std = 1.4826 * mad                 # MAD -> std-equivalent for normals
    ideal[col] = round(med, 2)
    tol[col] = round(float(max(robust_std, floor)), 2)


def build_profile(df: pd.DataFrame, session_dir: str, *, name="me",
                  handedness="right", min_good=5, with_skeletons=True,
                  raw_dirs=None) -> dict:
    arc_good, arc_method = select_good(df, session_dir, min_good)
    form_good, form_method = select_form_good(df, min_good)

    # `ideal` = the app's SCORED targets -- only trustworthy in-plane form
    # metrics. The arc angles go in `diagnostic` (shown, never scored): a single
    # wide camera foreshortens them, so they read high and can't be a real target
    # until court-corner / 2-cam calibration (2026-07-05 audit).
    ideal, tol = {}, {}
    diagnostic, diag_tol = {}, {}
    for col, floor in FORM_METRICS:
        _add_ideal(ideal, tol, form_good, col, floor)
    for col, floor in ARC_METRICS:
        _add_ideal(diagnostic, diag_tol, arc_good, col, floor)

    skeletons = {"load": None, "release": None, "follow": None}
    skel_note = ""
    if with_skeletons and {"clip", "shot_in_clip"}.issubset(form_good.columns):
        skeletons, skel_note = build_ideal_skeletons(form_good, handedness, raw_dirs)

    note = (f"Arc ideals from {len(arc_good)} {arc_method}; "
            f"form ideals from {len(form_good)} {form_method}.")
    if skel_note:
        note += f" {skel_note}"
    from shotlab.textbook import profile_block
    profile = {
        "name": name,
        "handedness": handedness,
        "note": note,
        "n_good": int(len(arc_good)),
        "n_form": int(len(form_good)),
        "source_method": arc_method,
        "form_method": form_method,
        "ideal": ideal,            # YOUR own norm (median of your good shots), SCORED
        "tolerance": tol,
        # foreshortened arc angles: shown for reference, NOT scored (see above)
        "diagnostic": diagnostic,
        "diagnostic_tolerance": diag_tol,
        "textbook": profile_block(),  # universal targets, SEPARATE from `ideal`
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
    from shotlab.curate import apply_excludes, load_excludes
    n0 = len(df)
    df = apply_excludes(df, args.session_dir)     # drop curated junk + layups
    ex, lay = load_excludes(args.session_dir)
    if ex or lay:
        print(f"  curation: dropped {n0 - len(df)} shots "
              f"({len(ex)} excluded, {len(lay)} layups)")
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
