"""Feel review: the user watches each verified make/miss from BOTH cameras and
records what the cameras can't measure.

Three jobs, one pass (designed with the user 2026-07-15):
  1. FEEL verdict (good/okay/off) -> `felt_good`, the top-priority pool for
     export_profile's personal ideal + correlate_feel.
  2. CONTEXT (movement/setup) -> human-verified keys for per-context ideals
     (the pipeline's own movement_dir/shot_setup guesses are often 'unknown').
  3. FAULT tags -> camera-blind mechanics (guide hand!) plus eye-vs-metric
     cross-checks (e.g. 'elbow flared' vs the 3D flare reading), and the miss
     DIRECTION (short/long = depth, which the wide camera cannot see).

Deliberately NOT asked: knee bend, release/entry angle, jump height -- the
system measures those better than eyes do.

Data: <session>/feel_review.json = {"<clip>|<shot_in_clip>": entry}. Entries
are joined into session_shots.csv by `apply_review` (columns build_session
preserves across rebuilds -- see USER_REVIEW_COLS).
"""

from __future__ import annotations

import json
import os

# ---------------------------------------------------------------- vocabulary
FEEL = ["good", "okay", "off"]
MOVEMENT = ["set/standing", "moving left", "moving right",
            "stepping in", "fading back"]
SETUP = ["catch-and-shoot", "off the dribble"]
FAULTS = {
    "feet/base": ["feet not set", "base crooked/narrow",
                  "drifted sideways in air", "landed off balance",
                  "no legs (all arms)"],
    "arm/hand": ["elbow flared", "ball dipped/long windup", "release too low",
                 "guide hand interfered", "no follow-through hold"],
    "rhythm": ["rushed", "hitched/paused"],
}
MISS_DIR = ["short", "long", "left", "right", "in-and-out"]

# columns apply_review writes; build_session must carry these through a rebuild
USER_REVIEW_COLS = ["feel", "review_movement", "review_setup", "review_tags",
                    "miss_dir", "review_note"]

# wide clip <-> close (S8) clip pairing per two-camera session, filename-matched.
# wide_time = close_time + offset (shotlab.sync.sync_clips convention).
DEFAULT_PAIRS = [("PXL_20260710_175751234.mp4", "20260710_135805"),
                 ("PXL_20260710_180449842.mp4", "20260710_140431"),
                 ("PXL_20260710_181146426.mp4", "20260710_141132"),
                 ("PXL_20260710_181811930.mp4", "20260710_141758")]

# review window: approach footwork ... landing balance (user-approved 2026-07-15)
PRE_S = 3.0     # before the first tracked flight frame (the gather + approach)
POST_S = 1.5    # after the last tracked flight frame (rim + landing)


def review_path(session_dir: str) -> str:
    return os.path.join(session_dir, "feel_review.json")


def load_review(session_dir: str) -> dict:
    p = review_path(session_dir)
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_entry(session_dir: str, key: str, entry: dict) -> dict:
    """Merge one shot's review into feel_review.json (saved per answer, so a
    review session can stop/resume at any point)."""
    review = load_review(session_dir)
    review[key] = entry
    with open(review_path(session_dir), "w", encoding="utf-8") as f:
        json.dump(review, f, indent=1)
    return review


def review_candidates(df, truth: dict | None) -> list[dict]:
    """Shots the user should review, chronological: verified makes/misses when
    ground truth exists (non-shots never appear), else the curated heuristic
    calls. `df` = the session_shots dataframe (already curated upstream)."""
    out = []
    for _, r in df.iterrows():
        key = f"{r['clip']}|{int(r['shot_in_clip'])}"
        if truth:
            label = truth.get(key)
            if label not in ("make", "miss"):
                continue
            made = label == "make"
        else:
            if r.get("made") not in (True, "True", False, "False"):
                continue
            made = r.get("made") in (True, "True")
        out.append({"key": key, "clip": str(r["clip"]),
                    "shot_in_clip": int(r["shot_in_clip"]), "made": made,
                    "shot_num": int(r["shot_num"]) if "shot_num" in r else None})
    return out


def shot_windows(wide_path: str) -> dict[int, tuple[float, float]]:
    """{shot_in_clip: (t0, t1)} review window in WIDE-clip seconds, from the
    cached detection + real PTS: PRE_S before the flight through POST_S after."""
    from .detect_cache import _path, deserialize_detection
    from .video_io import frame_times_cached
    with open(_path(wide_path), encoding="utf-8") as f:
        _, shots = deserialize_detection(json.load(f))
    times = frame_times_cached(wide_path) or {}
    out = {}
    for s in shots:
        f0, f1 = int(s.frames[0]), int(s.frames[-1])
        t0 = times.get(f0, f0 / 30.0)
        t1 = times.get(f1, f1 / 30.0)
        out[int(s.index)] = (max(0.0, t0 - PRE_S), t1 + POST_S)
    return out


def close_window(wide_t0: float, wide_t1: float, offset: float
                 ) -> tuple[float, float]:
    """Map a wide-clip window onto the close clip. wide_time = close_time +
    offset, so close_time = wide_time - offset."""
    return max(0.0, wide_t0 - offset), wide_t1 - offset


def apply_review(session_dir: str) -> int:
    """Join feel_review.json into session_shots.csv: felt_good (good->True,
    off->False, okay stays None/neutral) + the review columns. Returns the
    number of shots updated. Idempotent -- re-run any time."""
    import pandas as pd
    review = load_review(session_dir)
    csv = os.path.join(session_dir, "session_shots.csv")
    df = pd.read_csv(csv)
    for col in USER_REVIEW_COLS:
        if col not in df.columns:
            df[col] = None
    df["felt_good"] = df.get("felt_good")
    n = 0
    for i, r in df.iterrows():
        e = review.get(f"{r['clip']}|{int(r['shot_in_clip'])}")
        if not e:
            continue
        feel = e.get("feel")
        df.at[i, "felt_good"] = (True if feel == "good"
                                 else False if feel == "off" else None)
        df.at[i, "feel"] = feel
        df.at[i, "review_movement"] = e.get("movement")
        df.at[i, "review_setup"] = e.get("setup")
        df.at[i, "review_tags"] = ";".join(e.get("tags", [])) or None
        df.at[i, "miss_dir"] = e.get("miss_dir")
        df.at[i, "review_note"] = e.get("note") or None
        n += 1
    df.to_csv(csv, index=False)
    return n
