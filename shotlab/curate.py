"""Per-session shot curation: exclude junk shots (bad pose on the court, false
non-shot detections) and separate layups from the jump-shot analysis.

Shot detection + far-shot pose are imperfect, and a quick human review catches
what heuristics can't. The verdicts live in <session>/exclude.json:

  {"exclude": [13, 15, 54, ...],   # not a real jump shot / pose failed -> drop
   "layups":  [6, 35, 59]}         # real makes, wrong TYPE -> out of jumper study

Everything form-facing (profile, film room, coaching galleries) filters through
apply_excludes so one review cleans the whole analysis.
"""

from __future__ import annotations

import json
import os


def load_excludes(session_dir):
    """Return (excluded_shot_nums, layup_shot_nums) as sets of ints."""
    p = os.path.join(session_dir, "exclude.json")
    if not os.path.exists(p):
        return set(), set()
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
    except (json.JSONDecodeError, OSError):
        return set(), set()
    return set(int(x) for x in d.get("exclude", [])), \
        set(int(x) for x in d.get("layups", []))


def apply_excludes(df, session_dir, drop_layups=True):
    """Drop excluded (and, by default, layup) shots from a session dataframe by
    shot_num. No-op when there's no exclude.json or no shot_num column."""
    if "shot_num" not in df.columns:
        return df
    ex, lay = load_excludes(session_dir)
    drop = ex | (lay if drop_layups else set())
    if not drop:
        return df
    return df[~df["shot_num"].astype("Int64").isin(drop)]


def save_excludes(session_dir, exclude=None, layups=None):
    """Write/replace the exclude.json (used by the curation UI / setup)."""
    os.makedirs(session_dir, exist_ok=True)
    with open(os.path.join(session_dir, "exclude.json"), "w", encoding="utf-8") as f:
        json.dump({"exclude": sorted(set(int(x) for x in (exclude or []))),
                   "layups": sorted(set(int(x) for x in (layups or [])))}, f, indent=2)
