"""Remap shot-keyed sidecar files after a re-detection renumbers shots.

Ground-truth labels (make_truth.json, keyed "clip.mp4|shot_in_clip") and
curation (exclude.json, keyed by session-level shot_num) survive a rebuild only
if the numbering didn't change -- but a re-detection CAN change it (e.g. the
v21 chunk-seam fix recovers shots the old windows lost, shifting every number
after the insertion). Shots are matched OLD->NEW by frame-range overlap within
each clip (two detections of the same flight overlap; distinct flights never
do), which is immune to renumbering.

Before rebuilding, snapshot the old numbering (see shot_map_pre_v21.json:
{clip.mp4: {shot_in_clip: [first_frame, last_frame]}}) and copy
session_shots.csv to session_shots_pre_v21.csv. Then rebuild, then:

    python tools/remap_shot_keys.py data/out/session_0710            # dry-run
    python tools/remap_shot_keys.py data/out/session_0710 --apply

--apply rewrites make_truth.json / exclude.json in place (originals kept as
*_pre_v21.json). Unmatched old shots (detection genuinely changed) are
reported and dropped -- re-label those in the audit view.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd


def _load_new_map(clips) -> dict:
    """{clip.mp4: {shot_in_clip: (f0, f1)}} from the CURRENT track caches."""
    out = {}
    for clip in clips:
        stem = os.path.splitext(clip)[0]
        tp = os.path.join("data", "out", stem, f"{stem}_track.json")
        if not os.path.exists(tp):
            print(f"  !! no track cache for {clip} -- skipping")
            continue
        with open(tp, encoding="utf-8") as f:
            data = json.load(f)
        out[clip] = {int(s["index"]): (int(s["frames"][0]), int(s["frames"][-1]))
                     for s in data["shots"]}
    return out


def _match(old: dict, new: dict) -> dict:
    """old shot_in_clip -> new shot_in_clip by max frame-range overlap."""
    mapping, used = {}, set()
    for oi, (of0, of1) in sorted(old.items()):
        best, best_ov = None, 0
        for ni, (nf0, nf1) in new.items():
            if ni in used:
                continue
            ov = min(of1, nf1) - max(of0, nf0) + 1
            if ov > best_ov:
                best, best_ov = ni, ov
        if best is not None and best_ov > 0:
            mapping[oi] = best
            used.add(best)
    return mapping


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session", help="session dir, e.g. data/out/session_0710")
    ap.add_argument("--apply", action="store_true",
                    help="rewrite the sidecars (default: dry-run report)")
    a = ap.parse_args(argv)
    d = a.session

    snap_p = os.path.join(d, "shot_map_pre_v21.json")
    if not os.path.exists(snap_p):
        print(f"no {snap_p} -- nothing to remap against")
        return 1
    with open(snap_p, encoding="utf-8") as f:
        old_map = {c: {int(k): (int(v[0]), int(v[1])) for k, v in m.items()}
                   for c, m in json.load(f).items()}
    new_map = _load_new_map(list(old_map))

    per_clip = {}          # clip -> {old shot_in_clip: new shot_in_clip}
    changed = unmatched = 0
    for clip, old in old_map.items():
        m = _match(old, new_map.get(clip, {}))
        per_clip[clip] = m
        miss = sorted(set(old) - set(m))
        gained = sorted(set(new_map.get(clip, {})) - set(m.values()))
        moved = {o: n for o, n in m.items() if o != n}
        changed += len(moved)
        unmatched += len(miss)
        print(f"{clip}: {len(old)} old -> {len(new_map.get(clip, {}))} new | "
              f"renumbered {len(moved)}{' ' + str(moved) if moved else ''}"
              f"{' | UNMATCHED old ' + str(miss) if miss else ''}"
              f"{' | NEW (unlabeled) ' + str(gained) if gained else ''}")

    # --- make_truth.json ("clip.mp4|shot_in_clip" -> label) ------------------
    tp = os.path.join(d, "make_truth.json")
    if os.path.exists(tp):
        with open(tp, encoding="utf-8") as f:
            truth = json.load(f)
        new_truth, dropped = {}, []
        for key, label in truth.items():
            clip, _, idx = key.rpartition("|")
            ni = per_clip.get(clip, {}).get(int(idx))
            if ni is None:
                dropped.append(key)
            else:
                new_truth[f"{clip}|{ni}"] = label
        print(f"make_truth: {len(truth)} labels -> {len(new_truth)} remapped"
              f"{', DROPPED ' + str(dropped) if dropped else ''}")
        if a.apply:
            shutil.copy(tp, os.path.join(d, "make_truth_pre_v21.json"))
            with open(tp, "w", encoding="utf-8") as f:
                json.dump(new_truth, f, indent=1)

    # --- exclude.json (session-level shot_num lists) --------------------------
    ep = os.path.join(d, "exclude.json")
    old_csv = os.path.join(d, "session_shots_pre_v21.csv")
    new_csv = os.path.join(d, "session_shots.csv")
    if os.path.exists(ep) and os.path.exists(old_csv) and os.path.exists(new_csv):
        odf = pd.read_csv(old_csv)
        ndf = pd.read_csv(new_csv)
        old_by_num = {int(r.shot_num): (r.clip, int(r.shot_in_clip))
                      for r in odf.itertuples()}
        new_num = {(r.clip, int(r.shot_in_clip)): int(r.shot_num)
                   for r in ndf.itertuples()}
        with open(ep, encoding="utf-8") as f:
            exc = json.load(f)
        out, drops = {}, []
        for field in ("exclude", "layups"):
            vals = []
            for num in exc.get(field, []):
                if int(num) not in old_by_num:
                    drops.append(f"{field}:{num}")
                    continue
                clip, oi = old_by_num[int(num)]
                ni = per_clip.get(clip, {}).get(oi)
                nn = new_num.get((clip, ni)) if ni is not None else None
                if nn is None:
                    drops.append(f"{field}:{num}")
                else:
                    vals.append(nn)
            out[field] = sorted(set(vals))
        print(f"exclude.json: {exc} -> {out}"
              f"{' | DROPPED ' + str(drops) if drops else ''}")
        if a.apply:
            shutil.copy(ep, os.path.join(d, "exclude_pre_v21.json"))
            with open(ep, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)

    print(f"\nsummary: {changed} renumbered, {unmatched} old shots unmatched"
          + ("" if a.apply else "  (dry-run -- nothing written)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
