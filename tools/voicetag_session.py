#!/usr/bin/env python
"""Apply post-session VOICE tags to a built session.

Once you've recorded a workout narrating each shot ("good" / "bad, flare" / ...)
and run build_session, this reads each clip's audio, transcribes the phrases,
maps each to the shot you'd just taken (by time), and writes `felt_good` +
`felt_reasons` into session_shots.csv -- so `export_profile` then builds your
profile from the shots you called good.

Usage:
  python tools/voicetag_session.py data/out/session --model models/vosk-small
                                   [--raw-dir data/raw/Hoops]

`--model` is an offline Vosk model dir (`pip install vosk` + a small model from
https://alphacephei.com/vosk/models). The transcription is the only part that
needs your real audio to validate; the phrase->shot mapping + CSV write are
tested (tests/test_voicetag.py).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.session import parse_clip_time
from shotlab.voicetag import assign_to_shots, transcribe_vosk


def _clip_audio_path(clip_name, raw_dirs):
    for d in raw_dirs:
        p = os.path.join(d, clip_name)
        if os.path.exists(p):
            return p
    return None


def apply_tags_to_session(session_dir, *, model_path=None, phrases_by_clip=None,
                          raw_dirs=None, max_gap_s=4.0, write=True):
    """Tag a session from voice. `phrases_by_clip` (clip -> [{t, text}]) injects
    a transcript for testing; otherwise each clip's audio is transcribed with
    Vosk (needs model_path). Returns a summary dict; writes the CSV in place
    when `write`."""
    raw_dirs = raw_dirs or [os.path.join("data", "raw", "Hoops"),
                            os.path.join("data", "raw")]
    csv = os.path.join(session_dir, "session_shots.csv")
    df = pd.read_csv(csv)
    if "felt_good" not in df.columns:
        df["felt_good"] = None
    if "felt_reasons" not in df.columns:
        df["felt_reasons"] = ""

    n_good = n_bad = 0
    reasons_seen = {}
    for clip, group in df.groupby("clip"):
        clip_start = parse_clip_time(clip)
        if clip_start is None:
            continue
        shots = []
        for i, row in group.iterrows():
            rel = (pd.to_datetime(row["abs_time"]) - clip_start).total_seconds()
            shots.append({"id": int(row["shot_in_clip"]), "t": float(rel), "row": i})

        if phrases_by_clip is not None:
            phrases = phrases_by_clip.get(clip, [])
        else:
            audio = _clip_audio_path(clip, raw_dirs)
            if audio is None:
                print(f"  {clip}: raw clip not found, skipped")
                continue
            phrases = transcribe_vosk(audio, model_path)

        tags = assign_to_shots(phrases, shots, max_gap_s=max_gap_s)
        by_id = {s["id"]: s["row"] for s in shots}
        for sid, tag in tags.items():
            row = by_id.get(sid)
            if row is None:
                continue
            df.at[row, "felt_good"] = (tag["outcome"] == "good")
            df.at[row, "felt_reasons"] = ",".join(tag["reasons"])
            if tag["outcome"] == "good":
                n_good += 1
            else:
                n_bad += 1
            for r in tag["reasons"]:
                reasons_seen[r] = reasons_seen.get(r, 0) + 1

    if write:
        df.to_csv(csv, index=False)
    return {"good": n_good, "bad": n_bad, "reasons": reasons_seen,
            "tagged": n_good + n_bad, "shots": len(df)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--model", default=None, help="offline Vosk model dir")
    ap.add_argument("--raw-dir", action="append", default=None)
    args = ap.parse_args(argv)
    if not args.model:
        print("need --model (an offline Vosk model dir) to transcribe audio")
        return 1
    s = apply_tags_to_session(args.session_dir, model_path=args.model,
                              raw_dirs=args.raw_dir)
    print(f"tagged {s['tagged']}/{s['shots']} shots: "
          f"{s['good']} good, {s['bad']} bad  reasons={s['reasons']}")
    print("now re-run tools/export_profile.py to rebuild your profile from "
          "the felt-good shots.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
