#!/usr/bin/env python
"""Build the film room for a session: per-shot phase closeups (load / release /
follow, cropped tight to the body + skeleton) and an arrow-key gallery HTML.

Usage:
  python tools/film_room.py data/out/session_0703 --which makes
Writes <session>/film_room.html (+ closeups/ cache).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.closeups import build_shot_closeups, film_room_html


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--which", choices=["makes", "misses", "all"], default="makes")
    ap.add_argument("--handedness", default="auto")
    args = ap.parse_args(argv)
    only = {"makes": True, "misses": False, "all": None}[args.which]

    print(f"building {args.which} closeups (pose re-run per shot) ...")
    cl = build_shot_closeups(args.session_dir, only_made=only,
                             handedness=args.handedness)
    print(f"{len(cl)} shots with closeups")
    if not cl:
        print("no closeups built"); return 1
    html = film_room_html(cl)
    out = os.path.join(args.session_dir, "film_room.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
