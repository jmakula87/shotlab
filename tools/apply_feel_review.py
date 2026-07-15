#!/usr/bin/env python
"""Join feel_review.json answers into session_shots.csv (felt_good + context +
fault tags + miss direction) so export_profile / correlate_feel pick them up.
Same as the dashboard's "Apply reviews" button; idempotent.

Usage:  python tools/apply_feel_review.py --session data/out/session_0710
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.feelreview import apply_review


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    a = ap.parse_args(argv)
    n = apply_review(a.session)
    print(f"applied {n} reviewed shots into {a.session}/session_shots.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
