"""BLIND make/miss labelling over the pre-cut per-shot review clips.

Why this exists rather than the dashboard audit view: that view showed the
pipeline's and the model's call BEFORE you answered, preselected that call, and
put "Save & next" one click away -- so clicking through wrote the MODEL's opinion
into make_truth.json, the file the model is then trained on. Both 2026-08-04
reviewers called it a rubber-stamping machine. It has since been failed closed,
but confirmatory labels want a purpose-built tool.

The design rules, all deliberate:
  * NOTHING is shown that could anchor you -- no pipeline call, no model call, no
    probability, no prior label until you ask for it with `v`.
  * One keypress per shot, saved immediately. No confirm step to rubber-stamp.
  * Order is SHUFFLED by default. Outcomes cluster in runs in this data (clip 2
    attempts 12-22 are all misses), so labelling in sequence lets a streak carry
    you -- you start seeing what you expect.
  * Provenance is recorded per label: how it was produced, when, how long you
    looked, and whether you changed a previous answer.
  * Model agreement is revealed only AFTER the pass, via --reveal.

Labels are written to <session>/make_truth.json in the SAME
{key: "make"|"miss"|"notshot"|"unsure"} format every other tool already reads.
Provenance goes to <session>/make_truth_meta.json alongside it.

Run (SYSTEM python):
  python -X utf8 tools/label_shots.py --session data/out/session_0729
  python -X utf8 tools/label_shots.py --session data/out/session_0729 --reveal
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import cv2
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import guiview as gv

KEYS = {ord("m"): "make", ord("n"): "miss",
        ord("b"): "notshot", ord("u"): "unsure"}
PRETTY = {"make": "MAKE", "miss": "miss", "notshot": "NOT a shot", "unsure": "unsure"}


def _load(path, default):
    try:
        return json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _shots(session, clipdir, only_unlabelled, truth):
    """[(key, clip, shot_in_clip, video_path)] for shots that have a review clip."""
    df = pd.read_csv(os.path.join(session, "session_shots.csv"))
    out = []
    for _, r in df.iterrows():
        clip = str(r["clip"])
        idx = int(r["shot_in_clip"])
        key = f"{clip}|{idx}"
        if only_unlabelled and key in truth:
            continue
        stem = os.path.splitext(os.path.basename(clip))[0]
        vp = os.path.join(clipdir, f"{stem}_s{idx:03d}_wide.mp4")
        if os.path.exists(vp):
            out.append((key, clip, idx, vp))
    return out


def reveal(session):
    """Post-hoc: where do you and the model disagree? Only meaningful AFTER a
    blind pass, which is why it is a separate mode and not an on-screen hint."""
    truth = _load(os.path.join(session, "make_truth.json"), {})
    meta = _load(os.path.join(session, "make_truth_meta.json"), {})
    pred = _load(os.path.join(session, "make_pred.json"), {})
    if not truth:
        print("no labels yet -- run the blind pass first")
        return 0
    both = [(k, v, pred[k]) for k, v in truth.items()
            if k in pred and v in ("make", "miss")]
    if not both:
        print(f"{len(truth)} labels, but no overlapping model predictions "
              f"(make_pred.json missing or keyed differently)")
        return 0
    agree = [k for k, v, p in both if (v == "make") == bool(p.get("made"))]
    print(f"{len(truth)} labels; {len(both)} comparable to the model")
    print(f"agreement: {len(agree)}/{len(both)} = {len(agree)/len(both):.0%}")
    print("\n⚠️  This is NOT a model accuracy measurement unless these labels are")
    print("   blind and this model was not fitted on this session. Check")
    print("   models/<model>.trained_on.json before quoting the number.\n")
    dis = [(k, v, p) for k, v, p in both if (v == "make") != bool(p.get("made"))]
    if dis:
        print(f"{len(dis)} DISAGREEMENTS (yours vs model, most confident first):")
        for k, v, p in sorted(dis, key=lambda t: -abs((t[2].get("prob") or .5) - .5)):
            print(f"  {k:<44} you={v:<6} model={'make' if p.get('made') else 'miss':<6}"
                  f" p={p.get('prob', float('nan')):.2f}")
    src = {}
    for k in truth:
        src[meta.get(k, {}).get("source", "unknown")] = \
            src.get(meta.get(k, {}).get("source", "unknown"), 0) + 1
    print(f"\nlabel provenance: {src}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--session", required=True)
    ap.add_argument("--clips", default=None,
                    help="review-clip dir (default <session>/review_clips)")
    ap.add_argument("--all", action="store_true",
                    help="include already-labelled shots (default: only new ones)")
    ap.add_argument("--in-order", action="store_true",
                    help="do NOT shuffle. Off by default on purpose: outcomes come "
                         "in runs, and sequence lets a streak anchor you.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reveal", action="store_true",
                    help="post-hoc model agreement + provenance; no labelling")
    a = ap.parse_args(argv)

    if a.reveal:
        return reveal(a.session)

    clipdir = a.clips or os.path.join(a.session, "review_clips")
    tpath = os.path.join(a.session, "make_truth.json")
    mpath = os.path.join(a.session, "make_truth_meta.json")
    truth = _load(tpath, {})
    meta = _load(mpath, {})

    shots = _shots(a.session, clipdir, not a.all, truth)
    if not shots:
        print(f"nothing to label in {clipdir} "
              f"({len(truth)} already labelled; --all to revisit)")
        return 0
    if not a.in_order:
        random.Random(a.seed).shuffle(shots)

    print(f"{len(shots)} shots to label   [m]ake  [n]miss  [b] not a shot  "
          f"[u]nsure   |  [space] pause  [r]eplay  [left/right] move  "
          f"[v] show my previous answer  [q]uit")
    win = "label shots -- BLIND (no model call shown)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    i, saved, t_shot = 0, 0, time.time()
    show_prev = False
    while 0 <= i < len(shots):
        key, clip, idx, vp = shots[i]
        cap = cv2.VideoCapture(vp)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 960)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 540)
        scale = gv.display_scale(w, h)
        delay = max(1, int(1000.0 / max(fps, 1.0)))
        playing, choice = True, None
        t_shot = time.time()
        while choice is None:
            ok, frame = cap.read()
            if not ok:                                  # loop the clip
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok:
                    break
            disp = gv.to_display(frame, scale)
            bar = f"{i+1}/{len(shots)}   shot {idx}   {os.path.basename(clip)[:22]}"
            cv2.putText(disp, bar, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(disp, bar, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 1, cv2.LINE_AA)
            hint = "m=make  n=miss  b=not a shot  u=unsure"
            cv2.putText(disp, hint, (10, disp.shape[0] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(disp, hint, (10, disp.shape[0] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            if show_prev and key in truth:
                p = f"your previous answer: {PRETTY.get(truth[key], truth[key])}"
                cv2.putText(disp, p, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(disp, p, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (60, 220, 255), 1, cv2.LINE_AA)
            cv2.imshow(win, disp)
            k = cv2.waitKey(delay if playing else 40) & 0xFF
            if k in KEYS:
                choice = KEYS[k]
            elif k == ord(" "):
                playing = not playing
            elif k == ord("r"):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            elif k == ord("v"):
                show_prev = not show_prev
            elif k in (81, ord(",")):                   # left
                i = max(0, i - 1); break
            elif k in (83, ord(".")):                   # right
                i = min(len(shots) - 1, i + 1); break
            elif k in (ord("q"), 27):
                cap.release(); cv2.destroyAllWindows()
                print(f"\nstopped. {saved} labelled this run, "
                      f"{len(truth)} total in {tpath}")
                return 0
            if not playing:
                cap.set(cv2.CAP_PROP_POS_FRAMES,
                        max(0, cap.get(cv2.CAP_PROP_POS_FRAMES) - 1))
        cap.release()
        if choice is not None:
            prev = truth.get(key)
            truth[key] = choice
            meta[key] = {"label": choice, "source": "human_blind",
                         "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                         "seconds": round(time.time() - t_shot, 1),
                         "changed_from": prev if prev != choice else None,
                         "shown_previous": bool(show_prev and prev is not None)}
            json.dump(truth, open(tpath, "w", encoding="utf-8"), indent=2)
            json.dump(meta, open(mpath, "w", encoding="utf-8"), indent=2)
            saved += 1
            i += 1
    cv2.destroyAllWindows()
    secs = [m.get("seconds", 0) for m in meta.values() if isinstance(m, dict)]
    pace = f", median {sorted(secs)[len(secs)//2]:.1f}s/shot" if secs else ""
    print(f"\ndone: {saved} labelled this run, {len(truth)} total{pace}")
    print(f"  labels     -> {tpath}")
    print(f"  provenance -> {mpath}")
    print("  model agreement: python -X utf8 tools/label_shots.py "
          f"--session {a.session} --reveal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
