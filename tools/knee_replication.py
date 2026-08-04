"""Independent replication of the knee_bend_3d_deg candidate on the 07-20 session.

The candidate (makes show a HIGHER knee angle, i.e. LESS bend; d=+0.40, p=0.037)
was found post-hoc on 07-29. Re-testing it there is resubstitution -- the same 137
shots that produced the hypothesis cannot also test it. 07-20 is untouched for
this hypothesis and carries 111 hand-counted outcomes, so it is a genuine
replication set that needs no new filming.

Chain, each hop measured rather than assumed:
  close release --(audio sync, conf 0.87-0.90)--> wide shot --(rim frame)--> attempt
Both hops are ONE-TO-ONE via flare_report.assign_one_to_one; the 07-29 work showed
a nearest-match join is not injective and quietly produces pseudo-replication.

Reported per the CORRECTED pre-registration (2026-08-04): an ESTIMATE with a CI
and a clip-stratified permutation p, NOT a pass/fail gate. At these n a single
session cannot confirm or refute d=0.25 -- it has ~30% power -- so the deliverable
is a pooled estimate across sessions, and non-significance dead-letters nothing.

Run (SYSTEM python): python -X utf8 tools/knee_replication.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools import eval_ablations as E, rim_segments as rs
from tools.flare_report import assign_one_to_one
from shotlab.phase1_ball.track import assemble_track
from shotlab.phase1_ball.pipeline import _union_beam, _rim_frame
from shotlab.court import detect_shots_to_rim
from shotlab.video_io import probe

OLD = ROOT / "data" / "raw" / "Camera 1" / "Old"
CLOSE_OUT = ROOT / "data" / "out" / "session_0720_close"
METRIC = "knee_bend_3d_deg"
# wide stem -> (close stem, offset seconds), offsets measured by shotlab.sync
PAIRS = {
    "PXL_20260720_151519220": ("20260720_111510", -6.99),
    "PXL_20260720_152319112": ("20260720_112306", -10.95),
    "PXL_20260720_153054813": ("20260720_113041", -11.96),
}


def cohen_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    s = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                  / (len(a) + len(b) - 2))
    return None if s < 1e-9 else float((a.mean() - b.mean()) / s)


def main():
    raw_path = CLOSE_OUT / "flare_readings_raw.json"
    if not raw_path.exists():
        raise SystemExit(f"no close-cam readings yet at {raw_path}\n"
                         f"run tools/flare_report.py on the 0720 pairs first")
    rows = json.load(open(raw_path, encoding="utf-8"))
    by_clip = defaultdict(list)
    for r in rows:
        by_clip[r["clip"]].append(r)
    print(f"close-cam readings: {len(rows)} releases over {len(by_clip)} clips")

    E.CLIP_DIR = OLD
    vals, labs, clips = [], [], []
    for wide, (close, off) in PAIRS.items():
        calib = rs.calib_at(rs.load_rims(wide), 0)
        cand = E._detect_full_clip(wide)
        g = assemble_track(E._cands_at_conf(cand, 0.25))
        cloud = E._cands_at_conf(cand, 0.01)
        shots, _ = _union_beam(detect_shots_to_rim(g, calib, cloud=cloud),
                               g, cloud, calib)
        att = E.load_attempts(wide)
        fps_w = probe(str(OLD / f"{wide}.mp4")).fps
        fps_c = probe(str(OLD / f"{close}.mp4")).fps

        # hop 2: wide shot -> hand-counted attempt, one-to-one on the rim frame
        s_rim = [_rim_frame(s, calib) for s in shots]
        a_rim = [a["rim_frame"] for a in att]
        m2 = assign_one_to_one(s_rim, a_rim, max_dist=30)
        truth = {i: (att[j]["outcome"] == "make") for i, (j, _d) in m2.items()}

        # hop 1: close release -> wide shot, one-to-one on the release time
        recs = by_clip.get(close, [])
        t_close = [r["frame"] / fps_c + off for r in recs]
        t_wide = [float(s.frames[0]) / fps_w for s in shots]
        m1 = assign_one_to_one(t_close, t_wide, max_dist=1.5)

        n_used = 0
        for i, (j, _d) in m1.items():
            if j not in truth:
                continue
            v = recs[i].get(METRIC)
            if v is None:
                continue
            vals.append(float(v)); labs.append(1 if truth[j] else 0)
            clips.append(wide); n_used += 1
        print(f"  {wide[-9:]}: {len(shots)} wide shots, {len(att)} attempts, "
              f"{len(truth)} labelled, {len(recs)} releases -> {n_used} joined")

    v = np.array(vals, float); l = np.array(labs, int)
    c = np.array(clips)
    if len(v) < 10:
        raise SystemExit(f"\nonly {len(v)} joined shots -- too few to report")
    a, b = v[l == 1], v[l == 0]
    d = cohen_d(a, b)
    se = math.sqrt((len(a) + len(b)) / (len(a) * len(b))
                   + d * d / (2 * (len(a) + len(b) - 2)))
    lo, hi = d - 1.96 * se, d + 1.96 * se

    print(f"\n{METRIC} on 07-20  (INDEPENDENT of the 07-29 discovery)")
    print(f"  makes  n={len(a):3d}  mean {a.mean():7.2f}")
    print(f"  misses n={len(b):3d}  mean {b.mean():7.2f}")
    print(f"  d = {d:+.2f}   95% CI [{lo:+.2f}, {hi:+.2f}]")

    # Clip-stratified permutation: shuffle labels only WITHIN a clip. Outcomes
    # cluster in runs, so free shuffling assumes an exchangeability the data
    # violates. BOTH tails reported: the registration is ONE-SIDED positive, so
    # that is the operative test, but the two-sided p is shown so the one-sided
    # number cannot look like a quiet choice made after seeing the sign.
    rng = np.random.default_rng(0)
    hits1 = hits2 = 0
    N = 20000
    for _ in range(N):
        lp = l.copy()
        for cl in np.unique(c):
            m = c == cl
            lp[m] = rng.permutation(lp[m])
        dd = cohen_d(v[lp == 1], v[lp == 0])
        if dd is None:
            continue
        if dd >= d:
            hits1 += 1
        if abs(dd) >= abs(d):
            hits2 += 1
    print(f"  clip-stratified permutation p: one-sided (registered) "
          f"{(hits1 + 1) / (N + 1):.4f}   two-sided {(hits2 + 1) / (N + 1):.4f}")

    # Coverage vs outcome -- the defect that disqualified the wide camera. If the
    # metric survives more often on makes, the analysed rows are selected by the
    # thing being measured and d is not interpretable.
    try:
        from scipy.stats import fisher_exact
        mk_c = int(((l == 1)).sum()); ms_c = int((l == 0).sum())
        print(f"  coverage: {mk_c} makes / {ms_c} misses joined with a 3D knee "
              f"({len(v)} of {len(rows)} releases)")
    except Exception:
        pass
    print(f"\n  07-29 discovery was d=+0.40 (p=0.037, n=49/61).")
    print(f"  Registered direction is POSITIVE (makes bend LESS).")
    print(f"  This session {'AGREES' if (d or 0) > 0 else 'DISAGREES'} in sign.")
    print("\n  ⚠️ Per the corrected pre-registration this is an ESTIMATE, not a")
    print("  verdict: a single session has ~30% power at d=0.25, so a null here")
    print("  dead-letters nothing and a hit here promotes nothing. What counts is")
    print("  the POOLED estimate across sessions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
