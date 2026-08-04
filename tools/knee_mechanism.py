"""Is knee_bend_3d_deg a real signal, or a proxy for something already measured?

The candidate replicated (07-29 d=+0.40, 07-20 d=+0.40, pooled [+0.11,+0.69]),
but a replicated correlation with no mechanism is still a bandaid waiting to
happen. Two ways it could be standing in for something else:

  * ANOTHER METRIC on the same row. If tempo or jump height carries the same
    make/miss effect AND correlates strongly with knee, then the knee number is
    that thing wearing a different name.
  * FATIGUE. If both bend and make% drift over a session, a within-session time
    trend manufactures the association without any per-shot relationship.

Both are checked on BOTH sessions, because a mechanism that only appears in one
is not a mechanism. Every metric here comes from the same close-cam row as the
knee, so no extra join is needed and no new failure mode is introduced.

Run (SYSTEM python): python -X utf8 tools/knee_mechanism.py
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

KNEE = "knee_bend_3d_deg"
OTHERS = ["tempo_dip_to_release_s", "jump_height_ft", "release_height_ft",
          "balance_drift_px_per_ht", "elbow_angle_at_release_3d_deg",
          "knee_bend_deg", "flare_deg", "follow_through_hold_s"]

SESSIONS = {
    "07-29": dict(
        wide_dir=ROOT / "data/raw/Camera 1",
        close_dir=ROOT / "data/raw/Camera 2/upright",
        readings=ROOT / "data/out/session_0729/flare_readings_raw.json",
        pairs={"PXL_20260729_155320813": ("20260729_115341", 22.64),
               "PXL_20260729_155914855-001": ("20260729_115926", 13.12),
               "PXL_20260729_160743954": ("20260729_120729", -12.47),
               "PXL_20260729_161439291-002": ("20260729_121450", 13.04)}),
    "07-20": dict(
        wide_dir=ROOT / "data/raw/Camera 1/Old",
        close_dir=ROOT / "data/raw/Camera 1/Old",
        readings=ROOT / "data/out/session_0720_close/flare_readings_raw.json",
        pairs={"PXL_20260720_151519220": ("20260720_111510", -6.99),
               "PXL_20260720_152319112": ("20260720_112306", -10.95),
               "PXL_20260720_153054813": ("20260720_113041", -11.96)}),
}


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or len(b) < 3:
        return None
    s = math.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1))
                  / (len(a)+len(b)-2))
    return None if s < 1e-9 else float((a.mean()-b.mean())/s)


def joined_rows(cfg):
    """[(row, made, clip, close_time_s)] one-to-one on both hops."""
    rows = json.load(open(cfg["readings"], encoding="utf-8"))
    by = defaultdict(list)
    for r in rows:
        by[r["clip"]].append(r)
    E.CLIP_DIR = cfg["wide_dir"]
    out = []
    for wide, (close, off) in cfg["pairs"].items():
        calib = rs.calib_at(rs.load_rims(wide), 0)
        cand = E._detect_full_clip(wide)
        g = assemble_track(E._cands_at_conf(cand, 0.25))
        cloud = E._cands_at_conf(cand, 0.01)
        shots, _ = _union_beam(detect_shots_to_rim(g, calib, cloud=cloud),
                               g, cloud, calib)
        att = E.load_attempts(wide)
        fw = probe(str(cfg["wide_dir"] / f"{wide}.mp4")).fps
        fc = probe(str(cfg["close_dir"] / f"{close}.mp4")).fps
        m2 = assign_one_to_one([_rim_frame(s, calib) for s in shots],
                               [a["rim_frame"] for a in att], max_dist=30)
        truth = {i: (att[j]["outcome"] == "make") for i, (j, _d) in m2.items()}
        recs = by.get(close, [])
        m1 = assign_one_to_one([r["frame"] / fc + off for r in recs],
                               [float(s.frames[0]) / fw for s in shots], max_dist=1.5)
        for i, (j, _d) in m1.items():
            if j in truth:
                out.append((recs[i], truth[j], wide, recs[i]["frame"] / fc))
    return out


def main():
    for name, cfg in SESSIONS.items():
        rows = joined_rows(cfg)
        knee = [(r.get(KNEE), made, clip, t) for r, made, clip, t in rows
                if r.get(KNEE) is not None]
        v = np.array([k for k, _m, _c, _t in knee], float)
        l = np.array([1 if m else 0 for _k, m, _c, _t in knee], int)
        print(f"\n=== {name}: {len(v)} joined shots with a 3D knee "
              f"({int(l.sum())} makes) ===")
        print(f"  knee d = {cohen_d(v[l==1], v[l==0]):+.2f}\n")

        print(f"  {'other metric':<34}{'its own d':>10}{'r with knee':>13}{'n':>6}")
        print("  " + "-" * 63)
        for m in OTHERS:
            pair = [(r.get(KNEE), r.get(m), made) for r, made, _c, _t in rows
                    if r.get(KNEE) is not None and r.get(m) is not None]
            if len(pair) < 10:
                print(f"  {m:<34}{'--':>10}{'--':>13}{len(pair):>6}")
                continue
            k = np.array([p[0] for p in pair], float)
            x = np.array([p[1] for p in pair], float)
            lab = np.array([1 if p[2] else 0 for p in pair], int)
            d = cohen_d(x[lab == 1], x[lab == 0])
            r = float(np.corrcoef(k, x)[0, 1]) if x.std() > 1e-9 else float("nan")
            flag = "  <-- proxy suspect" if (d is not None and abs(d) > 0.3
                                             and abs(r) > 0.5) else ""
            print(f"  {m:<34}{(f'{d:+.2f}' if d is not None else '--'):>10}"
                  f"{r:>+13.2f}{len(k):>6}{flag}")

        # fatigue: does knee drift with time, and does make% drift with it too?
        t = np.array([tt for _k, _m, _c, tt in knee], float)
        if len(t) > 20 and t.std() > 1e-9:
            rk = float(np.corrcoef(t, v)[0, 1])
            rm = float(np.corrcoef(t, l)[0, 1])
            print(f"\n  fatigue check (time within clip):")
            print(f"    corr(time, knee)  = {rk:+.2f}")
            print(f"    corr(time, made)  = {rm:+.2f}")
            print(f"    a shared time trend would manufacture the association;")
            print(f"    both must be non-trivial for that to be the explanation.")
    print("\nA metric is a PROXY only if it carries a similar d AND correlates")
    print("strongly with the knee. Similar d with low r means two separate signals.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
