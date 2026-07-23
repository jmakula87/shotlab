"""Classify WHY each hand-counted attempt was missed by the baseline pipeline:
detection vs tracker vs segmenter. Uses the cached full-clip candidate cloud from
tools/eval_ablations.py (no re-detect), so it is fast.

For each missed attempt (rim_frame F), in a +-W frame window:
  DETECTION : no YOLO@0.25 candidate ever comes within the shot gate of the rim
              -> the detector never saw the ball at the rim (note if @0.01 did).
  TRACKER   : a @0.25 candidate reached the rim, but the assembled track did not
              follow it there -> greedy association lost the ball.
  SEGMENTER : the track DID reach the rim but no shot was emitted -> detect_shots_
              to_rim rejected it; we report the local stats that point at the gate
              (too few points / not enough launch drop / bad parabola fit).

Run (SYSTEM python): python -X utf8 tools/diagnose_misses.py --clip PXL_20260720_151519220
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools import rim_segments as rs
from tools import eval_ablations as E
from shotlab.phase1_ball.track import assemble_track
from shotlab.arc import fit_parabola_ransac

W = 40                      # half-window (frames) around the attempt's rim frame
LAUNCH_DROP = 200.0
MIN_POINTS = 8


def _near_rim(track_or_cands, calib, f0, f1):
    """min distance to the rim over frames in [f0,f1] for a {frame: cand} map."""
    best = np.inf
    for f, c in track_or_cands.items():
        if f0 <= f <= f1:
            if isinstance(c, list):
                for cc in c:
                    best = min(best, np.hypot(cc.cx - calib.rim_x, cc.cy - calib.rim_y))
            else:
                best = min(best, np.hypot(c.cx - calib.rim_x, c.cy - calib.rim_y))
    return best


def _seg_reason(track, calib, F):
    """If the track reaches the rim near F, mimic detect_shots_to_rim's walk-back
    to report which gate would drop it. Returns a short reason string."""
    gate = calib.shot_gate_px
    frames = np.array(sorted(f for f in track if F - W <= f <= F + W))
    if len(frames) == 0:
        return "no track pts in window"
    y = np.array([track[int(f)].cy for f in frames])
    x = np.array([track[int(f)].cx for f in frames])
    d = np.hypot(x - calib.rim_x, y - calib.rim_y)
    if d.min() >= gate:
        return f"track min-dist {d.min():.0f}px >= gate {gate:.0f}"
    i = int(np.argmin(d))
    # walk back to launch (>= LAUNCH_DROP below rim), stop at big frame gaps
    j = i
    while j > 0 and (y[j] - calib.rim_y) < LAUNCH_DROP:
        if frames[j] - frames[j - 1] > 45:
            break
        j -= 1
    npts = i - j + 1
    drop = y[j] - y[i]
    if npts < MIN_POINTS:
        return f"too few points (n={npts} < {MIN_POINTS}); launch not reached"
    if drop < 0.8 * LAUNCH_DROP:
        return f"insufficient launch drop ({drop:.0f}px < {0.8*LAUNCH_DROP:.0f})"
    fit = fit_parabola_ransac(x[j:i+1], y[j:i+1], threshold_px=8.0)
    if fit is None:
        return f"RANSAC fit failed on {npts}-pt segment (gather-poisoned?)"
    if fit.coeffs[0] >= 0:
        return "fit not a downward arc"
    if fit.n_used < 7:
        return f"fit inliers {fit.n_used} < 7"
    rel, ent = fit.release_angle_deg(), fit.entry_angle_deg(calib.rim_x)
    if min(rel, ent) > 78:
        return f"78 deg gate (rel {rel:.0f}, ent {ent:.0f})"
    return f"passes local gates (n={npts}, drop={drop:.0f}) -- inspect full-clip context"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    args = ap.parse_args(argv)
    clip = args.clip

    rim_doc = rs.load_rims(clip)
    calib = rs.calib_at(rim_doc, 0)
    attempts = E.load_attempts(clip)
    eval_json = json.load(open(E.HANDCOUNT_DIR / f"{clip}_eval.json"))
    cache = json.load(open(E.CAND_CACHE / f"{clip}_cloud01.json"))
    raw = {int(k): v for k, v in cache.items()}
    base = E._cands_at_conf(raw, 0.25)
    cloud = E._cands_at_conf(raw, 0.01)
    track = assemble_track(base)

    # recompute matched attempts at baseline to know which were missed
    produced = E._segment_full_clip(base, rim_doc, max(raw) + 1)
    tp, fp, fn = E.match(produced, attempts, tol=30)
    print(f"clip {clip}: rim ({calib.rim_x:.0f},{calib.rim_y:.0f}) gate {calib.shot_gate_px:.0f}; "
          f"{len(fn)} missed of {len(attempts)}")

    buckets = {"DETECTION": [], "TRACKER": [], "SEGMENTER": []}
    for a in fn:
        F = int(a["rim_frame"])
        f0, f1 = F - W, F + W
        det025 = _near_rim(base, calib, f0, f1)
        det001 = _near_rim(cloud, calib, f0, f1)
        trk = _near_rim({f: c for f, c in track.items()}, calib, f0, f1)
        gate = calib.shot_gate_px
        if det025 >= gate:
            kind = "DETECTION"
            note = f"@0.25 min-dist {det025:.0f}px (>gate); @0.01 {det001:.0f}px"
        elif trk >= gate:
            kind = "TRACKER"
            note = f"cand reached rim ({det025:.0f}px) but track min-dist {trk:.0f}px"
        else:
            kind = "SEGMENTER"
            note = _seg_reason(track, calib, F)
        buckets[kind].append((a["attempt_id"], F, a.get("outcome"), note))

    for kind, items in buckets.items():
        print(f"\n=== {kind}: {len(items)} ===")
        for aid, F, out, note in items:
            print(f"  #{aid:>2} @f{F:<6} {out:<4} {note}")

    print(f"\nSUMMARY  detection={len(buckets['DETECTION'])}  "
          f"tracker={len(buckets['TRACKER'])}  segmenter={len(buckets['SEGMENTER'])}  "
          f"(of {len(fn)} misses)")
    print("=> the biggest bucket is the lever.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
