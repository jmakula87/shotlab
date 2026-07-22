"""Adversarial probe: GT-only ORACLE-TRACK through both segmenters, instrumented.

Uses ONLY the human labels + the rim calibrations recorded in
process/step1a_rim_oracle_ceiling.txt (no video decode, no YOLO), so it is
read-only and deterministic.

Questions answered:
  1. How many labeled flight WINDOWS exist (the max recoverable "shots")?
  2. Per window: does GT ever come within shot_gate_px of the auto rim?
     Does the window contain a point >= launch_drop (200px) below the rim?
     Is the 0.8*launch_drop (160px) y-drop satisfiable within the window?
  3. Run the REAL assemble_track on a GT-only candidate stream and feed it to
     (a) detect_shots_to_rim with the recorded rim  (b) segment_shots.
     Instrument detect_shots_to_rim's walk-back: does j cross window
     boundaries (jumping the huge unlabeled gaps)?
"""
import json, sys, collections
from pathlib import Path
import numpy as np

ROOT = Path(r"C:\Users\jmaku\Desktop\ShotLab")
sys.path.insert(0, str(ROOT))
from shotlab.phase1_ball.detect import BallCandidate
from shotlab.phase1_ball.track import assemble_track, segment_shots
from shotlab.court import Calibration, detect_shots_to_rim
from shotlab.arc import fit_parabola_ransac

RIMS = {  # from process/step1a_rim_oracle_ceiling.txt
    "PXL_20260720_151519220": (1134, 470, 16),
    "PXL_20260720_152319112": (1306, 462, 19),
    "PXL_20260720_153054813": (972, 469, 20),
}

labels = json.load(open(ROOT / "data/labels/ball_labels_0720.json"))
by_clip = collections.defaultdict(list)
for L in labels:
    by_clip[L["clip"]].append(L)

WINDOW_GAP = 30  # labeled windows are contiguous (gap-1) separated by 100s-1000s

def windows(frames):
    runs, cur = [], []
    for f in sorted(frames):
        if cur and f - cur[-1] > WINDOW_GAP:
            runs.append(cur); cur = []
        cur.append(f)
    if cur:
        runs.append(cur)
    return runs

def instrumented_rim_segment(track, calib, max_rim_gap=20, launch_drop=200.0,
                             min_points=8, threshold_px=8.0, win_of=None):
    """Copy of court.detect_shots_to_rim with logging (read-only probe)."""
    frames = np.array(sorted(track))
    xy = np.array([[track[int(f)].cx, track[int(f)].cy] for f in frames])
    x, y = xy[:, 0], xy[:, 1]
    dist = np.hypot(x - calib.rim_x, y - calib.rim_y)
    fidx = {int(f): i for i, f in enumerate(frames)}
    near = frames[dist < calib.shot_gate_px]
    print(f"    near-rim frames: {len(near)}")
    if len(near) == 0:
        return 0
    events, cur = [], [near[0]]
    for a, b in zip(near, near[1:]):
        if b - a <= max_rim_gap:
            cur.append(b)
        else:
            events.append(cur); cur = [b]
    events.append(cur)
    print(f"    rim events: {len(events)}")
    kept = 0
    seen_launch = set()
    for ev in events:
        t_rim = min(ev, key=lambda f: dist[fidx[int(f)]])
        i = fidx[int(t_rim)]
        j = i
        while j > 0 and (y[j] - calib.rim_y) < launch_drop:
            j -= 1
        w_i = win_of[int(frames[i])] if win_of else "?"
        w_j = win_of[int(frames[j])] if win_of else "?"
        cross = " *** CROSSED WINDOWS ***" if w_i != w_j else ""
        tag = f"ev rim f{int(t_rim)} (win {w_i}) -> launch f{int(frames[j])} (win {w_j}){cross}"
        if int(frames[j]) in seen_launch:
            print(f"    {tag}  -> DEDUP-SKIPPED (seen launch)")
            continue
        seen_launch.add(int(frames[j]))
        npts = i - j + 1
        drop = y[j] - y[i]
        if npts < min_points or drop < 0.8 * launch_drop:
            print(f"    {tag}  -> REJECT npts={npts} drop={drop:.0f}px (need >=8 & >=160)")
            continue
        seg = slice(j, i + 1)
        fit = fit_parabola_ransac(x[seg], y[seg], threshold_px=threshold_px)
        if fit is None:
            print(f"    {tag}  -> REJECT ransac fit None (npts={npts}, span f{int(frames[j])}-f{int(frames[i])})")
            continue
        if fit.coeffs[0] >= 0:
            print(f"    {tag}  -> REJECT concave-up")
            continue
        if fit.n_used < 7:
            print(f"    {tag}  -> REJECT n_used={fit.n_used}<7")
            continue
        rel = fit.release_angle_deg(); ent = fit.entry_angle_deg(calib.rim_x)
        if min(rel, ent) > 78:
            print(f"    {tag}  -> REJECT near-vertical rel={rel:.0f} ent={ent:.0f}")
            continue
        print(f"    {tag}  -> SHOT (npts={npts} drop={drop:.0f} rel={rel:.0f} ent={ent:.0f})")
        kept += 1
    return kept


tot_win = tot_rim = tot_gap = 0
for clip in sorted(by_clip):
    Ls = by_clip[clip]
    rim_x, rim_y, half_w = RIMS[clip]
    gate = max(2.0 * half_w, 90.0)
    calib = Calibration(session=clip, image_w=1920, image_h=1080,
                        rim_x=rim_x, rim_y=rim_y, rim_radius_px=half_w,
                        shot_gate_px=gate)
    gt = {L["frame"]: (L["cx0"] + L["px"], L["cy0"] + L["py"], L["r"])
          for L in Ls if L.get("present")}
    allf = [L["frame"] for L in Ls]
    wins = windows(allf)
    win_of = {}
    for wi, w in enumerate(wins, 1):
        for f in w:
            win_of[f] = wi
    print(f"\n=== {clip}  rim=({rim_x},{rim_y}) gate={gate:.0f}  "
          f"labeled={len(allf)} present={len(gt)}  WINDOWS={len(wins)}")
    for wi, w in enumerate(wins, 1):
        pf = [f for f in w if f in gt]
        if not pf:
            print(f"  win{wi}: f{w[0]}-f{w[-1]} n={len(w)}  NO present GT")
            continue
        xs = np.array([gt[f][0] for f in pf]); ys = np.array([gt[f][1] for f in pf])
        d = np.hypot(xs - rim_x, ys - rim_y)
        below = ys - rim_y
        print(f"  win{wi}: f{w[0]}-f{w[-1]} n={len(w)} present={len(pf)} "
              f"minRimDist={d.min():.0f}px reach={'YES' if d.min()<gate else 'NO'} "
              f"maxBelowRim={below.max():.0f}px (need>=200 for walk-back stop; "
              f">=160 drop) heightSpan={ys.max()-ys.min():.0f}px")
    tot_win += len(wins)

    # GT-only candidate stream -> real tracker
    cands = {f: [BallCandidate(f, gx, gy, r, 0.99)] for f, (gx, gy, r) in gt.items()}
    track = assemble_track(cands)
    print(f"  oracle-track: {len(track)} pts (of {len(gt)} GT)")
    print(f"  -- rim-anchored detect_shots_to_rim (instrumented):")
    n_rim = instrumented_rim_segment(track, calib, win_of=win_of)
    segs = segment_shots(track)
    print(f"  -- gap-based segment_shots: {len(segs)} shots "
          f"({[ (int(s.frames[0]), int(s.frames[-1])) for s in segs ]})")
    tot_rim += n_rim
    tot_gap += len(segs)

print(f"\nTOTAL labeled windows (max recoverable) = {tot_win}")
print(f"GT-ORACLE-TRACK  ->  rim-anchored shots = {tot_rim}   gap-based shots = {tot_gap}")
