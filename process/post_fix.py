"""Post-anchor-fix analysis of the close camera. Run AFTER flare_report re-runs.

Three questions, in order of what they can settle:

(1) release_frame_delta -- does the release compute_form re-finds from pose alone
    agree with the flare detector's release? This is the only independent check on
    the close camera's release, which no ball hand-off can confirm. Tight => a
    pose-corroborated confidence tier is arguable ON EVIDENCE. Wide => the close
    camera cannot do release-anchored metrics at all.

(2) wide vs close, SAME metric, SAME shot -- the direct camera-dependence test.
    Before the fix it read r ~ 0, but the close side was measuring the wrong second
    of video, so that number meant nothing.

(3) make/miss d, with a ONE-TO-ONE join (14 wide shots had taken 2 close releases
    each) and release_conf respected.

Comparison against the pre-fix artifact is printed wherever both exist, because the
size of the shift IS the evidence that the bug mattered.
"""
import csv, json, math, re, sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\jmaku\Desktop\ShotLab")
sys.path.insert(0, str(ROOT))
from tools import eval_ablations as E
import tools.flare_report as FR
from shotlab.metric_ranges import in_range

FPS = 30.0
PAIRS = {
    "20260729_115341": ("PXL_20260729_155320813", 22.64),
    "20260729_115926": ("PXL_20260729_155914855-001", 13.12),
    "20260729_120729": ("PXL_20260729_160743954", -12.47),
    "20260729_121450": ("PXL_20260729_161439291-002", 13.04),
}
METRICS = ["knee_bend_deg", "knee_bend_3d_deg",
           "balance_drift_px_per_ht",
           "elbow_angle_at_release_deg", "elbow_angle_at_release_3d_deg",
           "follow_through_hold_s",
           "tempo_dip_to_release_s", "jump_height_ft", "release_height_ft"]
RELEASE_ANCHORED = {"elbow_angle_at_release_deg", "tempo_dip_to_release_s",
                    "release_vs_apex_s"}
OUT = ROOT / "data/out/session_0729"

new = json.load(open(OUT / "flare_readings_raw.json", encoding="utf-8"))
try:
    old = json.load(open(OUT / "flare_readings_raw.MISANCHORED.json", encoding="utf-8"))
except FileNotFoundError:
    old = []
print(f"fixed artifact: {len(new)} rows, clips {dict(Counter(r['clip'] for r in new))}")
print(f"pre-fix       : {len(old)} rows\n")

# ---- (1) does the pose-refound release agree with the flare detection? --------
d = [r["release_frame_delta"] for r in new if r.get("release_frame_delta") is not None]
if d:
    a = np.array(d, float)
    print("(1) release_frame_delta  (pose-refound release  MINUS  flare release)")
    print(f"    n={len(a)}  median={np.median(a):+.1f}f  mean={a.mean():+.1f}f  "
          f"sd={a.std(ddof=1):.1f}f")
    print(f"    |delta| <=1f: {100*np.mean(np.abs(a)<=1):.0f}%   "
          f"<=2f: {100*np.mean(np.abs(a)<=2):.0f}%   "
          f"<=5f: {100*np.mean(np.abs(a)<=5):.0f}%   "
          f"range [{a.min():+.0f}, {a.max():+.0f}]")
    print("    tight => the two independent release estimates corroborate each other")
else:
    print("(1) no release_frame_delta emitted -- did the re-run use the new code?")
conf = Counter(r.get("release_conf") for r in new)
print(f"    release_conf: {dict(conf)}\n")

# ---- shared join machinery ----------------------------------------------------
sess = list(csv.DictReader(open(OUT / "session_shots.csv", newline="", encoding="utf-8")))


def clip_start(stem):
    m = re.search(r"_(\d{8})_(\d{2})(\d{2})(\d{2})(\d{3})", stem)
    dd, hh, mm, ss, ms = m.groups()
    return datetime(int(dd[:4]), int(dd[4:6]), int(dd[6:]),
                    int(hh), int(mm), int(ss), int(ms) * 1000)


wide_row, shot_truth = {}, {}
for wide_stem in {v[0] for v in PAIRS.values()}:
    recs = [r for r in sess if Path(r["clip"]).stem == wide_stem]
    for r in recs:
        wide_row[(wide_stem, int(r["shot_in_clip"]))] = r
    att = E.load_attempts(wide_stem)
    if not recs or not att:
        continue
    t0 = clip_start(wide_stem)
    est = [(r, (datetime.fromisoformat(r["t"]) - t0).total_seconds() * FPS) for r in recs]
    lead = float(np.median([min(att, key=lambda a: abs(a["rim_frame"] - f))["rim_frame"] - f
                            for _r, f in est]))
    cand = sorted(((abs(a["rim_frame"] - (f + lead)), i, a)
                   for i, (_r, f) in enumerate(est) for a in att), key=lambda t: t[0])
    ur, ua = set(), set()
    for dist, i, a in cand:
        if dist > 30 or i in ur or a["attempt_id"] in ua:
            continue
        ur.add(i); ua.add(a["attempt_id"])
        shot_truth[(wide_stem, int(est[i][0]["shot_in_clip"]))] = a["outcome"] == "make"

FR.WIDE_DIR = str(ROOT / "data" / "raw" / "Camera 1")
_wt = {w: FR.wide_shot_times(w) for w, _o in PAIRS.values()}


def join(rows):
    """close rows -> {(wide_stem, shot_in_clip): row}, ONE-TO-ONE (nearest wins)."""
    buckets = defaultdict(list)
    for r in rows:
        wide_stem, off = PAIRS[r["clip"]]
        wtimes = _wt[wide_stem]
        if not wtimes:
            continue
        ptime = r["frame"] / FPS + off
        j = int(np.argmin([abs(t - ptime) for t, _ in wtimes]))
        dt = abs(wtimes[j][0] - ptime)
        if dt < 1.5:
            buckets[(wide_stem, j + 1)].append((dt, r))
    return {k: min(v, key=lambda t: t[0])[1] for k, v in buckets.items()}


new_j, old_j = join(new), join(old) if old else {}

# ---- (2) the direct camera-dependence test -----------------------------------
print("(2) SAME metric, SAME shot, two cameras")
print(f"    {'metric':<30}{'n':>5}{'wide':>9}{'close':>9}{'r':>8}{'rho':>8}   pre-fix r")
print("    " + "-" * 76)
for m in METRICS[:4]:
    def pairs(j):
        xs, ys = [], []
        for key, r in j.items():
            w = wide_row.get(key)
            wv = w.get(m) if w else None
            cv = r.get(m)
            if not wv or cv is None:
                continue
            wv = float(wv)
            if not in_range(m, wv):
                continue
            xs.append(wv); ys.append(float(cv))
        return np.array(xs), np.array(ys)
    x, y = pairs(new_j)
    if len(x) < 5:
        print(f"    {m:<30}{len(x):>5}   -- too few paired reads --")
        continue
    r_p = float(np.corrcoef(x, y)[0, 1])
    r_s = float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])
    ox, oy = pairs(old_j) if old_j else (np.array([]), np.array([]))
    prev = f"{float(np.corrcoef(ox, oy)[0, 1]):+.2f}" if len(ox) >= 5 else "n/a"
    print(f"    {m:<30}{len(x):>5}{x.mean():>9.2f}{y.mean():>9.2f}"
          f"{r_p:>8.2f}{r_s:>8.2f}      {prev}")

# ---- (2b) 2D vs 3D of the SAME joint, WITHIN the close camera -----------------
# Needs no second camera, so it is available now and at full n. The 2D angle is an
# image-plane projection; the 3D one is metric. If they disagree, the projection is
# injecting geometry-dependent noise into every 2D form number we have ever shipped.
print("\n(2b) same joint, same frame, 2D projection vs metric 3D (close cam only)")
for two_d, three_d in [("knee_bend_deg", "knee_bend_3d_deg"),
                       ("elbow_angle_at_release_deg", "elbow_angle_at_release_3d_deg")]:
    xs = [(float(r[two_d]), float(r[three_d])) for r in new
          if r.get(two_d) is not None and r.get(three_d) is not None]
    if len(xs) < 5:
        print(f"     {two_d:<32} n={len(xs)}  -- too few --")
        continue
    x = np.array([a for a, _ in xs]); y = np.array([b for _, b in xs])
    r_p = float(np.corrcoef(x, y)[0, 1])
    print(f"     {two_d:<32} n={len(x):<4} 2D mean {x.mean():6.1f}  "
          f"3D mean {y.mean():6.1f}  r={r_p:+.2f}  "
          f"median |diff|={np.median(np.abs(x-y)):5.1f} deg")
# how often does the 2D censoring window disagree with the anatomical one?
k2 = [r.get("knee_bend_deg") for r in new]
k3 = [r.get("knee_bend_3d_deg") for r in new]
only3 = sum(1 for a, b in zip(k2, k3) if a is None and b is not None)
print(f"     shots where 3D gives a knee but the 30-150 window dropped the 2D one: {only3}")

# ---- (2c) THE PRE-REGISTERED SHARP TEST: 2D vs 3D within the WIDE camera ------
# The wide camera is oblique, so this is where an image-plane projection should
# distort if it distorts anywhere. The close camera is a profile view -- knee
# flexion is already in its image plane, so 2D~3D agreement there proves little.
# Reading committed to in advance:
#   wide agreement MUCH worse than close  -> the projection is the defect
#   wide agreement ~ close                -> projection is NOT what broke the wide
#                                            camera; coverage/scale is, and that was
#                                            already established independently.
print("\n(2c) SHARP TEST -- 2D vs metric 3D within the WIDE (oblique) camera")
print("     compare against the close (profile) numbers in 2b")
for two_d, three_d in [("knee_bend_deg", "knee_bend_3d_deg"),
                       ("elbow_angle_at_release_deg", "elbow_angle_at_release_3d_deg")]:
    xy = [(float(r[two_d]), float(r[three_d])) for r in sess
          if r.get(two_d) and r.get(three_d)]
    if len(xy) < 5:
        print(f"     {two_d:<32} n={len(xy)}  -- absent from the CSV, or too few --")
        continue
    x = np.array([a for a, _ in xy]); y = np.array([b for _, b in xy])
    r_p = float(np.corrcoef(x, y)[0, 1])
    print(f"     {two_d:<32} n={len(x):<4} 2D mean {x.mean():6.1f}  "
          f"3D mean {y.mean():6.1f}  r={r_p:+.2f}  "
          f"median |diff|={np.median(np.abs(x-y)):5.1f} deg")
w3 = [r.get("knee_bend_3d_deg") for r in sess]
w2 = [r.get("knee_bend_deg") for r in sess]
print(f"     wide knee reads: 2D present {sum(1 for v in w2 if v)}, "
      f"3D present {sum(1 for v in w3 if v)} of {len(sess)} shots "
      f"-- does 3D also recover COVERAGE, or only accuracy?")

# ---- (3) make/miss, one-to-one, release_conf respected ------------------------
print("\n(3) close-cam make/miss, ONE-TO-ONE join")
print(f"    {'metric':<30}{'makes':>9}{'misses':>9}{'d':>7}{'p':>8}{'n':>9}{'floor':>7}")
print("    " + "-" * 79)
ps = []
for m in METRICS:
    a, b = [], []
    for key, r in new_j.items():
        t = shot_truth.get(key)
        v = r.get(m)
        if t is None or v is None:
            continue
        if m in RELEASE_ANCHORED and r.get("release_conf") not in ("high", "medium"):
            continue                       # the gate that was being bypassed
        (a if t else b).append(float(v))
    if len(a) < 5 or len(b) < 5:
        print(f"    {m:<30}{'--':>9}{'--':>9}{'--':>7}{'--':>8}{f'{len(a)}/{len(b)}':>9}"
              f"   (gated out)" if m in RELEASE_ANCHORED else
              f"    {m:<30}{'--':>9}{'--':>9}{'--':>7}{'--':>8}{f'{len(a)}/{len(b)}':>9}")
        continue
    a, b = np.array(a), np.array(b)
    s = math.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    if s == 0:
        continue
    dd = (a.mean() - b.mean()) / s
    both = np.concatenate([a, b]); obs = abs(a.mean() - b.mean())
    rng = np.random.default_rng(0); h = 0
    for _ in range(20000):
        rng.shuffle(both)
        if abs(both[:len(a)].mean() - both[len(a):].mean()) >= obs:
            h += 1
    p = (h + 1) / 20001
    floor = 2.8 / math.sqrt((len(a) + len(b)) / 2)
    tag = "  <-- CLEARS ITS FLOOR" if (p < 0.05 and abs(dd) >= floor) else ""
    print(f"    {m:<30}{a.mean():>9.2f}{b.mean():>9.2f}{dd:>7.2f}{p:>8.4f}"
          f"{f'{len(a)}/{len(b)}':>9}{floor:>7.2f}{tag}")
    ps.append((m, p))
k = max(len(ps), 1)
print(f"\n    {len(ps)} tested -> Bonferroni p < {0.05/k:.4f}; survives:",
      [m for m, p in ps if p < 0.05 / k] or "NOTHING")

# ---- (4) is CLOSE-cam coverage outcome-correlated? ---------------------------
# This is the defect the review found on the WIDE camera (pose survived on 0.43 of
# makes vs 0.27 of misses), which selects the analysed rows by the thing being
# measured. Applying the same test to the close camera, and specifically to
# knee_bend_3d_deg -- the one surviving candidate. Assuming it is clean because
# the close camera is better would be exactly the error being corrected.
print("\n(4) close-cam coverage vs OUTCOME (the defect found on the wide camera)")
try:
    from scipy.stats import fisher_exact
    ok = True
except Exception:
    ok = False
tot_mk = sum(1 for k2, r in new_j.items() if shot_truth.get(k2) is True)
tot_ms = sum(1 for k2, r in new_j.items() if shot_truth.get(k2) is False)
print(f"    joined shots with truth: {tot_mk} makes / {tot_ms} misses")
print(f"    {'metric':<32}{'make cov':>10}{'miss cov':>10}{'p':>9}")
print("    " + "-" * 61)
for m in ["knee_bend_3d_deg", "knee_bend_deg", "balance_drift_px_per_ht",
          "elbow_angle_at_release_3d_deg"]:
    a = sum(1 for k2, r in new_j.items()
            if shot_truth.get(k2) is True and r.get(m) is not None)
    b = sum(1 for k2, r in new_j.items()
            if shot_truth.get(k2) is False and r.get(m) is not None)
    if not tot_mk or not tot_ms:
        continue
    if ok:
        _, p = fisher_exact([[a, tot_mk - a], [b, tot_ms - b]])
        ps_ = f"{p:>9.3f}"
    else:
        ps_ = f"{'--':>9}"
    print(f"    {m:<32}{a}/{tot_mk} ({a/tot_mk:.2f}){'':>1}{b}/{tot_ms} "
          f"({b/tot_ms:.2f}){ps_}")
print("    a significant gap here would mean the surviving rows are selected by")
print("    the outcome, and the candidate's d is not interpretable as form.")
