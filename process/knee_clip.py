"""Is knee_bend_3d_deg's d=+0.40 a real within-shooter effect, or a CLIP artifact?

Shots are not exchangeable: they come in 4 clips filmed over ~28 minutes, and the
clips differ in make rate, camera position and fatigue. If a clip happens to have
both a higher make rate AND a higher knee angle, a POOLED comparison manufactures
an effect that exists in no clip. The pooled permutation test cannot see this,
because it shuffles labels across clips.

Two checks:
  (a) d within each clip separately -- does the effect exist anywhere on its own?
  (b) a CLIP-STRATIFIED permutation, shuffling make/miss only WITHIN a clip, which
      is the correct null for data blocked this way.
"""
import csv, json, math, re, sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\jmaku\Desktop\ShotLab")
sys.path.insert(0, str(ROOT))
from tools import eval_ablations as E
import tools.flare_report as FR

FPS = 30.0
PAIRS = {
    "20260729_115341": ("PXL_20260729_155320813", 22.64),
    "20260729_115926": ("PXL_20260729_155914855-001", 13.12),
    "20260729_120729": ("PXL_20260729_160743954", -12.47),
    "20260729_121450": ("PXL_20260729_161439291-002", 13.04),
}
METRIC = "knee_bend_3d_deg"
OUT = ROOT / "data/out/session_0729"
rows = json.load(open(OUT / "flare_readings_raw.json", encoding="utf-8"))
sess = list(csv.DictReader(open(OUT / "session_shots.csv", newline="", encoding="utf-8")))


def clip_start(stem):
    m = re.search(r"_(\d{8})_(\d{2})(\d{2})(\d{2})(\d{3})", stem)
    dd, hh, mm, ss, ms = m.groups()
    return datetime(int(dd[:4]), int(dd[4:6]), int(dd[6:]), int(hh), int(mm), int(ss), int(ms)*1000)


shot_truth = {}
for wide_stem in {v[0] for v in PAIRS.values()}:
    recs = [r for r in sess if Path(r["clip"]).stem == wide_stem]
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
buckets = defaultdict(list)
for r in rows:
    wide_stem, off = PAIRS[r["clip"]]
    wt = _wt[wide_stem]
    if not wt:
        continue
    ptime = r["frame"] / FPS + off
    j = int(np.argmin([abs(t - ptime) for t, _ in wt]))
    if abs(wt[j][0] - ptime) < 1.5:
        buckets[(wide_stem, j + 1)].append((abs(wt[j][0] - ptime), r))
joined = {k: min(v, key=lambda t: t[0])[1] for k, v in buckets.items()}

# clip -> (values, labels)
per_clip = defaultdict(lambda: ([], []))
for k, r in joined.items():
    t = shot_truth.get(k)
    v = r.get(METRIC)
    if t is None or v is None:
        continue
    per_clip[k[0]][0].append(float(v))
    per_clip[k[0]][1].append(1 if t else 0)


def cohen_d(v, l):
    a = v[l == 1]; b = v[l == 0]
    if len(a) < 2 or len(b) < 2:
        return None
    s = math.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    return None if s < 1e-9 else (a.mean() - b.mean()) / s


print(f"{METRIC}: per-clip breakdown")
print(f"  {'clip':<26}{'n mk/ms':>10}{'make%':>8}{'mean mk':>10}{'mean ms':>10}{'d':>8}")
print("  " + "-" * 74)
allv, alll, allc = [], [], []
for i, (clip, (v, l)) in enumerate(sorted(per_clip.items())):
    v = np.array(v, float); l = np.array(l, int)
    d = cohen_d(v, l)
    mk = v[l == 1]; ms = v[l == 0]
    print(f"  {clip:<26}{f'{len(mk)}/{len(ms)}':>10}{100*l.mean():>7.0f}%"
          f"{mk.mean():>10.1f}{ms.mean():>10.1f}"
          f"{(f'{d:+.2f}' if d is not None else '--'):>8}")
    allv.append(v); alll.append(l); allc.append(np.full(len(v), i))
v = np.concatenate(allv); l = np.concatenate(alll); c = np.concatenate(allc)
print(f"\n  POOLED d = {cohen_d(v, l):+.2f}  (n={int((l==1).sum())}/{int((l==0).sum())})")
print("  If the per-clip d's straddle zero while pooled is positive, the pooled")
print("  number is a between-clip artifact (Simpson's paradox), not a shot effect.\n")

# clip-stratified permutation: shuffle labels WITHIN each clip only
obs = abs(cohen_d(v, l))
rng = np.random.default_rng(0)
hits = 0
N = 20000
for _ in range(N):
    lp = l.copy()
    for i in np.unique(c):
        m = c == i
        lp[m] = rng.permutation(lp[m])
    dd = cohen_d(v, lp)
    if dd is not None and abs(dd) >= obs:
        hits += 1
print(f"  clip-STRATIFIED permutation p = {(hits+1)/(N+1):.4f}   "
      f"(pooled/unstratified was 0.0365)")
print("  This is the correct null for shots blocked into clips. If it is much")
print("  weaker than the pooled p, the clip structure was doing the work.")
