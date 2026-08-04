"""Is the follow-through effect FORM, or the shooter REACTING to the outcome?

follow_through_hold_s counts frames after release while the wrist stays above the
shoulder. The ball needs ~1s to reach the rim and the window here is 1.0s, so a
shooter who can see the shot is good/bad partway through the flight could hold or
drop accordingly. That would produce a real make/miss correlation that is USELESS
for coaching -- making causes the hold, not the reverse.

Discriminating test, no re-run needed: hold >= t is derivable from the scalar, so
the whole survival curve is already on disk. Read WHERE the curves separate.
  separation from t=1     -> form (present at the instant of release)
  separation only after ~10f (0.33s) -> consistent with outcome knowledge
Also checks the two ways this metric can lie: right-censoring at the window edge,
and holds truncated by pose dropout rather than by the wrist actually dropping.
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
POST = 30                     # the window body_metrics passes
PAIRS = {
    "20260729_115341": ("PXL_20260729_155320813", 22.64),
    "20260729_115926": ("PXL_20260729_155914855-001", 13.12),
    "20260729_120729": ("PXL_20260729_160743954", -12.47),
    "20260729_121450": ("PXL_20260729_161439291-002", 13.04),
}
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

mk = [r["follow_through_hold_s"] for k, r in joined.items()
      if shot_truth.get(k) is True and r.get("follow_through_hold_s") is not None]
ms = [r["follow_through_hold_s"] for k, r in joined.items()
      if shot_truth.get(k) is False and r.get("follow_through_hold_s") is not None]
mk_f = np.round(np.array(mk) * FPS).astype(int)
ms_f = np.round(np.array(ms) * FPS).astype(int)
print(f"makes n={len(mk_f)}  misses n={len(ms_f)}")
print(f"mean hold: make {mk_f.mean()/FPS:.3f}s  miss {ms_f.mean()/FPS:.3f}s\n")

print("censoring / degenerate-value check")
for nm, a in (("make", mk_f), ("miss", ms_f)):
    print(f"  {nm}: at window ceiling ({POST}f) {np.mean(a>=POST)*100:5.1f}%   "
          f"zero-length {np.mean(a<=0)*100:5.1f}%   "
          f"median {np.median(a):.0f}f   max {a.max()}f")
print("  a ceiling pile-up means the metric is right-censored and the true gap is")
print("  UNDERstated; a zero pile-up means pose dropout, not a real dropped wrist.\n")

print("survival: P(hold >= t) by outcome -- WHERE do the curves separate?")
print(f"  {'t (frames)':>11}{'t (s)':>8}{'P make':>9}{'P miss':>9}{'gap':>8}   "
      f"{'Fisher p':>9}")
print("  " + "-" * 62)
try:
    from scipy.stats import fisher_exact
    have_scipy = True
except Exception:
    have_scipy = False
for t in list(range(1, 13)) + [15, 18, 21, 24, 27, 30]:
    a = int(np.sum(mk_f >= t)); b = int(np.sum(ms_f >= t))
    pa, pb = a/len(mk_f), b/len(ms_f)
    if have_scipy:
        _, p = fisher_exact([[a, len(mk_f)-a], [b, len(ms_f)-b]])
        ps = f"{p:>9.3f}"
    else:
        ps = f"{'--':>9}"
    star = "  <--" if have_scipy and p < 0.05 else ""
    print(f"  {t:>11}{t/FPS:>8.2f}{pa:>9.2f}{pb:>9.2f}{pa-pb:>8.2f}   {ps}{star}")

print("\n  EARLY separation (t<=6, i.e. within 0.2s of release) => form.")
print("  LATE-ONLY separation (emerging past t~10) => consistent with the shooter")
print("  reacting to a flight they can already read. Coaching value differs totally.")
