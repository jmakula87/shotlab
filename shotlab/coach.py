"""The coaching layer: turn a session's numbers into plain-English feedback —
what you did well, what to work on, per-shot grades, and a comparison to the
research-backed ideal targets.

Honesty rules baked in:
- Absolute angles are foreshortened by the single camera, so per-shot grading is
  done RELATIVE TO YOUR OWN norm for each zone (outlier detection) -- reliable
  regardless of the camera. Absolute-vs-ideal is reported as "aim for", flagged
  as approximate until court calibration.
- The one ideal that's well-established: a ~45 deg ENTRY angle into the rim
  (maximizes the effective opening). Consistency (low spread) matters as much as
  any single number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .session import fatigue_trends, consistency_stats

# Research-backed "aim for" reference (the gold standard to work toward). The
# entry-angle universal is shared with shotlab.textbook so the number can't
# drift between the coach and the profile's textbook block.
from .textbook import TEXTBOOK as _TB
IDEAL = {
    "entry_angle_deg": (_TB["entry_angle_deg"]["target"],
                        "≈45° entry maximizes the rim opening"),
    "release_angle_deg": (52.0, "~50–55° release for a typical jumper"),
}


def grade_shots(df: pd.DataFrame) -> pd.DataFrame:
    """Grade each shot vs YOUR OWN zone norm. A shot is 'off' when one of its
    metrics is an outlier (>=1.5 sd) for that zone -- i.e. something differed from
    your normal shot from that spot. Returns a per-shot grade + the issue."""
    metrics = ["release_angle_deg", "entry_angle_deg", "apex_height_ft", "knee_bend_deg"]
    nice = {"release_angle_deg": "release", "entry_angle_deg": "entry",
            "apex_height_ft": "apex", "knee_bend_deg": "knee bend"}
    from .metric_ranges import in_range
    rows = []
    for zone, g in df.groupby("zone"):
        for _, row in g.iterrows():
            worst = None
            for m in metrics:
                # skip absent values AND physically-impossible reads -- an artifact
                # (e.g. a 172deg "knee bend") must not set the zone norm or get
                # graded a form fault against it (2026-07-06 final sweep, D4/#7)
                if m not in g.columns or pd.isna(row.get(m)) or not in_range(m, row[m]):
                    continue
                vals = g[m][g[m].map(lambda v: in_range(m, v))]
                med, std = vals.median(), vals.std()
                if std and std > 1e-6:
                    z = (row[m] - med) / std
                    if abs(z) >= 1.5 and (worst is None or abs(z) > abs(worst[1])):
                        worst = (m, z, row[m], med)
            if worst is None:
                grade, issue = "clean", "matched your norm for this spot"
            else:
                m, z, val, med = worst
                grade = "off"
                issue = (f"{nice[m]} was {'high' if z > 0 else 'low'} "
                         f"({val:.0f} vs your {zone} norm {med:.0f})")
            rows.append({"shot_num": row.get("shot_num"), "zone": zone,
                         "made": row.get("made"), "grade": grade, "issue": issue})
    out = pd.DataFrame(rows)
    return out.sort_values("shot_num").reset_index(drop=True) if not out.empty else out


def _fmt(x):
    return f"{x:+.1f}"


def generate_review(df: pd.DataFrame) -> dict:
    """Produce {summary, strengths[], improvements[], focus[]} for a session."""
    strengths, improvements, focus = [], [], []
    if df.empty:
        return {"summary": "No shots.", "strengths": [], "improvements": [], "focus": []}

    n = len(df)
    dur = df["elapsed_min"].max()

    # ---- fatigue ----
    tr = fatigue_trends(df).set_index("metric") if not fatigue_trends(df).empty else None
    if tr is not None:
        for m, label in [("release_angle_deg", "release angle"),
                         ("entry_angle_deg", "entry angle"),
                         ("knee_bend_deg", "knee bend")]:
            if m in tr.index and tr.loc[m, "slope_per_min"] is not None:
                drop = tr.loc[m, "start"] - tr.loc[m, "end"]
                if m == "knee_bend_deg":
                    # rising knee angle = LESS bend = losing legs
                    if tr.loc[m, "slope_per_min"] > 0.15:
                        improvements.append(
                            f"Your knee bend faded as the session went on "
                            f"(~{abs(drop):.0f}° less load by the end) — a legs-tiring "
                            f"sign. Shorter sets or a rest between them could keep your base under you.")
                elif drop > 5:
                    improvements.append(
                        f"Your {label} dropped ~{drop:.0f}° from start to end — "
                        f"fatigue flattening your shot. Watch your last few reps per set.")
                elif abs(drop) <= 3:
                    strengths.append(f"Your {label} held steady across all {dur:.0f} "
                                     f"minutes — good stamina on that piece.")

    # ---- make% fatigue ----
    if "made" in df.columns:
        m = df[df["made"].isin([True, False])]
        if len(m) >= 8:
            half = m["elapsed_min"].median()
            p1 = 100 * (m[m["elapsed_min"] <= half]["made"] == True).mean()
            p2 = 100 * (m[m["elapsed_min"] > half]["made"] == True).mean()
            if p1 - p2 >= 8:
                improvements.append(
                    f"Make% fell from ~{p1:.0f}% to ~{p2:.0f}% as you tired "
                    f"(low-confidence number, but the *direction* is telling) — your "
                    f"makes live in the fresh reps. Conditioning is a shooting skill.")
            elif p2 >= p1:
                strengths.append(f"You held (or improved) your make% late — "
                                 f"{p1:.0f}%→{p2:.0f}%. Mentally tough finishing.")

    # ---- consistency: best & worst zone ----
    cons = consistency_stats(df)
    if "zone" in df.columns:
        zc = df.groupby("zone")
        # within-zone entry spread = repeatability from a spot (needs >=4 shots)
        spreads = {z: g["entry_angle_deg"].std() for z, g in zc
                   if "entry_angle_deg" in g and g["entry_angle_deg"].notna().sum() >= 4}
        if spreads:
            best = min(spreads, key=spreads.get)
            worst = max(spreads, key=spreads.get)
            strengths.append(f"Your **{best}** shot was your most repeatable "
                             f"(tightest entry-angle spread).")
            if worst != best:
                improvements.append(f"Your **{worst}** shot scattered the most — "
                                    f"least repeatable. Park there for a focused set.")
                focus.append(f"Groove the **{worst}** spot — most reps, smallest motion, "
                             f"chase repeatability before range.")

        # WHERE you actually score -- the money spot is the best make%, not the
        # tightest spread (audit D14a: the tightest spot was 1-for-6). Also flag a
        # spot you shoot a lot from but rarely make. Make% is low-confidence.
        if "made" in df.columns:
            m = df[df["made"].isin([True, False])]
            rate = {z: (g["made"] == True).mean() for z, g in m.groupby("zone")
                    if len(g) >= 6}
            cnt = m["zone"].value_counts()
            if rate:
                bestm = max(rate, key=rate.get)
                strengths.append(f"You score best from **{bestm}** "
                                 f"({rate[bestm]*100:.0f}% on {int(cnt.get(bestm, 0))} shots) "
                                 f"— your money spot (make% is a low-confidence heuristic).")
                for z in list(cnt.index)[:2]:
                    if z in rate and cnt[z] >= 0.30 * len(m) and rate[z] < 0.30:
                        improvements.append(
                            f"You took **{int(cnt[z])}** shots from **{z}** but made only "
                            f"**{rate[z]*100:.0f}%** — that volume isn't paying off; "
                            f"either it's not your spot or your form drifts there.")
                        break

    # ---- volume per zone (where you practiced) ----
    if "zone" in df.columns:
        vc = df["zone"].value_counts()
        if len(vc):
            focus.append(f"You took most shots from **{vc.index[0]}** ({vc.iloc[0]}). "
                         f"If that's not a game spot for you, mix the floor more.")

    # ---- entry vs ideal (flagged approximate) ----
    if "entry_angle_deg" in df.columns and df["entry_angle_deg"].notna().sum() >= 5:
        from .metric_ranges import gate
        avg = gate(df, "entry_angle_deg")["entry_angle_deg"].mean()   # drop artifacts
        focus.append(f"Aim toward a ~45° entry angle (yours averages {avg:.0f}°, but "
                     f"that number is foreshortened by the camera — calibrate next "
                     f"session for a true read).")

    summary = (f"{n} shots over {dur:.0f} min. "
               + ("Strong stamina and repeatability in spots; "
                  if strengths else "")
               + ("clear fatigue signal to manage." if improvements else "solid all-round."))
    return {"summary": summary, "strengths": strengths,
            "improvements": improvements, "focus": focus}


def arc_from_angles(release_deg: float, entry_deg: float, n: int = 40):
    """Reconstruct a NORMALIZED arc shape from a shot's release & entry angles
    (x from 0=release to 1=rim). Lets us draw 'your arc vs the ideal arc' from the
    angles we already have, without storing the full trajectory. Returns (x, h)
    with h scaled to a unit horizontal span."""
    import math
    tr = math.tan(math.radians(max(1.0, min(89.0, release_deg))))
    te = math.tan(math.radians(max(1.0, min(89.0, entry_deg))))
    # h(0)=0, slope(0)=tr, slope(1)=-te  ->  h = a x^2 + b x
    b = tr
    a = (-te - tr) / 2.0
    x = np.linspace(0, 1, n)
    h = a * x * x + b * x
    return x, h


IDEAL_ARC = (52.0, 45.0)   # release, entry -> the reference arc to aim at


def recommend_drills(df: pd.DataFrame, max_drills: int = 5) -> list[str]:
    """Concrete drills prescribed from this session's weaknesses."""
    drills = []
    if df.empty:
        return drills

    # 0) target your STRONGEST make-driver -- the mechanic that most tracks with
    #    YOUR makes. Drills used to ignore the driver analysis entirely (audit
    #    D14b). Make-correlation is low-confidence, so this is a lean, not a law.
    # (expected make-direction, drill text). Only prescribe when the MEASURED
    # direction matches the drill's premise -- else it would coach the OPPOSITE of
    # the data (2026-07-06 final sweep caught "let it go later" prescribed on a
    # session where makes released earlier). release_vs_apex dropped (low-conf).
    from .correlate import correlate_makes
    _DRIVER_DRILLS = {
        "follow_through_hold_s": ("higher",
            "**Freeze the follow-through:** hold your wrist snapped until the ball "
            "hits the rim, every rep — your makes hold it longer than your misses."),
        "knee_bend_deg": ("lower",          # lower knee angle = deeper bend
            "**Load the legs:** exaggerate a deeper dip on a set of 25 — your makes "
            "come with more knee bend. Power from the legs, not the arm."),
        "release_angle_deg": ("higher",
            "**Softer arc:** shoot over a raised obstacle so you drop it in rather "
            "than line-drive it — your makes carry more arc."),
        "tempo_dip_to_release_s": ("lower",
            "**One motion:** dip-and-up in a single beat; your makes are quicker "
            "from the load to the release."),
        "balance_drift_px_per_ht": ("lower",
            "**Land where you left:** shoot with a piece of tape under your lead "
            "foot and land on it — your makes drift less."),
    }
    try:
        assocs = correlate_makes(df.to_dict("records"))
        for a in sorted(assocs, key=lambda a: abs(a.cohen_d or 0), reverse=True):
            spec = _DRIVER_DRILLS.get(a.metric)
            if (spec and a.cohen_d is not None and abs(a.cohen_d) >= 0.2
                    and a.confidence != "insufficient" and a.direction == spec[0]):
                drills.append(spec[1])
                break
    except Exception:
        pass

    # 1) weakest (least repeatable) zone -> form reps there
    if "zone" in df.columns:
        spreads = {z: g["entry_angle_deg"].std() for z, g in df.groupby("zone")
                   if g["entry_angle_deg"].notna().sum() >= 4}
        if spreads:
            worst = max(spreads, key=spreads.get)
            drills.append(f"**Form-shooting, {worst}:** 3×25 makes from your {worst} "
                          f"spot — identical motion every rep, chase a tight group "
                          f"before worrying about makes. (Your least repeatable spot.)")

    # 2) fatigue -> conditioning
    tr = fatigue_trends(df)
    tr = tr.set_index("metric") if not tr.empty else None
    fatigued = False
    if tr is not None:
        for m in ("release_angle_deg", "entry_angle_deg"):
            if m in tr.index and tr.loc[m, "slope_per_min"] is not None \
                    and (tr.loc[m, "start"] - tr.loc[m, "end"]) > 5:
                fatigued = True
    if "made" in df.columns:
        mm = df[df["made"].isin([True, False])]
        if len(mm) >= 8:
            half = mm["elapsed_min"].median()
            p1 = (mm[mm["elapsed_min"] <= half]["made"] == True).mean()
            p2 = (mm[mm["elapsed_min"] > half]["made"] == True).mean()
            if p1 - p2 >= 0.08:
                fatigued = True
    if fatigued:
        drills.append("**Conditioning ladder:** 10 shots → sprint baseline-to-"
                      "baseline → repeat ×5. Trains you to hold your form tired "
                      "(your shot flattened late this session).")

    # 3) flat entry -> arc drill
    if "entry_angle_deg" in df.columns and df["entry_angle_deg"].notna().sum() >= 5:
        from .metric_ranges import gate
        if gate(df, "entry_angle_deg")["entry_angle_deg"].mean() < 45:   # drop artifacts
            drills.append("**Arc drill:** shoot over a raised obstacle (chair on a "
                          "table / partner's reach); make yourself drop the ball IN "
                          "rather than line-drive it. Aim for a softer, ~45° entry.")

    # 4) one-spot heavy -> spread the floor
    if "zone" in df.columns:
        vc = df["zone"].value_counts()
        if len(vc) and vc.iloc[0] > 0.45 * len(df):
            drills.append(f"**Star drill:** 5 makes from each of 5 spots around the "
                          f"arc before moving on. You took {int(vc.iloc[0])}/{len(df)} "
                          f"from {vc.index[0]} — broaden the floor.")

    # 5) baseline consistency drill (always useful)
    drills.append("**Beat-your-spread:** pick your money spot, take 25, and try to "
                  "shrink your entry-angle spread vs last session — consistency is "
                  "the whole game.")
    return drills[:max_drills]


def rank_shots(df: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Rank shots by 'ideal form/arc' so you can review your best ones.

    Honest scoring given foreshortening: rewards a MADE shot, a 'clean' grade
    (consistent motion vs your own norm), and a softer arc (higher apex + an
    entry angle in a good window rather than flat). Returns the top N with a
    'why' note and clip / shot_in_clip so they can be pulled for review."""
    if df.empty:
        return pd.DataFrame()
    g = grade_shots(df)[["shot_num", "grade"]]
    d = df.merge(g, on="shot_num", how="left") if "shot_num" in df else df.copy()
    # null physically-impossible reads so they don't skew the ranking z-scores
    # (best_shots.csv feeds the profile's arc pool; 2026-07-06 final sweep, #7)
    from .metric_ranges import in_range
    for _m in ("apex_height_ft", "entry_angle_deg", "release_angle_deg", "knee_bend_deg"):
        if _m in d.columns:
            d[_m] = d[_m].where(d[_m].map(lambda v: in_range(_m, v)))

    def zscore(col):
        if col not in d:
            return pd.Series(0.0, index=d.index)
        return d.groupby("zone")[col].transform(
            lambda s: (s - s.mean()) / s.std() if s.std() and s.std() > 1e-6 else s * 0)

    apex_z = zscore("apex_height_ft").fillna(0)          # softer arc = higher apex
    score = apex_z.copy()
    if "made" in d:
        score = score + (d["made"] == True).astype(float) * 2.0
    if "grade" in d:
        score = score + (d["grade"] == "clean").astype(float) * 1.0
    if "entry_angle_deg" in d:                            # penalize flat / extreme
        e = d["entry_angle_deg"]
        score = score - ((e < 40) | (e > 70)).astype(float) * 1.0
    d = d.assign(ideal_score=score.round(2))

    def why(r):
        bits = []
        if r.get("made") == True:
            bits.append("made")
        if r.get("grade") == "clean":
            bits.append("clean form")
        if r.get("apex_height_ft", 0) and apex_z.loc[r.name] > 0.5:
            bits.append("soft arc")
        return ", ".join(bits) or "solid"

    d["why"] = [why(r) for _, r in d.iterrows()]
    cols = [c for c in ["shot_num", "clip", "shot_in_clip", "elapsed_min", "zone",
                        "release_angle_deg", "entry_angle_deg", "apex_height_ft",
                        "made", "grade", "ideal_score", "why"] if c in d.columns]
    return d.sort_values("ideal_score", ascending=False)[cols].head(top).reset_index(drop=True)


def review_markdown(review: dict) -> str:
    md = [f"**{review['summary']}**", ""]
    if review["strengths"]:
        md.append("### ✅ What you did well")
        md += [f"- {s}" for s in review["strengths"]]
    if review["improvements"]:
        md.append("\n### ⚠️ What to work on")
        md += [f"- {s}" for s in review["improvements"]]
    if review["focus"]:
        md.append("\n### 🎯 Focus next session")
        md += [f"- {s}" for s in review["focus"]]
    return "\n".join(md)
