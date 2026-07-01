"""Make-correlation engine -- "which of MY mechanics go with MY makes?"

The long-term holy grail of the tool: instead of comparing you to a textbook
ideal, correlate YOUR form/arc metrics against YOUR make/miss outcomes and
surface the mechanics that actually track with the ball going in.

Honesty first (this is why it stays advisory until calibration footage lands):
  * make/miss is a LOW-confidence geometric heuristic (`make.classify_make`);
  * one side-on camera foreshortens depth-dependent metrics (elbow flare,
    squareness) -- those carry an extra caveat;
  * a workout is a few dozen shots, so samples per metric are small.
So we report an *effect size* (standardized mean difference, Cohen's d) and a
*permutation p-value* (assumption-free, robust at small n), gate on a minimum
count of made AND missed shots, and never promote a finding above "medium".
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


# (record field, human label, depth_dependent?). depth_dependent metrics are
# foreshortened on a single side-on camera -> extra caveat in the note.
CANDIDATE_METRICS = [
    ("release_angle_deg", "Release angle", False),
    ("entry_angle_deg", "Entry angle", False),
    ("apex_height_ft", "Apex height", False),
    ("knee_bend_deg", "Knee bend depth", False),
    ("release_vs_apex_s", "Release timing vs apex", False),
    ("elbow_angle_at_release_deg", "Elbow angle at release", True),
    ("follow_through_hold_s", "Follow-through hold", False),
    ("balance_drift_px_per_ht", "Balance drift", False),
    ("backspin_rpm", "Backspin", False),
]

# Units for the human summary (display only).
_UNITS = {
    "release_angle_deg": "°", "entry_angle_deg": "°",
    "elbow_angle_at_release_deg": "°", "knee_bend_deg": "°",
    "apex_height_ft": " ft", "release_vs_apex_s": " s",
    "follow_through_hold_s": " s", "backspin_rpm": " rpm",
    "balance_drift_px_per_ht": "",
}


@dataclass
class MetricMakeAssoc:
    metric: str
    label: str
    n_made: int
    n_miss: int
    mean_made: float | None
    mean_miss: float | None
    diff: float | None              # mean_made - mean_miss
    cohen_d: float | None           # standardized effect size
    point_biserial_r: float | None  # corr(metric, made)
    p_perm: float | None            # permutation p-value (two-sided on |diff|)
    confidence: str                 # medium | low | insufficient
    direction: str                  # "higher" | "lower" | ""
    note: str = ""

    def as_row(self) -> dict:
        return asdict(self)


def _as_bool(made) -> bool | None:
    """ShotRecord.made is True/False/None (None = unclassified)."""
    if made is True or made is False:
        return made
    return None


def _pair_values(rows, field, label_field="made"):
    """(values, label01) for rows where the metric and the binary label are both
    known. `label_field` is "made" (make/miss) or "felt_good" (subjective feel)."""
    vals, lab = [], []
    for r in rows:
        v = r.get(field) if isinstance(r, dict) else getattr(r, field, None)
        m = r.get(label_field) if isinstance(r, dict) else getattr(r, label_field, None)
        m = _as_bool(m)
        if v is None or m is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(fv):        # NaN/inf (missing pose/spin) == missing
            continue
        vals.append(fv)
        lab.append(1 if m else 0)
    return np.array(vals, float), np.array(lab, int)


def _permutation_p(vals, lab, n_perm, seed) -> float:
    """Two-sided permutation p on the absolute difference of group means."""
    made, miss = vals[lab == 1], vals[lab == 0]
    obs = abs(made.mean() - miss.mean())
    rng = np.random.default_rng(seed)
    n_made = int(lab.sum())
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(vals)
        if abs(perm[:n_made].mean() - perm[n_made:].mean()) >= obs - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)      # add-one smoothing (never 0)


def _confidence(n_made, n_miss, p, d, min_n) -> str:
    if n_made < min_n or n_miss < min_n:
        return "insufficient"
    # capped at "medium" on purpose: the make label itself is low-confidence.
    if p is not None and p < 0.05 and d is not None and abs(d) >= 0.5:
        return "medium"
    return "low"


def correlate_label(rows, *, label_field="made", min_n=8, n_perm=2000,
                    seed=0) -> list[MetricMakeAssoc]:
    """Associate each candidate metric with a binary outcome across the shots.

    `label_field` = "made" (make/miss) or "felt_good" (your subjective good/off
    tag). `rows` is any iterable of ShotRecord or dict rows. Returns one
    association per candidate, sorted by |Cohen's d| with sufficient-n,
    significant findings first; insufficient-n metrics sort last. (In the
    result, n_made/mean_made = the positive class, n_miss/mean_miss = negative.)
    """
    rows = list(rows)
    out: list[MetricMakeAssoc] = []
    for field, label, depth in CANDIDATE_METRICS:
        vals, lab = _pair_values(rows, field, label_field)
        n_made, n_miss = int((lab == 1).sum()), int((lab == 0).sum())
        if n_made == 0 or n_miss == 0:
            out.append(MetricMakeAssoc(
                field, label, n_made, n_miss, None, None, None, None, None,
                None, "insufficient", "",
                note="need both outcomes with this metric"))
            continue
        made, miss = vals[lab == 1], vals[lab == 0]
        mean_made, mean_miss = float(made.mean()), float(miss.mean())
        diff = mean_made - mean_miss
        # pooled SD for Cohen's d
        sd = float(np.sqrt(((made.var(ddof=0) * n_made +
                             miss.var(ddof=0) * n_miss) / (n_made + n_miss))))
        d = diff / sd if sd > 1e-9 else 0.0
        # point-biserial = Pearson(metric, made01)
        if vals.std() > 1e-9 and lab.std() > 1e-9:
            r = float(np.corrcoef(vals, lab)[0, 1])
        else:
            r = 0.0
        enough = n_made >= min_n and n_miss >= min_n
        p = _permutation_p(vals, lab, n_perm, seed) if enough else None
        conf = _confidence(n_made, n_miss, p, d, min_n)
        direction = "higher" if diff > 0 else ("lower" if diff < 0 else "")
        note = ("depth-dependent on a single side-on camera -- treat as a hint"
                if depth else "")
        if not enough:
            note = (note + "; " if note else "") + \
                f"only {min(n_made, n_miss)} of the rarer outcome (need {min_n})"
        out.append(MetricMakeAssoc(
            field, label, n_made, n_miss, round(mean_made, 2),
            round(mean_miss, 2), round(diff, 3), round(d, 3), round(r, 3),
            None if p is None else round(p, 4), conf, direction, note))

    def _key(a: MetricMakeAssoc):
        rank = {"medium": 0, "low": 1, "insufficient": 2}[a.confidence]
        mag = abs(a.cohen_d) if a.cohen_d is not None else -1.0
        return (rank, -mag)

    out.sort(key=_key)
    return out


def correlate_makes(rows, **kw) -> list[MetricMakeAssoc]:
    """Which mechanics track with the ball going IN (make/miss heuristic)."""
    return correlate_label(rows, label_field="made", **kw)


def correlate_feel(rows, **kw) -> list[MetricMakeAssoc]:
    """Which mechanics track with shots that FELT good (your subjective tag) --
    a personalization signal that sidesteps the weak make/miss detector."""
    return correlate_label(rows, label_field="felt_good", **kw)


# subject -> (positive word, negative word, empty-message, reliability caveat)
_SUBJECTS = {
    "makes": ("makes", "misses",
              "Not enough cleanly-classified makes AND misses yet to correlate "
              "form with outcomes. Keep filming -- this engine sharpens with volume.",
              "only as reliable as the make detection"),
    "feel": ("good-feeling shots", "off-feeling shots",
             "Not enough good- AND off-tagged shots yet. Tag shots by feel as you "
             "film -- this learns YOUR ideal as the tags add up.",
             "based on your own feel tags"),
}


def summarize_drivers(assocs: list[MetricMakeAssoc], subject="makes") -> str:
    """Plain-English review of what tracks with the outcome, honest about the
    heuristic signal and small samples. subject = "makes" or "feel"."""
    pos, neg, empty_msg, caveat = _SUBJECTS.get(subject, _SUBJECTS["makes"])
    real = [a for a in assocs if a.confidence in ("medium", "low")
            and a.cohen_d is not None]
    if not real:
        return empty_msg
    lines = [f"**What tracks with your {pos}** (advisory -- samples are small):"]
    shown = [a for a in real if a.confidence == "medium"] or real[:3]
    for a in shown[:4]:
        unit = _UNITS.get(a.metric, "")
        strength = "stands out" if a.confidence == "medium" else "leans"
        lines.append(
            f"- **{a.label}** {strength}: {pos} ~{abs(a.diff):.2f}{unit} "
            f"{a.direction} than {neg} "
            f"({a.mean_made}{unit} vs {a.mean_miss}{unit}; "
            f"d={a.cohen_d}, p={a.p_perm}, n={a.n_made}/{a.n_miss})."
            + (f" Note: {a.note}." if a.note else ""))
    lines.append(f"_Correlation, not proof of cause -- and {caveat}. "
                 "Personalized ideals firm up with more shots._")
    return "\n".join(lines)


def summarize_make_drivers(assocs: list[MetricMakeAssoc]) -> str:
    return summarize_drivers(assocs, subject="makes")


def summarize_feel_drivers(assocs: list[MetricMakeAssoc]) -> str:
    return summarize_drivers(assocs, subject="feel")
