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


# metrics measured AT the release frame -- only trustworthy when the release
# itself was found confidently (else the value is noise about a guessed frame)
_RELEASE_ANCHORED = {"release_vs_apex_s", "tempo_dip_to_release_s",
                     "elbow_angle_at_release_deg",
                     # the metric-3D twin is anchored to the SAME release estimate,
                     # so it inherits the same trust ceiling. Omitting it would have
                     # let the 3D elbow past a gate its 2D counterpart cannot pass
                     # (2026-08-04) -- a new metric name silently escaping an
                     # existing rule is exactly how the release_conf bypass happened.
                     "elbow_angle_at_release_3d_deg"}

# ⛔ NOT a form metric, whatever its p-value: follow_through_hold_s is measured
# ENTIRELY after the ball has left the hand, over a 1.0s window, while the ball
# needs ~1s to reach the rim. Measured 2026-08-04 on 49 makes / 61 misses: the
# survival curves P(hold >= t) are IDENTICAL (1.00 vs 1.00) through t=10 frames
# and separate only from t~15, significant at t=18-24 -- i.e. 0.6-0.8s post
# release, when the ball is at the rim and the shooter already knows. The plain
# mechanism is that a miss sends the shooter to rebound and a make does not.
# It cleared Bonferroni (d=+0.52, p=0.0054) and is still not usable for coaching:
# making causes the hold, not the reverse. Any future "follow-through" driver must
# show separation EARLY (t <= 6) before it means anything.
_OUTCOME_REACTIVE = {"follow_through_hold_s"}


def _as_bool(made) -> bool | None:
    """ShotRecord.made is True/False/None (None = unclassified)."""
    if made is True or made is False:
        return made
    return None


def _pair_values(rows, field, label_field="made"):
    """(values, label01) for rows where the metric and the binary label are both
    known. `label_field` is "made" (make/miss) or "felt_good" (subjective feel).

    Physically-implausible reads are dropped through metric_ranges.in_range (the
    SAME gate the profile ideals use), so a mis-detected sub-90 deg "elbow at
    release" can't inflate a make-driver -- both decision surfaces see the same
    real reads (2026-07-06 audit). Release-anchored metrics additionally require a
    trustworthy release (release_conf high/medium): a make-driver carried entirely
    by low-confidence wrist apexes (release_vs_apex_s was d=-1.2 vs ~0 on high-conf
    only) is an artifact, not a signal (2026-07-06 final sweep)."""
    from .metric_ranges import in_range
    gate_conf = field in _RELEASE_ANCHORED
    vals, lab = [], []
    for r in rows:
        v = r.get(field) if isinstance(r, dict) else getattr(r, field, None)
        m = r.get(label_field) if isinstance(r, dict) else getattr(r, label_field, None)
        m = _as_bool(m)
        if v is None or m is None:
            continue
        if not in_range(field, v):     # NaN/inf/artifact == missing
            continue
        if gate_conf:
            rc = r.get("release_conf") if isinstance(r, dict) else getattr(r, "release_conf", None)
            if rc is not None and rc not in ("high", "medium"):
                continue               # a KNOWN-low-conf release -> not a driver read
        vals.append(float(v))
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
        if field in _OUTCOME_REACTIVE:
            # "excluded" keeps it out of `real` in summarize_drivers, so it can be
            # neither reported as a driver NOR quoted as the largest effect in the
            # null message. Measured and dead-lettered, not silently dropped.
            out.append(MetricMakeAssoc(
                field, label, 0, 0, None, None, None, None, None,
                None, "excluded", "",
                note="measured entirely AFTER release, over a window in which the "
                     "shooter can already read the flight -- separation appears only "
                     "0.6-0.8s post-release (2026-08-04). Outcome causes it."))
            continue
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
        # .get, not [], so a new confidence tier sorts last instead of raising --
        # "excluded" (outcome-reactive) is the first such tier.
        rank = {"medium": 0, "low": 1, "insufficient": 2}.get(a.confidence, 3)
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

    # Nothing reached medium confidence => report the NULL, with the effect size
    # this sample could actually have seen. Listing the three largest low-confidence
    # results as things that "lean" turns noise into advice: on 2026-08-03 that
    # printed "entry angle leans" for d=0.15, p=0.38 on 137 shots, where the
    # detectable floor was d~0.34 and nothing came close.
    best = [a for a in real if a.confidence == "medium"]
    if not best:
        n = max((a.n_made + a.n_miss) for a in real)
        floor = 2.8 / float(np.sqrt(max(n, 4) / 2))
        top = max(real, key=lambda a: abs(a.cohen_d))
        return (f"**Nothing separates your {pos} from your {neg}** in this sample. "
                f"With n={n} the smallest effect detectable is about d={floor:.2f}; "
                f"the largest seen is {top.label} at d={top.cohen_d} (p={top.p_perm}), "
                f"which is inside the noise. That is a real result, not a gap in the "
                f"data -- {caveat}. More shots would lower the floor.")

    lines = [f"**What tracks with your {pos}** (advisory -- samples are small):"]
    shown = best
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
