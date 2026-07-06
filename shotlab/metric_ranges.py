"""Physically-plausible ranges per metric -- the single source of truth for what
counts as a real measurement vs a pose/tracking artifact.

A value outside its range is a mis-detection (a 172 deg "knee bend" is a
straight-leg pose; a 45 deg "elbow at release" is a slewed mid-push frame; a
0.00s follow-through is a floor, not a hold) and must not skew a profile ideal
OR a make/miss correlation. Both surfaces (tools/export_profile.py building the
ideals, shotlab/correlate.py building the make-drivers) gate through here, so
they can't disagree about which reads are real (2026-07-06 audit: correlate was
feeding raw ungated values and inflating the elbow make-driver via sub-90 deg
artifact reads).

None on either end = no gate on that end.
"""

from __future__ import annotations

import numpy as np

VALID_RANGE = {
    "knee_bend_deg": (30.0, 150.0),          # >150 = not actually bent (artifact)
    "elbow_angle_at_release_deg": (90.0, 180.0),
    "tempo_dip_to_release_s": (0.05, 2.0),   # <0.05s = sub-frame floor
    "follow_through_hold_s": (0.02, 3.0),    # 0.00 = no hold measured
    "balance_drift_px_per_ht": (0.0, 3.0),
    "release_angle_deg": (10.0, 80.0),
    "entry_angle_deg": (10.0, 80.0),
}


def in_range(col: str, val) -> bool:
    """True if `val` is a finite, physically-plausible reading of `col`.
    Ungated columns pass any finite value."""
    try:
        fv = float(val)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(fv):
        return False
    lo, hi = VALID_RANGE.get(col, (None, None))
    if lo is not None and fv < lo:
        return False
    if hi is not None and fv > hi:
        return False
    return True
