"""Universal shooting ideals -- the few numbers that ARE the same for everyone,
kept SEPARATE from the personal profile (which is the mean of your own good
shots).

Only physics-backed universals belong here. Body-form angles (elbow bend, knee
depth, tempo, follow-through) are player-specific and optimized to your body, so
they stay personal -- chasing a pro's numbers there can hurt a shot that already
works. What qualifies as universal:

  - ENTRY ANGLE ~45 deg: maximizes the rim opening (shot-science / Noah data);
    the ball's geometry at the rim doesn't care whose shot it is.
  - ELBOW FLARE ~0 deg: a tucked elbow (in the shoulder->rim vertical plane)
    keeps the ball on line; flaring it out sideways pushes shots off-line. Bad
    for every shooter. BUT flare is out-of-plane, so a single camera cannot see
    it -- it needs the 2-camera 3D rig (twocam.elbow_flare) to measure. It's a
    real universal target that only comes online with the second camera.

Each entry carries `measurable_now` so consumers know whether we can actually
compare against it on the current 1-camera setup.
"""

from __future__ import annotations

TEXTBOOK = {
    "entry_angle_deg": {
        "target": 45.0,
        "tolerance": 5.0,
        "measurable_now": True,
        "why": "~45 deg entry maximizes the rim opening -- true for every shooter.",
    },
    "elbow_flare_deg": {
        "target": 0.0,
        "tolerance": 5.0,
        "measurable_now": False,
        "needs": "2-camera 3D (flare is out-of-plane; twocam.elbow_flare measures "
                 "it once the second camera is calibrated).",
        "why": "a tucked elbow keeps the ball on line; flaring it out pushes "
               "shots off-line -- bad for everyone.",
    },
}


def grade(metric: str, value):
    """Compare a measured value to its textbook target.

    Returns (within_tolerance, signed_delta, spec) or None when there's no
    textbook target for the metric, no value, or it isn't measurable on the
    current rig (so callers never present a comparison we can't actually make)."""
    spec = TEXTBOOK.get(metric)
    if spec is None or value is None or not spec.get("measurable_now", False):
        return None
    delta = round(float(value) - spec["target"], 1)
    return (abs(delta) <= spec["tolerance"], delta, spec)


def profile_block() -> dict:
    """The textbook ideals as shipped in a profile: target + tolerance + the
    universal 'why', plus whether it's live on the current rig. Kept as a
    SEPARATE block from the personal `ideal` so nothing blends the two."""
    return {k: {"target": v["target"], "tolerance": v["tolerance"],
                "measurable_now": v["measurable_now"], "why": v["why"],
                **({"needs": v["needs"]} if "needs" in v else {})}
            for k, v in TEXTBOOK.items()}
