"""Universal shooting ideals -- the few numbers that ARE the same for everyone,
kept SEPARATE from the personal profile (which is the mean of your own good
shots).

Only physics-backed universals belong here. Body-form angles (elbow bend, knee
depth, tempo, follow-through) are player-specific and optimized to your body, so
they stay personal -- chasing a pro's numbers there can hurt a shot that already
works. What qualifies as universal:

  - ENTRY ANGLE ~45 deg: maximizes the rim opening (shot-science / Noah's 100M+
    tracked shots); the ball's geometry at the rim doesn't care whose shot it is.
  - RELEASE ANGLE ~50-52 deg: the launch angle that PRODUCES a ~45 deg entry
    (the ball leaves below the rim, so the optimal launch is always >45). Peer-
    reviewed motion capture clusters good shooters near 51-52 deg; shorter
    players carry slightly MORE arc (lower release -> steeper optimal), so ~52
    fits a <=6'0" guard. Semi-universal (drifts a few deg with distance).
  - ELBOW FLARE ~0 deg: a tucked elbow (in the shoulder->rim vertical plane)
    keeps the ball on line; flaring it out sideways pushes shots off-line. Bad
    for every shooter. Out-of-plane -> needs the 2-camera 3D rig to measure.

`measurable_now` says whether we can compare on the CURRENT rig; `blocked_by`
names what unblocks it. NOTE: the arc angles ARE computed by the pipeline but a
single wide camera FORESHORTENS them (reads high), so they aren't trustworthy
against these targets until court calibration -- hence measurable_now=False for
them, unblocked by "calibration", not the 2nd camera.
"""

from __future__ import annotations

_CALIB_NOTE = ("a single wide camera foreshortens the arc (reads high), so the "
               "raw degrees aren't comparable to this target until we calibrate "
               "-- mark the court corners with the known dimensions, or use the "
               "2-camera rig.")

TEXTBOOK = {
    "entry_angle_deg": {
        "target": 45.0,
        "tolerance": 3.0,
        "measurable_now": False,
        "blocked_by": "calibration",
        "needs": _CALIB_NOTE,
        "why": "~45 deg entry maximizes the rim opening -- true for every shooter "
               "(Noah's 100M+ shots; higher arcs lose make%).",
    },
    "release_angle_deg": {
        "target": 52.0,
        "tolerance": 4.0,
        "measurable_now": False,
        "blocked_by": "calibration",
        "needs": _CALIB_NOTE,
        "why": "~50-52 deg launch produces the 45 deg entry; shorter guards carry "
               "a touch more arc (~52 for <=6'0\").",
    },
    "elbow_flare_deg": {
        "target": 0.0,
        "tolerance": 5.0,
        "measurable_now": False,
        "blocked_by": "2nd camera",
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
                **({"blocked_by": v["blocked_by"]} if "blocked_by" in v else {}),
                **({"needs": v["needs"]} if "needs" in v else {})}
            for k, v in TEXTBOOK.items()}
