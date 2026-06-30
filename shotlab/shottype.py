"""Auto shot-type tagging -- label each shot by FORM and SETUP.

Two axes, both heuristic and honestly confidence-labelled (one side-on camera,
no court homography yet):

  form  -- jumper | layup | floater | unknown
           A shot from mid/far range is almost certainly a jumper (medium conf).
           Only near-the-rim shots can be a layup/floater, and only with a flat /
           low arc -- those calls stay LOW confidence until calibration lands.

  setup -- catch_and_shoot | on_the_move | off_dribble | unknown
           Reuses the movement-into-the-shot signal (`movement_dir`) and adds
           dribble detection: a bounce in the ball's vertical path in the ~1.5 s
           before release means the shooter put it on the floor first.

Zone/distance is already tagged by `court.zone_for_release`; this adds the
qualitative type a box score would carry.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class ShotType:
    form: str = "unknown"           # jumper | layup | floater | unknown
    form_conf: str = "na"           # medium | low | na
    setup: str = "unknown"          # catch_and_shoot | on_the_move | off_dribble
    setup_conf: str = "na"
    note: str = ""

    def as_row(self) -> dict:
        return asdict(self)


def classify_form(depth, apex_height_ft, release_angle_deg) -> tuple[str, str]:
    """form, confidence from where the shot was taken + how lobbed the arc was."""
    if depth in ("mid", "far"):
        return "jumper", "medium"          # range rules out a layup/floater
    if depth == "near":
        flat = release_angle_deg is not None and release_angle_deg < 30
        low_apex = apex_height_ft is not None and apex_height_ft < 1.3
        if flat and low_apex:
            return "layup", "low"          # close, flat, barely any arc
        if apex_height_ft is not None and apex_height_ft < 2.2:
            return "floater", "low"        # close but lobbed up
        return "jumper", "low"             # close-range jumper (ambiguous)
    return "unknown", "na"


def detect_dribble(ball_track, rel_frame, fps, *, lookback_s=1.5,
                   min_samples=6) -> tuple[bool | None, str]:
    """Did the ball bounce off the floor in the window before release?

    A dribble shows up as the ball descending then rebounding -- a local bottom
    (max image-y) flanked by higher points -- with an amplitude well above
    keypoint noise. Returns (dribbled, confidence); dribbled is None when there
    aren't enough pre-shot ball samples to tell."""
    lo = int(rel_frame - lookback_s * fps)
    ys, rs = [], []
    for f in range(lo, int(rel_frame) + 1):
        bc = ball_track.get(f)
        if bc is not None:
            ys.append(float(bc.cy))
            rs.append(float(bc.r))
    if len(ys) < min_samples:
        return None, "low"                 # too sparse to judge
    ys = np.asarray(ys)
    rmed = float(np.median(rs)) or 1.0
    amp = ys.max() - ys.min()
    if amp < 2.0 * rmed:                    # essentially flat -> no bounce
        return False, "low"
    # count prominent local bottoms (image-y maxima) with real prominence
    bounces = 0
    for i in range(1, len(ys) - 1):
        if ys[i] >= ys[i - 1] and ys[i] >= ys[i + 1]:
            prominence = ys[i] - min(ys[max(0, i - 2)], ys[min(len(ys) - 1, i + 2)])
            if prominence > 1.5 * rmed:
                bounces += 1
    return (bounces >= 1), "low"


def classify_setup(movement_dir, dribbled) -> tuple[str, str]:
    if dribbled is True:
        return "off_dribble", "low"
    if movement_dir == "set":
        return "catch_and_shoot", "low"
    if movement_dir in ("left", "right"):
        return "on_the_move", "low"
    return "unknown", "na"


def classify_shot_type(*, depth, apex_height_ft, release_angle_deg,
                       movement_dir="unknown", ball_track=None, rel_frame=None,
                       fps=30.0) -> ShotType:
    """Combine form + setup into one tag. Pass ball_track + rel_frame to enable
    dribble detection (omit to fall back to the movement signal only)."""
    form, fconf = classify_form(depth, apex_height_ft, release_angle_deg)
    dribbled = None
    if ball_track is not None and rel_frame is not None:
        dribbled, _ = detect_dribble(ball_track, rel_frame, fps)
    setup, sconf = classify_setup(movement_dir, dribbled)
    note = "" if dribbled is not None else "dribble not assessed (sparse pre-shot ball track)"
    return ShotType(form=form, form_conf=fconf, setup=setup, setup_conf=sconf,
                    note=note)
