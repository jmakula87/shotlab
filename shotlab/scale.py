"""Real-world scale from the rim.

A regulation rim is ~18 in across (1.5 ft), and we already detect it, so its
pixel width is a stable, measured-once ruler -- better than the per-shot ball
diameter (which jitters with detection noise and changes size as the ball moves
in depth). This converts pixel heights to feet: apex-above-rim, release height,
jump height.

Honesty: the rim's scale is exact at the RIM's depth plane. The ball at its apex
sits ~over the rim, so apex-above-rim is the most trustworthy (MEDIUM). Release
and jump happen at the SHOOTER's depth (nearer/farther than the rim), so they
carry a perspective error and are lower-confidence -- labeled accordingly.
"""

from __future__ import annotations

RIM_WIDTH_FT = 1.5          # regulation rim inner diameter ~18 in

# Anthropometry: nose sits ~92% up the stature, the ankle joint ~4.5%, so the
# nose->ankle vertical span is ~87.5% of standing height. That span is what a
# side-on pose measures cleanly (both landmarks are reliable and roughly in the
# frontal plane), so it's our body ruler.
NOSE_TO_ANKLE_FRAC = 0.875


def px_per_foot_from_rim(rim_radius_px, rim_width_ft: float = RIM_WIDTH_FT):
    """Pixels-per-foot from the rim's visible half-width (rim_radius_px). Returns
    None if the rim size is unusable."""
    if rim_radius_px is None or rim_radius_px <= 0:
        return None
    return (2.0 * rim_radius_px) / rim_width_ft


def parse_height(text):
    """Parse a height into feet: 5'10\" / 5'10 / 5ft10 / 6' / 5.83 / 70in / 70\".
    Returns a float in feet, or None for empty/unparseable input."""
    if text is None:
        return None
    s = str(text).strip().lower()
    if not s:
        return None
    import re
    # feet [+ optional inches], with an optional trailing inch mark
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(?:'|ft|feet)\s*(\d+(?:\.\d+)?)?"
                 r"\s*(?:\"|in|inch|inches)?$", s)
    if m:
        inches = float(m.group(2)) if m.group(2) else 0.0
        return float(m.group(1)) + inches / 12.0
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(?:\"|in|inch|inches)$", s)
    if m:                                   # inches only
        return float(m.group(1)) / 12.0
    try:                                    # bare number = decimal feet
        return float(s)
    except ValueError:
        return None


def px_per_foot_from_body(nose_to_ankle_px, shooter_height_ft: float):
    """Pixels-per-foot at the SHOOTER's depth, from their known height.

    The rim ruler is exact only at the rim's depth; the shooter stands closer to
    the camera (measured ~2.5x closer on this court), so rim-scaled release/jump
    heights run that many times too large. The shooter's own body IS the ruler
    at the shooter's depth: nose->ankle pixels / (height * NOSE_TO_ANKLE_FRAC).
    Returns None if inputs are unusable."""
    if not nose_to_ankle_px or nose_to_ankle_px <= 0 or not shooter_height_ft:
        return None
    return nose_to_ankle_px / (shooter_height_ft * NOSE_TO_ANKLE_FRAC)


def height_ft(dy_px, ppf):
    """Convert a pixel height (already a positive-up delta) to feet."""
    if ppf is None or ppf <= 0 or dy_px is None:
        return None
    return dy_px / ppf


def apex_above_rim_ft(ball_apex_y_px, rim_y_px, ppf):
    """How far the ball's arc peak clears the rim, in feet. Image y grows DOWN,
    so a peak above the rim has apex_y < rim_y -> positive feet. Can be negative
    (a flat shot that never gets above the rim line)."""
    if ball_apex_y_px is None or rim_y_px is None:
        return None
    return height_ft(rim_y_px - ball_apex_y_px, ppf)


def release_height_ft(ball_release_y_px, ground_y_px, ppf):
    """Height of the ball at release above the ground (ankle line), in feet."""
    if ball_release_y_px is None or ground_y_px is None:
        return None
    return height_ft(ground_y_px - ball_release_y_px, ppf)


def jump_height_ft(low_body_y_px, high_body_y_px, ppf):
    """Vertical body travel (lowest body point in the load -> highest at the
    peak), in feet. Both are image-y (down); low point has the LARGER y."""
    if low_body_y_px is None or high_body_y_px is None:
        return None
    return height_ft(low_body_y_px - high_body_y_px, ppf)
