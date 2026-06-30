"""Make/miss classification from the ball trajectory near the rim.

This is the hardest signal on consumer footage -- the ball is small and often
lost against clutter right at the rim. We use a geometric heuristic and report a
confidence, never a false certainty:

  make  -- after reaching the rim the ball continues essentially straight DOWN
           through the hoop (stays near rim_x, passes below rim_y).
  miss  -- the ball deflects sideways or upward off the rim/backboard.

Confidence is LOW whenever the post-rim trajectory is sparse. Treat session
make% as indicative, not exact, until footage/rim resolution improves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MakeResult:
    made: bool | None
    confidence: str          # medium | low | na
    note: str = ""


def classify_make(shot, ball_track, calib, lookahead: int = 40) -> MakeResult:
    rim_x, rim_y = calib.rim_x, calib.rim_y
    last_f = int(shot.frames[-1])

    # collect ball positions from the rim approach through a short lookahead
    pts = []
    for f in range(int(shot.frames[0]), last_f + lookahead + 1):
        bc = ball_track.get(f)
        if bc is not None:
            pts.append((f, bc.cx, bc.cy))
    if len(pts) < 4:
        return MakeResult(None, "na", "ball lost near the rim")

    arr = np.array([(x, y) for _, x, y in pts], float)
    # index of closest approach to the rim
    d = np.linalg.norm(arr - calib.rim, axis=1)
    k = int(d.argmin())
    if d[k] > calib.shot_gate_px:
        return MakeResult(None, "na", "did not reach the rim")

    after = arr[k:]
    if len(after) < 3:
        return MakeResult(None, "low", "no trajectory after the rim")

    # MAKE signature: ball drops below the rim while staying laterally near rim_x
    below = after[:, 1] > rim_y + 0.5 * calib.rim_radius_px
    near_x = np.abs(after[:, 0] - rim_x) < 1.4 * calib.rim_radius_px
    drops_through = bool(np.any(below & near_x))

    # MISS signature: ball ends up well to the side of the rim, or bounces up
    ends_aside = abs(after[-1, 0] - rim_x) > 2.2 * calib.rim_radius_px
    bounces_up = after[-1, 1] < rim_y - calib.rim_radius_px

    if drops_through and not ends_aside:
        return MakeResult(True, "low", "ball dropped through rim line")
    if ends_aside or bounces_up:
        return MakeResult(False, "low", "ball deflected away from rim")
    return MakeResult(None, "low", "ambiguous post-rim path")
