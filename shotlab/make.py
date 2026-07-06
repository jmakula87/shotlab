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
    rim_frame: int | None = None   # frame of closest rim approach (for audio timing)


def classify_make(shot, ball_track, calib, lookahead: int = 40,
                  fps: float = 30.0) -> MakeResult:
    rim_x, rim_y = calib.rim_x, calib.rim_y
    rr = float(calib.rim_radius_px)
    last_f = int(shot.frames[-1])

    # collect ball positions from the rim approach through a short lookahead
    pts = []
    for f in range(int(shot.frames[0]), last_f + lookahead + 1):
        bc = ball_track.get(f)
        if bc is not None:
            pts.append((f, bc.cx, bc.cy))
    if len(pts) < 4:
        return MakeResult(None, "na", "ball lost near the rim")

    frames = np.array([f for f, _, _ in pts])
    arr = np.array([(x, y) for _, x, y in pts], float)
    d = np.linalg.norm(arr - calib.rim, axis=1)
    k = int(d.argmin())                         # closest approach to the rim
    rim_frame = int(frames[k])
    if d[k] > calib.shot_gate_px:
        return MakeResult(None, "na", "did not reach the rim", rim_frame=rim_frame)

    # Judge only a SHORT window (~0.5s) after closest approach, and STOP at a big
    # tracker jump -- the old code trusted the far-away LAST tracked point (often a
    # rebound / next possession ~1.3s later), so the verdict flipped with rim size
    # and post-net noise (audit D9). The window makes the make/miss call about the
    # ball's pass through the rim, not where it wandered afterward.
    win_end = min(len(arr), k + 1 + int(round(0.5 * max(fps, 1.0))))
    seg = [arr[k]]
    for i in range(k + 1, win_end):
        if np.linalg.norm(arr[i] - arr[i - 1]) > 4.0 * rr:   # tracker jumped -> stop
            break
        seg.append(arr[i])
    seg = np.array(seg)
    if len(seg) < 2:
        return MakeResult(None, "low", "no clean trajectory after the rim",
                          rim_frame=rim_frame)

    near_x = np.abs(seg[:, 0] - rim_x) < 1.4 * rr
    below = seg[:, 1] > rim_y + 0.5 * rr
    started_at_rim = seg[0, 1] <= rim_y + 0.5 * rr
    # MAKE: the ball passes DOWN through the rim cylinder (from at/above the rim to
    # below it, staying laterally within the rim)
    drops_through = bool(started_at_rim and np.any(below & near_x))
    max_lateral = float(np.max(np.abs(seg[:, 0] - rim_x)))   # within the window only
    ends_up = bool(seg[-1, 1] < rim_y - rr and not np.any(below))

    if drops_through and max_lateral < 2.2 * rr:
        return MakeResult(True, "low", "ball passed down through the rim",
                          rim_frame=rim_frame)
    if max_lateral > 2.2 * rr or ends_up:
        return MakeResult(False, "low", "ball deflected away from the rim",
                          rim_frame=rim_frame)
    return MakeResult(None, "low", "ambiguous post-rim path", rim_frame=rim_frame)
