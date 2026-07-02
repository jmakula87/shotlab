"""Phase 2 form metrics, computed per shot from the pose timeseries synced to
the ball track.

Each metric carries an explicit confidence that honors the camera angle, per the
2026 survey: in-plane angles on a square camera are trustworthy; depth-dependent
ones (elbow flare, squareness) are LOW and labeled as such.

Release frame is found by syncing pose to the ball: among frames around the shot
start, the one where the ball is closest to the shooting wrist is the release.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np

from .pose import FramePose, joint_angle, side_keys, L
from ..scale import release_height_ft, jump_height_ft, px_per_foot_from_body


@dataclass
class FormMetric:
    name: str
    value: float | None
    confidence: str           # high | medium | low | na
    note: str = ""


# Flip if 'left'/'right' come out reversed for a camera setup (one-line anchor).
LEFT_RIGHT_FLIP = False


@dataclass
class ReleaseEstimate:
    """Where the ball leaves the shooting hand, to sub-frame precision."""
    frame: int               # integer release frame (divergence onset)
    t: float                 # sub-frame release time, in frame units (>= frame)
    confidence: str          # high | medium | low
    diverging: bool          # ball separation grows after release (true hand-off)
    note: str = ""


@dataclass
class ShotForm:
    shot: int
    release_frame: int
    movement_dir: str = "unknown"   # left | right | set | unknown
    release_t: float = 0.0          # sub-frame release time (frame units)
    release_conf: str = "na"        # confidence in the ball/pose release sync
    metrics: list = field(default_factory=list)

    def as_row(self) -> dict:
        row = {"shot": self.shot, "release_frame": self.release_frame,
               "release_t": round(self.release_t, 2),
               "release_conf": self.release_conf,
               "movement_dir": self.movement_dir}
        for m in self.metrics:
            row[m.name] = m.value
            row[m.name + "_conf"] = m.confidence
        return row


def movement_direction(rel_f, poses, keys, rim_xy, fps,
                       window_s=0.30, set_thresh=0.12) -> str:
    """Which way the shooter was moving INTO the shot, relative to facing the rim.

    Uses the hip-center trajectory in the ~0.3s before release. Small lateral
    travel -> 'set' (catch-and-shoot). Otherwise the sign of (facing × velocity)
    in the image gives a consistent left/right per camera setup (flip with
    LEFT_RIGHT_FLIP if the labels read backwards on your footage)."""
    if rim_xy is None or not poses:
        return "unknown"
    lo = rel_f - int(window_s * fps)
    hips = []
    for f in range(lo, rel_f + 1):
        fp = poses.get(f)
        if fp is None:
            continue
        if fp.vis[L["l_hip"]] >= 0.3 and fp.vis[L["r_hip"]] >= 0.3:
            hips.append((fp.xy[L["l_hip"]] + fp.xy[L["r_hip"]]) / 2.0)
    if len(hips) < 3:
        return "unknown"
    v = hips[-1] - hips[0]                       # image-space movement
    fp_rel = poses.get(rel_f)
    bh = _body_height_px(fp_rel, keys) if fp_rel else float("nan")
    if not (bh == bh) or bh < 1:
        bh = 1.0
    if (v[0] ** 2 + v[1] ** 2) ** 0.5 / bh < set_thresh:
        return "set"
    shooter = hips[-1]
    f_vec = (rim_xy[0] - shooter[0], rim_xy[1] - shooter[1])   # facing the rim
    cross = f_vec[0] * v[1] - f_vec[1] * v[0]
    right = (cross > 0) != LEFT_RIGHT_FLIP
    return "right" if right else "left"


def _vis_ok(fp: FramePose, names, thr=0.4) -> bool:
    return all(fp.v(n) >= thr for n in names)


def detect_handedness(poses, frames, default="right", thr=0.4) -> str:
    """Which hand shoots, from pose alone: the shooting wrist rises HIGHEST
    through the shot (releases and follows through above the guide hand), so the
    wrist with the smaller minimum image-y over the shot window is the shooter's.
    Falls back to `default` when neither wrist is reliably visible."""
    lo, hi = int(frames[0]), int(frames[-1])
    l_min = r_min = float("inf")
    for f in range(lo, hi + 1):
        fp = poses.get(f)
        if fp is None:
            continue
        if fp.v("l_wrist") >= thr:
            l_min = min(l_min, float(fp.pt("l_wrist")[1]))
        if fp.v("r_wrist") >= thr:
            r_min = min(r_min, float(fp.pt("r_wrist")[1]))
    if l_min == float("inf") and r_min == float("inf"):
        return default
    return "left" if l_min < r_min else "right"


def find_release(shot, ball_track, poses, handedness, fps=30.0) -> ReleaseEstimate:
    """Where the ball leaves the shooting hand, to sub-frame precision.

    Two signals, combined so neither footage type breaks:
      - BALL divergence (`_ball_release`): the ball sits at the wrist until
        release, then separates -- the onset of that divergence is a sharp,
        accurate release. Great when the ball is tracked THROUGH the release.
      - POSE wrist-apex (`_wrist_apex`): the shooting wrist peaks at the snap.
        This needs no ball, so it survives when the detector picks the ball up
        late.

    On far/small-ball footage the detector doesn't lock the ball until it's
    ~0.4s into flight, so the ball-divergence "release" lands well after the true
    snap (on an arm-already-down frame). We detect that case -- the wrist apex
    sits clearly BEFORE the ball release -- and use the apex instead. When the
    ball is tracked cleanly the two coincide (apex not earlier), so we keep the
    sharper ball estimate; the synthetic hand-off test still lands on it."""
    keys = side_keys(handedness)
    ball_est = _ball_release(shot, ball_track, poses, keys)
    apex = _wrist_apex(shot, poses, keys, fps)
    if apex is not None:
        af, at = apex
        # ball release lagging the wrist snap by >~0.12s == detected-late ball
        if ball_est.frame - af > 0.12 * max(fps, 1.0):
            return ReleaseEstimate(frame=af, t=at, confidence="medium",
                                   diverging=True,
                                   note="pose wrist-apex (ball detected late in flight)")
    return ball_est


def _wrist_apex(shot, poses, keys, fps):
    """The frame + sub-frame time of peak shooting-wrist extension (the snap),
    from pose alone. Searches a pre-window before flight start (the true release
    precedes the ball's first tracked flight frame). Returns (frame, t) or None."""
    start = int(shot.frames[0])
    pre = int(round(0.6 * max(fps, 1.0)))
    post = max(4, int(round(0.15 * max(fps, 1.0))))
    lo = max(min(poses) if poses else start, start - pre)
    best_f, best_y = None, float("inf")
    for f in range(lo, start + post + 1):
        fp = poses.get(f)
        if fp is None:
            continue
        if fp.v(keys["wrist"]) >= 0.4 and fp.v(keys["shoulder"]) >= 0.4:
            y = float(fp.pt(keys["wrist"])[1])       # image y: smaller = higher
            if y < best_y:
                best_y, best_f = y, f
    if best_f is None:
        return None
    # only trust it as a release if the wrist is actually up (above the shoulder)
    fp = poses[best_f]
    if float(fp.pt(keys["wrist"])[1]) >= float(fp.pt(keys["shoulder"])[1]):
        return None
    # sub-frame apex via a parabolic vertex fit on wrist-y at the peak +/- 1
    y0 = best_y
    ym = float(poses[best_f - 1].pt(keys["wrist"])[1]) if poses.get(best_f - 1) else None
    yp = float(poses[best_f + 1].pt(keys["wrist"])[1]) if poses.get(best_f + 1) else None
    t = float(best_f)
    if ym is not None and yp is not None:
        denom = ym - 2 * y0 + yp
        if abs(denom) > 1e-6:
            t = best_f + max(-1.0, min(1.0, 0.5 * (ym - yp) / denom))
    return best_f, t


def _ball_release(shot, ball_track, poses, keys) -> ReleaseEstimate:
    """Ball-divergence release: the END of the in-hand minimum-distance cluster
    (onset of divergence), with sub-frame interpolation across the half-radius
    separation crossing. Low-confidence min-distance fallback when the hand-off
    is never clean."""
    start = int(shot.frames[0])
    lo = max(min(poses) if poses else start, start - 12)
    window = range(lo, start + 8)

    samples = []  # (frame, dist, ball_radius, wrist_visibility)
    for f in window:
        fp = poses.get(f)
        bc = ball_track.get(f)
        if fp is None or bc is None:
            continue
        wrist = fp.pt(keys["wrist"])
        d = float(np.hypot(bc.cx - wrist[0], bc.cy - wrist[1]))
        samples.append((f, d, float(bc.r), fp.v(keys["wrist"])))

    if len(samples) < 2:
        return ReleaseEstimate(frame=start, t=float(start), confidence="low",
                               diverging=False,
                               note="ball/pose not tracked at the hand-off")

    dists = [s[1] for s in samples]
    rmed = float(np.median([s[2] for s in samples])) or 1.0
    imin = int(np.argmin(dists))
    dmin = dists[imin]
    tol = 0.6 * rmed

    # Extend the held cluster forward from the minimum while still ~in hand; its
    # last frame is the divergence onset.
    j = imin
    while j + 1 < len(samples) and dists[j + 1] <= dmin + tol:
        j += 1
    rel_f = samples[j][0]

    # Divergence check: separation rises over the next couple of samples.
    rises = 0
    for k in range(j, min(j + 3, len(samples)) - 1):
        if dists[k + 1] > dists[k] + 1e-6:
            rises += 1
    diverging = rises >= 1

    # Sub-frame: interpolate where separation crosses half a ball radius.
    rel_t = float(rel_f)
    if j + 1 < len(samples):
        f0, d0 = samples[j][0], dists[j]
        f1, d1 = samples[j + 1][0], dists[j + 1]
        sep_thr = dmin + 0.5 * rmed
        if d1 > d0 and d0 <= sep_thr <= d1 and f1 > f0:
            frac = (sep_thr - d0) / (d1 - d0)
            rel_t = f0 + frac * (f1 - f0)

    wrist_vis_ok = samples[j][3] >= 0.4
    if diverging and wrist_vis_ok and rises >= 2:
        conf = "high"
    elif diverging and wrist_vis_ok:
        conf = "medium"
    else:
        conf = "low"

    if not diverging:
        rel_f = samples[imin][0]                     # min-distance fallback
        rel_t = float(rel_f)
        return ReleaseEstimate(frame=rel_f, t=rel_t, confidence="low",
                               diverging=False,
                               note="no clean separation; min-distance fallback")
    return ReleaseEstimate(frame=rel_f, t=rel_t, confidence=conf, diverging=True)


def find_release_frame(shot, ball_track, poses, handedness, fps=30.0) -> int:
    """Back-compat shim: the integer release frame. See find_release for the full
    sub-frame estimate and confidence."""
    return find_release(shot, ball_track, poses, handedness, fps=fps).frame


def _body_height_px(fp: FramePose, keys) -> float:
    sh = fp.pt(keys["shoulder"])
    ank = fp.pt(keys["ankle"])
    h = abs(ank[1] - sh[1])
    return h if h > 1 else float("nan")


def _shooter_ppf(poses, frames, rel_f, shooter_height_ft):
    """Pixels-per-foot at the SHOOTER's depth from their known height.

    Measured over the GATHER-through-release window only ([load .. just after
    release]), which is depth-stable -- the shooter is planted at their spot.
    Restricting to this window keeps the ruler honest if the ball flight is long
    and the shooter starts drifting in to rebound while the ball is still up
    (later frames would be at a different, nearer depth). Within the window, p90
    of the nose->(lower ankle) span picks the most-extended frame (release/rise,
    not the crouched load) = the true body length. Needs nose + an ankle visible
    on >=3 frames, else None (rim fallback).

    NB: the per-shot correction vs the rim ruler is genuinely large and variable
    (measured 0.7x-4.5x on real footage) -- on a wide clip the shooter stands
    several times closer to the camera than the far rim, so rim-scaled heights
    are that many times too big; on a moved-in camera where rim and shooter sit
    at similar depths the two rulers roughly agree. That spread is correct, not
    noise."""
    if not shooter_height_ft:
        return None
    lo = min(frames[0] - 20, rel_f - 20)
    hi = rel_f + 8                                    # through release, before rebound
    spans = []
    for f in range(int(lo), int(hi) + 1):
        fp = poses.get(f)
        if fp is None or fp.v("nose") < 0.5:
            continue
        anks = [float(fp.pt(n)[1]) for n in ("l_ankle", "r_ankle") if fp.v(n) >= 0.5]
        if not anks:
            continue
        d = max(anks) - float(fp.pt("nose")[1])       # lower ankle below the nose
        if d > 1:
            spans.append(d)
    if len(spans) < 3:
        return None
    return px_per_foot_from_body(float(np.percentile(spans, 90)), shooter_height_ft)


def _apex_frame(poses, frames) -> int | None:
    """Frame of highest body position (hip midpoint y minimal)."""
    best_f, best_y = None, float("inf")
    for f in frames:
        fp = poses.get(f)
        if fp is None:
            continue
        hip = (fp.xy[L["l_hip"]] + fp.xy[L["r_hip"]]) / 2
        if hip[1] < best_y:
            best_y, best_f = hip[1], f
    return best_f


def _hip_y(poses, f) -> float | None:
    fp = poses.get(f)
    if fp is None:
        return None
    return float(((fp.xy[L["l_hip"]] + fp.xy[L["r_hip"]]) / 2)[1])


def _apex_subframe(poses, frames) -> float | None:
    """Sub-frame apex via a parabolic vertex fit on hip-y at the peak and its two
    neighbours -- so release-vs-apex timing isn't quantized to whole frames."""
    af = _apex_frame(poses, frames)
    if af is None:
        return None
    y0, ym, yp = _hip_y(poses, af), _hip_y(poses, af - 1), _hip_y(poses, af + 1)
    if y0 is None or ym is None or yp is None:
        return float(af)
    denom = ym - 2 * y0 + yp
    if abs(denom) < 1e-6:
        return float(af)
    off = 0.5 * (ym - yp) / denom
    return af + max(-1.0, min(1.0, off))


def _elbow_angle_at(poses, f, keys) -> float | None:
    fp = poses.get(f)
    if fp is None or not _vis_ok(fp, [keys["shoulder"], keys["elbow"], keys["wrist"]]):
        return None
    return joint_angle(fp.pt(keys["shoulder"]), fp.pt(keys["elbow"]),
                       fp.pt(keys["wrist"]))


def _elbow_angle_at_t(poses, t, keys) -> float | None:
    """Elbow angle at a sub-frame release time, interpolated between the two
    bracketing frames when both are visible (else the nearest visible one)."""
    f0 = int(np.floor(t))
    frac = t - f0
    a0, a1 = _elbow_angle_at(poses, f0, keys), _elbow_angle_at(poses, f0 + 1, keys)
    if a0 is None:
        return a1
    if a1 is None or frac <= 0:
        return a0
    return a0 + frac * (a1 - a0)


def compute_form(shot, ball_track, poses, fps, *, handedness="right",
                 camera_angle="side_on", rim_xy=None, px_per_foot=None,
                 shooter_height_ft=None) -> ShotForm:
    if handedness == "auto":
        handedness = detect_handedness(poses, shot.frames)
    keys = side_keys(handedness)
    side_on = camera_angle == "side_on"
    rel = find_release(shot, ball_track, poses, handedness, fps=fps)
    rel_f, rel_t = rel.frame, rel.t
    move = movement_direction(rel_f, poses, keys, rim_xy, fps)
    frames = [int(f) for f in shot.frames]
    span = range(max(min(poses) if poses else frames[0], frames[0] - 20),
                 frames[-1] + 1)
    # depth-correct ruler from the shooter's known height (falls back to rim).
    # Measured over the planted gather->release window, NOT the full flight
    # (the shooter drifts toward the camera to rebound during flight).
    body_ppf = _shooter_ppf(poses, frames, rel_f, shooter_height_ft)

    metrics: list[FormMetric] = []

    # ---- 1. elbow angle at release (sub-frame interpolated) -------------
    fp = poses.get(rel_f)
    if fp and _vis_ok(fp, [keys["shoulder"], keys["elbow"], keys["wrist"]]):
        ang = _elbow_angle_at_t(poses, rel_t, keys)
        if ang is None:                       # bracket frame invisible: use rel_f
            ang = joint_angle(fp.pt(keys["shoulder"]), fp.pt(keys["elbow"]),
                              fp.pt(keys["wrist"]))
        conf = "medium" if side_on else "low"
        note = "" if side_on else "front-on: in-plane elbow angle ok, but flare is out-of-plane"
        if abs(rel_t - rel_f) > 1e-3:
            note = (note + "; " if note else "") + "sub-frame interpolated to release"
        metrics.append(FormMetric("elbow_angle_at_release_deg", round(ang, 1), conf, note))
    else:
        metrics.append(FormMetric("elbow_angle_at_release_deg", None, "na",
                                  "shooting arm not clearly visible at release"))

    # ---- 2. knee bend depth (min knee angle in the load) ----------------
    knee_angs, load_f, load_min = [], None, float("inf")
    for f in span:
        if f > rel_f:
            break
        fp = poses.get(f)
        if fp and _vis_ok(fp, [keys["hip"], keys["knee"], keys["ankle"]]):
            k = joint_angle(fp.pt(keys["hip"]), fp.pt(keys["knee"]),
                            fp.pt(keys["ankle"]))
            knee_angs.append(k)
            if k < load_min:                 # deepest bend = bottom of the load
                load_min, load_f = k, f
    if knee_angs:
        conf = "high" if side_on else "low"
        metrics.append(FormMetric("knee_bend_deg", round(min(knee_angs), 1), conf,
                                  "" if side_on else "knee flexion is sagittal; needs side-on"))
    else:
        metrics.append(FormMetric("knee_bend_deg", None, "na", "legs not visible"))

    # ---- 2b. shot tempo: dip bottom -> release (quickness) --------------
    if load_f is not None and rel_t >= load_f:
        tempo = (rel_t - load_f) / fps
        metrics.append(FormMetric("tempo_dip_to_release_s", round(tempo, 3), "high",
                                  "time from your deepest load to release (lower = quicker)"))
    else:
        metrics.append(FormMetric("tempo_dip_to_release_s", None, "na",
                                  "load/release not both tracked"))

    # ---- 3. release vs jump apex (seconds, both sub-frame) --------------
    apex_t = _apex_subframe(poses, span)
    if apex_t is not None:
        dt = (rel_t - apex_t) / fps          # + = release after apex (late)
        metrics.append(FormMetric("release_vs_apex_s", round(dt, 3), "high",
                                  "+ = released after the peak (late)"))
    else:
        metrics.append(FormMetric("release_vs_apex_s", None, "na", "body not tracked"))

    # ---- 4. follow-through hold (s the wrist stays snapped) -------------
    hold = 0
    fp0 = poses.get(rel_f)
    if fp0 and _vis_ok(fp0, [keys["shoulder"], keys["wrist"], keys["elbow"]]):
        for f in range(rel_f, frames[-1] + 1):
            fp = poses.get(f)
            if fp is None or not _vis_ok(fp, [keys["shoulder"], keys["wrist"]]):
                break
            wrist = fp.pt(keys["wrist"])
            shoulder = fp.pt(keys["shoulder"])
            # snapped == wrist still at/above shoulder height (image y smaller)
            if wrist[1] <= shoulder[1] + 0.15 * _body_height_px(fp, keys):
                hold += 1
            else:
                break
        metrics.append(FormMetric("follow_through_hold_s", round(hold / fps, 3),
                                  "medium",
                                  "wrist-height proxy; finger snap needs a hand model"))
    else:
        metrics.append(FormMetric("follow_through_hold_s", None, "na",
                                  "shooting hand not visible at release"))

    # ---- 5. balance drift (CoM horizontal travel / body height) --------
    hips, heights = [], []
    for f in span:
        fp = poses.get(f)
        if fp and _vis_ok(fp, [keys["shoulder"], keys["ankle"]]):
            hips.append(((fp.xy[L["l_hip"]] + fp.xy[L["r_hip"]]) / 2)[0])
            heights.append(_body_height_px(fp, keys))
    if len(hips) >= 3 and np.nanmedian(heights) > 1:
        drift = (np.nanmax(hips) - np.nanmin(hips)) / np.nanmedian(heights)
        metrics.append(FormMetric("balance_drift_px_per_ht", round(float(drift), 3),
                                  "medium", "lateral lean/drift through the shot"))
    else:
        metrics.append(FormMetric("balance_drift_px_per_ht", None, "na", "body not tracked"))

    # ---- 6. squareness (front-on only, LOW) ----------------------------
    if not side_on:
        fp = poses.get(rel_f) or poses.get(frames[0])
        if fp and _vis_ok(fp, ["l_shoulder", "r_shoulder"]):
            ls, rs = fp.pt("l_shoulder"), fp.pt("r_shoulder")
            tilt = np.degrees(np.arctan2(rs[1] - ls[1], rs[0] - ls[0]))
            # normalize to deviation from horizontal shoulder line
            dev = ((tilt + 180) % 180)
            dev = dev - 180 if dev > 90 else dev
            metrics.append(FormMetric("squareness_deg", round(float(dev), 1), "low",
                                      "out-of-plane; 1-camera estimate only"))
        else:
            metrics.append(FormMetric("squareness_deg", None, "na", "shoulders not visible"))
    else:
        metrics.append(FormMetric("squareness_deg", None, "na",
                                  "needs a front-on session"))

    # ---- 7. real-feet heights (rim-scaled; shooter is off the rim's depth
    #         plane, so these are LOW confidence) -----------------------------
    _height_metrics(metrics, poses, ball_track, keys, span, rel_f, px_per_foot,
                    body_ppf=body_ppf)

    return ShotForm(shot=shot.index, release_frame=rel_f, movement_dir=move,
                    release_t=rel_t, release_conf=rel.confidence, metrics=metrics)


def _height_metrics(metrics, poses, ball_track, keys, span, rel_f, rim_ppf,
                    body_ppf=None):
    """Release height (ball above the ankle line) + jump height (body's vertical
    travel), both in feet.

    Prefers the SHOOTER-height ruler (`body_ppf`) when available: it's measured
    at the shooter's own depth, so these vertical distances come out honest
    (MEDIUM confidence). Without it, falls back to the rim ruler -- exact only
    at the rim's depth, so ~2.5x too large for the nearer shooter (LOW)."""
    ppf = body_ppf or rim_ppf
    if body_ppf:
        note, conf = "body-scaled from your height", "medium"
    else:
        note, conf = "rim-scaled; shooter off the rim's depth plane", "low"
    fp = poses.get(rel_f)
    rh = None
    if ppf and fp and _vis_ok(fp, [keys["ankle"]]):
        bc = ball_track.get(rel_f)
        ball_y = bc.cy if bc is not None else (
            fp.pt(keys["wrist"])[1] if _vis_ok(fp, [keys["wrist"]]) else None)
        rh = release_height_ft(ball_y, fp.pt(keys["ankle"])[1], ppf)
    metrics.append(FormMetric("release_height_ft",
                              None if rh is None else round(rh, 2),
                              conf if rh is not None else "na", note))

    jh = _jump_height(poses, span, ppf)
    metrics.append(FormMetric("jump_height_ft",
                              None if jh is None else round(jh, 2),
                              conf if jh is not None else "na",
                              note + "; ankle-based (squat excluded)"))


def _lower_ankle_y(poses, f) -> float | None:
    """Per-frame image-y of the LOWER (larger-y) ankle. This only rises when
    BOTH feet are airborne: a step lifts one foot, a squat lifts neither, so the
    series isolates true flight. Needs BOTH ankles visible -- with one occluded
    the lower-of-two logic degenerates to whichever leg the model kept, which on
    far/small footage is exactly the unreliable one."""
    fp = poses.get(f)
    if fp is None or fp.v("l_ankle") < 0.4 or fp.v("r_ankle") < 0.4:
        return None
    return max(float(fp.pt("l_ankle")[1]), float(fp.pt("r_ankle")[1]))


def _jump_height(poses, span, ppf):
    """Jump height in feet from ankle flight, not hip travel. The old hip-based
    max-minus-min counted the load SQUAT as jump (hips drop ~a foot in the
    load); ankles sit on the ground line through the squat and rise by exactly
    the jump, so: ground = high percentile of the lower-ankle series (grounded
    frames dominate the span), peak = its minimum during flight. The peak is
    taken on a median-3 smoothed series so a single-frame pose glitch can't
    fake a flight."""
    ys = [y for y in (_lower_ankle_y(poses, f) for f in span) if y is not None]
    if not ppf or len(ys) < 7:
        return None
    sm = [float(np.median(ys[max(0, i - 1):i + 2])) for i in range(len(ys))]
    ground = float(np.percentile(sm, 80))     # grounded ground-line, squat-proof
    peak = float(min(sm))
    jh = jump_height_ft(ground, peak, ppf)
    # physics gate: past a world-class vertical it's a tracking failure, not a
    # jump -- report nothing rather than a garbage number
    if jh is not None and jh > 4.0:
        return None
    return jh
