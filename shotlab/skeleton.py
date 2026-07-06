"""Ideal per-phase skeletons for the phone app's overlay (profile v2).

The app can ghost a "gold ideal skeleton" over your live shot at the release
frame (and, later, load/follow) so you see YOUR best form on top of the current
one. Profile v1 shipped only numeric targets; this builds the skeletons by
re-running pose on your good shots and averaging their joints.

How it stays honest:
  - Coordinates are stored the SAME way the app's runtime landmarks are:
    normalized [0,1] image coords (x = px/W, y = px/H). The app re-warps the
    ideal onto the live skeleton by shoulder-center + shoulder->hip length
    (overlay.js:drawIdealAligned), so we canonicalize the exact same way here:
    center on the shoulder midpoint, scale by the shoulder->hip length. Shots
    taken at different spots/sizes then average into one clean shape.
  - A pose-QUALITY gate (visibility >= MIN_VIS on the joints a phase needs) means
    a foreshortened wide-angle shot where the shooter is small doesn't muddy the
    average -- clean side-on shots dominate. We report how many shots contributed.

Only the raw clip + the cached ball track (data/out/<clip>/<clip>_track.json,
which already holds the shots + their frames) are needed -- no re-detection.
"""

from __future__ import annotations

import os

import numpy as np

from .detect_cache import _load as _load_track
from .video_io import probe, iter_frames
from .phase2_pose.pose import PoseExtractor, side_keys, joint_angle, L

# Joints the overlay draws (BlazePose-33 subset) -- we output all 33 for index
# alignment but these are the ones that matter.
_DRAWN = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
MIN_VIS = 0.5            # a joint must be at least this visible to contribute
FOLLOW_S = 0.20         # seconds after release for the follow-through pose
_PRE, _POST = 30, 12    # frame window padding around a shot for pose


def _canonical(fp, w, h):
    """A pose -> (33,2) ASPECT-TRUE coords centered on the shoulder midpoint and
    scaled by the shoulder->hip length. Returns (canon_xy, vis) or None.

    Works in PIXEL space -- it must NOT normalize x by width and y by height
    separately, which stored an aspect-distorted shape and made the phone's gold
    overlay warp on a non-square/portrait canvas (elbow read 169 vs 157 deg;
    audit D7). Unit is shoulder->hip lengths, so it's aspect-free and the app can
    align it isotropically. (w,h kept for signature compatibility.)"""
    xy = fp.xy.astype(float).copy()      # pixels; isotropic
    s_mid = (xy[11] + xy[12]) / 2.0
    h_mid = (xy[23] + xy[24]) / 2.0
    scale = float(np.hypot(*(s_mid - h_mid)))
    if scale < 1e-6:
        return None
    return (xy - s_mid) / scale, fp.vis


def _knee_angle(fp, keys):
    return joint_angle(fp.pt(keys["hip"]), fp.pt(keys["knee"]), fp.pt(keys["ankle"]))


def _load_frame(poses, lo, rel_f, keys):
    """Deepest-crouch frame (min shooting-side knee angle) in [lo, rel_f]."""
    best_f, best = None, float("inf")
    for f in range(lo, rel_f + 1):
        fp = poses.get(f)
        if fp is None:
            continue
        if all(fp.v(n) >= MIN_VIS for n in (keys["hip"], keys["knee"], keys["ankle"])):
            k = _knee_angle(fp, keys)
            if k == k and k < best:
                best, best_f = k, f
    return best_f


def _follow_frame(poses, rel_f, hi, fps):
    """A frame ~FOLLOW_S after release with a pose (nearest within +/-3)."""
    target = rel_f + int(round(FOLLOW_S * fps))
    for d in [0, 1, -1, 2, -2, 3, -3]:
        f = target + d
        if lo_ok(f, rel_f, hi) and poses.get(f) is not None:
            return f
    return None


def lo_ok(f, rel_f, hi):
    return rel_f <= f <= hi


def _phase_frames(shot, track, poses, fps, handedness):
    """Locate the load / release / follow pose frames.

    Release uses the SAME `find_release` the metrics do (ball-divergence onset
    when the ball is tracked through the hand-off, wrist-apex fallback only when
    the ball is detected late) -- so the film-room release IMAGE and the elbow
    number printed beside it are from one and the same frame, and the image is
    the true ball-departure instant rather than the follow-through lockout.
    """
    from .phase2_pose.form import find_release, detect_handedness
    hand = handedness
    if hand == "auto":
        hand = detect_handedness(poses, [int(f) for f in shot.frames])
    keys = side_keys(hand)

    lo = int(shot.frames[0]) - _PRE
    hi = int(shot.frames[-1]) + _POST
    rel_f = find_release(shot, track or {}, poses, hand, fps=fps).frame
    if poses.get(rel_f) is None:
        return {"load": None, "release": None, "follow": None}, keys
    return {
        "load": _load_frame(poses, lo, rel_f, keys),
        "release": rel_f,
        "follow": _follow_frame(poses, rel_f, hi, fps),
    }, keys


# joints each phase's canonical frame must have visible to count
_PHASE_REQ = {
    "load": ("hip", "knee", "ankle"),
    "release": ("shoulder", "elbow", "wrist"),
    "follow": ("shoulder", "elbow", "wrist"),
}


def _shot_skeletons(video_path, shot, track, fps, w, h, handedness):
    """Decode just this shot's window, run pose, and return
    {phase: (canon_xy (33,2), vis (33,))} for the phases we can trust."""
    lo = max(0, int(shot.frames[0]) - _PRE)
    hi = int(shot.frames[-1]) + _POST
    ext = PoseExtractor(fps=fps, variant="full", smooth=True)
    poses = {}
    try:
        for idx, frame in iter_frames(video_path, start=lo, stop=hi + 1):
            fp = ext.process_frame(idx, frame)
            if fp is not None:
                poses[idx] = fp
    finally:
        ext.close()
    if not poses:
        return {}
    frames, keys = _phase_frames(shot, track, poses, fps, handedness)
    out = {}
    for phase, f in frames.items():
        if f is None:
            continue
        fp = poses.get(f)
        need = [keys[j] for j in _PHASE_REQ[phase]] + ["l_shoulder", "r_shoulder", "l_hip", "r_hip"]
        if not all(fp.v(n) >= MIN_VIS for n in need):
            continue
        if not _phase_valid(phase, fp, keys):    # drop degenerate detections
            continue
        canon = _canonical(fp, w, h)
        if canon is not None:
            out[phase] = canon
    return out


def _phase_valid(phase, fp, keys):
    """Sanity-gate a phase pose so a bad shot doesn't pollute the average.
    Release: the shooting wrist must actually be above the shoulder (arm up).
    Follow: wrist at least above the hip (still snapped, not hanging).
    Load: the knee must actually be bent (a real crouch, not just standing)."""
    wy = fp.pt(keys["wrist"])[1]
    if phase == "release":
        return wy < fp.pt(keys["shoulder"])[1]
    if phase == "follow":
        return wy < fp.pt(keys["hip"])[1]
    if phase == "load":
        return _knee_angle(fp, keys) < 165.0
    return True


def _nominal(canon_xy):
    """Map shoulder-centered canonical coords to sane normalized [0,1] display
    coords (the app re-warps by shoulder/hip, so the constants only keep values
    in range -- they don't affect the final overlay)."""
    NOM = 0.18
    out = np.empty_like(canon_xy)
    out[:, 0] = 0.5 + canon_xy[:, 0] * NOM
    out[:, 1] = 0.42 + canon_xy[:, 1] * NOM
    return out


def build_skeletons(good_shots, raw_dirs, handedness="auto"):
    """Average clean per-phase poses across the good shots into ideal skeletons.

    good_shots: iterable of (clip_basename, shot_in_clip).
    raw_dirs:   dirs to look for the raw clip in (first hit wins).
    Returns (skeletons, stats) where skeletons = {load|release|follow: [ {x,y,
    visibility} x33 ] or None} and stats = {phase: n_contributing_shots}.
    """
    # group requested shots by clip so each clip's track loads once
    by_clip = {}
    for clip, sic in good_shots:
        by_clip.setdefault(clip, set()).add(int(sic))

    acc = {p: {"xy": np.zeros((33, 2)), "w": np.zeros(33)} for p in _PHASE_REQ}
    stats = {p: 0 for p in _PHASE_REQ}

    for clip, sics in by_clip.items():
        video_path = _find_clip(clip, raw_dirs)
        if video_path is None:
            continue
        loaded = _load_track(video_path)     # (params, track, shots) or None
        if loaded is None:
            continue
        _params, track, shots = loaded
        info = probe(video_path)
        wanted = {s.index: s for s in shots if s.index in sics}
        for s in wanted.values():
            sk = _shot_skeletons(video_path, s, track, info.fps,
                                 info.width, info.height, handedness)
            for phase, (canon, vis) in sk.items():
                wt = np.clip(vis, 0.0, 1.0)
                acc[phase]["xy"] += canon * wt[:, None]
                acc[phase]["w"] += wt
                stats[phase] += 1

    skeletons = {}
    for phase, a in acc.items():
        if stats[phase] == 0:
            skeletons[phase] = None
            continue
        w = np.where(a["w"] > 1e-6, a["w"], 1.0)
        mean = a["xy"] / w[:, None]
        disp = _nominal(mean)
        vis = np.clip(a["w"] / max(stats[phase], 1), 0.0, 1.0)
        skeletons[phase] = [
            {"x": round(float(disp[i, 0]), 4),
             "y": round(float(disp[i, 1]), 4),
             "visibility": round(float(vis[i]), 3)}
            for i in range(33)
        ]
    return skeletons, stats


def _find_clip(clip, raw_dirs):
    for d in raw_dirs:
        p = os.path.join(d, clip)
        if os.path.exists(p):
            return p
    return None
