"""Side-by-side shot comparison from annotated stills.

For two shots, finds the key phases of the shooting motion (load / rise /
release / follow-through), draws the skeleton at each with RED DOTS on the elbow
and knee plus the joint angles, crops to the shooter, and tiles them into one
comparison image (one row per shot, one column per phase). Lets you eyeball
"good shot: deeper knee, elbow at 90 / weak shot: stood tall, elbow flared."

Angles are foreshortened by the single camera, so they're labelled as a guide;
the VISUAL pose comparison is the point.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from .phase2_pose.pose import L, joint_angle, side_keys
from .phase1_ball.overlay import _SKELETON

_RED = (40, 40, 230)
_CYAN = (230, 230, 40)
_WHITE = (245, 245, 245)


def _nearest_pose(poses, f, span=6):
    for d in range(span + 1):
        for ff in (f - d, f + d):
            if ff in poses:
                return poses[ff]
    return None


def key_phases(shot, poses, rel_f, fps, keys) -> dict:
    """Return {phase_label: frame} for load / rise / release / follow-through."""
    # load = deepest knee bend (min knee angle) in the ~0.6s before release
    lo = rel_f - int(0.6 * fps)
    best_f, best_k = rel_f, 999.0
    for f in range(lo, rel_f + 1):
        fp = poses.get(f)
        if fp is None or fp.vis[L[keys["knee"]]] < 0.3:
            continue
        k = joint_angle(fp.pt(keys["hip"]), fp.pt(keys["knee"]), fp.pt(keys["ankle"]))
        if k == k and k < best_k:
            best_k, best_f = k, f
    load_f = best_f
    rise_f = (load_f + rel_f) // 2
    follow_f = rel_f + int(0.18 * fps)
    return {"load": load_f, "rise": rise_f, "release": rel_f, "follow-through": follow_f}


def annotate_still(frame, fp, keys, label, scale_h=360):
    """Draw skeleton + elbow/knee dots + angles, crop to the shooter."""
    img = frame.copy()
    # skeleton
    for a, b in _SKELETON:
        if fp.vis[a] >= 0.3 and fp.vis[b] >= 0.3:
            cv2.line(img, tuple(fp.xy[a].astype(int)), tuple(fp.xy[b].astype(int)),
                     _CYAN, 2, cv2.LINE_AA)
    # elbow + knee dots and angles
    for joint, a, b, c in [("elbow", keys["shoulder"], keys["elbow"], keys["wrist"]),
                           ("knee", keys["hip"], keys["knee"], keys["ankle"])]:
        if all(fp.v(n) >= 0.3 for n in (a, b, c)):
            ang = joint_angle(fp.pt(a), fp.pt(b), fp.pt(c))
            p = fp.pt(b).astype(int)
            cv2.circle(img, tuple(p), 9, _RED, -1, cv2.LINE_AA)
            cv2.circle(img, tuple(p), 9, _WHITE, 1, cv2.LINE_AA)
            cv2.putText(img, f"{joint} {ang:.0f}", (p[0] + 12, p[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, _RED, 2, cv2.LINE_AA)
    # crop to the visible keypoints (the shooter), padded
    vis = fp.xy[fp.vis >= 0.3]
    if len(vis) >= 4:
        x0, y0 = vis.min(0); x1, y1 = vis.max(0)
        pad = 0.30 * max(x1 - x0, y1 - y0)
        h, w = img.shape[:2]
        x0, x1 = int(max(0, x0 - pad)), int(min(w, x1 + pad))
        y0, y1 = int(max(0, y0 - pad)), int(min(h, y1 + pad))
        if x1 - x0 > 10 and y1 - y0 > 10:
            img = img[y0:y1, x0:x1]
    # label banner
    sc = scale_h / img.shape[0]
    img = cv2.resize(img, (int(img.shape[1] * sc), scale_h))
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (30, 30, 30), -1)
    cv2.putText(img, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _WHITE, 2,
                cv2.LINE_AA)
    return img


def _detect_clip(video_path, weights, calib, handedness, imgsz=640, max_frames=8000):
    """Detect ALL shots + pose for a clip once (so comparing two shots from the
    same clip costs one detection pass)."""
    from .detect_cache import detect_or_load
    from .phase2_pose.pipeline import run_phase2
    from .video_io import probe
    info = probe(video_path)
    stride = max(1, round(info.fps / 40), -(-info.n_frames // 7000))
    track, all_shots = detect_or_load(video_path, weights, calib, stride,
                                      max_frames, imgsz=imgsz)
    p2 = run_phase2(video_path, all_shots, track, handedness=handedness,
                    rim_xy=(calib.rim_x, calib.rim_y))
    rel = {f.shot: f.release_frame for f in p2.forms}
    shots = {s.index: s for s in all_shots}
    return info, shots, p2.poses, rel


def compare_shots(clip_a, shot_a, clip_b, shot_b, *, weights, out_path,
                  handedness="right", labels=("A", "B")):
    """Render a comparison PNG for two shots. Re-detects each clip (OpenVINO is
    fast). Returns out_path."""
    from .court import auto_calibrate
    keys = side_keys(handedness)

    # detect each unique clip just once
    clip_cache = {}
    for clip in {clip_a, clip_b}:
        calib = auto_calibrate(clip, os.path.basename(clip))
        clip_cache[clip] = _detect_clip(clip, weights, calib, handedness)

    rows = []
    phase_order = ["load", "rise", "release", "follow-through"]
    for clip, sidx, tag in [(clip_a, shot_a, labels[0]), (clip_b, shot_b, labels[1])]:
        info, shots, poses, rel = clip_cache[clip]
        shot = shots.get(sidx)
        if shot is None:
            raise ValueError(f"shot {sidx} not found in {os.path.basename(clip)}")
        rel_f = rel.get(sidx, int(shot.frames[0]))
        phases = key_phases(shot, poses, rel_f, info.fps, keys)
        cells = []
        for ph in phase_order:
            fp = _nearest_pose(poses, phases[ph])
            if fp is None:
                cell = np.full((360, 270, 3), 40, np.uint8)
                cv2.putText(cell, f"{ph}: no pose", (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, _WHITE, 1)
            else:
                from .video_io import iter_frames
                frame = None
                for _, fr in iter_frames(clip, start=fp.frame_idx, stop=fp.frame_idx + 1):
                    frame = fr
                cell = annotate_still(frame, fp, keys, f"{tag} | {ph}")
            cells.append(cell)
        # pad cells to same width
        w = max(c.shape[1] for c in cells)
        cells = [cv2.copyMakeBorder(c, 0, 0, 0, w - c.shape[1], cv2.BORDER_CONSTANT,
                                    value=(20, 20, 20)) for c in cells]
        rows.append(np.hstack(cells))

    w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT,
                               value=(20, 20, 20)) for r in rows]
    grid = np.vstack(rows)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, grid)
    return out_path
