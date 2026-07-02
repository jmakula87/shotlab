"""Validate Phase-2 form metrics against a synthetic skeleton performing a shot
with KNOWN mechanics. This tests the metric math (no MediaPipe needed); real
keypoint accuracy is validated separately on real footage."""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.phase2_pose.pose import FramePose, L, joint_angle
from shotlab.phase2_pose.form import compute_form


class FakeShot:
    def __init__(self, frames):
        self.index = 1
        self.frames = np.array(frames)


class FakeBall:
    def __init__(self, cx, cy):
        self.cx, self.cy, self.r, self.conf = cx, cy, 8.0, 0.9


def make_pose(idx, joints):
    """joints: name -> (x, y). Unset joints are 0 with visibility 1."""
    xy = np.zeros((33, 2))
    vis = np.ones(33)
    z = np.zeros(33)
    for name, (x, y) in joints.items():
        xy[L[name]] = [x, y]
    return FramePose(idx, xy, vis, z)


def build_shot_sequence():
    """Right-handed, side-on shooter. Image y grows DOWN.
    - frames 0..9: dip (knee bends to ~110 deg, body lowers)
    - frames 10..18: rise + jump (hips go UP = y decreases), release at 18
    - frames 19..27: follow-through hold (wrist stays high), then arm drops
    Release elbow angle set to ~95 deg.
    """
    poses, ball = {}, {}
    release_f = 18
    apex_f = 18
    for f in range(0, 32):
        # vertical body offset: lowers during dip, peaks (most negative) at apex
        if f <= apex_f:
            body_dy = -3.0 * f            # rising: y decreases
        else:
            body_dy = -3.0 * apex_f + 4.0 * (f - apex_f)   # coming down
        # knee angle: 170 -> 110 by frame 9, back to ~175 by release
        if f <= 9:
            knee = 170 - (60 * f / 9.0)
        else:
            knee = 110 + (65 * (f - 9) / 9.0)
        knee = max(105, min(178, knee))

        hip_y = 300 + body_dy
        # place ankle fixed-ish (feet), knee from knee angle in sagittal plane
        ankle = (200, 470)
        hip = (205, hip_y)
        # knee point: interpolate so angle(hip,knee,ankle) ~ target. Simple: put
        # knee horizontally offset by how bent it is.
        bend = (180 - knee)
        knee_pt = (200 + bend * 0.8, (hip_y + 470) / 2)

        shoulder = (210, hip_y - 90)
        # arm: elbow below/under, wrist up near head. At release set ~95 deg.
        if f >= release_f:
            # follow-through: wrist high above shoulder, elbow extended
            held = f <= release_f + 6
            wrist = (240, shoulder[1] - (40 if held else +40))
            elbow = (228, shoulder[1] - 5)
        else:
            wrist = (235, shoulder[1] + 10)
            elbow = (225, shoulder[1] + 30)

        joints = {
            "r_shoulder": shoulder, "l_shoulder": (195, shoulder[1]),
            "r_elbow": elbow, "r_wrist": wrist,
            "r_hip": hip, "l_hip": (195, hip_y),
            "r_knee": knee_pt, "r_ankle": ankle, "l_ankle": (190, 470),
        }
        poses[f] = make_pose(f, joints)
        # ball sits at the wrist until release, then flies
        if f <= release_f:
            ball[f] = FakeBall(*wrist)
        else:
            ball[f] = FakeBall(wrist[0] + 8 * (f - release_f),
                               wrist[1] - 6 * (f - release_f))
    return poses, ball, release_f


def test_release_frame_sync():
    poses, ball, rel = build_shot_sequence()
    shot = FakeShot(list(range(14, 30)))
    sf = compute_form(shot, ball, poses, fps=60, handedness="right",
                      camera_angle="side_on")
    # divergence-onset detector lands on the true release (ball leaves at rel)
    assert abs(sf.release_frame - rel) <= 1, sf.release_frame


def test_release_subframe_and_confidence():
    from shotlab.phase2_pose.form import find_release
    poses, ball, rel = build_shot_sequence()
    shot = FakeShot(list(range(14, 30)))
    est = find_release(shot, ball, poses, handedness="right")
    # integer frame is the onset; sub-frame time sits between onset and next frame
    assert est.frame == rel, est.frame
    assert rel <= est.t <= rel + 1, est.t
    # clean synthetic hand-off with visible wrist -> high confidence + diverging
    assert est.diverging is True
    assert est.confidence == "high", est.confidence
    # the ShotForm carries the sync confidence and sub-frame time through
    sf = compute_form(shot, ball, poses, fps=60, camera_angle="side_on")
    assert sf.release_conf == "high"
    assert abs(sf.release_t - est.t) < 1e-6


def test_late_ball_falls_back_to_wrist_apex():
    """Far/small ball the detector only locks onto ~0.4s into flight: the ball
    'release' would land late (arm already down). find_release should instead use
    the pose wrist-apex and recover the true release."""
    from shotlab.phase2_pose.form import find_release
    poses, _ball, rel = build_shot_sequence()      # true release / wrist apex = 18
    # ball tracked ONLY late, already flying up-and-away from the lowered wrist
    late_ball = {f: FakeBall(300 + 8 * (f - 28), 100 - 6 * (f - 28))
                 for f in range(28, 32)}
    shot = FakeShot(list(range(28, 32)))           # flight detected late
    est = find_release(shot, late_ball, poses, handedness="right", fps=60)
    assert abs(est.frame - rel) <= 2, est.frame    # NOT the late ball frame (28)
    assert "apex" in est.note, est.note


def test_clean_ball_still_wins_over_apex():
    """When the ball IS tracked through the release, keep the sharper ball
    estimate (the apex must not hijack clean footage)."""
    from shotlab.phase2_pose.form import find_release
    poses, ball, rel = build_shot_sequence()
    shot = FakeShot(list(range(14, 30)))
    est = find_release(shot, ball, poses, handedness="right", fps=60)
    assert abs(est.frame - rel) <= 1, est.frame
    assert "apex" not in est.note


def test_jump_height_is_ankle_based_not_squat():
    """The load squat drops the hips ~a foot; that must NOT count as jump.
    Jump = how far the (lower) ankle line rises when both feet are airborne."""
    from shotlab.phase2_pose.form import _jump_height
    poses = {}
    for f in range(0, 30):
        ankle_y = 470.0 - (20.0 if 18 <= f <= 22 else 0.0)   # 20px flight
        hip_y = 300.0 + (40.0 if 5 <= f <= 12 else 0.0)      # 40px squat
        poses[f] = make_pose(f, {"l_ankle": (190, ankle_y), "r_ankle": (200, ankle_y),
                                 "l_hip": (195, hip_y), "r_hip": (205, hip_y)})
    jh = _jump_height(poses, range(0, 30), ppf=40.0)
    assert abs(jh - 0.5) < 0.05, jh          # 20px / 40ppf = 0.5 ft, squat excluded


def test_jump_height_ignores_single_foot_step():
    """A step into the shot lifts ONE foot; the lower-ankle series stays on the
    ground line, so no phantom jump."""
    from shotlab.phase2_pose.form import _jump_height
    poses = {}
    for f in range(0, 30):
        r_y = 470.0 - (30.0 if 5 <= f <= 10 else 0.0)        # stepping foot
        poses[f] = make_pose(f, {"l_ankle": (190, 470.0), "r_ankle": (200, r_y)})
    jh = _jump_height(poses, range(0, 30), ppf=40.0)
    assert abs(jh) < 0.05, jh


def test_jump_height_physics_gate():
    """An impossible jump (ankles rise 8 ft worth of px) is a tracking failure
    and must come back None, not a number."""
    from shotlab.phase2_pose.form import _jump_height
    poses = {}
    for f in range(0, 30):
        y = 470.0 - (320.0 if 15 <= f <= 20 else 0.0)   # 320px @ 40ppf = 8 ft
        poses[f] = make_pose(f, {"l_ankle": (190, y), "r_ankle": (200, y)})
    assert _jump_height(poses, range(0, 30), ppf=40.0) is None


def test_jump_height_ignores_single_frame_glitch():
    """One bad pose frame (both ankles jump 60px for a single frame) must not
    read as flight -- the median-3 smoothing kills it."""
    from shotlab.phase2_pose.form import _jump_height
    poses = {}
    for f in range(0, 30):
        y = 410.0 if f == 12 else 470.0                      # 1-frame glitch
        poses[f] = make_pose(f, {"l_ankle": (190, y), "r_ankle": (200, y)})
    jh = _jump_height(poses, range(0, 30), ppf=40.0)
    assert abs(jh) < 0.05, jh


def test_release_subframe_no_divergence_is_low_conf():
    """A ball that never leaves the hand (no shot) -> low-confidence fallback."""
    from shotlab.phase2_pose.form import find_release
    poses, ball, _ = build_shot_sequence()
    # pin the ball to the wrist for every frame (never diverges)
    held = {}
    for f, fp in poses.items():
        w = fp.pt("r_wrist")
        held[f] = FakeBall(float(w[0]), float(w[1]))
    shot = FakeShot(list(range(14, 30)))
    est = find_release(shot, held, poses, handedness="right")
    assert est.diverging is False
    assert est.confidence == "low"


def test_knee_bend_recovered():
    poses, ball, rel = build_shot_sequence()
    shot = FakeShot(list(range(14, 30)))
    sf = compute_form(shot, ball, poses, fps=60, camera_angle="side_on")
    m = {x.name: x for x in sf.metrics}
    kb = m["knee_bend_deg"]
    assert kb.value is not None
    assert kb.confidence == "high"           # side-on -> high confidence
    assert 100 <= kb.value <= 125, kb.value   # min knee angle ~110


def test_release_vs_apex_and_followthrough():
    poses, ball, rel = build_shot_sequence()
    shot = FakeShot(list(range(14, 30)))
    sf = compute_form(shot, ball, poses, fps=60, camera_angle="side_on")
    m = {x.name: x for x in sf.metrics}
    # apex == release here, so dt ~ 0
    assert abs(m["release_vs_apex_s"].value) < 0.05
    # follow-through hold is positive (wrist held high ~6 frames @60fps ~ 0.1s)
    ft = m["follow_through_hold_s"]
    assert ft.value is not None and ft.value > 0


def test_auto_handedness():
    from shotlab.phase2_pose.form import detect_handedness
    # right wrist rises higher (smaller y) than left across the window -> "right"
    poses = {}
    for f in range(0, 10):
        poses[f] = make_pose(f, {"r_wrist": (240, 100 - f * 5),   # climbs high
                                 "l_wrist": (180, 260)})          # stays low
    assert detect_handedness(poses, list(range(0, 10))) == "right"
    # mirror it -> "left"
    poses2 = {}
    for f in range(0, 10):
        poses2[f] = make_pose(f, {"l_wrist": (180, 100 - f * 5),
                                  "r_wrist": (240, 260)})
    assert detect_handedness(poses2, list(range(0, 10))) == "left"
    # nothing visible -> default
    empty = {f: make_pose(f, {}) for f in range(3)}
    for fp in empty.values():
        fp.vis[:] = 0.0
    assert detect_handedness(empty, [0, 1, 2], default="right") == "right"


def test_tempo_dip_to_release():
    poses, ball, rel = build_shot_sequence()
    shot = FakeShot(list(range(14, 30)))
    sf = compute_form(shot, ball, poses, fps=60, camera_angle="side_on")
    m = {x.name: x for x in sf.metrics}
    tempo = m["tempo_dip_to_release_s"]
    # deepest knee ~frame 9, release ~frame 18 @60fps -> ~0.15s, and positive
    assert tempo.value is not None and 0.05 < tempo.value < 0.4, tempo.value


def test_squareness_na_on_sideon():
    poses, ball, rel = build_shot_sequence()
    shot = FakeShot(list(range(14, 30)))
    sf = compute_form(shot, ball, poses, fps=60, camera_angle="side_on")
    m = {x.name: x for x in sf.metrics}
    assert m["squareness_deg"].confidence == "na"   # needs front-on


def test_elbow_confidence_drops_fronton():
    poses, ball, rel = build_shot_sequence()
    shot = FakeShot(list(range(14, 30)))
    sf_side = compute_form(shot, ball, poses, fps=60, camera_angle="side_on")
    sf_front = compute_form(shot, ball, poses, fps=60, camera_angle="front_on")
    side = {x.name: x for x in sf_side.metrics}["elbow_angle_at_release_deg"]
    front = {x.name: x for x in sf_front.metrics}["elbow_angle_at_release_deg"]
    assert side.confidence == "medium"
    assert front.confidence == "low"     # honest about out-of-plane flare


if __name__ == "__main__":
    import traceback
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in funcs:
        try:
            fn(); print(f"PASS {fn.__name__}"); passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(funcs)} passed")
    sys.exit(0 if passed == len(funcs) else 1)
