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


def build_clean_ball_late_apex():
    """Models the real release physics: the ball-CENTER starts moving off the
    wrist landmark at frame 18 (it rolls out to the fingertips), and the arm
    reaches FULL EXTENSION -- the true release -- at the wrist apex, frame 22.
    So the ball hand-off (~18) precedes peak extension (~22) by a few frames,
    exactly as on real footage. Image y grows downward (smaller = higher)."""
    poses, ball = {}, {}
    rel = 18
    for f in range(0, 32):
        shoulder_y = 210 - 2 * min(f, 22)               # body rises through 22
        if f <= 22:
            wrist_y = shoulder_y - 20 - 3 * f            # wrist climbs to a peak at 22
        else:
            wrist_y = shoulder_y - 20 - 3 * 22 + 5 * (f - 22)
        wrist = (240, float(wrist_y))
        joints = {
            "r_shoulder": (210, shoulder_y), "l_shoulder": (195, shoulder_y),
            "r_elbow": (228, wrist_y + 25), "r_wrist": wrist,
            "r_hip": (205, shoulder_y + 90), "l_hip": (195, shoulder_y + 90),
            "r_knee": (200, shoulder_y + 140), "r_ankle": (200, 470),
            "l_ankle": (190, 470),
        }
        poses[f] = make_pose(f, joints)
        if f <= rel:
            ball[f] = FakeBall(*wrist)                   # in hand until release
        else:                                            # flies up-and-away -> diverges
            ball[f] = FakeBall(wrist[0] + 10 * (f - rel), wrist[1] - 4 * (f - rel))
    return poses, ball, rel


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


def test_far_ball_is_not_a_confident_release():
    """A ball tracked far from the wrist (never in hand) must NOT be graded a
    confident hand-off just because its distance recedes -- the proximity gate
    (2026-07-06 audit: curated shots 28/24/21 shipped high-conf with the ball
    280-620px from the wrist)."""
    from shotlab.phase2_pose.form import find_release
    poses, ball = {}, {}
    for f in range(0, 24):
        wrist_y = 210 - 8 * f if f <= 8 else 146 + 8 * (f - 8)   # real peak at ~8
        wrist = (240, float(wrist_y))
        joints = {"r_shoulder": (210, 200), "l_shoulder": (195, 200),
                  "r_elbow": (228, wrist_y + 20), "r_wrist": wrist,
                  "r_hip": (205, 290), "l_hip": (195, 290),
                  "r_knee": (200, 360), "r_ankle": (200, 470), "l_ankle": (190, 470)}
        poses[f] = make_pose(f, joints)
        ball[f] = FakeBall(900.0 + 8 * f, 100.0)    # 600+px away, receding
    shot = FakeShot(list(range(6, 22)))
    est = find_release(shot, ball, poses, handedness="right", fps=30)
    assert est.diverging is False, est
    assert est.confidence == "low", est.confidence


def test_no_overhead_apex_is_not_confirmed():
    """The ball diverges cleanly but the arm never rises overhead -> no pose
    corroboration -> unconfirmed low, never a high-confidence release (2026-07-06
    audit: apex-None used to pass the ball estimate's confidence verbatim)."""
    from shotlab.phase2_pose.form import find_release
    poses, ball = {}, {}
    for f in range(0, 20):
        wrist = (240.0 + 4 * f, 230.0)              # stays BELOW the shoulder (200)
        joints = {"r_shoulder": (210, 200), "l_shoulder": (195, 200),
                  "r_elbow": (228, 215), "r_wrist": wrist,
                  "r_hip": (205, 290), "l_hip": (195, 290),
                  "r_knee": (200, 360), "r_ankle": (200, 470), "l_ankle": (190, 470)}
        poses[f] = make_pose(f, joints)
        ball[f] = (FakeBall(*wrist) if f <= 8
                   else FakeBall(wrist[0] + 12 * (f - 8), wrist[1] - 8 * (f - 8)))
    shot = FakeShot(list(range(2, 18)))
    est = find_release(shot, ball, poses, handedness="right", fps=30)
    assert est.confidence == "low", est.confidence
    assert est.diverging is False


def test_wrist_still_rising_at_window_edge_is_rejected():
    """If the wrist is still rising at the search-window edge, the 'apex' is the
    window clip (elbow mid-push ~90deg), not the snap -> reject (2026-07-06
    audit: curated shot 107 saturated the boundary at high confidence)."""
    from shotlab.phase2_pose.form import _wrist_apex, side_keys
    poses = {}
    for f in range(0, 40):
        wrist_y = 300.0 - 5.0 * f                   # rising through the whole window
        poses[f] = make_pose(f, {"r_shoulder": (210, 250), "r_wrist": (240, wrist_y)})
    shot = FakeShot(list(range(6, 22)))
    assert _wrist_apex(shot, poses, side_keys("right"), fps=30) is None


def test_late_ball_still_uses_peak_extension():
    """Far/small ball the detector only locks onto ~0.4s into flight: the release
    is still the pose wrist-apex (peak extension), and because the ball hand-off
    doesn't line up with the peak it comes back unconfirmed (low confidence)."""
    from shotlab.phase2_pose.form import find_release
    poses, _ball, rel = build_shot_sequence()      # wrist apex = 18
    # ball tracked ONLY late, already flying up-and-away from the lowered wrist
    late_ball = {f: FakeBall(300 + 8 * (f - 28), 100 - 6 * (f - 28))
                 for f in range(28, 32)}
    shot = FakeShot(list(range(28, 32)))           # flight detected late
    est = find_release(shot, late_ball, poses, handedness="right", fps=60)
    assert abs(est.frame - rel) <= 2, est.frame    # peak extension, not the late ball
    assert "apex" in est.note, est.note
    assert est.confidence == "low"                 # hand-off didn't confirm the peak


def test_release_anchors_to_peak_extension_confirmed_by_handoff():
    """Release is peak arm extension (the wrist apex), NOT ball-center divergence:
    the ball rolls off the fingertips a hand-length past the wrist, so its
    divergence fires mid-push at a still-bent elbow. The ball hand-off a few
    frames earlier CONFIRMS the apex (high confidence)."""
    from shotlab.phase2_pose.form import find_release, _wrist_apex, side_keys
    poses, ball, _ = build_clean_ball_late_apex()
    shot = FakeShot(list(range(14, 30)))
    af, _ = _wrist_apex(shot, poses, side_keys("right"), fps=30)
    est = find_release(shot, ball, poses, handedness="right", fps=30)
    assert est.frame == af, (est.frame, af)            # peak extension = release
    assert est.confidence == "high"                    # ball hand-off confirmed it
    assert est.diverging is True


def test_pumpfake_gives_no_confident_release():
    """A pump/aborted shot: the arm RISES (wrist above the shoulder) but the ball
    never leaves the hand. There's a wrist apex, but with NO clean ball hand-off
    it must come back LOW confidence and NOT diverging -- so nothing downstream
    trusts a non-release as a confident release (2026-07-05 audit #3)."""
    from shotlab.phase2_pose.form import find_release
    poses, ball = {}, {}
    for f in range(0, 24):
        wrist_y = 210 - 6 * f if f <= 10 else 150       # rises above shoulder, holds
        wrist = (240, float(wrist_y))
        joints = {"r_shoulder": (210, 200), "l_shoulder": (195, 200),
                  "r_elbow": (228, wrist_y + 20), "r_wrist": wrist,
                  "r_hip": (205, 290), "l_hip": (195, 290),
                  "r_knee": (200, 360), "r_ankle": (200, 470), "l_ankle": (190, 470)}
        poses[f] = make_pose(f, joints)
        ball[f] = FakeBall(float(wrist[0]), float(wrist[1]))   # ball pinned to wrist
    shot = FakeShot(list(range(6, 22)))
    est = find_release(shot, ball, poses, handedness="right", fps=30)
    assert est.diverging is False, est
    assert est.confidence == "low", est.confidence


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


def test_shooter_height_gives_body_scaled_jump():
    """With a known shooter height, JUMP height uses the body ruler (MEDIUM,
    'body-scaled') instead of the rim ruler (LOW). Release height stays LOW
    either way (its depth differs from the body plane)."""
    poses, ball, rel = build_shot_sequence()
    shot = FakeShot(list(range(14, 30)))
    # add a visible nose so the body ruler can measure nose->ankle
    for f, fp in poses.items():
        fp.xy[L["nose"]] = [212, fp.pt("r_shoulder")[1] - 20]  # above shoulders
        fp.vis[L["nose"]] = 1.0
    rim_ppf = 20.0
    sf_rim = compute_form(shot, ball, poses, fps=60, camera_angle="side_on",
                          px_per_foot=rim_ppf)
    sf_body = compute_form(shot, ball, poses, fps=60, camera_angle="side_on",
                           px_per_foot=rim_ppf, shooter_height_ft=70 / 12.0)
    rim_jh = next(m for m in sf_rim.metrics if m.name == "jump_height_ft")
    body_jh = next(m for m in sf_body.metrics if m.name == "jump_height_ft")
    assert rim_jh.confidence == "low"
    assert body_jh.confidence == "medium"
    assert "body-scaled" in body_jh.note
    # release height is LOW in both (depth-limited) and floor-referenced
    for sf in (sf_rim, sf_body):
        rh = next(m for m in sf.metrics if m.name == "release_height_ft")
        if rh.value is not None:
            assert rh.confidence == "low"
            assert "floor" in rh.note and "2-cam" in rh.note


def test_no_nose_jump_falls_back_to_rim_scale():
    """No visible nose -> body ruler can't measure -> jump is rim-scaled (LOW),
    so a height flag never silently produces a garbage body scale."""
    poses, ball, rel = build_shot_sequence()
    for fp in poses.values():                     # nose not detected
        fp.vis[L["nose"]] = 0.0
    shot = FakeShot(list(range(14, 30)))
    sf = compute_form(shot, ball, poses, fps=60, camera_angle="side_on",
                      px_per_foot=20.0, shooter_height_ft=70 / 12.0)
    jh = next(m for m in sf.metrics if m.name == "jump_height_ft")
    assert jh.confidence in ("low", "na")
    assert "rim-scaled" in jh.note


def test_release_height_referenced_to_floor_not_airborne_ankle():
    """Release height must use the planted floor, not the (raised) ankle at a
    jump -- otherwise an airborne release reads too low by the jump height."""
    from shotlab.phase2_pose.form import _ground_line
    poses, ball, rel = build_shot_sequence()
    span = range(0, 32)
    ground = _ground_line(poses, span)
    # the synthetic ankles sit at y=470 the whole clip -> floor at 470
    assert ground is not None and abs(ground - 470) < 5


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
    """A ball that never leaves the hand AND no overhead arm motion (no shot) ->
    low-confidence fallback (no ball divergence, no wrist apex to fall back on)."""
    from shotlab.phase2_pose.form import find_release
    poses, ball, _ = build_shot_sequence()
    held = {}
    for f, fp in poses.items():
        # keep the wrist BELOW the shoulder the whole time (arm never shoots),
        # so there's no valid overhead apex -- a true no-shot
        fp.xy[L["r_wrist"]] = [235, fp.pt("r_shoulder")[1] + 30]
        w = fp.pt("r_wrist")
        held[f] = FakeBall(float(w[0]), float(w[1]))   # ball pinned to the wrist
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
