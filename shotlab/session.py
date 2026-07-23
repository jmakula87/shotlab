"""Session analytics: stitch many clips into one timeline and surface trends.

Answers the "am I getting more tired / worse over the session?" questions:
  - filename timestamps (PXL_YYYYMMDD_HHMMSS) give each clip's wall-clock start;
    frame index / fps gives the offset within a clip -> every shot gets an
    absolute time and an elapsed-minutes-into-session value.
  - per-shot metrics (arc, optional knee bend, optional spin), zone, and
    make/miss are collected into one table.
  - fatigue trends = linear fit of each metric vs elapsed time (slope = drift).

Per-clip results are cached to data/out/<clip>/<clip>_shots_session.json so the
(expensive) detection only runs once.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
from dataclasses import dataclass, asdict, field, fields as _dc_fields

import numpy as np
import pandas as pd

from .court import Calibration, filter_shots_by_rim, zone_for_release
from .make import classify_make
from .shottype import classify_shot_type
from .scale import px_per_foot_from_rim, apex_above_rim_ft

_TS_RE = re.compile(r"(\d{8})_(\d{6})(\d{0,3})")


def _round2(x):
    return None if x is None else round(float(x), 2)


def parse_clip_time(path: str) -> _dt.datetime | None:
    """PXL_20260628_191733118.mp4 -> datetime(2026,06,28,19,17,33,118ms)."""
    m = _TS_RE.search(os.path.basename(path))
    if not m:
        return None
    d, t, ms = m.groups()
    try:
        base = _dt.datetime.strptime(d + t, "%Y%m%d%H%M%S")
        if ms:
            base = base.replace(microsecond=int(ms.ljust(3, "0")) * 1000)
        return base
    except ValueError:
        return None


@dataclass
class ShotRecord:
    clip: str
    abs_time: str                 # ISO; absolute wall-clock of the release
    elapsed_min: float            # minutes since the session's first shot
    shot_in_clip: int
    release_angle_deg: float | None = None
    entry_angle_deg: float | None = None
    apex_height_ft: float | None = None
    apex_above_rim_ft: float | None = None   # ball arc peak above the rim (rim-scaled)
    release_height_ft: float | None = None   # ball height at release above the floor
    jump_height_ft: float | None = None      # vertical body travel (load->peak)
    n_points: int = 0
    rim_dist_px: float | None = None
    rim_dx_px: float | None = None    # release point vs rim, image px (sign = camera-dependent)
    rim_dy_px: float | None = None    # release point below rim = positive (image y)
    zone: str = ""
    side: str = ""
    depth: str = ""
    shot_form: str = "unknown"      # jumper | layup | floater | unknown
    shot_setup: str = "unknown"     # catch_and_shoot | on_the_move | off_dribble
    movement_dir: str = "unknown"   # left | right | set | unknown (into the shot)
    knee_bend_deg: float | None = None
    release_vs_apex_s: float | None = None
    tempo_dip_to_release_s: float | None = None   # quickness: deepest load -> release
    elbow_angle_at_release_deg: float | None = None
    follow_through_hold_s: float | None = None
    balance_drift_px_per_ht: float | None = None
    release_conf: str = "na"        # confidence in the ball/pose release sync
    backspin_rpm: float | None = None
    made: object = None           # True / False / None
    make_conf: str = "na"
    felt_good: object = None      # True / False / None -- user's subjective tag
    is_real: bool = True          # False = degenerate detection (stationary/phantom/
    quality_note: str = ""        #  too-long); auto-dropped by curation (audit D2)

    def row(self) -> dict:
        return asdict(self)


def _cache_path(video_path: str) -> str:
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join("data", "out", stem, f"{stem}_shots_session.json")


# Bump when the record-building LOGIC changes in a way the schema/params don't
# capture (e.g. a metric formula). The ShotRecord field set is folded in
# automatically, so adding/removing a record field invalidates caches on its own.
_CACHE_VERSION = 21   # v20: VFR fix -- abs_time + audio rim window from real
                      #      container PTS, not frame/nominal-fps
                      # v21: chunk windows detect `_CHUNK_OVERLAP` past their
                      #      end + seam dedup (boundary shots were lost)
# NOTE: is_real is DETECTION-derived (fit/frames/rim), not pose -- ideally it'd be
# recomputed at load time so a threshold tweak doesn't force a pose re-run. Deferred
# (needs a full Calibration, not fully reconstructable from the detection cache).


def _record_cache_sig(*, detector_name, weights, imgsz, stride, max_frames,
                      with_pose, with_spin, handedness, with_audio=False,
                      shooter_height_ft=None, video_path=None,
                      calib=None, tiles=None, conf=0.25, use_beam=False,
                      make_model=None) -> str:
    """A signature for a clip's cached records. Any change to the record schema,
    the detection/pose params, the VIDEO's content (size+mtime -- an in-place
    re-trim/re-transcode keeps the basename the cache is filed under), the
    weights' content (a retrain re-exported to the same path), or an explicit
    calibration invalidates the cache, so a re-run recomputes instead of
    silently returning stale rows. calib=None hashes as 'auto': auto-calibration
    is deterministic given the video content, which the signature now carries."""
    from .detect_cache import _weights_id, _code_hash
    from .video_io import video_id
    schema = ",".join(f.name for f in _dc_fields(ShotRecord))
    calib_id = ("auto" if calib is None else ",".join(
        str(round(float(v), 1)) for v in
        (calib.rim_x, calib.rim_y, calib.rim_radius_px, calib.shot_gate_px)))
    raw = "|".join(str(x) for x in [
        _CACHE_VERSION, _code_hash(), schema, detector_name, _weights_id(weights),
        imgsz, stride, max_frames, with_pose, with_spin, handedness, with_audio,
        shooter_height_ft, tiles, round(float(conf), 3), bool(use_beam),
        (os.path.basename(make_model) if make_model else "geom"),
        video_id(video_path) if video_path else "none", calib_id])
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _real_time(frame, times, info) -> float:
    """Real (capture-time) seconds of `frame` since clip start.

    Uses the container's per-frame PTS when available: these phones record
    VARIABLE frame rate (a Pixel clip measured 30fps then 24fps against a
    nominal 27), so frame/nominal-fps drifts by whole seconds across a long
    clip -- enough to point the audio make/miss window at silence and skew the
    session timeline (2026-07-15 audit). PTS measure PLAYBACK time, so slow-mo
    files (stored at 30fps playback) are scaled back to capture time -- audio
    and wall clock both run in real time. Falls back to frame/fps when no
    usable PTS exist."""
    if times:
        t = times.get(int(frame))
        if t is not None:
            cfps = getattr(info, "container_fps", 0.0) or info.fps
            return float(t) * (cfps / info.fps) if info.fps else float(t)
    return frame / info.fps


def _records_from_shots(shots, track, video_path, calib, info, clip_start, *,
                        do_spin, with_pose, handedness, audio=None,
                        shooter_height_ft=None, times=None,
                        make_model=None) -> list[ShotRecord]:
    """Build ShotRecords for a set of shots (pose + spin + make + zone). Shared by
    the whole-clip pass and each window of the long-clip chunker; frame indices in
    `shots`/`track` are absolute, so the timestamps come out right either way.

    `make_model` (path to a make_visual joblib): use the LEARNED visual make/miss
    model as the primary classifier (geometric `classify_make` = ~coin-flip on real
    footage, measured 2026-07-23; a re-fit visual model is 81% LOCO). Geometric is
    the fallback when the visual model can't extract features."""
    from .phase1_ball.pipeline import metrics_for_shot
    _mv_model = None
    if make_model:
        try:
            import shotlab.make_visual as _mv
            _mv_model = _mv.load(make_model)
        except (FileNotFoundError, ImportError, Exception):
            _mv_model = None

    ppf_rim = px_per_foot_from_rim(calib.rim_radius_px)   # rim = 18in ruler

    forms = {}
    if with_pose and shots:
        from .phase2_pose.pipeline import run_phase2
        p2 = run_phase2(video_path, shots, track, handedness=handedness,
                        camera_angle="side_on", rim_xy=(calib.rim_x, calib.rim_y),
                        px_per_foot=ppf_rim, shooter_height_ft=shooter_height_ft)
        forms = {fm.shot: fm for fm in p2.forms}

    from .court import shot_quality
    records = []
    for s in shots:
        mm = metrics_for_shot(s, rim_x=calib.rim_x)
        # release point = the LOWEST fit-INLIER (not argmax over all points, which
        # picks the stale walk-back outlier and mislocates the zone; audit D12)
        mask = getattr(s.fit, "inlier_mask", None)
        in_idx = (np.where(mask)[0] if mask is not None and len(mask) == len(s.ys)
                  and mask.sum() >= 3 else np.arange(len(s.ys)))
        apex_i = int(in_idx[np.argmax(s.ys[in_idx])])   # lowest inlier point = release
        z = zone_for_release((s.xs[apex_i], s.ys[apex_i]), calib)
        real_ok, real_note = shot_quality(s, calib, info.fps)
        # release time = first tracked frame of the flight
        rel_frame = int(s.frames[0])
        t_off = _real_time(rel_frame, times, info)
        abs_t = (clip_start + _dt.timedelta(seconds=t_off)) if clip_start else None

        rec = ShotRecord(
            clip=os.path.basename(video_path),
            abs_time=abs_t.isoformat() if abs_t else "",
            elapsed_min=0.0,            # filled in by build_session
            shot_in_clip=s.index,
            release_angle_deg=mm.release_angle_deg,
            entry_angle_deg=mm.entry_angle_deg,
            apex_height_ft=mm.apex_height_ft,
            n_points=mm.n_points,
            rim_dist_px=s.meta.get("rim_dist_px"),
            rim_dx_px=z.get("rim_dx_px"), rim_dy_px=z.get("rim_dy_px"),
            zone=z["zone"], side=z["side"], depth=z["depth"],
            is_real=bool(real_ok), quality_note="" if real_ok else real_note,
        )
        # ball arc peak above the rim, in feet (rim-scaled; ball is ~at rim depth
        # here, so this is the most trustworthy real-feet number)
        rec.apex_above_rim_ft = _round2(
            apex_above_rim_ft(float(np.min(s.ys[in_idx])), calib.rim_y, ppf_rim))
        if do_spin:
            from .phase3_spin.spin import estimate_spin
            from .video_io import iter_frames
            lo, hi = int(s.frames[0]), int(s.frames[-1])
            sp = estimate_spin(iter_frames(video_path, start=lo, stop=hi + 1),
                               track, s, info.fps)
            rec.backspin_rpm = sp.backspin_rpm
        sf = forms.get(s.index)
        if sf is not None:
            by = {m.name: m for m in sf.metrics}
            def _mv(name):
                m = by.get(name)
                return m.value if m else None
            rec.knee_bend_deg = _mv("knee_bend_deg")
            rec.release_vs_apex_s = _mv("release_vs_apex_s")
            rec.tempo_dip_to_release_s = _mv("tempo_dip_to_release_s")
            rec.release_height_ft = _mv("release_height_ft")
            rec.jump_height_ft = _mv("jump_height_ft")
            rec.elbow_angle_at_release_deg = _mv("elbow_angle_at_release_deg")
            rec.follow_through_hold_s = _mv("follow_through_hold_s")
            rec.balance_drift_px_per_ht = _mv("balance_drift_px_per_ht")
            rec.release_conf = sf.release_conf
            rec.movement_dir = sf.movement_dir
        # auto shot-type tag (form + setup); release frame from pose sync if we
        # have it, else the flight's first frame.
        st_rel = sf.release_frame if sf is not None else rel_frame
        stype = classify_shot_type(
            depth=rec.depth, apex_above_rim_ft=rec.apex_above_rim_ft,
            release_angle_deg=rec.release_angle_deg,
            movement_dir=rec.movement_dir, ball_track=track,
            rel_frame=st_rel, fps=info.fps)
        rec.shot_form, rec.shot_setup = stype.form, stype.setup
        mk = classify_make(s, track, calib, fps=info.fps)   # geometric (+ rim_frame)
        made, mconf = mk.made, mk.confidence
        if _mv_model is not None:                            # LEARNED visual model wins
            import shotlab.make_visual as _mv
            feats = _mv.shot_features(video_path, s, calib, track=track)
            if feats is not None:
                v_made, v_prob = _mv.predict(_mv_model, feats)
                made = v_made
                mconf = "high" if abs(v_prob - 0.5) > 0.3 else "med"
        # Audio ONLY fills a genuine unknown (measured 2026-07-23: on a loud outdoor
        # court fuse_make never flips a decided call and mislabels filled unknowns as
        # "miss" 13/20 -- so it is demoted from the fuse to a last-resort tiebreak).
        if made is None and audio is not None and audio[0] is not None:
            from .audio import audio_make_hint, fuse_make
            rim_f = mk.rim_frame if mk.rim_frame is not None else int(s.frames[-1])
            rim_t = _real_time(rim_f, times, info)
            hint = audio_make_hint(audio[0], audio[1], rim_t)
            made, mconf = fuse_make(None, mk.confidence, hint)
        rec.made, rec.make_conf = made, mconf
        records.append(rec)
    return records


def _chunk_cache_path(video_path: str, start: int) -> str:
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join("data", "out", stem, f"{stem}_chunk_{start}.json")


# Frames each detection window reads PAST its end. Disjoint windows silently
# LOSE a shot whose launch sits at the end of one window and rim event at the
# start of the next -- neither window sees the whole arc (2026-07-15 audit,
# ~0.5-1% of shots at 7000-frame chunks). 600 frames = 20s at 30fps / 5s at
# 120fps, both far longer than any launch->rim arc plus walk-back.
_CHUNK_OVERLAP = 600


def _windows(n_frames: int, chunk_frames: int, overlap: int = _CHUNK_OVERLAP):
    """(w0, w1, stop) per window: [w0, w1) is the window's OWNERSHIP span (cache
    key = w0, windows advance by chunk_frames), [w0, stop) is what it detects
    over -- reaching `overlap` past w1 so a seam-straddling shot is seen whole
    by the earlier window."""
    ov = min(int(overlap), int(chunk_frames))
    out = []
    for w0 in range(0, n_frames, chunk_frames):
        w1 = min(n_frames, w0 + chunk_frames)
        out.append((w0, w1, min(n_frames, w1 + ov)))
    return out


def _merge_seam_pairs(kept: list, new_pairs: list, overlap: int = _CHUNK_OVERLAP
                      ) -> list:
    """Append window k+1's (record, shot) pairs onto `kept`, dropping seam
    duplicates: a shot whose frame range intersects an already-kept shot is the
    same flight seen by both windows (only possible inside the overlap zone).
    The fuller arc wins -- the later window starts mid-flight at the seam, so
    its walk-back is truncated and its metrics are the worse ones."""
    for rec, s in new_pairs:
        f0, f1 = int(s.frames[0]), int(s.frames[-1])
        dup = None
        for i in range(len(kept) - 1, -1, -1):
            ks = kept[i][1]
            k0, k1 = int(ks.frames[0]), int(ks.frames[-1])
            if k1 < f0 - overlap:          # kept is time-ordered; stop early
                break
            if f0 <= k1 and k0 <= f1:
                dup = i
                break
        if dup is None:
            kept.append((rec, s))
        elif len(s.frames) > len(kept[dup][1].frames):
            kept[dup] = (rec, s)
    return kept


def _process_chunked(video_path, calib, info, clip_start, *, weights, imgsz,
                     stride, chunk_frames, do_spin, with_pose, handedness,
                     use_cache, sig, audio=None, shooter_height_ft=None,
                     times=None, tiles=None, conf=0.25, use_beam=False,
                     make_model=None):
    """Process a long clip in absolute frame WINDOWS so each window fits the
    background-job time cap and is cached on its own -- a kill resumes at the next
    window instead of re-detecting from frame 0.

    Each window's cache holds BOTH its detection (track+shots) and its records, so
    resuming skips the expensive detection too. Windows are merged (shots
    renumbered) and the full-clip track cache is written so render/compare don't
    re-detect the long clip."""
    from .detect_cache import (detect_window, save_detection, _params,
                               serialize_detection, deserialize_detection)
    n = info.n_frames
    merged_track, pairs = {}, []
    for w0, w1, stop in _windows(n, chunk_frames):
        ccache = _chunk_cache_path(video_path, w0)
        recs = None
        if use_cache and os.path.exists(ccache):
            try:
                with open(ccache, encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("sig") == sig:        # same code/schema/params
                    track, shots = deserialize_detection(data["detection"])
                    recs = [ShotRecord(**r) for r in data["records"]]
            except (json.JSONDecodeError, KeyError, TypeError):
                recs = None                       # corrupt/old chunk -> redo
        if recs is None:
            track, shots = detect_window(video_path, weights, calib, int(stride),
                                         w0, stop, imgsz=imgsz, tiles=tiles,
                                         conf=conf, use_beam=use_beam)
            recs = _records_from_shots(shots, track, video_path, calib, info,
                                       clip_start, do_spin=do_spin,
                                       with_pose=with_pose, handedness=handedness,
                                       audio=audio,
                                       shooter_height_ft=shooter_height_ft,
                                       times=times, make_model=make_model)
            os.makedirs(os.path.dirname(ccache), exist_ok=True)
            with open(ccache, "w", encoding="utf-8") as f:
                json.dump({"sig": sig,
                           "detection": serialize_detection(track, shots, {}),
                           "records": [r.row() for r in recs]}, f)
        merged_track.update(track)
        pairs = _merge_seam_pairs(pairs, list(zip(recs, shots)))
    all_records = [p[0] for p in pairs]
    merged_shots = [p[1] for p in pairs]

    # renumber shots 1..N across the whole clip (records + track-cache shots stay
    # in lock-step so the dashboard/render share one numbering)
    for i, (rec, s) in enumerate(zip(all_records, merged_shots), start=1):
        rec.shot_in_clip = i
        s.index = i
    # whole-clip track cache, keyed as a full (max_frames=None) detection so
    # render_shots/compare load it instead of re-detecting the long clip
    save_detection(video_path, merged_track, merged_shots,
                   _params(video_path, weights, imgsz, stride, None, calib, tiles, conf))
    return all_records


def process_clip(video_path: str, calib: Calibration | None = None, *,
                 detector_name="motion", weights=None, imgsz=768, stride=1,
                 max_frames=None, chunk_frames=None, with_pose=False,
                 with_spin="auto", handedness="right",
                 use_cache=True, with_audio=False,
                 shooter_height_ft=None, tiles=None, conf=0.25,
                 use_beam=False, make_model=None) -> list[ShotRecord]:
    """Detect rim-anchored shots in one clip and return ShotRecords.

    If calib is None the rim is auto-detected for THIS clip (the tripod may move
    between clips, so per-clip calibration is the default).

    `chunk_frames` (yolo only): process clips longer than this in frame windows of
    that size, caching each window, so arbitrarily long exercise clips finish
    within the job time cap and resume after a kill."""
    cache = _cache_path(video_path)
    sig = _record_cache_sig(detector_name=detector_name, weights=weights,
                            imgsz=imgsz, stride=stride, max_frames=max_frames,
                            with_pose=with_pose, with_spin=with_spin,
                            handedness=handedness, with_audio=with_audio,
                            shooter_height_ft=shooter_height_ft,
                            video_path=video_path, calib=calib, tiles=tiles,
                            conf=conf, use_beam=use_beam, make_model=make_model)
    # (make_model participates in the signature so switching classifiers recomputes)
    if use_cache and os.path.exists(cache):
        try:
            with open(cache, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and data.get("sig") == sig:
                return [ShotRecord(**r) for r in data["records"]]
            # old bare-list format or sig mismatch (code/params changed) -> stale
        except (json.JSONDecodeError, KeyError, TypeError):
            pass   # corrupt cache -> recompute

    if calib is None:
        from .court import auto_calibrate
        calib = auto_calibrate(video_path, os.path.basename(video_path))
        if calib is None:
            return []   # no rim found -> can't anchor shots in this clip

    from .phase1_ball.pipeline import run_phase1
    from .video_io import probe
    if detector_name == "motion":
        from .phase1_ball.detect import MotionBallDetector
        detector = MotionBallDetector()
    elif detector_name == "motion+color":
        from .phase1_ball.detect import MotionColorBallDetector
        detector = MotionColorBallDetector()
    elif detector_name == "yolo":
        detector = None        # YOLO path uses the cached-detection helper below
    else:
        from .phase1_ball.detect import ColorBallDetector
        detector = ColorBallDetector()

    info = probe(video_path)
    clip_start = parse_clip_time(video_path)
    # real per-frame PTS: these phones record VFR, so frame/nominal-fps drifts
    # by seconds over a long clip (see _real_time). One cached grab() pass;
    # None (no usable PTS) -> the frame/fps fallback.
    from .video_io import frame_times_cached
    times = frame_times_cached(video_path)
    # auto-stride: detect at ~40 effective fps (30fps->1, 120fps->3), and thin
    # very long clips further so a single clip fits the background-job time cap.
    if stride == "auto":
        stride = max(1, round(info.fps / 40), -(-info.n_frames // 7000))
    do_spin = (with_spin is True) or (with_spin == "auto" and info.is_slowmo)
    audio = None
    if with_audio:
        from .audio import extract_audio
        audio = extract_audio(video_path)     # (samples, sr) once per clip

    # long-clip auto-chunk: window the detection so it fits the cap + resumes
    if (detector_name == "yolo" and chunk_frames
            and info.n_frames > chunk_frames):
        records = _process_chunked(
            video_path, calib, info, clip_start,
            weights=weights or "yolo11n.pt", imgsz=imgsz, stride=stride,
            chunk_frames=int(chunk_frames), do_spin=do_spin, with_pose=with_pose,
            handedness=handedness, use_cache=use_cache, sig=sig, audio=audio,
            shooter_height_ft=shooter_height_ft, times=times, tiles=tiles,
            conf=conf, use_beam=use_beam, make_model=make_model)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "w", encoding="utf-8") as f:
            json.dump({"sig": sig, "records": [r.row() for r in records]}, f, indent=2)
        return records

    # calib -> rim-anchored shots. YOLO detection is CACHED per clip (detect once,
    # reuse for rendering/comparison/re-analysis) -- the big speedup.
    if detector_name == "yolo":
        from .detect_cache import detect_or_load
        track, shots = detect_or_load(video_path, weights or "yolo11n.pt", calib,
                                      int(stride), max_frames, imgsz=imgsz,
                                      use_cache=use_cache, tiles=tiles, conf=conf,
                                      use_beam=use_beam)
    else:
        res = run_phase1(video_path, detector=detector, calib=calib,
                         stride=int(stride), max_frames=max_frames)
        track, shots = res.track, res.shots

    records = _records_from_shots(shots, track, video_path, calib, info,
                                  clip_start, do_spin=do_spin,
                                  with_pose=with_pose, handedness=handedness,
                                  audio=audio, shooter_height_ft=shooter_height_ft,
                                  times=times, make_model=make_model)

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump({"sig": sig, "records": [r.row() for r in records]}, f, indent=2)
    return records


def build_session(records: list[ShotRecord]) -> pd.DataFrame:
    """Combine ShotRecords across clips, sort by time, add elapsed_min."""
    rows = [r.row() for r in records]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df["abs_time"] != ""].copy()
    df["t"] = pd.to_datetime(df["abs_time"])
    df = df.sort_values("t").reset_index(drop=True)
    t0 = df["t"].iloc[0]
    df["elapsed_min"] = (df["t"] - t0).dt.total_seconds() / 60.0
    df["shot_num"] = np.arange(1, len(df) + 1)
    return df


def volume_stats(df: pd.DataFrame) -> dict:
    """Reps + makes + longest make-streak for a session (gamification)."""
    out = {"shots": len(df), "attempts": 0, "makes": 0, "make_pct": None,
           "longest_make_streak": 0}
    if df.empty or "made" not in df.columns:
        return out
    m = df.copy()
    if "elapsed_min" in m:
        m = m.sort_values("elapsed_min")
    seq = [v in (True, "True", 1) for v in m["made"]
           if v in (True, False, "True", "False", 1, 0)]
    out["attempts"] = len(seq)
    out["makes"] = sum(seq)
    out["make_pct"] = round(100 * out["makes"] / len(seq), 1) if seq else None
    best = cur = 0
    for v in seq:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    out["longest_make_streak"] = best
    return out


def aggregate_sessions(out_dir: str = "data/out") -> pd.DataFrame:
    """Roll every built session (data/out/session_*/session_shots.csv) into one
    row per session, dated from its shots, so you can track progress across days.
    Returns a DataFrame sorted by date (empty if no sessions yet)."""
    import glob
    from .curate import apply_excludes
    rows = []
    for d in sorted(glob.glob(os.path.join(out_dir, "session_*"))):
        csv = os.path.join(d, "session_shots.csv")
        if not os.path.exists(csv):
            continue
        df = apply_excludes(pd.read_csv(csv), d)   # match every other surface
        if df.empty:
            continue
        t = pd.to_datetime(df.get("abs_time"), errors="coerce").dropna()
        date = t.min().date().isoformat() if len(t) else ""
        vs = volume_stats(df)
        row = {"session": os.path.basename(d), "date": date, "shots": len(df),
               "makes": vs["makes"], "make_pct": vs["make_pct"],
               "longest_streak": vs["longest_make_streak"]}
        # consistency (within-zone std) per metric, so progress tracks BOTH the
        # mean AND how repeatable you are -- consistency matters as much as level.
        cons = consistency_stats(df)
        wz = (dict(zip(cons["metric"], cons["within_zone_std"]))
              if not cons.empty else {})
        for met in ["release_angle_deg", "entry_angle_deg", "apex_height_ft",
                    "knee_bend_deg"]:
            short = met.replace("_deg", "").replace("_ft", "")
            if met in df.columns and df[met].notna().any():
                from .metric_ranges import gate       # match the gated std_ below
                row["avg_" + short] = round(float(gate(df, met)[met].mean()), 1)
            if met in wz and wz[met] == wz[met]:        # not NaN
                row["std_" + short] = round(float(wz[met]), 1)
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values("date").reset_index(drop=True) if not out.empty else out


def consistency_progress(agg: pd.DataFrame) -> pd.DataFrame:
    """Across built sessions, is each metric's spread (std_*) shrinking over time?

    Lower std = more repeatable, so a NEGATIVE slope is improvement. Needs >=2
    sessions with the column present; returns one row per tracked metric."""
    if agg is None or agg.empty or len(agg) < 2:
        return pd.DataFrame()
    out = []
    for col in [c for c in agg.columns if c.startswith("std_")]:
        sub = agg[[col]].dropna()
        if len(sub) < 2:
            continue
        y = sub[col].to_numpy(float)
        x = np.arange(len(y), dtype=float)            # session order
        slope = float(np.polyfit(x, y, 1)[0]) if len(y) >= 2 else 0.0
        out.append({
            "metric": col[len("std_"):],
            "first_std": round(float(y[0]), 1),
            "latest_std": round(float(y[-1]), 1),
            "delta": round(float(y[-1] - y[0]), 1),
            "slope_per_session": round(slope, 2),
            "improving": bool(slope < 0),             # tighter spread over time
        })
    return pd.DataFrame(out)


def prescribe_target(df: pd.DataFrame) -> dict:
    """Pick the ONE thing to work on next: the least-repeatable metric (highest
    within-zone spread relative to its own level, so metrics of different units
    compare fairly). Returns {target_metric, within_zone_std, cv, reason} or {}."""
    cons = consistency_stats(df)
    if cons.empty:
        return {}
    best = None
    for _, row in cons.iterrows():
        m = row["metric"]
        wz = row.get("within_zone_std")
        if wz is None or wz != wz or m not in df.columns:
            continue
        from .metric_ranges import gate
        level = float(gate(df, m)[m].abs().mean())
        cv = wz / level if level > 1e-9 else wz          # spread relative to level
        if best is None or cv > best["cv"]:
            best = {"target_metric": m, "within_zone_std": round(float(wz), 2),
                    "cv": round(float(cv), 3)}
    if best is None:
        return {}
    best["reason"] = f"least repeatable ({best['target_metric']}, within-zone spread {best['within_zone_std']})"
    return best


def drill_effectiveness(sessions: list) -> pd.DataFrame:
    """Did the metric a session TOLD you to work on actually get more repeatable
    the NEXT session? `sessions` = date-ordered list of
    {name, target_metric, stds:{metric: within_zone_std}}. One row per hand-off."""
    rows = []
    for prev, cur in zip(sessions, sessions[1:]):
        tgt = prev.get("target_metric")
        if not tgt:
            continue
        before = prev.get("stds", {}).get(tgt)
        after = cur.get("stds", {}).get(tgt)
        if before is None or after is None:
            continue
        rows.append({"worked_on": tgt, "from_session": prev.get("name"),
                     "checked_session": cur.get("name"),
                     "std_before": round(float(before), 2),
                     "std_after": round(float(after), 2),
                     "improved": bool(after < before)})
    return pd.DataFrame(rows)


def consistency_stats(df: pd.DataFrame, metrics=None) -> pd.DataFrame:
    """How REPEATABLE are you? Spread (std dev) of each metric, plus whether you
    get more erratic as you tire.

    The headline number is the WITHIN-ZONE std: pooling the spread within each
    court zone removes the cross-zone confound (different spots foreshorten
    differently), so it reflects true shot-to-shot consistency from a given spot.
    first/second-half within-zone std shows if your shot scatters more when tired.
    """
    metrics = metrics or ["release_angle_deg", "entry_angle_deg",
                          "apex_height_ft", "apex_above_rim_ft", "knee_bend_deg",
                          "tempo_dip_to_release_s", "release_height_ft"]
    if df.empty or "elapsed_min" not in df:
        return pd.DataFrame()
    half = df["elapsed_min"].median()

    def pooled_within_zone_std(sub, m):
        g = sub.groupby("zone")[m]
        s, c = g.std(), g.count()
        s = s.dropna()
        w = (c[s.index] - 1)
        return float((s * w).sum() / w.sum()) if w.sum() > 0 else float("nan")

    rows = []
    for m in metrics:
        if m not in df.columns:
            continue
        from .metric_ranges import gate
        sub = gate(df[["elapsed_min", "zone", m]].dropna(subset=[m]), m)
        if len(sub) < 4:
            continue
        wz1 = pooled_within_zone_std(sub[sub["elapsed_min"] <= half], m)
        wz2 = pooled_within_zone_std(sub[sub["elapsed_min"] > half], m)
        # round per-unit: 1dp for degrees is fine, but a seconds std ~0.05 rounds
        # to a meaningless "0.0/0.1" at 1dp (audit D18b) -- give seconds 3dp, feet 2
        dec = 3 if m.endswith("_s") else 2 if (m.endswith("_ft") or m.endswith("per_ht")) else 1
        rows.append({
            "metric": m,
            "overall_std": round(float(sub[m].std()), dec),
            "within_zone_std": round(pooled_within_zone_std(sub, m), dec),
            "first_half_std": round(wz1, dec),
            "second_half_std": round(wz2, dec),
            "more_erratic_when_tired": bool(wz2 > wz1) if wz2 == wz2 and wz1 == wz1 else None,
        })
    return pd.DataFrame(rows)


_FATIGUE_METRICS = ["release_angle_deg", "entry_angle_deg", "apex_height_ft",
                    "apex_above_rim_ft", "knee_bend_deg", "tempo_dip_to_release_s",
                    "release_height_ft", "jump_height_ft"]


def fatigue_breakdown(df: pd.DataFrame, metrics=None) -> pd.DataFrame:
    """Which part of your shot degrades MOST as the session wears on -- so you
    know if it's your legs, your arc, or your release that goes first.

    Ranks metrics by their first-half -> second-half change, standardized by each
    metric's own spread (so degrees, seconds, and feet compare fairly). The top
    row is flagged `fades_most`."""
    metrics = metrics or _FATIGUE_METRICS
    if df.empty or "elapsed_min" not in df:
        return pd.DataFrame()
    half = df["elapsed_min"].median()
    rows = []
    for m in metrics:
        if m not in df.columns:
            continue
        from .metric_ranges import gate
        sub = gate(df[["elapsed_min", m]].dropna(), m)
        if len(sub) < 6:
            continue
        sd = float(sub[m].std())
        f1 = float(sub[sub["elapsed_min"] <= half][m].mean())
        f2 = float(sub[sub["elapsed_min"] > half][m].mean())
        if not (f1 == f1 and f2 == f2) or sd < 1e-9:
            continue
        rows.append({"metric": m, "first_half": round(f1, 2),
                     "second_half": round(f2, 2), "change": round(f2 - f1, 2),
                     "change_in_sd": round((f2 - f1) / sd, 2)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_abs"] = out["change_in_sd"].abs()
    out = out.sort_values("_abs", ascending=False).drop(columns="_abs").reset_index(drop=True)
    out["fades_most"] = [i == 0 for i in range(len(out))]
    return out


def mean_drift(agg: pd.DataFrame) -> pd.DataFrame:
    """Across built sessions, is a metric's LEVEL creeping over time (e.g. your
    release angle getting flatter session to session)? One row per avg_ metric;
    `drifting` flags a >5% move from the first session."""
    if agg is None or agg.empty or len(agg) < 2:
        return pd.DataFrame()
    rows = []
    for col in [c for c in agg.columns if c.startswith("avg_")]:
        sub = agg[col].dropna()
        if len(sub) < 2:
            continue
        y = sub.to_numpy(float)
        slope = float(np.polyfit(np.arange(len(y), dtype=float), y, 1)[0])
        base = abs(y[0]) if abs(y[0]) > 1e-9 else 1.0
        rows.append({"metric": col[len("avg_"):], "first": round(float(y[0]), 2),
                     "latest": round(float(y[-1]), 2), "delta": round(float(y[-1] - y[0]), 2),
                     "slope_per_session": round(slope, 3),
                     "drifting": bool(abs(y[-1] - y[0]) / base > 0.05)})
    return pd.DataFrame(rows)


def fatigue_trends(df: pd.DataFrame, metrics=None) -> pd.DataFrame:
    """Linear fit of each metric vs elapsed_min. Negative slope = declines as the
    session goes on (a fatigue signal)."""
    metrics = metrics or ["release_angle_deg", "entry_angle_deg", "apex_height_ft",
                          "apex_above_rim_ft", "knee_bend_deg",
                          "tempo_dip_to_release_s", "release_height_ft",
                          "jump_height_ft", "backspin_rpm"]
    out = []
    for m in metrics:
        if m not in df.columns:
            continue
        from .metric_ranges import gate
        sub = gate(df[["elapsed_min", m]].dropna(), m)   # drop artifact reads
        if len(sub) < 3:
            out.append({"metric": m, "n": len(sub), "slope_per_min": None,
                        "start": None, "end": None, "trend": "insufficient data"})
            continue
        x = sub["elapsed_min"].to_numpy()
        y = sub[m].to_numpy()
        slope, intercept = np.polyfit(x, y, 1)
        start, end = intercept, intercept + slope * x.max()
        trend = "declining" if slope < 0 else "rising"
        out.append({"metric": m, "n": len(sub),
                    "slope_per_min": round(float(slope), 3),
                    "start": round(float(start), 1), "end": round(float(end), 1),
                    "trend": trend})
    return pd.DataFrame(out)
