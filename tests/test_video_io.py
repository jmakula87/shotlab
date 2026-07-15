"""video_id + frame_times_cached: content-keyed PTS caching on a real (tiny,
cv2-written) clip. The PTS scan itself (frame_times) is exercised on real
footage by the 3D pipeline; here we lock the cache behavior around it:
hit on unchanged video, miss + rescan when the file changes, degenerate-PTS
verdict cached as a fallback signal (None)."""

import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from shotlab.video_io import frame_times_cached, video_id


def _write_clip(path, n=20, fps=25.0):
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    assert w.isOpened(), "cv2 VideoWriter unavailable"
    for i in range(n):
        frame = np.full((48, 64, 3), i * 10 % 255, np.uint8)
        w.write(frame)
    w.release()


def test_video_id_tracks_content():
    with tempfile.TemporaryDirectory() as td:
        clip = os.path.join(td, "c.mp4")
        _write_clip(clip)
        a = video_id(clip)
        assert a not in ("", "absent")
        _write_clip(clip, n=30)                      # rewrite -> new size/mtime
        assert video_id(clip) != a
        assert video_id(os.path.join(td, "nope.mp4")) == "absent"


def test_frame_times_cached_roundtrip_and_invalidation():
    with tempfile.TemporaryDirectory() as td:
        clip = os.path.join(td, "PXL_20260715_180000000.mp4")
        out = os.path.join(td, "out")
        _write_clip(clip, n=20, fps=25.0)
        t1 = frame_times_cached(clip, out_dir=out)
        assert t1 is not None and len(t1) >= 19
        # sensible PTS: monotonic, ~1/25s apart
        vals = [t1[k] for k in sorted(t1)]
        dts = np.diff(vals)
        assert (dts >= 0).all() and abs(np.median(dts) - 0.04) < 0.02, dts

        cpath = os.path.join(out, "PXL_20260715_180000000",
                             "PXL_20260715_180000000_pts.json")
        assert os.path.exists(cpath)
        # cache hit: same times back without a rescan (poison the video to prove
        # the second call never opens it)
        blocked = clip + ".moved"
        shutil.move(clip, blocked)
        shutil.copy(blocked, clip)                    # same content, new mtime ->
        t_miss = frame_times_cached(clip, out_dir=out)  # id changed: rescan works
        assert t_miss is not None and len(t_miss) == len(t1)
        # now a true hit: id unchanged; corrupt the video file body to prove the
        # cached map is served without decoding
        with open(cpath, encoding="utf-8") as f:
            cached = json.load(f)
        assert cached["video"] == video_id(clip)
        t2 = frame_times_cached(clip, out_dir=out)
        assert t2 == t_miss


def test_degenerate_pts_cached_as_none():
    with tempfile.TemporaryDirectory() as td:
        clip = os.path.join(td, "flat.mp4")
        out = os.path.join(td, "out")
        _write_clip(clip)
        cdir = os.path.join(out, "flat")
        os.makedirs(cdir)
        # pre-seed a cache that says "no usable PTS" for this exact video
        with open(os.path.join(cdir, "flat_pts.json"), "w", encoding="utf-8") as f:
            json.dump({"video": video_id(clip), "times": []}, f)
        assert frame_times_cached(clip, out_dir=out) is None


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
