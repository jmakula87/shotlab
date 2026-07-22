"""Frame-ranged rim calibration — handles a camera that MOVES within one clip.

The 2026-07-22 dual review found clip1 (PXL_20260720_151519220) has real shots
approaching the rim at x~1130 early and x~1244 late: a 110px within-clip spread,
i.e. the tripod moved mid-clip. A single per-clip `Calibration` (calibrate.py)
cannot model that. This stores a LIST of rims, each scoped to a frame range,
and resolves the right one per frame.

Format `config/rim_<clip>.json`:
  {"clip": "...", "image_w": 1920, "image_h": 1080,
   "rims": [{"rim_x":1134,"rim_y":470,"rim_radius_px":16,"shot_gate_px":90,
             "f0":0,"f1":2000,"note":"early position"}, ...]}

A single fixed rim is just one entry with f0=0, f1=None (whole clip).
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shotlab.court import Calibration

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def rim_path(clip: str) -> Path:
    return CONFIG_DIR / f"rim_{clip}.json"


def load_rims(clip: str) -> dict | None:
    p = rim_path(clip)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_rims(doc: dict) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    p = rim_path(doc["clip"])
    # keep rims sorted by start frame so resolution is deterministic
    doc["rims"] = sorted(doc["rims"], key=lambda r: r["f0"])
    with open(p, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    return p


def add_rim(clip: str, image_w: int, image_h: int, *, rim_x, rim_y,
            rim_radius_px, f0=0, f1=None, gate_mult=2.0, note="") -> dict:
    """Append (or start) a frame-ranged rim entry. f1=None means 'to end of clip'."""
    doc = load_rims(clip) or {"clip": clip, "image_w": int(image_w),
                              "image_h": int(image_h), "rims": []}
    doc["rims"].append({
        "rim_x": float(rim_x), "rim_y": float(rim_y),
        "rim_radius_px": float(rim_radius_px),
        "shot_gate_px": max(gate_mult * float(rim_radius_px), 90.0),
        "f0": int(f0), "f1": (None if f1 is None else int(f1)), "note": note})
    return doc


def calib_at(doc: dict, frame: int) -> Calibration | None:
    """The Calibration whose frame range covers `frame` (last match wins on
    overlap, so a later-added correction supersedes)."""
    hit = None
    for r in doc["rims"]:
        f0, f1 = r["f0"], r["f1"]
        if frame >= f0 and (f1 is None or frame < f1):
            hit = r
    if hit is None:
        return None
    return Calibration(
        session=doc["clip"], image_w=doc["image_w"], image_h=doc["image_h"],
        rim_x=hit["rim_x"], rim_y=hit["rim_y"],
        rim_radius_px=hit["rim_radius_px"], shot_gate_px=hit["shot_gate_px"],
        note=hit.get("note", ""))


def segments(doc: dict, n_frames: int) -> list[tuple[int, int, Calibration]]:
    """Non-overlapping [f0,f1) spans each with its resolved Calibration, covering
    [0, n_frames). Built by resolving calib_at at every rim boundary — so shots
    are segmented under the correct rim for their part of the clip."""
    bounds = {0, n_frames}
    for r in doc["rims"]:
        bounds.add(max(0, r["f0"]))
        bounds.add(n_frames if r["f1"] is None else min(n_frames, r["f1"]))
    bs = sorted(b for b in bounds if 0 <= b <= n_frames)
    out = []
    for a, b in zip(bs, bs[1:]):
        if b <= a:
            continue
        c = calib_at(doc, a)
        if c is not None:
            out.append((a, b, c))
    return out


def _selftest():
    doc = add_rim("CLIP", 1920, 1080, rim_x=1134, rim_y=470, rim_radius_px=16,
                  f0=0, f1=2000, note="early")
    doc = {"clip": "CLIP", "image_w": 1920, "image_h": 1080, "rims": doc["rims"]}
    doc["rims"].append({"rim_x": 1244, "rim_y": 461, "rim_radius_px": 16,
                        "shot_gate_px": 90, "f0": 2000, "f1": None, "note": "late"})
    assert calib_at(doc, 100).rim_x == 1134, "early rim"
    assert calib_at(doc, 5000).rim_x == 1244, "late rim"
    assert calib_at(doc, 2000).rim_x == 1244, "boundary is [f0,f1)"
    segs = segments(doc, 6000)
    assert segs[0][:2] == (0, 2000) and segs[0][2].rim_x == 1134, segs
    assert segs[1][:2] == (2000, 6000) and segs[1][2].rim_x == 1244, segs
    # single fixed rim covers everything
    d2 = add_rim("C2", 1920, 1080, rim_x=900, rim_y=460, rim_radius_px=18)
    assert calib_at(d2, 0).rim_x == 900 and calib_at(d2, 99999).rim_x == 900
    assert len(segments(d2, 500)) == 1
    print("rim_segments selftest OK")


if __name__ == "__main__":
    _selftest()
