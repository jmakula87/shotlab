"""Audio-assisted make/miss.

Make/miss is our weakest visual signal (the ball is small and often lost at the
rim). Sound is a cheap, independent cue: a MISS usually has a loud, sharp
rim/backboard CLANG, while a MAKE is a soft swish or near-silent drop-through.
We measure how loud the moment the ball reaches the rim is, relative to the
clip's baseline, and fuse that hint with the visual classifier.

Honest limits: dribbles, talking, wind, and other courts' noise all pollute
this, so the audio hint is LOW confidence on its own -- its value is confirming
(or contradicting) the visual call. All pure-numpy; `extract_audio` shells out
to ffmpeg only for real clips.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import wave

import numpy as np


def _rms_frames(x: np.ndarray, sr: int, frame_ms: float = 10.0) -> np.ndarray:
    n = max(1, int(sr * frame_ms / 1000.0))
    if len(x) < n:
        return np.array([float(np.sqrt(np.mean(x ** 2) + 1e-12))]) if len(x) else np.array([0.0])
    nf = len(x) // n
    fr = x[:nf * n].reshape(nf, n)
    return np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12)


def rim_contact_ratio(samples, sr, rim_time_s, pre=0.1, post=0.7):
    """Peak loudness in a window around the rim moment, relative to the clip's
    baseline loudness. High = a loud contact (clang); ~1 = clean/quiet."""
    x = np.asarray(samples, float)
    if x.size == 0:
        return None
    x = x / (np.max(np.abs(x)) + 1e-9)
    baseline = float(np.median(_rms_frames(x, sr)))
    lo = int(max(0, (rim_time_s - pre) * sr))
    hi = int(min(len(x), (rim_time_s + post) * sr))
    win = x[lo:hi]
    if win.size == 0:
        return None
    peak = float(np.max(_rms_frames(win, sr)))
    return peak / (baseline + 1e-6)


def audio_make_hint(samples, sr, rim_time_s, *, loud=4.0, clean=2.0) -> dict:
    """A make/miss hint from sound alone. loud contact -> miss; clean/soft ->
    make; in between -> ambiguous."""
    ratio = rim_contact_ratio(samples, sr, rim_time_s)
    if ratio is None:
        return {"made": None, "confidence": "na", "ratio": None, "note": "no audio"}
    r = round(ratio, 2)
    if ratio >= loud:
        return {"made": False, "confidence": "low", "ratio": r,
                "note": "loud rim/backboard contact"}
    if ratio <= clean:
        return {"made": True, "confidence": "low", "ratio": r,
                "note": "clean/soft -- no hard contact"}
    return {"made": None, "confidence": "low", "ratio": r, "note": "ambiguous loudness"}


def fuse_make(visual_made, visual_conf, audio_hint) -> tuple:
    """Combine the visual call with the audio hint. Agreement bumps confidence a
    notch; disagreement keeps the visual call but drops to low confidence."""
    am = audio_hint.get("made")
    if visual_made is None:
        return (am, "low") if am is not None else (None, "na")
    if am is None:
        return (visual_made, visual_conf)
    if am == visual_made:
        bump = {"low": "medium", "na": "low"}.get(visual_conf, visual_conf)
        return (visual_made, bump)
    return (visual_made, "low")            # sight and sound disagree -> uncertain


def extract_audio(video_path: str, sr: int = 16000):
    """Decode a clip's audio to mono float32 at `sr` via ffmpeg. Returns
    (samples, sr) or (None, sr) if the clip has no audio / ffmpeg is missing."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", str(sr),
             "-vn", tmp.name],
            capture_output=True)
        if r.returncode != 0 or not os.path.getsize(tmp.name):
            return None, sr
        with wave.open(tmp.name, "rb") as w:
            frames = w.readframes(w.getnframes())
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return data, sr
    except (FileNotFoundError, OSError, wave.Error):
        return None, sr
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
