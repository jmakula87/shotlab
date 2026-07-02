"""Temporal sync of two-camera recordings from their AUDIO.

Both phones hear the same court: the session ritual is one loud clap or hard
ball-bounce after both cameras are rolling, and every dribble/rim contact after
it is a free extra anchor. Cross-correlating the two clips' loudness ONSET
envelopes (not raw samples -- phones sit at different distances with different
mics and AGC, so absolute waveforms don't match; the *timing* of loudness jumps
does) recovers the time offset between the recordings to a few milliseconds --
well under one video frame at 30 fps (33 ms).

Convention: `offset_s` is how far clip B's clock runs BEHIND clip A's, i.e. an
event at time `t` in A happens at `t - offset_s` in B. Equivalently, ADD
`offset_s` to a B timestamp to express it on A's clock.
"""

from __future__ import annotations

import numpy as np

from .audio import extract_audio, _rms_frames

ENV_MS = 5.0                      # envelope resolution (200 Hz)


def onset_envelope(samples, sr, frame_ms: float = ENV_MS) -> np.ndarray:
    """Positive loudness JUMPS per frame -- sharp transients (clap, bounce, rim
    clang) become spikes; steady noise (wind, traffic) flattens out."""
    x = np.asarray(samples, float)
    if x.size == 0:
        return np.array([])
    x = x / (np.max(np.abs(x)) + 1e-9)
    rms = _rms_frames(x, sr, frame_ms)
    d = np.diff(rms, prepend=rms[:1])
    env = np.maximum(d, 0.0)
    # keep only decisive transients: the noise floor's tiny jumps would
    # otherwise dominate the correlation on quiet/distant-mic recordings.
    # The clip's max onset is always a genuine event (the clap / loudest
    # contact), so a fraction of it is a scale-free threshold.
    m = float(env.max())
    if m <= 1e-12:
        return env
    env = np.where(env >= 0.05 * m, env, 0.0)
    return env / m


def sync_offset(samples_a, sr_a, samples_b, sr_b,
                max_offset_s: float = 60.0,
                frame_ms: float = ENV_MS) -> tuple[float, float] | None:
    """Offset between two recordings of the same scene, from audio alone.

    Returns (offset_s, confidence 0..1) or None if either side has no usable
    audio. offset_s follows the module convention (add to B's timestamps to get
    A's clock). Confidence = correlation peak prominence vs the field of other
    lags; treat < ~0.3 as a failed lock (re-sync from a clean clap).
    """
    ea = onset_envelope(samples_a, sr_a, frame_ms)
    eb = onset_envelope(samples_b, sr_b, frame_ms)
    if ea.size < 10 or eb.size < 10:
        return None
    # each envelope's TRUE frame rate: _rms_frames uses int(sr*frame_ms/1000)
    # samples per frame, so at 44.1 kHz a "5 ms" frame is really 4.9887 ms.
    # Left uncorrected the two clocks drift frames apart across a clip and no
    # single lag aligns more than one event -- resample both onto one clock.
    fr = 1000.0 / frame_ms
    fra = sr_a / max(1, int(sr_a * frame_ms / 1000.0))
    frb = sr_b / max(1, int(sr_b * frame_ms / 1000.0))
    ea = np.interp(np.arange(0, ea.size / fra, 1.0 / fr) * fra,
                   np.arange(ea.size), ea)
    eb = np.interp(np.arange(0, eb.size / frb, 1.0 / fr) * frb,
                   np.arange(eb.size), eb)
    max_lag = int(max_offset_s * fr)

    # FFT correlation of the raw sparse envelopes. No mean subtraction: the
    # thresholded envelope is near-zero almost everywhere, and subtracting a
    # mean would turn the FFT zero-padding into a big constant block whose
    # self-correlation swamps the spike alignments we're after.
    n = int(2 ** np.ceil(np.log2(ea.size + eb.size)))
    fa = np.fft.rfft(ea, n)
    fb = np.fft.rfft(eb, n)
    corr = np.fft.irfft(fa * np.conj(fb), n)
    lags = np.concatenate([np.arange(0, min(max_lag + 1, n // 2)),
                           np.arange(-min(max_lag, n // 2 - 1), 0)])
    vals = corr[lags]                                     # negative idx wraps

    k = int(np.argmax(vals))
    lag = int(lags[k])
    peak = float(vals[k])

    # sub-frame refinement: parabolic fit through the peak's neighbours
    frac = 0.0
    vm = corr[(lag - 1) % n]
    vp = corr[(lag + 1) % n]
    denom = vm - 2 * peak + vp
    if abs(denom) > 1e-12:
        frac = float(np.clip(0.5 * (vm - vp) / denom, -1.0, 1.0))

    # confidence = how decisively the winner beats the best OTHER lag (outside
    # +/-0.5 s). A true lock aligns every clap/bounce/clang at one lag, so the
    # runner-up only ever matches a single event: high ratio. Unrelated audio
    # has runner-ups as good as the winner: ~0.
    guard = int(0.5 * fr)
    field = vals[np.abs(lags - lag) > guard]
    if peak <= 1e-12:
        conf = 0.0
    elif field.size:
        second = max(float(field.max()), 0.0)
        conf = float(np.clip(1.0 - second / peak, 0.0, 1.0))
    else:
        conf = 1.0
    return (lag + frac) / fr, conf


def sync_clips(path_a: str, path_b: str, sr: int = 16000,
               max_offset_s: float = 60.0) -> tuple[float, float] | None:
    """Sync two video files by their audio tracks. Returns (offset_s,
    confidence) on A's clock, or None when either clip has no audio."""
    a, _ = extract_audio(path_a, sr)
    b, _ = extract_audio(path_b, sr)
    if a is None or b is None:
        return None
    return sync_offset(a, sr, b, sr, max_offset_s=max_offset_s)
