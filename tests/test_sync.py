"""Audio-sync validation on synthetic two-microphone recordings with a KNOWN
offset: same court sounds, different start times, gains, sample rates, and
noise floors -- the solver must recover the offset to well under a video frame."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.sync import sync_offset, onset_envelope


def court_audio(sr, dur_s, events, gain=1.0, noise=0.005, seed=0):
    """A recording: silence + background noise + sharp decaying bursts (clap,
    bounces, rim hits) at the given times."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, noise, int(sr * dur_s))
    for t, amp in events:
        i = int(t * sr)
        if 0 <= i < len(x) - sr // 10:
            burst = amp * np.exp(-np.arange(sr // 10) / (sr * 0.01))
            x[i:i + sr // 10] += burst * rng.normal(0, 1, sr // 10)
    return x * gain


EVENTS = [(2.0, 1.0),                      # the sync clap
          (5.3, 0.4), (6.1, 0.4), (7.0, 0.5),   # dribbles
          (9.8, 0.7), (14.2, 0.6)]              # rim contacts


def test_recovers_known_offset():
    """B starts 3.37 s after A (same 48k rate)."""
    true = 3.37
    sr = 48000
    a = court_audio(sr, 20, EVENTS, seed=1)
    b = court_audio(sr, 20, [(t - true, m) for t, m in EVENTS], seed=2)
    off, conf = sync_offset(a, sr, b, sr)
    assert abs(off - true) < 0.02, off          # under a 30fps frame (33 ms)
    assert conf > 0.3, conf


def test_mixed_sample_rates_and_gain():
    """Phone A 48 kHz loud, phone B 44.1 kHz quiet + noisier -- timing still wins."""
    true = -1.85                                # B started BEFORE A
    a = court_audio(48000, 20, EVENTS, gain=1.0, seed=3)
    b = court_audio(44100, 20, [(t - true, m) for t, m in EVENTS],
                    gain=0.25, noise=0.02, seed=4)
    off, conf = sync_offset(a, 48000, b, 44100)
    assert abs(off - true) < 0.02, off
    assert conf > 0.3, conf


def test_single_clap_is_enough():
    true = 7.5
    sr = 16000
    a = court_audio(sr, 30, [(9.0, 1.0)], seed=5)
    b = court_audio(sr, 30, [(9.0 - true, 1.0)], seed=6)
    off, conf = sync_offset(a, sr, b, sr)
    assert abs(off - true) < 0.02, off


def test_no_common_events_low_confidence():
    """Unrelated recordings must not report a confident lock."""
    sr = 16000
    a = court_audio(sr, 20, [(3.0, 1.0), (11.0, 0.5)], seed=7)
    b = court_audio(sr, 20, [(6.5, 0.8), (15.0, 0.6)], seed=8)
    res = sync_offset(a, sr, b, sr)
    assert res is not None
    _, conf = res
    assert conf < 0.5, conf


def test_empty_audio_returns_none():
    assert sync_offset(np.array([]), 16000, np.zeros(16000), 16000) is None


def test_envelope_spikes_on_transients():
    sr = 16000
    x = court_audio(sr, 10, [(4.0, 1.0)], noise=0.001, seed=9)
    env = onset_envelope(x, sr)
    peak_t = int(np.argmax(env)) * 5.0 / 1000.0
    assert abs(peak_t - 4.0) < 0.05, peak_t


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} passed")
