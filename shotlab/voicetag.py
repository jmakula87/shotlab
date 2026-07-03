"""Post-session voice tagging: pull good/bad + reason tags from what you SAID
during a recorded workout.

You wear a mic and, right after each shot, say a short phrase from a fixed
vocabulary ("good" / "bad" / "bad, flare" / "bad, off hand"). The phone's camera
app records your voice on the clip's audio track. Afterwards an offline
speech-to-text pass transcribes the audio, and this module maps each phrase to
the nearest shot by time and extracts the tags.

Why a fixed vocabulary: recognition on 25 min of outdoor audio is far more
reliable when constrained to a handful of known words than on free speech. Why
tag reasons at all: only the camera-BLIND ones (elbow flare, guide/off hand)
truly need your voice -- the measurable form (knee, tempo, balance) the metrics
already recover from good-vs-bad. The rest are optional feels.

This file is the pure parsing + shot-assignment core (unit-tested). The STT
engine (`transcribe_vosk`, offline + grammar-restricted to `vocab()`) is wired
but validated on the first real recorded session.
"""

from __future__ import annotations

# Outcome words. "off" is deliberately NOT a bad-synonym -- it collides with
# "off hand" / "off balance"; say "bad".
OUTCOME = {
    "good": {"good", "nice", "money", "swish"},
    "bad": {"bad", "brick", "miss"},
}

# reason keyword(s) -> canonical tag. The first two are camera-blind (the reason
# voice tagging exists); the rest are optional feels that also show in metrics.
REASONS = {
    "flare": {"flare", "flared", "flaring"},              # elbow flare (else needs 2-cam)
    "off_hand": {"off hand", "offhand", "guide hand"},    # guide-hand influence
    "short": {"short"},
    "long": {"long", "strong"},
    "rushed": {"rushed", "rush", "quick"},
    "balance": {"balance", "off balance", "leaned"},
}


def vocab() -> list[str]:
    """Every word/phrase the STT grammar should allow, so recognition is
    restricted to our vocabulary (robust on noisy outdoor audio)."""
    words = set()
    for s in OUTCOME.values():
        words |= s
    for s in REASONS.values():
        words |= s
    return sorted(words)


def parse_phrase(text):
    """A spoken phrase -> (outcome, reasons). outcome in {'good','bad',None};
    reasons = canonical tags found. Order-independent; 'bad' wins over 'good'
    if both slip in (reasons are only ever added after 'bad')."""
    t = (text or "").lower()
    outcome = None
    if any(w in t for w in OUTCOME["bad"]):
        outcome = "bad"
    elif any(w in t for w in OUTCOME["good"]):
        outcome = "good"
    reasons = [tag for tag, kws in REASONS.items() if any(k in t for k in kws)]
    return outcome, reasons


def assign_to_shots(phrases, shots, max_gap_s: float = 4.0):
    """Attach each spoken phrase to the shot it refers to.

    You speak right AFTER a shot, so a phrase maps to the most recent shot whose
    release is at or before the phrase, within `max_gap_s`. Later phrases for the
    same shot overwrite earlier ones (a spoken correction wins).

    phrases: list of dicts {t, text} (or {t, outcome, reasons}). shots: list of
    dicts {id, t} (release time, seconds, session clock). Returns
    {shot_id: {outcome, reasons, text}} for the shots that got a usable tag."""
    shots_sorted = sorted(shots, key=lambda s: s["t"])
    out = {}
    for ph in sorted(phrases, key=lambda p: p["t"]):
        if "outcome" in ph:
            outcome, reasons = ph["outcome"], ph.get("reasons", [])
        else:
            outcome, reasons = parse_phrase(ph.get("text", ""))
        if outcome is None:
            continue                                     # noise / not a tag
        # most recent shot at or before this phrase, within the gap
        best = None
        for s in shots_sorted:
            if s["t"] <= ph["t"] + 1e-9 and ph["t"] - s["t"] <= max_gap_s:
                best = s
            elif s["t"] > ph["t"]:
                break
        if best is not None:
            out[best["id"]] = {"outcome": outcome, "reasons": reasons,
                               "text": ph.get("text", "")}
    return out


def transcribe_vosk(audio_path: str, model_path: str, sr: int = 16000):
    """Offline speech-to-text over a recorded clip's audio, restricted to our
    vocabulary via a Vosk grammar (robust on outdoor noise). Returns a list of
    {t, text} phrases (t = phrase start, session-clock seconds).

    Needs `pip install vosk` and a small model dir (~50MB, offline). Wired but
    validated on the first real recorded session -- kept out of the unit tests
    (which cover the pure parser/assignment above).
    """
    import json
    from .audio import extract_audio

    try:
        from vosk import Model, KaldiRecognizer            # lazy: optional dep
    except ImportError as e:                                # pragma: no cover
        raise RuntimeError(
            "voice tagging needs Vosk: `pip install vosk` and download a small "
            "model (https://alphacephei.com/vosk/models), then pass its path."
        ) from e

    samples, _ = extract_audio(audio_path, sr)
    if samples is None:                                    # pragma: no cover
        return []
    import numpy as np
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()

    grammar = json.dumps(vocab() + ["[unk]"])
    rec = KaldiRecognizer(Model(model_path), sr, grammar)
    rec.SetWords(True)
    phrases = []
    CHUNK = sr * 2                                          # 2s of bytes
    for i in range(0, len(pcm), CHUNK * 2):
        if rec.AcceptWaveform(pcm[i:i + CHUNK * 2]):
            r = json.loads(rec.Result())
            if r.get("text") and r.get("result"):
                phrases.append({"t": r["result"][0]["start"], "text": r["text"]})
    r = json.loads(rec.FinalResult())
    if r.get("text") and r.get("result"):
        phrases.append({"t": r["result"][0]["start"], "text": r["text"]})
    return phrases
