// Spoken feedback: turn the per-shot deltas into a short coaching cue and speak
// it aloud (browser text-to-speech). Built for the headphones-in, eyes-on-court
// flow -- you shoot and HEAR what to fix, no need to look at the phone.
//
// The cues correct back toward YOUR OWN profile ideal (which is the mean of your
// good shots), so a deviation is "off from your normal", not "wrong vs a pro".
// For metrics with a clear better direction we only cue the side worth fixing
// (e.g. not bending enough); the other side is a good deviation and stays quiet.

import { byPriority } from "./analyze.js";

// key -> { hi, lo }: the spoken cue when the measured value is ABOVE / BELOW your
// ideal. A missing side means that direction is a GOOD deviation -> no cue.
const CUES = {
  // your own release-elbow angle; either way is just "off your normal"
  elbow_angle_at_release_deg: { hi: "elbow's a little high", lo: "elbow's a little low" },
  // knee angle: HIGHER = straighter legs = less bend (the fixable side)
  knee_bend_deg: { hi: "bend your knees a bit more" },
  // dip->release seconds: HIGHER = slower into the shot
  tempo_dip_to_release_s: { hi: "a touch slow into your release" },
  // follow-through hold: LOWER = you cut it short
  follow_through_hold_s: { lo: "hold your follow-through longer" },
  // sideways hip drift: HIGHER = you drifted off balance
  balance_drift_px_per_ht: { hi: "watch your balance" },
  // (release_vs_apex_s removed as a cue -- low-confidence, sign-unstable signal;
  // 2026-07-06 final sweep)
};

const _cap = s => s.charAt(0).toUpperCase() + s.slice(1);

// deltas: the array from compareToProfile. Returns a short spoken string.
export function spokenFeedback(deltas, { max = 2 } = {}) {
  if (!deltas || !deltas.length)
    return "Couldn't read that one clearly.";
  const cues = [];
  for (const d of byPriority(deltas)) {      // drivers first, elbow last (D14c)
    if (d.fault === false || (d.fault === undefined && d.within)) continue;
    const c = CUES[d.key];
    if (!c) continue;
    const cue = d.delta > 0 ? c.hi : c.lo;   // hi = above ideal, lo = below
    if (cue) cues.push(cue);
  }
  if (!cues.length) return "Dialed. That's your shot.";
  return _cap(cues.slice(0, max).join(", and ")) + ".";
}

export function ttsSupported() {
  return typeof window !== "undefined" && "speechSynthesis" in window;
}

// Speak a phrase now, cancelling anything mid-sentence so rapid shots don't
// queue a backlog of stale feedback. Best-effort -- never throws.
export function speak(text, { rate = 1.05, pitch = 1, lang = "en-US" } = {}) {
  if (!ttsSupported() || !text) return;
  try {
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = rate; u.pitch = pitch; u.lang = lang;
    window.speechSynthesis.speak(u);
  } catch (_) { /* TTS is a nicety; never let it break the shot loop */ }
}

export function stopSpeaking() {
  if (ttsSupported()) {
    try { window.speechSynthesis.cancel(); } catch (_) {}
  }
}
