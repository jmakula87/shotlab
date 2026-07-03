// Person enrollment ("scan me"): capture YOUR body size at your shooting spot,
// so live mode locks onto you and ignores far objects, other people, and the
// frames where you've run off to rebound. The app's pose model returns only the
// single most-prominent figure -- which isn't always you -- so we gate every
// detected pose by how well its size matches your scan before trusting it.
//
// The size proxy is the nose->(lower ankle) span in pixels: stable, needs a
// full body in frame (so object/partial detections fail it), and it naturally
// changes when you move toward/away from the camera -- which is why "at my
// shooting spot" reads as a match and "running to rebound" doesn't.

export function bodyPx(lm, H) {
  if (!lm) return null;
  const nose = lm[0];
  if ((nose.visibility ?? 1) < 0.5) return null;
  const ankles = [27, 28].map(i => lm[i]).filter(a => (a.visibility ?? 1) >= 0.5);
  if (!ankles.length) return null;
  const ay = Math.max(...ankles.map(a => a.y));
  const span = (ay - nose.y) * H;
  return span > 1 ? span : null;
}

export function poseCenterX(lm, W) {
  const hips = [23, 24].map(i => lm[i]).filter(p => (p.visibility ?? 1) >= 0.5);
  if (!hips.length) return null;
  return hips.reduce((s, p) => s + p.x, 0) / hips.length * W;
}

function median(a) {
  const s = [...a].sort((x, y) => x - y);
  return s.length ? s[Math.floor(s.length / 2)] : null;
}

// Collect body-size samples over the scan, then finalize a reference.
export class Enroller {
  constructor() { this.samples = []; }
  add(lm, H) { const b = bodyPx(lm, H); if (b) this.samples.push(b); }
  get count() { return this.samples.length; }
  // needs a handful of clean full-body reads to be trustworthy
  finish(minSamples = 8) {
    if (this.samples.length < minSamples) return null;
    return { bodyPx: median(this.samples), n: this.samples.length };
  }
}

// Is this pose plausibly YOU (matching your scanned size)? Without an
// enrollment we accept everything (old behavior). Generous band so your natural
// depth wobble at the spot still matches, but a far object / near kid doesn't.
export function matchesEnrollment(lm, enr, H, lo = 0.62, hi = 1.6) {
  if (!enr || !enr.bodyPx) return true;
  const b = bodyPx(lm, H);
  if (b === null) return false;         // no full body -> not a trustworthy subject
  const r = b / enr.bodyPx;
  return r > lo && r < hi;
}
