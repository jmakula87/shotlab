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

// Secondary size proxy: the shoulder-midpoint -> hip-midpoint torso span. It
// survives when the ankles leave frame (the close framing recommended for form
// study), so live tracking isn't dropped whenever your feet clip out -- we fall
// back to matching torso size instead of hard-rejecting the frame.
export function torsoPx(lm, H) {
  if (!lm) return null;
  const sh = [11, 12].map(i => lm[i]).filter(p => (p.visibility ?? 1) >= 0.5);
  const hp = [23, 24].map(i => lm[i]).filter(p => (p.visibility ?? 1) >= 0.5);
  if (!sh.length || !hp.length) return null;
  const sy = sh.reduce((s, p) => s + p.y, 0) / sh.length;
  const hy = hp.reduce((s, p) => s + p.y, 0) / hp.length;
  const span = Math.abs(hy - sy) * H;
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

// Collect body-size samples over the scan, then finalize a reference. We keep
// BOTH the full-body (nose->ankle) and torso (shoulder->hip) spans so live
// matching has a fallback when the feet clip out of frame.
export class Enroller {
  constructor() { this.samples = []; this.torso = []; }
  add(lm, H) {
    const b = bodyPx(lm, H); if (b) this.samples.push(b);
    const t = torsoPx(lm, H); if (t) this.torso.push(t);
  }
  get count() { return this.samples.length; }
  // needs a handful of clean full-body reads to be trustworthy
  finish(minSamples = 8) {
    if (this.samples.length < minSamples) return null;
    return { bodyPx: median(this.samples),
             torsoPx: this.torso.length ? median(this.torso) : null,
             n: this.samples.length };
  }
}

// Is this pose plausibly YOU (matching your scanned size)? Without an
// enrollment we accept everything (old behavior). Generous band so your natural
// depth wobble at the spot still matches, but a far object / near kid doesn't.
// Prefer the full-body span; if the feet are out of frame, fall back to the
// torso span (slightly wider band, since torso alone is noisier) rather than
// dropping the frame outright.
export function matchesEnrollment(lm, enr, H, lo = 0.62, hi = 1.6) {
  if (!enr || !enr.bodyPx) return true;
  const b = bodyPx(lm, H);
  if (b !== null) { const r = b / enr.bodyPx; return r > lo && r < hi; }
  if (enr.torsoPx) {
    const t = torsoPx(lm, H);
    if (t !== null) { const r = t / enr.torsoPx; return r > lo * 0.9 && r < hi * 1.1; }
  }
  return false;                         // no usable size read -> not trustworthy
}
