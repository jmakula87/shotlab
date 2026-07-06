// Live-camera mode: run pose continuously, keep a rolling buffer, and auto-fire
// when a SHOT completes (the shooting wrist rises above the head to a peak, then
// starts back down = release + follow-through). Fires once per shot, then a
// refractory pause before the next. Enables "instant feedback after each shot".
//
// Live camera needs a secure context (HTTPS or localhost) for getUserMedia.

import { L, sideKeys } from "./analyze.js";

// Pure + unit-tested: index of a clean OVERHEAD release in a wrist/nose series
// (image y: smaller = higher). Release = the highest wrist point that is above
// the head AND a local minimum (rose to it, then fell). -1 if none.
export function releaseIndex(wristY, noseY) {
  let peak = -1, peakY = Infinity;
  for (let i = 0; i < wristY.length; i++) {
    if (wristY[i] < noseY[i] && wristY[i] < peakY) { peakY = wristY[i]; peak = i; }
  }
  if (peak <= 0 || peak >= wristY.length - 1) return -1;          // need neighbours
  const rose = wristY[peak] < wristY[peak - 1];
  const fell = wristY[peak] <= wristY[peak + 1];
  return (rose && fell) ? peak : -1;
}

export class ShotDetector {
  constructor({ handedness = "right", bufferS = 3.0, refractoryS = 1.2,
                maxSettleS = 1.5 } = {}) {
    this.keys = sideKeys(handedness);
    this.bufferS = bufferS;
    this.refractoryS = refractoryS;
    this.maxSettleS = maxSettleS;    // cap the wait for the follow-through to end
    this.buf = [];                   // [{ t, lm }]
    this.lastFire = -Infinity;
  }

  // Feed one frame (t in seconds, lm = 33 landmarks, W/H = video px). Returns a
  // shot { frames, releaseT } the moment a shot completes, else null.
  feed(t, lm, W, H) {
    if (lm) this.buf.push({ t, lm });
    while (this.buf.length && t - this.buf[0].t > this.bufferS) this.buf.shift();
    if (this.buf.length < 5 || t - this.lastFire < this.refractoryS) return null;

    const wristY = [], noseY = [], shoulderY = [];
    for (const f of this.buf) {
      wristY.push(f.lm[this.keys.wrist].y * H);
      noseY.push(f.lm[L.nose].y * H);
      shoulderY.push(f.lm[this.keys.shoulder].y * H);
    }
    const idx = releaseIndex(wristY, noseY);
    if (idx < 0) return null;

    // Require a genuine UPSTROKE: the wrist was below the head (arm down) at some
    // frame before the peak. Without this, jitter on a wrist held overhead mints
    // phantom shots (audit D11).
    let roseFromBelow = false;
    for (let i = 0; i < idx; i++) if (wristY[i] > noseY[i]) { roseFromBelow = true; break; }
    if (!roseFromBelow) return null;

    // Fire only once the FOLLOW-THROUGH is actually captured: wait until the
    // wrist comes back DOWN below the shoulder (hold complete) or we hit the max
    // wait. The old fixed 0.15s fired mid-follow-through and truncated the
    // follow-through-hold measurement -- his #1 make-driver (audit D3).
    const peakT = this.buf[idx].t;
    const wristDown = wristY[wristY.length - 1] > shoulderY[shoulderY.length - 1];
    const done = (idx < wristY.length - 1) && (wristDown || t - peakT >= this.maxSettleS);
    if (done && peakT - this.lastFire >= this.refractoryS) {
      const shot = { frames: this.buf.slice(), releaseT: peakT };
      this.lastFire = t;             // refractory from the fire, not the peak
      this.buf = [];                 // clear so a held wrist can't re-fire the shot
      return shot;
    }
    return null;
  }
}

export async function startCamera(video) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia)
    throw new Error("camera API unavailable (needs HTTPS or localhost)");
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  return stream;
}

export function stopCamera(video) {
  const s = video.srcObject;
  if (s) s.getTracks().forEach(t => t.stop());
  video.srcObject = null;
}
