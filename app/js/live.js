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
  constructor({ handedness = "right", bufferS = 2.5, refractoryS = 1.2,
                settleS = 0.15 } = {}) {
    this.keys = sideKeys(handedness);
    this.bufferS = bufferS;
    this.refractoryS = refractoryS;
    this.settleS = settleS;
    this.buf = [];            // [{ t, lm }]
    this.lastFire = -Infinity;
  }

  // Feed one frame (t in seconds, lm = 33 landmarks, W/H = video px). Returns a
  // shot { frames, releaseT } the moment a shot completes, else null.
  feed(t, lm, W, H) {
    if (lm) this.buf.push({ t, lm });
    while (this.buf.length && t - this.buf[0].t > this.bufferS) this.buf.shift();
    if (this.buf.length < 5) return null;

    const wristY = [], noseY = [];
    for (const f of this.buf) {
      wristY.push(f.lm[this.keys.wrist].y * H);
      noseY.push(f.lm[L.nose].y * H);
    }
    const idx = releaseIndex(wristY, noseY);
    if (idx < 0) return null;

    const peakT = this.buf[idx].t;
    // fire only once the follow-through has settled and we're past refractory
    if (t - peakT >= this.settleS && peakT - this.lastFire >= this.refractoryS) {
      this.lastFire = peakT;
      return { frames: this.buf.slice(), releaseT: peakT };
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
