// Shot analysis from the pose timeseries: phases, release, angles, and a
// comparison against the loaded profile's ideal targets.
//
// IMPORTANT: MediaPipe x,y are normalized 0..1 over a NON-square frame, so raw
// coords are aspect-distorted. Every angle/height here is computed in PIXEL
// space (x*W, y*H) to stay correct.

export const L = {
  nose: 0,
  l_shoulder: 11, r_shoulder: 12, l_elbow: 13, r_elbow: 14,
  l_wrist: 15, r_wrist: 16, l_hip: 23, r_hip: 24,
  l_knee: 25, r_knee: 26, l_ankle: 27, r_ankle: 28,
};

export function sideKeys(hand) {
  const s = hand === "left" ? "l" : "r";
  return {
    shoulder: L[`${s}_shoulder`], elbow: L[`${s}_elbow`], wrist: L[`${s}_wrist`],
    hip: L[`${s}_hip`], knee: L[`${s}_knee`], ankle: L[`${s}_ankle`],
  };
}

const px = (lm, i, W, H) => ({ x: lm[i].x * W, y: lm[i].y * H,
                               v: lm[i].visibility ?? 1 });

export function jointAngle(a, b, c) {
  const v1 = { x: a.x - b.x, y: a.y - b.y };
  const v2 = { x: c.x - b.x, y: c.y - b.y };
  const m1 = Math.hypot(v1.x, v1.y), m2 = Math.hypot(v2.x, v2.y);
  if (m1 < 1e-6 || m2 < 1e-6) return null;
  let cos = (v1.x * v2.x + v1.y * v2.y) / (m1 * m2);
  cos = Math.max(-1, Math.min(1, cos));
  return Math.acos(cos) * 180 / Math.PI;
}

const visOK = (fp, W, H, idxs, thr = 0.4) =>
  idxs.every(i => (fp[i].visibility ?? 1) >= thr);

// frames: [{ t, lm }] where lm is the 33-landmark array. W,H = video pixels.
export function analyzeShot(frames, { hand = "right", W, H, fps = 30 } = {}) {
  const K = sideKeys(hand);
  const series = frames.map(({ t, lm }) => {
    const wrist = px(lm, K.wrist, W, H);
    const hipC = { x: (lm[L.l_hip].x + lm[L.r_hip].x) / 2 * W,
                   y: (lm[L.l_hip].y + lm[L.r_hip].y) / 2 * H };
    let elbow = null;
    if (visOK(lm, W, H, [K.shoulder, K.elbow, K.wrist]))
      elbow = jointAngle(px(lm, K.shoulder, W, H), px(lm, K.elbow, W, H),
                         px(lm, K.wrist, W, H));
    let knee = null;
    if (visOK(lm, W, H, [K.hip, K.knee, K.ankle]))
      knee = jointAngle(px(lm, K.hip, W, H), px(lm, K.knee, W, H),
                        px(lm, K.ankle, W, H));
    return { t, wristY: wrist.y, hipY: hipC.y, elbow, knee };
  });

  if (!series.length) return null;

  // release ~ highest wrist point (min y in image space)
  let relIdx = 0;
  for (let i = 1; i < series.length; i++)
    if (series[i].wristY < series[relIdx].wristY) relIdx = i;

  // load ~ deepest knee bend (min knee angle) at or before release
  let loadIdx = null, minKnee = Infinity;
  for (let i = 0; i <= relIdx; i++)
    if (series[i].knee != null && series[i].knee < minKnee) {
      minKnee = series[i].knee; loadIdx = i;
    }

  // follow-through hold: frames after release the wrist stays near/above shoulder
  const followIdx = Math.min(series.length - 1, relIdx + Math.round(0.25 * fps));

  const elbowAtRelease = _nearestNonNull(series, relIdx, "elbow");
  const kneeBend = (minKnee === Infinity) ? null : minKnee;

  return {
    frameCount: series.length,
    phases: { load: loadIdx, release: relIdx, follow: followIdx },
    metrics: {
      elbow_at_release_deg: round1(elbowAtRelease),
      knee_bend_deg: round1(kneeBend),
    },
    series,
  };
}

function _nearestNonNull(series, idx, key) {
  for (let d = 0; d < series.length; d++) {
    if (series[idx + d] && series[idx + d][key] != null) return series[idx + d][key];
    if (series[idx - d] && series[idx - d][key] != null) return series[idx - d][key];
  }
  return null;
}

const round1 = x => (x == null ? null : Math.round(x * 10) / 10);

// Compare measured metrics to the profile's ideal targets -> deltas + feedback.
export function compareToProfile(metrics, profile) {
  const ideal = (profile && profile.ideal) || {};
  const out = [];
  for (const [key, meas] of Object.entries(metrics)) {
    if (meas == null || ideal[key] == null) continue;
    const target = ideal[key];
    const delta = round1(meas - target);
    const tol = (profile.tolerance && profile.tolerance[key]) ?? 8;
    out.push({ key, measured: meas, ideal: target, delta,
               within: Math.abs(delta) <= tol });
  }
  return out;
}

export function feedbackLines(deltas) {
  const label = {
    elbow_at_release_deg: "Elbow at release",
    knee_bend_deg: "Knee bend",
  };
  const off = deltas.filter(d => !d.within);
  if (!deltas.length)
    return ["Couldn't read your form clearly — try filming closer / better light."];
  if (!off.length) return ["✅ Dialed — everything's within your normal range."];
  return off.map(d => {
    const dir = d.delta > 0 ? "more open / higher" : "more bent / lower";
    return `⚠️ ${label[d.key] || d.key}: ${Math.abs(d.delta)}° ${dir} than your ideal ` +
           `(${d.measured}° vs ${d.ideal}°).`;
  });
}
