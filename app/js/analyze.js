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
    const shoulder = px(lm, K.shoulder, W, H);
    const ankle = px(lm, K.ankle, W, H);
    const hipC = { x: (lm[L.l_hip].x + lm[L.r_hip].x) / 2 * W,
                   y: (lm[L.l_hip].y + lm[L.r_hip].y) / 2 * H };
    let elbow = null;
    if (visOK(lm, W, H, [K.shoulder, K.elbow, K.wrist]))
      elbow = jointAngle(shoulder, px(lm, K.elbow, W, H), wrist);
    let knee = null;
    if (visOK(lm, W, H, [K.hip, K.knee, K.ankle]))
      knee = jointAngle(px(lm, K.hip, W, H), px(lm, K.knee, W, H), ankle);
    const bodyH = Math.abs(ankle.y - shoulder.y) || null;
    return { t, wristY: wrist.y, hipY: hipC.y, hipX: hipC.x,
             shoulderY: shoulder.y, bodyH, elbow, knee };
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

  // tempo: deepest load -> release (quickness)
  const tempo = (loadIdx != null && relIdx >= loadIdx)
    ? series[relIdx].t - series[loadIdx].t : null;

  // follow-through hold: seconds after release the wrist stays at/above shoulder
  let hold = 0;
  for (let i = relIdx; i < series.length - 1; i++) {
    const s = series[i];
    if (s.shoulderY != null && s.wristY <= s.shoulderY + 0.15 * (s.bodyH || 1e9))
      hold = series[i + 1].t - series[relIdx].t;
    else break;
  }

  // balance drift: horizontal hip travel over the shot, normalized by body height
  const hips = series.map(s => s.hipX).filter(v => v != null);
  const heights = series.map(s => s.bodyH).filter(v => v != null);
  let drift = null;
  if (hips.length >= 3 && heights.length) {
    const medH = heights.slice().sort((a, b) => a - b)[Math.floor(heights.length / 2)];
    if (medH > 1) drift = (Math.max(...hips) - Math.min(...hips)) / medH;
  }

  return {
    frameCount: series.length,
    phases: { load: loadIdx, release: relIdx, follow: followIdx },
    metrics: {
      elbow_angle_at_release_deg: round1(elbowAtRelease),
      knee_bend_deg: round1(kneeBend),
      tempo_dip_to_release_s: tempo == null ? null : Math.round(tempo * 1000) / 1000,
      follow_through_hold_s: Math.round(hold * 1000) / 1000,
      balance_drift_px_per_ht: drift == null ? null : Math.round(drift * 1000) / 1000,
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

export const METRIC_LABEL = {
  elbow_angle_at_release_deg: ["Elbow at release", "°"],
  knee_bend_deg: ["Knee bend", "°"],
  tempo_dip_to_release_s: ["Tempo (dip→release)", "s"],
  follow_through_hold_s: ["Follow-through hold", "s"],
  balance_drift_px_per_ht: ["Balance drift", ""],
  release_angle_deg: ["Release angle", "°"],
  entry_angle_deg: ["Entry angle", "°"],
};

export function feedbackLines(deltas) {
  const off = deltas.filter(d => !d.within);
  if (!deltas.length)
    return ["Couldn't read your form clearly — try filming closer / better light."];
  if (!off.length) return ["✅ Dialed — everything's within your normal range."];
  return off.map(d => {
    const [lbl, u] = METRIC_LABEL[d.key] || [d.key, ""];
    const dir = d.delta > 0 ? "higher" : "lower";
    return `⚠️ ${lbl}: ${Math.abs(d.delta)}${u} ${dir} than your ideal ` +
           `(${d.measured}${u} vs ${d.ideal}${u}).`;
  });
}
