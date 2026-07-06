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

  // release ~ ONSET of the wrist's overhead peak plateau: the first frame within
  // a small epsilon of the minimum y, NOT the global argmin. Jitter over a long
  // follow-through would otherwise place "release" anywhere in the plateau and
  // drift tempo/timing; the onset is stable and lands nearer the true release
  // (audit D6).
  let minY = Infinity, maxY = -Infinity;
  for (const s of series) {
    if (s.wristY < minY) minY = s.wristY;
    if (s.wristY > maxY) maxY = s.wristY;
  }
  const eps = Math.max(4, 0.06 * (maxY - minY));
  let relIdx = 0;
  for (let i = 0; i < series.length; i++)
    if (series[i].wristY <= minY + eps) { relIdx = i; break; }

  // load ~ deepest knee bend within ~0.8s BEFORE release (not the whole buffer,
  // so a pre-shot crouch can't masquerade as the load) (audit D6)
  const loadLo = Math.max(0, relIdx - Math.round(0.8 * fps));
  let loadIdx = null, minKnee = Infinity;
  for (let i = loadLo; i <= relIdx; i++)
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

// Metrics measured in the single-camera image plane that a wide view
// FORESHORTENS (they read high and can't be trusted against a real target until
// court-corner / 2-cam calibration). Never score or coach off these -- the repo
// marks them measurable_now:false, and scoring them was surfacing uncalibrated
// arc angles as if they were dialed-in personal targets (2026-07-05 audit).
export const CALIBRATION_GATED = new Set(["release_angle_deg", "entry_angle_deg"]);

// Which side of the ideal is an actual FAULT worth flagging. "both" = no clear
// better direction (elbow reads "off your normal" either way); "hi" = only when
// measured is ABOVE the ideal is it a fault; "lo" = only below. A better-than-
// ideal deviation on a one-sided metric is a GOOD thing and stays quiet -- the
// screen used to flag it as a fault (e.g. "balance drift lower than ideal";
// audit D13). Mirrors say.js's cue directions.
export const FAULT_SIDE = {
  elbow_angle_at_release_deg: "both",
  knee_bend_deg: "hi",              // higher angle = straighter legs = less bend
  tempo_dip_to_release_s: "hi",     // higher = slower into the shot
  follow_through_hold_s: "lo",      // lower = cut it short
  balance_drift_px_per_ht: "hi",    // higher = drifted off balance
};

// Cue priority: fix the biggest make-drivers first (follow-through, release
// timing), elbow LAST (his weakest signal). Was object-insertion order, which
// fired elbow first and crowded out follow-through (audit D14c).
export const CUE_PRIORITY = [
  "follow_through_hold_s", "release_vs_apex_s", "knee_bend_deg",
  "tempo_dip_to_release_s", "balance_drift_px_per_ht",
  "elbow_angle_at_release_deg",
];

function _isFault(key, delta, within) {
  if (within) return false;
  const side = FAULT_SIDE[key] || "both";
  return side === "both" || (side === "hi" && delta > 0) || (side === "lo" && delta < 0);
}

// Order deltas by coaching priority (drivers first), unlisted metrics last.
export function byPriority(deltas) {
  const rank = k => { const i = CUE_PRIORITY.indexOf(k); return i < 0 ? 99 : i; };
  return [...deltas].sort((a, b) => rank(a.key) - rank(b.key));
}

// Compare measured metrics to the profile's ideal targets -> deltas + feedback.
export function compareToProfile(metrics, profile) {
  const ideal = (profile && profile.ideal) || {};
  const out = [];
  for (const [key, meas] of Object.entries(metrics)) {
    if (meas == null || ideal[key] == null) continue;
    if (CALIBRATION_GATED.has(key)) continue;   // foreshortened -> not scored
    const target = ideal[key];
    const delta = round1(meas - target);
    const tol = (profile.tolerance && profile.tolerance[key]) ?? 8;
    const within = Math.abs(delta) <= tol;
    out.push({ key, measured: meas, ideal: target, delta, within,
               fault: _isFault(key, delta, within) });
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
  // only flag the FIXABLE side (a better-than-ideal deviation isn't a fault),
  // and lead with the biggest make-drivers (audit D13/D14c)
  const off = byPriority(deltas.filter(d => d.fault));
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
