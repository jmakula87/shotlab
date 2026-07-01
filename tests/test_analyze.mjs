// Node test for the app's analyzeShot metrics + profile key alignment.
// Run: node tests/test_analyze.mjs
import { analyzeShot, compareToProfile } from "../app/js/analyze.js";

let passed = 0, failed = 0;
const ok = (name, cond) => cond ? (passed++, console.log("PASS " + name))
                                : (failed++, console.log("FAIL " + name));

function lm(joints) {
  const a = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: 1 }));
  for (const [k, v] of Object.entries(joints)) a[k] = { visibility: 1, ...v };
  return a;
}
// right-handed shooter; wrist rises to a release at frame 6, knee bends mid-load
const frames = [];
for (let i = 0; i < 12; i++) {
  const wr = i <= 6 ? 0.7 - i * 0.08 : 0.22 + (i - 6) * 0.04;   // rise then fall
  const kn = i <= 5 ? 0.60 + i * 0.01 : 0.62;                    // knee point
  frames.push({ t: i / 30, lm: lm({
    12: { x: 0.50, y: 0.35 },   // r_shoulder
    14: { x: 0.55, y: 0.45 },   // r_elbow
    16: { x: 0.60, y: wr },     // r_wrist
    24: { x: 0.50, y: 0.55 },   // r_hip
    23: { x: 0.45, y: 0.55 },   // l_hip
    26: { x: 0.52, y: kn },     // r_knee
    28: { x: 0.50, y: 0.75 },   // r_ankle
    0: { x: 0.50, y: 0.30 },    // nose
  }) });
}

const a = analyzeShot(frames, { hand: "right", W: 1, H: 1, fps: 30 });
ok("returns metrics", a && a.metrics);
const keys = Object.keys(a.metrics);
ok("has profile-aligned elbow key (bug fix)",
   keys.includes("elbow_angle_at_release_deg") && !keys.includes("elbow_at_release_deg"));
ok("has tempo", "tempo_dip_to_release_s" in a.metrics);
ok("has follow-through", "follow_through_hold_s" in a.metrics);
ok("has balance drift", "balance_drift_px_per_ht" in a.metrics);
ok("elbow computed (non-null)", a.metrics.elbow_angle_at_release_deg != null);

// compareToProfile now aligns on the elbow key
const prof = { ideal: { elbow_angle_at_release_deg: a.metrics.elbow_angle_at_release_deg + 20 },
               tolerance: { elbow_angle_at_release_deg: 5 } };
const deltas = compareToProfile(a.metrics, prof);
const eb = deltas.find(d => d.key === "elbow_angle_at_release_deg");
ok("compareToProfile matches the elbow key", !!eb && eb.within === false);

console.log(`\n${passed}/${passed + failed} passed`);
process.exit(failed ? 1 : 0);
