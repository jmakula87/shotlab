// Node unit test for the "scan me" enrollment gate. Run: node tests/test_enroll.mjs
import { bodyPx, matchesEnrollment, Enroller } from "../app/js/enroll.js";

let passed = 0, failed = 0;
const ok = (name, cond) => cond ? (passed++, console.log("PASS " + name))
                                : (failed++, console.log("FAIL " + name));

// build a 33-landmark array; nose at y=noseY, ankles at ankleY (normalized)
function person(noseY, ankleY, vis = 1) {
  const lm = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, z: 0, visibility: vis }));
  lm[0] = { x: 0.5, y: noseY, z: 0, visibility: vis };        // nose
  lm[27] = { x: 0.48, y: ankleY, z: 0, visibility: vis };     // l ankle
  lm[28] = { x: 0.52, y: ankleY, z: 0, visibility: vis };     // r ankle
  return lm;
}
const H = 1000;

// bodyPx = (ankleY - noseY) * H
ok("bodyPx basic", Math.abs(bodyPx(person(0.2, 0.8), H) - 600) < 1e-6);
ok("bodyPx null when nose hidden", bodyPx(person(0.2, 0.8, 0.1), H) === null);
ok("bodyPx null when no ankle", (() => {
  const p = person(0.2, 0.8); p[27].visibility = 0; p[28].visibility = 0;
  return bodyPx(p, H) === null;
})());

// enrollment via the Enroller (median of samples)
const enr = new Enroller();
for (const ay of [0.78, 0.80, 0.82, 0.80, 0.81, 0.79, 0.80, 0.80, 0.80])
  enr.add(person(0.2, ay), H);
const e = enr.finish();
ok("enroller finishes with enough reads", e && Math.abs(e.bodyPx - 600) < 25);
ok("enroller null when too few", new Enroller().finish() === null);

// matching: same size matches, far object (small) + near thing (big) don't
ok("matches same size", matchesEnrollment(person(0.2, 0.8), e, H) === true);
ok("rejects far/small object", matchesEnrollment(person(0.45, 0.55), e, H) === false); // 100px
ok("rejects near/huge object", matchesEnrollment(person(0.0, 1.0), e, H) === false); // 1000px, 1.67x
ok("accepts a slight depth wobble", matchesEnrollment(person(0.22, 0.78), e, H) === true); // 560px
ok("no enrollment -> accept all", matchesEnrollment(person(0.45, 0.55), null, H) === true);
ok("enrolled but no full body -> reject",
   matchesEnrollment(person(0.2, 0.8, 0.1), e, H) === false);

// torso fallback: when the feet clip out of frame, match on shoulder->hip span
// instead of dropping the frame (2026-07-05 audit #14).
function personFull(noseY, shY, hipY, ankleY, ankleVis = 1) {
  const lm = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, z: 0, visibility: 1 }));
  lm[0] = { x: 0.5, y: noseY, z: 0, visibility: 1 };
  lm[11] = { x: 0.46, y: shY, z: 0, visibility: 1 };
  lm[12] = { x: 0.54, y: shY, z: 0, visibility: 1 };
  lm[23] = { x: 0.47, y: hipY, z: 0, visibility: 1 };
  lm[24] = { x: 0.53, y: hipY, z: 0, visibility: 1 };
  lm[27] = { x: 0.48, y: ankleY, z: 0, visibility: ankleVis };
  lm[28] = { x: 0.52, y: ankleY, z: 0, visibility: ankleVis };
  return lm;
}
const enrT = new Enroller();
for (let i = 0; i < 9; i++) enrT.add(personFull(0.20, 0.35, 0.55, 0.80), H);
const eT = enrT.finish();
ok("enroller captures a torso reference", eT && Math.abs(eT.torsoPx - 200) < 25);
ok("feet clipped but torso matches -> accept",
   matchesEnrollment(personFull(0.20, 0.35, 0.55, 0.80, 0), eT, H) === true);
ok("feet clipped and torso too small -> reject",
   matchesEnrollment(personFull(0.40, 0.46, 0.54, 0.70, 0), eT, H) === false); // torso 80px

console.log(`\n${passed}/${passed + failed} passed`);
process.exit(failed ? 1 : 0);
