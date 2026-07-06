// Node unit test for the live-mode shot trigger. Run: node tests/test_live.mjs
import { releaseIndex, ShotDetector } from "../app/js/live.js";

let passed = 0, failed = 0;
const ok = (name, cond) => cond ? (passed++, console.log("PASS " + name))
                                : (failed++, console.log("FAIL " + name));

// --- releaseIndex: clean overhead release (image y: smaller = higher) ---
{
  const wristY = [400, 300, 150, 90, 180, 300];   // rises to a peak (90) then falls
  const noseY = wristY.map(() => 200);
  ok("release detected at the overhead peak", releaseIndex(wristY, noseY) === 3);
}
{
  const wristY = [300, 280, 260, 250, 260];        // wrist never gets above the head
  const noseY = wristY.map(() => 200);
  ok("no release when wrist stays below head", releaseIndex(wristY, noseY) === -1);
}

// --- ShotDetector: fires exactly once for a shot, respects refractory ---
function lmWith(noseY, wristY) {
  const lm = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: 1 }));
  lm[0].y = noseY;    // nose
  lm[16].y = wristY;  // right wrist
  return lm;
}
{
  const det = new ShotDetector({ handedness: "right", refractoryS: 1.2 });
  // wrist dips low, rises above head to a peak, follows through down, at 30fps
  const wristSeq = [0.8, 0.75, 0.6, 0.35, 0.15, 0.30, 0.55, 0.7, 0.75, 0.78];
  let fires = 0;
  wristSeq.forEach((wy, i) => {
    const t = i / 30;
    const shot = det.feed(t, lmWith(0.4, wy), 1, 1);   // H=1 so y is used directly
    if (shot) fires++;
  });
  ok("fires once for a single shot", fires === 1);
}
{
  // too few frames -> never fires
  const det = new ShotDetector();
  let fires = 0;
  for (let i = 0; i < 3; i++)
    if (det.feed(i / 30, lmWith(0.4, 0.2), 1, 1)) fires++;
  ok("does not fire on too-few frames", fires === 0);
}
{
  // D11: a wrist held overhead with jitter has no upstroke-from-below -> no
  // phantom shots (the old detector re-fired every 1.2s)
  const det = new ShotDetector({ handedness: "right" });
  let fires = 0;
  for (let i = 0; i < 90; i++) {                    // 3s held high, tiny jitter
    const wy = 0.15 + (i % 2) * 0.01;              // always above the nose (0.4)
    if (det.feed(i / 30, lmWith(0.4, wy), 1, 1)) fires++;
  }
  ok("no phantom fire on a held-overhead wrist", fires === 0);
}
{
  // D3: the fired shot buffers PAST the peak (through a long follow-through hold)
  // so the hold is measurable, instead of firing 0.15s after the peak
  const det = new ShotDetector({ handedness: "right" });
  const seq = [0.8, 0.7, 0.5, 0.3, 0.15, 0.16, 0.16, 0.17, 0.16, 0.17, 0.16, 0.55, 0.75];
  let shot = null;
  seq.forEach((wy, i) => { const s = det.feed(i / 30, lmWith(0.4, wy), 1, 1); if (s) shot = s; });
  ok("fires after the follow-through completes", shot !== null);
  const afterPeak = shot ? shot.frames.filter(f => f.t > shot.releaseT).length : 0;
  ok("captures follow-through frames past the peak", afterPeak >= 5);
}

console.log(`\n${passed}/${passed + failed} passed`);
process.exit(failed ? 1 : 0);
