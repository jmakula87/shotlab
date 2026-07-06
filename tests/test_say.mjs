// Node unit test for the spoken-feedback text builder. Run: node tests/test_say.mjs
import { spokenFeedback } from "../app/js/say.js";

let passed = 0, failed = 0;
const ok = (name, cond) => cond ? (passed++, console.log("PASS " + name))
                                : (failed++, console.log("FAIL " + name));

const within = (key, delta) => ({ key, delta, within: true });
const off = (key, delta) => ({ key, delta, within: false });

// all within -> dialed
ok("all within -> dialed",
   /dialed/i.test(spokenFeedback([within("knee_bend_deg", 2)])));

// no deltas read -> couldn't read
ok("empty -> couldn't read", /couldn'?t read/i.test(spokenFeedback([])));

// knee too straight (angle above ideal) -> bend more
ok("knee straight -> bend more",
   /bend your knees/i.test(spokenFeedback([off("knee_bend_deg", 15)])));

// knee DEEPER than ideal (below) is a good deviation -> no cue -> dialed
ok("knee deeper -> no correction",
   /dialed/i.test(spokenFeedback([off("knee_bend_deg", -15)])));

// elbow high and low both cue
ok("elbow high", /elbow.*high/i.test(spokenFeedback([off("elbow_angle_at_release_deg", 10)])));
ok("elbow low", /elbow.*low/i.test(spokenFeedback([off("elbow_angle_at_release_deg", -10)])));

// short follow-through (below ideal) -> hold longer; long follow (above) -> no cue
ok("short follow -> hold longer",
   /hold your follow/i.test(spokenFeedback([off("follow_through_hold_s", -0.2)])));
ok("long follow -> good deviation, dialed",
   /dialed/i.test(spokenFeedback([off("follow_through_hold_s", 0.2)])));

// balance drift (above) -> watch balance
ok("drift -> watch balance",
   /balance/i.test(spokenFeedback([off("balance_drift_px_per_ht", 0.3)])));

// release_vs_apex_s is intentionally NOT cued (low-confidence, sign-unstable;
// 2026-07-06 final sweep) -> it never speaks, even when out of band
ok("release timing is not cued (demoted)",
   /dialed/i.test(spokenFeedback([off("release_vs_apex_s", -0.1)])));

// caps at 2 cues, joined
const two = spokenFeedback([off("knee_bend_deg", 15),
                            off("elbow_angle_at_release_deg", 10),
                            off("balance_drift_px_per_ht", 0.3)]);
ok("caps at 2 cues", (two.match(/, and /g) || []).length === 1);

// starts capitalized, ends with a period
ok("well-formed sentence", /^[A-Z].*\.$/.test(
   spokenFeedback([off("knee_bend_deg", 15)])));

console.log(`\n${passed}/${passed + failed} passed`);
process.exit(failed ? 1 : 0);
