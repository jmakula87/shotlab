// Node unit test for the feel-log CSV exporter. Run: node tests/test_feelcsv.mjs
import { collectFeelLogs, feelLogsToCsv, hasFeelLogs } from "../app/js/feelcsv.js";

let passed = 0, failed = 0;
const ok = (name, cond) => cond ? (passed++, console.log("PASS " + name))
                                : (failed++, console.log("FAIL " + name));

// a localStorage stand-in
function fakeStorage(obj) {
  const keys = Object.keys(obj);
  return { length: keys.length, key: i => keys[i], getItem: k => obj[k] };
}

const s1 = 1751400000000, s2 = 1751490000000;
const store = fakeStorage({
  ["shotlab_feel_" + s2]: JSON.stringify([
    { n: 1, feel: "good", heard: "good", t: 3.4567,
      metrics: { elbow_angle_at_release_deg: 95.2 } },
  ]),
  ["shotlab_feel_" + s1]: JSON.stringify([
    { n: 2, feel: "off", heard: "that was a brick", t: 8.1,
      metrics: { knee_bend_deg: 120.5 } },
  ]),
  shotlab_feel_corrupt: "{not json",
  unrelated_key: "x",
});

const logs = collectFeelLogs(store);
ok("collects only valid feel logs", logs.length === 2);
ok("sorted oldest first", logs[0].sessionId === s1 && logs[1].sessionId === s2);
ok("hasFeelLogs true", hasFeelLogs(store) === true);
ok("hasFeelLogs false on empty", hasFeelLogs(fakeStorage({})) === false);

const csv = feelLogsToCsv(logs);
const lines = csv.trim().split("\n");
ok("header + one row per entry", lines.length === 3);
ok("metric union in header", lines[0].includes("metric_knee_bend_deg")
                          && lines[0].includes("metric_elbow_angle_at_release_deg"));
ok("session iso stamped", lines[1].startsWith(new Date(s1).toISOString()));
ok("t rounded to ms", lines[2].includes("3.457"));
ok("heard text with spaces survives", lines[1].includes("that was a brick"));
// a metric absent from a session leaves an empty cell, not a shifted row
const header = lines[0].split(",").length;
ok("rows match header width", lines[1].split(",").length === header
                           && lines[2].split(",").length === header);
// quoting: a comma inside a heard phrase must not split the row
const q = feelLogsToCsv([{ sessionId: s1, entries: [
  { n: 1, feel: "good", heard: "yes, money", t: 1, metrics: {} }] }]);
ok("comma in text is quoted", q.includes('"yes, money"'));

console.log(`\n${passed}/${passed + failed} passed`);
process.exit(failed ? 1 : 0);
