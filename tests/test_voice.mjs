// Node unit test for the voice feel-parser. Run: node tests/test_voice.mjs
import { parseFeel } from "../app/js/voice.js";

let passed = 0, failed = 0;
const ok = (name, cond) => cond ? (passed++, console.log("PASS " + name))
                                : (failed++, console.log("FAIL " + name));

ok("plain good", parseFeel("good") === "good");
ok("plain off", parseFeel("off") === "off");
ok("phrase good", parseFeel("that felt really good") === "good");
ok("phrase off", parseFeel("no that was a brick") === "off");
ok("slang good", parseFeel("money") === "good");
ok("slang off", parseFeel("that was gross") === "off");
ok("case-insensitive", parseFeel("GOOD") === "good");
ok("unrelated -> null", parseFeel("what time is it") === null);
ok("empty -> null", parseFeel("") === null);
ok("good wins if first", parseFeel("good not bad") === "good");

console.log(`\n${passed}/${passed + failed} passed`);
process.exit(failed ? 1 : 0);
