// App wiring: pick a clip -> on-device pose per frame -> live skeleton overlay
// -> phase/metric analysis -> compare to profile -> feedback + release overlay.

import { initPose, poseBackend, detect } from "./pose.js";
import { analyzeShot, compareToProfile, feedbackLines, METRIC_LABEL } from "./analyze.js";
import { render, clear } from "./overlay.js";
import { startCamera, stopCamera, ShotDetector } from "./live.js";
import { VoiceFeel } from "./voice.js";
import { collectFeelLogs, feelLogsToCsv, hasFeelLogs } from "./feelcsv.js";
import { speak, spokenFeedback, stopSpeaking } from "./say.js";
import { Enroller, matchesEnrollment } from "./enroll.js";

// spoken feedback on unless the toggle is unchecked (headphones-in flow)
const ttsOn = () => { const el = $("speakFeedback"); return !el || el.checked; };

// your enrolled body size ("scan me") so live locks onto YOU, not passers-by /
// far objects / you mid-sprint to the rebound. null = not scanned (accept all).
let enrollment = null;
try { enrollment = JSON.parse(localStorage.getItem("shotlab_enroll") || "null"); } catch (_) {}
let scanning = false;

const $ = id => document.getElementById(id);
const statusEl = $("status"), video = $("video"), canvas = $("overlay");
const ctx = canvas.getContext("2d");

// Safe default so Live / analysis never throw on a null profile if the fetch or
// pose init fails (profile.handedness was crashing the Live button otherwise).
let profile = { name: "none", ideal: {}, handedness: "right" };
let frames = [];          // [{ t, lm }] collected during the analysis pass
let analysis = null;
let lastTs = -1;
let analyzing = false;

function setStatus(t) { statusEl.textContent = t; }

const withTimeout = (p, ms, what) => Promise.race([
  p, new Promise((_, rej) => setTimeout(
    () => rej(new Error(`${what} timed out after ${ms / 1000}s`)), ms)),
]);

async function boot() {
  // Load the profile independently of pose init: a pose failure must not leave
  // the profile (and Live) dead, and vice-versa.
  try {
    profile = await (await fetch("profile.json")).json();
    if (!profile.handedness) profile.handedness = "right";
    $("profileName").textContent = `profile: ${profile.name || "default"}`;
  } catch {
    $("profileName").textContent = "profile: none";
  }
  try {
    // Timeout so a stalled GPU/WASM init surfaces as an error instead of the app
    // hanging forever on "loading pose model…".
    const backend = await withTimeout(initPose(), 30000, "pose model load");
    $("engine").textContent = `engine: pose ${backend}`;
    setStatus("ready — pick a clip or go Live");
  } catch (e) {
    console.error(e);
    $("engine").textContent = "engine: failed";
    setStatus("⚠️ pose model failed to load (check connection). " + e.message);
  }
  if ("serviceWorker" in navigator)
    navigator.serviceWorker.register("sw.js").catch(() => {});
  updateExportButton();
}

$("file").addEventListener("change", e => {
  const f = e.target.files[0];
  if (!f) return;
  video.src = URL.createObjectURL(f);
  $("stage").hidden = false;
  $("report").hidden = true;
  $("release").hidden = true;
  frames = []; analysis = null; lastTs = -1;
  video.addEventListener("loadedmetadata", () => {
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    setStatus(`loaded ${video.videoWidth}×${video.videoHeight} — press Play to analyze`);
  }, { once: true });
});

$("play").addEventListener("click", () => {
  if (analyzing) return;
  startAnalysisPass();
});

$("showIdeal").addEventListener("change", () => { if (analysis) showRelease(); });
$("release").addEventListener("click", showRelease);

// One playback pass: run pose per displayed frame, draw live, collect landmarks.
function startAnalysisPass() {
  frames = []; lastTs = -1; analyzing = true;
  $("report").hidden = true; $("release").hidden = true;
  setStatus("analyzing… (pose running on-device)");
  video.currentTime = 0;
  video.play().then(() => pump()).catch(err => {
    analyzing = false; setStatus("couldn't play: " + err.message);
  });
}

function pump() {
  if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
    video.requestVideoFrameCallback(onFrame);
  } else {
    // fallback: rAF loop keyed off currentTime
    const step = () => { onFrame(0, { mediaTime: video.currentTime }); };
    video.addEventListener("timeupdate", step);
    video._fallbackStep = step;
  }
}

function onFrame(_now, meta) {
  if (!analyzing) return;
  const ts = Math.max(lastTs + 1, Math.round((meta.mediaTime ?? video.currentTime) * 1000));
  lastTs = ts;
  let lm = null;
  try { lm = detect(video, ts); } catch (e) { console.warn("detect", e); }
  if (lm) frames.push({ t: meta.mediaTime ?? video.currentTime, lm });
  render(ctx, lm, null, false);          // live: actual skeleton only
  if (!video.ended && "requestVideoFrameCallback" in HTMLVideoElement.prototype)
    video.requestVideoFrameCallback(onFrame);
}

video.addEventListener("ended", () => {
  if (!analyzing) return;
  analyzing = false;
  if (video._fallbackStep) video.removeEventListener("timeupdate", video._fallbackStep);
  finish();
});

function finish() {
  if (frames.length < 3) {
    setStatus(`only ${frames.length} pose frames — film closer / brighter and retry`);
    return;
  }
  analysis = analyzeShot(frames, { hand: profile.handedness || "right",
                                   W: video.videoWidth, H: video.videoHeight });
  const deltas = compareToProfile(analysis.metrics, profile);
  renderReport(analysis, deltas);
  $("release").hidden = false;
  showRelease();
  setStatus(`done — ${frames.length} frames analyzed`);
}

function frameNear(t) {
  let best = frames[0], bd = Infinity;
  for (const fr of frames) { const d = Math.abs(fr.t - t); if (d < bd) { bd = d; best = fr; } }
  return best;
}

function showRelease() {
  if (!analysis) return;
  const relT = analysis.series[analysis.phases.release].t;
  const fr = frameNear(relT);
  const idealLm = (profile.skeletons && profile.skeletons.release) || null;
  const drawNow = () => render(ctx, fr.lm, idealLm, $("showIdeal").checked);
  video.pause();
  video.addEventListener("seeked", drawNow, { once: true });
  video.currentTime = relT;
  drawNow();
}

function renderReport(a, deltas) {
  const rep = $("report"); rep.hidden = false;
  const rows = deltas.map(d => {
    const cls = d.fault ? "bad" : "good";   // a better-than-ideal deviation isn't a fault
    const sign = d.delta > 0 ? "+" : "";
    const [lbl, u] = METRIC_LABEL[d.key] || [d.key, ""];
    return `<div class="row"><span class="k">${lbl}</span>
      <span class="v">${d.measured}${u} <span class="delta ${cls}">(${sign}${d.delta}${u} vs ${d.ideal}${u})</span></span></div>`;
  }).join("");
  const fb = feedbackLines(deltas).map(l => `<li>${l}</li>`).join("");
  rep.innerHTML = `
    <div class="card"><h2>Your form vs your ideal</h2>${rows || "<p class='k'>no metrics read</p>"}</div>
    <div class="card feedback"><h2>Feedback</h2><ul>${fb}</ul></div>`;
  if (ttsOn()) speak(spokenFeedback(deltas));
}

const labelOf = k => ({ elbow_angle_at_release_deg: "Elbow at release",
                        knee_bend_deg: "Knee bend",
                        tempo_dip_to_release_s: "Tempo (dip→release)" }[k] || k);

// ---------------------------------------------------------------- live mode
let liveOn = false, liveTs = -1, liveDetector = null, liveCount = 0;
let liveShots = [], voice = null, feelGood = 0, feelOff = 0, sessionId = 0;

$("live").addEventListener("click", startLive);
$("stopLive").addEventListener("click", stopLive);
$("scan").addEventListener("click", scanMe);

// "Scan me": stand at your spot, whole body in frame, ~3s -> capture your size.
async function scanMe() {
  if (poseBackend() === "unknown") {
    setStatus("⚠️ pose engine isn't ready yet — wait for “ready”."); return;
  }
  try { await startCamera(video); } catch (e) { setStatus("⚠️ " + e.message); return; }
  $("stage").hidden = false;
  canvas.width = video.videoWidth || 720; canvas.height = video.videoHeight || 1280;
  const enr = new Enroller();
  const t0 = performance.now();
  scanning = true;
  const loop = () => {
    if (!scanning) return;
    const ts = Math.round(video.currentTime * 1000);
    let lm = null; try { lm = detect(video, ts); } catch (_) {}
    render(ctx, lm, null, false);
    if (lm) enr.add(lm, canvas.height);
    const el = performance.now() - t0;
    setStatus(`🔍 Scanning — stand at your spot, whole body in view… `
              + `${(Math.min(3000, el) / 1000).toFixed(1)}s (${enr.count} reads)`);
    if (el > 3000) {
      scanning = false;
      const e = enr.finish();
      if (e) {
        enrollment = e;
        try { localStorage.setItem("shotlab_enroll", JSON.stringify(e)); } catch (_) {}
        setStatus(`✅ Locked on — got your size (${e.n} reads). Now tap 🔴 Live.`);
      } else {
        setStatus("⚠️ Couldn't get a clean full-body read — try again with your "
                  + "whole body in the frame.");
      }
      stopCamera(video);
      return;
    }
    if ("requestVideoFrameCallback" in HTMLVideoElement.prototype)
      video.requestVideoFrameCallback(loop);
    else requestAnimationFrame(loop);
  };
  loop();
}

async function startLive() {
  if (poseBackend() === "unknown") {
    setStatus("⚠️ pose engine isn't ready yet — wait for “ready” or reload.");
    return;
  }
  try {
    await startCamera(video);
  } catch (e) {
    setStatus("⚠️ " + e.message);
    return;
  }
  $("stage").hidden = false; $("report").hidden = true; $("release").hidden = true;
  $("play").hidden = true; $("stopLive").hidden = false;
  $("liveFeed").hidden = false; $("liveFeed").innerHTML = "";
  canvas.width = video.videoWidth || 720; canvas.height = video.videoHeight || 1280;
  liveOn = true; liveTs = -1; liveCount = 0; liveShots = [];
  feelGood = 0; feelOff = 0; sessionId = Date.now();
  liveDetector = new ShotDetector({ handedness: profile.handedness || "right" });
  // hands-free feel tagging: say "good" / "off" after a shot
  voice = new VoiceFeel(onFeel);
  voice.start();
  const lock = enrollment ? "🔒 locked on you" : "⚠️ not scanned — tap 🔍 Scan me "
               + "first so it tracks YOU, not passers-by";
  const mic = voice.supported ? " · 🎙 say “good”/“off” after each shot" : "";
  setStatus(`live — shoot! · ${lock}${mic}`);
  liveLoop();
}

function stopLive() {
  liveOn = false;
  stopCamera(video);
  if (voice) voice.stop();
  stopSpeaking();
  $("stopLive").hidden = true; $("play").hidden = false;
  setStatus(`live stopped — tagged ${feelGood} good · ${feelOff} off`);
}

function onFeel(feel, heard) {
  // tag the most recent shot that isn't tagged yet
  const shot = [...liveShots].reverse().find(s => !s.feel);
  if (!shot) return;
  shot.feel = feel;
  if (feel === "good") feelGood++; else feelOff++;
  const badge = feel === "good" ? "👍 good" : "👎 off";
  shot.card.querySelector("h2").innerHTML += ` <span class="delta ${
    feel === "good" ? "good" : "bad"}">${badge}</span>`;
  // persist locally so the labels survive + can be exported later
  try {
    const key = "shotlab_feel_" + sessionId;
    const log = JSON.parse(localStorage.getItem(key) || "[]");
    log.push({ n: shot.n, feel, heard, t: shot.t, metrics: shot.metrics || {} });
    localStorage.setItem(key, JSON.stringify(log));
    updateExportButton();
  } catch (_) {}
  setStatus(`heard “${heard}” → shot ${shot.n} ${badge}  (${feelGood} good · ${feelOff} off)`);
}

// ------------------------------------------------- feel-log export (CSV)
function updateExportButton() {
  try { $("exportFeel").hidden = !hasFeelLogs(localStorage); } catch (_) {}
}

$("exportFeel").addEventListener("click", () => {
  const csv = feelLogsToCsv(collectFeelLogs(localStorage));
  const blob = new Blob([csv], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "shotlab_feel_log.csv";
  a.click();
  URL.revokeObjectURL(a.href);
});

function liveLoop() {
  if (!liveOn) return;
  const t = video.currentTime;
  const ts = Math.max(liveTs + 1, Math.round(t * 1000));
  liveTs = ts;
  let lm = null;
  try { lm = detect(video, ts); } catch (e) { /* keep going */ }
  // Only track + analyze a pose that matches YOUR scanned size. This drops the
  // wrong-subject grabs (far objects, other people, you sprinting to rebound at
  // a different distance) so it stops coaching them.
  const isMe = lm && matchesEnrollment(lm, enrollment, canvas.height);
  render(ctx, isMe ? lm : null, null, false);
  if (isMe) {
    const shot = liveDetector.feed(t, lm, canvas.width, canvas.height);
    if (shot) onLiveShot(shot);
  } else if (lm && enrollment) {
    setStatus("👀 looking for you — step to your spot (whole body in frame)");
  }
  if ("requestVideoFrameCallback" in HTMLVideoElement.prototype)
    video.requestVideoFrameCallback(liveLoop);
  else requestAnimationFrame(liveLoop);
}

function onLiveShot(shot) {
  liveCount += 1;
  const a = analyzeShot(shot.frames, { hand: profile.handedness || "right",
                                       W: canvas.width, H: canvas.height });
  if (!a) return;
  const deltas = compareToProfile(a.metrics, profile);
  const off = deltas.filter(d => d.fault).length;   // count faults, not good deviations
  const head = off === 0 ? "✅ dialed" : `⚠️ ${off} off`;
  const fb = feedbackLines(deltas).map(l => `<li>${l}</li>`).join("");
  const card = document.createElement("div");
  card.className = "card feedback";
  card.innerHTML = `<h2>Shot ${liveCount} — ${head}</h2><ul>${fb}</ul>`;
  $("liveFeed").prepend(card);          // newest on top
  if (ttsOn()) speak(spokenFeedback(deltas));   // hear the fix, hands-free
  liveShots.push({ n: liveCount, card, feel: null, t: shot.releaseT,
                   metrics: a.metrics });
}

boot();
