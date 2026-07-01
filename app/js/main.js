// App wiring: pick a clip -> on-device pose per frame -> live skeleton overlay
// -> phase/metric analysis -> compare to profile -> feedback + release overlay.

import { initPose, poseBackend, detect } from "./pose.js";
import { analyzeShot, compareToProfile, feedbackLines } from "./analyze.js";
import { render, clear } from "./overlay.js";

const $ = id => document.getElementById(id);
const statusEl = $("status"), video = $("video"), canvas = $("overlay");
const ctx = canvas.getContext("2d");

let profile = null;
let frames = [];          // [{ t, lm }] collected during the analysis pass
let analysis = null;
let lastTs = -1;
let analyzing = false;

function setStatus(t) { statusEl.textContent = t; }

async function boot() {
  try {
    const backend = await initPose();
    $("engine").textContent = `engine: pose ${backend}`;
    setStatus("ready — pick a shot clip");
  } catch (e) {
    console.error(e);
    setStatus("⚠️ pose model failed to load (check connection). " + e.message);
  }
  try {
    profile = await (await fetch("profile.json")).json();
    $("profileName").textContent = `profile: ${profile.name || "default"}`;
  } catch {
    profile = { name: "none", ideal: {} };
    $("profileName").textContent = "profile: none";
  }
  if ("serviceWorker" in navigator)
    navigator.serviceWorker.register("sw.js").catch(() => {});
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
    const cls = d.within ? "good" : "bad";
    const sign = d.delta > 0 ? "+" : "";
    return `<div class="row"><span class="k">${labelOf(d.key)}</span>
      <span class="v">${d.measured}° <span class="delta ${cls}">(${sign}${d.delta}° vs ${d.ideal}°)</span></span></div>`;
  }).join("");
  const fb = feedbackLines(deltas).map(l => `<li>${l}</li>`).join("");
  rep.innerHTML = `
    <div class="card"><h2>Your form vs your ideal</h2>${rows || "<p class='k'>no metrics read</p>"}</div>
    <div class="card feedback"><h2>Feedback</h2><ul>${fb}</ul></div>`;
}

const labelOf = k => ({ elbow_at_release_deg: "Elbow at release",
                        knee_bend_deg: "Knee bend" }[k] || k);

boot();
