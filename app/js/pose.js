// On-device pose via MediaPipe Tasks (BlazePose-33). Runs entirely in the
// browser (WASM/WebGL) -- no server, no upload. Same 33-landmark model the
// desktop pipeline uses, so metrics line up.

import { PoseLandmarker, FilesetResolver }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35";

// Keep VER in lock-step with the import URL above (static imports can't use a
// variable). 0.10.22 was never published to npm -> the import 404'd and took the
// whole module graph (and the app) down with it. 0.10.35 is the current latest.
const VER = "0.10.35";
const MODEL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

let landmarker = null;
let backend = "unknown";

async function _create(delegate) {
  const vision = await FilesetResolver.forVisionTasks(
    `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VER}/wasm`);
  return PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL, delegate },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
}

// Try GPU first (fast on phones), fall back to CPU if the device rejects it.
export async function initPose() {
  try {
    landmarker = await _create("GPU");
    backend = "GPU";
  } catch (e) {
    console.warn("GPU delegate failed, falling back to CPU:", e);
    landmarker = await _create("CPU");
    backend = "CPU";
  }
  return backend;
}

export function poseBackend() { return backend; }

// Returns the 33 landmarks [{x,y,z,visibility}] (x,y normalized 0..1) or null.
export function detect(video, timestampMs) {
  if (!landmarker) return null;
  const res = landmarker.detectForVideo(video, timestampMs);
  return (res && res.landmarks && res.landmarks.length) ? res.landmarks[0] : null;
}
