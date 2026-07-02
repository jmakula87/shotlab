// Minimal service worker: cache the local app shell so it installs and opens
// offline. MediaPipe WASM + the pose model load from their CDNs on first use
// (then the browser caches them); we don't precache those here.
// Bump this on every app-shell change so phones drop the old cached files
// (cache-first below would otherwise serve a stale/broken main.js forever).
const CACHE = "shotlab-formcheck-v6";
const SHELL = [
  ".", "index.html", "styles.css", "manifest.json", "icon.svg", "profile.json",
  "js/main.js", "js/pose.js", "js/analyze.js", "js/overlay.js", "js/live.js",
  "js/voice.js", "js/feelcsv.js", "js/say.js",
];

self.addEventListener("install", e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(keys =>
    Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
  ).then(() => self.clients.claim()));
});

self.addEventListener("fetch", e => {
  const url = new URL(e.request.url);
  if (url.origin !== self.location.origin) return;      // let CDN requests pass through
  e.respondWith(caches.match(e.request).then(hit => hit || fetch(e.request)));
});
