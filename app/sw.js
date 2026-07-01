// Minimal service worker: cache the local app shell so it installs and opens
// offline. MediaPipe WASM + the pose model load from their CDNs on first use
// (then the browser caches them); we don't precache those here.
const CACHE = "shotlab-formcheck-v1";
const SHELL = [
  ".", "index.html", "styles.css", "manifest.json", "icon.svg", "profile.json",
  "js/main.js", "js/pose.js", "js/analyze.js", "js/overlay.js", "js/live.js",
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
