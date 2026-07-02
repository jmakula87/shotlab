// Export the live feel-tag logs (localStorage "shotlab_feel_<sessionId>") as
// one CSV, so the phone labels can be merged with the desktop pipeline's
// records. Pure functions -- node-tested in tests/test_feelcsv.mjs; main.js
// wires them to a download button.

const KEY_PREFIX = "shotlab_feel_";

// Collect every feel log out of a localStorage-like object (length + key(i) +
// getItem). Returns [{sessionId, entries: [...]}] sorted oldest session first.
export function collectFeelLogs(storage) {
  const logs = [];
  for (let i = 0; i < storage.length; i++) {
    const key = storage.key(i);
    if (!key || !key.startsWith(KEY_PREFIX)) continue;
    const sessionId = Number(key.slice(KEY_PREFIX.length));
    if (!Number.isFinite(sessionId)) continue;
    try {
      const entries = JSON.parse(storage.getItem(key) || "[]");
      if (Array.isArray(entries) && entries.length)
        logs.push({ sessionId, entries });
    } catch (_) { /* a corrupt log shouldn't kill the export */ }
  }
  logs.sort((a, b) => a.sessionId - b.sessionId);
  return logs;
}

const csvCell = v => {
  if (v === null || v === undefined) return "";
  const s = String(v);
  return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
};

// One CSV across all sessions. Fixed columns first, then the union of every
// metric key seen (metric_<name>), so rows from sessions with different app
// versions still line up.
export function feelLogsToCsv(logs) {
  const metricKeys = [];
  for (const log of logs)
    for (const e of log.entries)
      for (const k of Object.keys(e.metrics || {}))
        if (!metricKeys.includes(k)) metricKeys.push(k);
  const head = ["session_utc", "session_id", "shot", "feel", "heard", "t_s",
                ...metricKeys.map(k => "metric_" + k)];
  const rows = [head];
  for (const log of logs) {
    const iso = new Date(log.sessionId).toISOString();
    for (const e of log.entries)
      rows.push([iso, log.sessionId, e.n, e.feel, e.heard,
                 e.t === null || e.t === undefined ? "" : Math.round(e.t * 1000) / 1000,
                 ...metricKeys.map(k => (e.metrics || {})[k])]);
  }
  return rows.map(r => r.map(csvCell).join(",")).join("\n") + "\n";
}

export function hasFeelLogs(storage) {
  return collectFeelLogs(storage).length > 0;
}
