// Hands-free feel-tagging: after a shot, just SAY how it felt ("good" / "off")
// and it tags the shot. Uses the Web Speech API (on-device in Chrome; needs
// HTTPS or localhost + mic permission). parseFeel() is pure + unit-tested.

const GOOD = ["good", "great", "nice", "money", "yes", "perfect", "clean",
              "pure", "cash", "wet", "smooth", "buttery"];
const OFF = ["off", "bad", "no", "miss", "ugly", "wrong", "gross", "brick",
             "nope", "yuck", "rushed", "short"];

// transcript -> "good" | "off" | null
export function parseFeel(transcript) {
  const t = (transcript || "").toLowerCase();
  const words = t.split(/[^a-z]+/).filter(Boolean);
  for (const w of words) {
    if (GOOD.includes(w)) return "good";
    if (OFF.includes(w)) return "off";
  }
  if (GOOD.some(g => t.includes(g))) return "good";
  if (OFF.some(o => t.includes(o))) return "off";
  return null;
}

export class VoiceFeel {
  constructor(onFeel) {
    this.onFeel = onFeel;
    this.on = false;
    const SR = (typeof window !== "undefined")
      && (window.SpeechRecognition || window.webkitSpeechRecognition);
    this.supported = !!SR;
    if (!SR) return;
    const r = new SR();
    r.continuous = true;
    r.interimResults = false;
    r.lang = "en-US";
    r.onresult = (e) => {
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const txt = e.results[i][0].transcript;
        const feel = parseFeel(txt);
        if (feel) this.onFeel(feel, txt.trim());
      }
    };
    r.onend = () => { if (this.on) { try { r.start(); } catch (_) {} } }; // keep alive
    r.onerror = () => {};
    this._r = r;
  }

  start() {
    if (this._r) { this.on = true; try { this._r.start(); } catch (_) {} }
  }

  stop() {
    this.on = false;
    if (this._r) { try { this._r.stop(); } catch (_) {} }
  }
}
