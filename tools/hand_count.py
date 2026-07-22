"""Hand-count every shot attempt in a full clip -> process/handcount/<clip>_attempts.csv.

The ground truth for tools/eval_ablations.py. Per both adversarial reviewers
(2026-07-22): count attempts FRESH by eye -- do NOT seed the list from the
pipeline's detections (that is the selection bias that made every prior number
meaningless). Log EVERY attempt, including airballs / bad misses the rim-anchored
detector cannot see -- those size the attempt-detection prize.

For each attempt, park the playhead near the moment the ball reaches the rim
(or where it would have) and press one key:
  m = MAKE  (went in; reached rim)
  n = MISS  (hit rim/backboard; reached rim)
  b = AIRBALL / bad miss (reached neither rim nor backboard)
Each logs one attempt at the CURRENT frame. Then keep watching.

Navigate:  SPACE play/pause,  d/a = +/-1,  e/q = +/-5,  c/z = +/-30,  g = jump
Other:     u = undo last attempt,  s = save,  ESC / close window = save & quit
The CSV autosaves on every log and on quit.

Usage:
  python -X utf8 tools/hand_count.py --clip PXL_20260720_151519220
  python -X utf8 tools/hand_count.py --selftest
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from shotlab.video_io import probe

CLIP_DIR = ROOT / "data" / "raw" / "Camera 1"
HANDCOUNT_DIR = ROOT / "process" / "handcount"
FIELDS = ["attempt_id", "rim_frame", "outcome", "reached", "note"]


def csv_path(clip):
    return HANDCOUNT_DIR / f"{clip}_attempts.csv"


def save_attempts(clip, attempts):
    HANDCOUNT_DIR.mkdir(parents=True, exist_ok=True)
    with open(csv_path(clip), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for a in attempts:
            w.writerow(a)


def load_attempts(clip):
    p = csv_path(clip)
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return [{k: row[k] for k in FIELDS} for row in csv.DictReader(f)]


def add_attempt(attempts, frame, outcome, reached, note=""):
    aid = (max((int(a["attempt_id"]) for a in attempts), default=0) + 1)
    attempts.append({"attempt_id": aid, "rim_frame": int(frame),
                     "outcome": outcome, "reached": reached, "note": note})
    # keep chronological so the CSV reads in playback order
    attempts.sort(key=lambda a: int(a["rim_frame"]))
    return attempts


def gui(clip):
    import cv2
    path = CLIP_DIR / f"{clip}.mp4"
    if not path.exists():
        raise SystemExit(f"clip not found: {path}")
    info = probe(str(path))
    cap = cv2.VideoCapture(str(path))
    attempts = load_attempts(clip)
    if attempts:
        print(f"resuming: {len(attempts)} existing attempts in {csv_path(clip).name}")
    frame, playing = 0, False
    win = f"hand_count {clip}  [m]make [n]miss [b]airball [u]undo [s]save [esc]quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def read(fno):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        ok, fr = cap.read()
        return fr if ok else None

    LOG = {ord('m'): ("make", "rim"), ord('n'): ("miss", "rim"),
           ord('b'): ("miss", "airball")}
    while True:
        fr = read(frame)
        if fr is None:
            frame = max(0, frame - 1); playing = False; continue
        disp = fr.copy()
        near = [a for a in attempts if abs(int(a["rim_frame"]) - frame) <= 45]
        cv2.putText(disp, f"f {frame}/{info.n_frames}  attempts={len(attempts)}"
                    f"  {'PLAY' if playing else 'PAUSE'}", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        for a in near:
            col = (0, 220, 0) if a["outcome"] == "make" else (
                (0, 150, 255) if a["reached"] == "rim" else (0, 0, 255))
            cv2.putText(disp, f"#{a['attempt_id']} {a['outcome']}/{a['reached']}"
                        f"@{a['rim_frame']}", (12, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        cv2.imshow(win, disp)
        k = cv2.waitKey(20 if playing else 30) & 0xFF
        if k == 27 or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        elif k == ord(' '): playing = not playing
        elif k == ord('d'): frame = min(info.n_frames - 1, frame + 1); playing = False
        elif k == ord('a'): frame = max(0, frame - 1); playing = False
        elif k == ord('e'): frame = min(info.n_frames - 1, frame + 5); playing = False
        elif k == ord('q'): frame = max(0, frame - 5); playing = False
        elif k == ord('c'): frame = min(info.n_frames - 1, frame + 30); playing = False
        elif k == ord('z'): frame = max(0, frame - 30); playing = False
        elif k == ord('g'):
            try:
                frame = max(0, min(info.n_frames - 1, int(input("jump to frame #: "))))
            except (ValueError, EOFError):
                pass
            playing = False
        elif k in LOG:
            outcome, reached = LOG[k]
            add_attempt(attempts, frame, outcome, reached)
            save_attempts(clip, attempts)
            print(f"logged #{attempts[-1]['attempt_id'] if False else ''} "
                  f"{outcome}/{reached} @ frame {frame}  (total {len(attempts)})")
        elif k == ord('u') and attempts:
            # undo the attempt nearest the current frame
            i = min(range(len(attempts)),
                    key=lambda j: abs(int(attempts[j]["rim_frame"]) - frame))
            removed = attempts.pop(i); save_attempts(clip, attempts)
            print(f"undid {removed}")
        elif k == ord('s'):
            save_attempts(clip, attempts); print(f"saved {csv_path(clip)}")
        if playing:
            frame = min(info.n_frames - 1, frame + 1)
    cap.release()
    cv2.destroyAllWindows()
    save_attempts(clip, attempts)
    n_air = sum(1 for a in attempts if a["reached"] == "airball")
    n_make = sum(1 for a in attempts if a["outcome"] == "make")
    print(f"saved {csv_path(clip)}: {len(attempts)} attempts "
          f"({n_make} makes, {n_air} airballs)")


def _selftest():
    import tempfile, os
    global HANDCOUNT_DIR
    orig = HANDCOUNT_DIR
    try:
        HANDCOUNT_DIR = Path(tempfile.mkdtemp())
        att = []
        add_attempt(att, 500, "miss", "rim")
        add_attempt(att, 100, "make", "rim")       # earlier -> sorts first
        add_attempt(att, 900, "miss", "airball")
        assert [int(a["rim_frame"]) for a in att] == [100, 500, 900], att
        assert [int(a["attempt_id"]) for a in att] == [2, 1, 3], "ids stable, order by frame"
        save_attempts("T", att)
        back = load_attempts("T")
        assert len(back) == 3 and back[0]["outcome"] == "make"
        assert back[2]["reached"] == "airball"
        # eval_ablations must read this CSV shape
        from tools.eval_ablations import load_attempts as eval_load
        globals_eval = eval_load.__globals__
        globals_eval["HANDCOUNT_DIR"] = HANDCOUNT_DIR
        rows = eval_load("T")
        assert len(rows) == 3 and rows[0]["reached"] == "rim"
        os.remove(csv_path("T"))
        print("hand_count selftest OK (CSV round-trips + eval_ablations reads it)")
    finally:
        HANDCOUNT_DIR = orig


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        _selftest(); return 0
    if not args.clip:
        ap.error("--clip required (or --selftest)")
    gui(args.clip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
