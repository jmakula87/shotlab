#!/usr/bin/env python
"""Export a session to a single shareable HTML file (charts + tables embedded).

Usage:
  python tools/export_report.py data/out/session_Hoops
Produces <session>/report.html -- open in any browser, no server needed.
"""
from __future__ import annotations

import base64
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _img_b64(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _shot_map_b64(df):
    """Render the per-shot release map to an embeddable PNG; None when the
    session CSV predates rim_dx_px/rim_dy_px."""
    if "rim_dx_px" not in df.columns or df["rim_dx_px"].dropna().empty:
        return None
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from shotlab.viz import draw_shot_map
    fig, ax = plt.subplots(figsize=(5.2, 5.6))
    draw_shot_map(ax, df)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _coach_thumbs(df, session_dir, raw_dirs=None):
    """A small mid-flight thumbnail (frame + ball trail) per shot, from the
    cached track. Cached to <session>/thumbs/shot_N.jpg; returns {shot_num: b64}."""
    import cv2
    from shotlab.detect_cache import _load as load_track
    raw_dirs = raw_dirs or [os.path.join("data", "raw", "Hoops"),
                            os.path.join("data", "raw")]
    tdir = os.path.join(session_dir, "thumbs")
    os.makedirs(tdir, exist_ok=True)
    thumbs = {}
    for clip, group in df.groupby("clip"):
        loaded = load_track(clip)
        by_idx = {}
        if loaded:
            _, _track, shots = loaded
            by_idx = {s.index: s for s in shots}
        raw = next((os.path.join(dd, clip) for dd in raw_dirs
                    if os.path.exists(os.path.join(dd, clip))), None)
        cap = cv2.VideoCapture(raw) if raw else None
        for _, row in group.iterrows():
            sn = int(row["shot_num"])
            fp = os.path.join(tdir, f"shot_{sn}.jpg")
            if not os.path.exists(fp) and cap is not None:
                s = by_idx.get(int(row["shot_in_clip"]))
                if s is None:
                    continue
                mid = int(s.frames[len(s.frames) // 2])
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
                ok, img = cap.read()
                if not ok:
                    continue
                for x, y in zip(s.xs, s.ys):
                    cv2.circle(img, (int(x), int(y)), 4, (0, 255, 255), -1)
                cv2.imwrite(fp, cv2.resize(img, (320, 180)),
                            [cv2.IMWRITE_JPEG_QUALITY, 80])
            if os.path.exists(fp):
                with open(fp, "rb") as f:
                    thumbs[sn] = base64.b64encode(f.read()).decode()
        if cap:
            cap.release()
    return thumbs


def _gallery(group, thumbs, metric, unit):
    cells = []
    for _, r in group.iterrows():
        b = thumbs.get(int(r["shot_num"]))
        if not b:
            continue
        v = r.get(metric)
        tag = f"#{int(r['shot_num'])}"
        if v is not None and str(v) not in ("nan", "None"):
            try:
                tag += f" · {float(v):.0f}{unit}"
            except (ValueError, TypeError):
                pass
        cells.append(f"<figure><img src='data:image/jpeg;base64,{b}'>"
                     f"<figcaption>{tag}</figcaption></figure>")
    return "<div class='gallery'>" + "".join(cells) + "</div>"


def _coaching_html(df, session_dir):
    """The coaching block: your makes vs misses, what they share / what's off,
    galleries of each, and drills. Built from the make-correlation + coach."""
    import re
    from shotlab.coach import generate_review, recommend_drills

    def _nm(v):
        return True if v in (True, "True") else (False if v in (False, "False") else None)
    made = df["made"].map(_nm) if "made" in df.columns else None
    if made is None or made.isin([True, False]).sum() < 6:
        return ""
    makes, misses = df[made == True], df[made == False]

    # the coachable levers with a KNOWN good direction, biggest gap first. Tempo
    # and balance-drift are left out here (noisy + ambiguous direction) -- they
    # still show with full stats in the make-drivers panel below.
    LEVERS = [("release_angle_deg", "release arc", "°"),
              ("knee_bend_deg", "knee bend", "°"),
              ("entry_angle_deg", "entry angle", "°"),
              ("follow_through_hold_s", "follow-through", "s")]
    diffs = []
    for col, label, unit in LEVERS:
        if col not in df.columns:
            continue
        gm, bm = makes[col].dropna(), misses[col].dropna()
        if len(gm) >= 4 and len(bm) >= 4:
            d = gm.mean() - bm.mean()
            sd = df[col].std()
            if sd and abs(d) / sd > 0.25:                 # a real gap
                diffs.append((abs(d) / sd, col, label, unit, gm.mean(), bm.mean(), d))
    diffs.sort(reverse=True)

    shared = "".join(
        f"<li><b>{label}</b>: your makes average <b>{gmean:.0f}{unit}</b> vs "
        f"<b>{bmean:.0f}{unit}</b> on misses — a {'higher' if d > 0 else 'lower'} "
        f"{label} is going in for you.</li>"
        for _eff, col, label, unit, gmean, bmean, d in diffs[:4]) or (
        "<li>No single form metric cleanly separated your makes from misses this "
        "session — your misses are more about touch than a broken mechanic.</li>")

    lead_metric = diffs[0][1] if diffs else "release_angle_deg"
    lead_unit = diffs[0][3] if diffs else "°"

    rev = generate_review(df)

    def _bold(s):
        return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    drills_html = "".join(f"<li>{_bold(dr)}</li>" for dr in recommend_drills(df))

    thumbs = _coach_thumbs(df, session_dir)
    mk_pct = (makes.shape[0] / max(1, (made.isin([True, False])).sum())) * 100

    return f"""
<div class='coach'>
<h2>🏀 Coaching — what's working and what to groove</h2>
<p class='lead'>{rev.get('summary', '')}</p>
<h3>What your makes have in common</h3><ul>{shared}</ul>
<h3>✅ Your makes ({makes.shape[0]}) — {mk_pct:.0f}% of classifiable shots</h3>
<p class='hint'>Sorted by shot number; the number under each is its {diffs[0][2] if diffs else 'release arc'}.
Look for the shared look — that's your rep to repeat.</p>
{_gallery(makes, thumbs, lead_metric, lead_unit)}
<h3>⚠️ Your misses ({misses.shape[0]}) — where the {diffs[0][2] if diffs else 'arc'} tends to slip</h3>
{_gallery(misses, thumbs, lead_metric, lead_unit)}
<h3>🎯 Work on this</h3><ul>{drills_html}</ul>
</div>"""


def _table(path, title):
    if not os.path.exists(path):
        return ""
    df = pd.read_csv(path)
    return f"<h2>{title}</h2>" + df.to_html(index=False, border=0,
                                            classes="t", justify="center")


def main(argv=None):
    d = (argv or sys.argv[1:])[0] if (argv or sys.argv[1:]) else "data/out/session_Hoops"
    shots_csv = os.path.join(d, "session_shots.csv")
    if not os.path.exists(shots_csv):
        print(f"no session_shots.csv in {d}")
        return 1
    df = pd.read_csv(shots_csv)
    from shotlab.curate import apply_excludes
    df = apply_excludes(df, d)               # drop curated junk + layups everywhere
    name = os.path.basename(d.rstrip("/\\"))

    # headline numbers
    n = len(df)
    dur = df["elapsed_min"].max() if "elapsed_min" in df else 0
    make = ""
    if "made" in df.columns:
        m = df[df["made"].isin([True, False])]
        if len(m):
            half = m["elapsed_min"].median()
            p = 100 * (m["made"] == True).mean()
            p1 = 100 * (m[m["elapsed_min"] <= half]["made"] == True).mean()
            p2 = 100 * (m[m["elapsed_min"] > half]["made"] == True).mean()
            make = (f"<div class='kpi'><b>{p:.0f}%</b><span>make (low-conf)</span></div>"
                    f"<div class='kpi'><b>{p2:.0f}%</b><span>2nd-half make "
                    f"({p2-p1:+.0f}% vs 1st)</span></div>")

    # real-feet + tempo KPIs (rim-scaled, low-conf but concrete)
    for col, lbl, unit in [("apex_above_rim_ft", "arc peak / rim", " ft"),
                           ("release_height_ft", "release ht", " ft"),
                           ("tempo_dip_to_release_s", "tempo", " s")]:
        if col in df.columns and df[col].notna().sum() >= 3:
            make += (f"<div class='kpi'><b>{df[col].mean():.2f}{unit}</b>"
                     f"<span>{lbl}</span></div>")

    # coach review (markdown -> simple HTML)
    review_html = ""
    rpath = os.path.join(d, "review.md")
    if os.path.exists(rpath):
        import re
        lines = open(rpath, encoding="utf-8").read().splitlines()
        html_lines, in_ul = [], False
        for ln in lines:
            ln = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", ln)
            if ln.startswith("### "):
                if in_ul:
                    html_lines.append("</ul>"); in_ul = False
                html_lines.append(f"<h3>{ln[4:]}</h3>")
            elif ln.startswith("- "):
                if not in_ul:
                    html_lines.append("<ul>"); in_ul = True
                html_lines.append(f"<li>{ln[2:]}</li>")
            elif ln.strip():
                if in_ul:
                    html_lines.append("</ul>"); in_ul = False
                html_lines.append(f"<p>{ln}</p>")
        if in_ul:
            html_lines.append("</ul>")
        review_html = ("<div class='review'><h2>📋 Coach review</h2>"
                       + "".join(html_lines) + "</div>")

    # make-correlation: what tracks with makes (advisory)
    make_drivers_html = ""
    if "made" in df.columns:
        import re
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from shotlab.correlate import correlate_makes, summarize_make_drivers
        def _nm(v):
            return True if v in (True, "True") else (False if v in (False, "False") else None)
        rows = df.to_dict("records")
        for r in rows:
            r["made"] = _nm(r.get("made"))
        assocs = correlate_makes(rows)
        summ = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", summarize_make_drivers(assocs))
        summ = re.sub(r"_(.+?)_", r"<i>\1</i>", summ)
        body = "".join(f"<li>{ln[2:]}</li>" if ln.startswith("- ") else f"<p>{ln}</p>"
                       for ln in summ.splitlines() if ln.strip())
        make_drivers_html = ("<div class='review'><h2>🎯 What tracks with your makes</h2>"
                             + body.replace("<li>", "<ul><li>", 1).replace(
                                 "</li><p>", "</li></ul><p>") + "</div>")

    chart = _img_b64(os.path.join(d, "session_chart.png"))
    chart_html = (f"<img src='data:image/png;base64,{chart}' style='max-width:100%'>"
                  if chart else "<p>(no chart)</p>")

    from shotlab.textbook import TEXTBOOK, grade
    tb_rows = []
    for metric, spec in TEXTBOOK.items():
        label = metric.replace("_deg", "").replace("_", " ")
        if spec["measurable_now"] and metric in df.columns and df[metric].notna().any():
            avg = float(df[metric].mean())
            g = grade(metric, avg)
            verdict = ("✅ on target" if g and g[0]
                       else f"{'+' if g and g[1] > 0 else ''}{g[1]}° off" if g else "—")
            tb_rows.append(f"<tr><td>{label}</td><td>{avg:.0f}°</td>"
                           f"<td>{spec['target']:.0f}°</td><td>{verdict}</td>"
                           f"<td class='why'>{spec['why']}</td></tr>")
        else:
            need = spec.get("needs", "not measured this session")
            tb_rows.append(f"<tr><td>{label}</td><td>—</td>"
                           f"<td>{spec['target']:.0f}°</td><td>needs 2nd camera</td>"
                           f"<td class='why'>{need}</td></tr>")
    textbook_html = (
        "<h2>📐 Textbook targets (universal — separate from your own norm)</h2>"
        "<table class='t'><tr><th>Metric</th><th>Your avg</th><th>Target</th>"
        "<th></th><th>Why it's universal</th></tr>" + "".join(tb_rows) + "</table>")

    coaching_html = _coaching_html(df, d)

    smap = _shot_map_b64(df)
    shot_map_html = ("<h2>🗺️ Shot map (release points vs the rim; dot = make, "
                     "X = miss — image-space, camera's view)</h2>"
                     f"<img src='data:image/png;base64,{smap}' "
                     "style='max-width:520px;display:block;margin:0 auto'>"
                     if smap else "")

    cols = [c for c in ["shot_num", "clip", "elapsed_min", "zone",
                        "shot_form", "shot_setup",
                        "release_angle_deg", "entry_angle_deg", "apex_height_ft",
                        "apex_above_rim_ft", "release_height_ft", "jump_height_ft",
                        "tempo_dip_to_release_s", "knee_bend_deg", "made"]
            if c in df.columns]
    shots_html = df[cols].round(2).to_html(index=False, border=0, classes="t",
                                           justify="center")

    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>ShotLab — {name}</title><style>
body{{font-family:system-ui,Arial,sans-serif;max-width:1000px;margin:24px auto;color:#1a1a1a;padding:0 16px}}
h1{{margin-bottom:4px}} .sub{{color:#777;margin-top:0}}
.kpis{{display:flex;gap:18px;flex-wrap:wrap;margin:18px 0}}
.kpi{{background:#f4f6f8;border-radius:10px;padding:14px 20px;text-align:center}}
.kpi b{{font-size:26px;display:block;color:#1e6fd8}} .kpi span{{font-size:12px;color:#666}}
table.t{{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0 20px}}
table.t th{{background:#1e6fd8;color:#fff;padding:6px 8px}}
table.t td{{padding:5px 8px;border-bottom:1px solid #eee;text-align:center}}
table.t tr:nth-child(even){{background:#fafafa}}
.review{{background:#f0f7ff;border-left:4px solid #1e6fd8;border-radius:8px;padding:6px 20px;margin:18px 0}}
.review h3{{margin:12px 0 4px;font-size:15px}} .review li{{margin:3px 0}}
.coach{{background:#f3fbf4;border-left:4px solid #2e7d32;border-radius:8px;padding:6px 20px;margin:18px 0}}
.coach h2{{margin-top:10px}} .coach h3{{margin:16px 0 4px;font-size:16px}}
.coach .lead{{font-size:15px}} .coach .hint{{color:#777;font-size:12px;margin:2px 0 8px}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px;margin:6px 0 12px}}
.gallery figure{{margin:0}} .gallery img{{width:100%;border-radius:6px;display:block}}
.gallery figcaption{{font-size:11px;color:#555;text-align:center;padding-top:2px}}
.note{{color:#777;font-size:12px;margin-top:24px;border-top:1px solid #eee;padding-top:10px}}
</style></head><body>
<h1>🏀 ShotLab — {name}</h1>
<p class='sub'>{n} shots · {dur:.0f} min session</p>
<div class='kpis'><div class='kpi'><b>{n}</b><span>shots</span></div>
<div class='kpi'><b>{dur:.0f} min</b><span>session</span></div>{make}</div>
{coaching_html}
{review_html}
{make_drivers_html}
{textbook_html}
{shot_map_html}
<h2>Metrics over the session (fatigue view)</h2>{chart_html}
{_table(os.path.join(d,'fatigue_trends.csv'),'Fatigue trends (slope/min; − = declines)')}
{_table(os.path.join(d,'consistency.csv'),'Consistency (std dev; within-zone = true repeatability)')}
{_table(os.path.join(d,'zone_summary.csv'),'By court zone')}
<h2>All shots</h2>{shots_html}
<p class='note'>Angles are foreshortened by the single-camera angle (consistent within a
zone, so per-zone trends are valid; absolute degrees are approximate until court
calibration). Make% is a low-confidence heuristic. Generated by ShotLab.</p>
</body></html>"""

    out = os.path.join(d, "report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
