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
.note{{color:#777;font-size:12px;margin-top:24px;border-top:1px solid #eee;padding-top:10px}}
</style></head><body>
<h1>🏀 ShotLab — {name}</h1>
<p class='sub'>{n} shots · {dur:.0f} min session</p>
<div class='kpis'><div class='kpi'><b>{n}</b><span>shots</span></div>
<div class='kpi'><b>{dur:.0f} min</b><span>session</span></div>{make}</div>
{review_html}
{make_drivers_html}
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
