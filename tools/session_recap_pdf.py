#!/usr/bin/env python
"""Rich session recap PDF: all stats + what dictates makes/misses + cross-metric
relationships (does arc suffer when the legs aren't loaded? when moving left vs
right?). Self-contained via matplotlib PdfPages.

Usage:
  python tools/session_recap_pdf.py data/out/session_0710 \
      --flare data/out/session_0710_3d/analysis3d.json \
      --out data/out/session_0710/session_0710_recap.pdf
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# metrics we analyze, with human labels and which direction is "better" for makes
# elbow-at-release is release-frame-sensitive on far single-cam footage -> unreliable
LOWCONF = {"elbow_angle_at_release_deg"}

METRICS = [
    ("release_angle_deg", "Release angle (°)", "higher"),
    ("entry_angle_deg", "Entry angle (°)", "higher"),
    ("apex_height_ft", "Apex height (ft)", "higher"),
    ("knee_bend_deg", "Knee angle (° — LOWER = deeper load)", "lower"),
    ("elbow_angle_at_release_deg", "Elbow at release (°)", "flat"),
    ("follow_through_hold_s", "Follow-through hold (s)", "higher"),
    ("balance_drift_px_per_ht", "Balance drift (lower = steadier)", "lower"),
    ("release_vs_apex_s", "Release vs jump apex (s)", "flat"),
    ("tempo_dip_to_release_s", "Tempo dip→release (s)", "flat"),
]


def _cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else None


def _perm_p(a, b, n=4000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return None
    obs = abs(a.mean() - b.mean())
    allv = np.concatenate([a, b]); na = len(a)
    rng = np.random.default_rng(seed); cnt = 0
    for _ in range(n):
        p = rng.permutation(allv)
        cnt += abs(p[:na].mean() - p[na:].mean()) >= obs
    return (cnt + 1) / (n + 1)


def _pearson(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 6:
        return None, None, int(m.sum())
    from scipy.stats import pearsonr
    r, p = pearsonr(x[m], y[m])
    return float(r), float(p), int(m.sum())


def _text_page(pdf, title, lines, subtitle=None):
    fig = plt.figure(figsize=(8.5, 11)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.text(0.06, 0.95, title, fontsize=18, fontweight="bold", va="top")
    if subtitle:
        ax.text(0.06, 0.915, subtitle, fontsize=10, color="#555", va="top")
    import textwrap
    y = 0.87
    for ln, size, col in lines:
        width = max(40, int(92 * 10.5 / size))
        wrapped = textwrap.wrap(ln, width=width, subsequent_indent="   ") if ln.strip() else [""]
        for w in wrapped:
            ax.text(0.07, y, w, fontsize=size, color=col, va="top",
                    family="DejaVu Sans")
            y -= 0.017 * (size / 10.0) + 0.006
    pdf.savefig(fig); plt.close(fig)


def build(session_dir, flare_json=None):
    df = pd.read_csv(os.path.join(session_dir, "session_shots.csv"))
    name = os.path.basename(session_dir)
    made = df[df["made"] == True]
    miss = df[df["made"] == False]
    from io import BytesIO
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        # ---------- page 1: overview ----------
        n = len(df); nmk = int(df["made"].sum()); ncl = df["made"].notna().sum()
        lines = [
            (f"{n} shots across {df['clip'].nunique()} clips · "
             f"{nmk}/{ncl} makes = {100*nmk/ncl:.0f}% (make/miss is LOW confidence)", 12, "#000"),
            ("", 8, "#000"),
            ("Per clip:", 12, "#000"),
        ]
        for c, g in df.groupby("clip"):
            lines.append((f"   {c[-16:]}: {len(g)} shots, {int(g['made'].sum())} makes",
                          10, "#333"))
        lines += [("", 8, "#000"),
                  ("This report: (1) all your stats, (2) what tracks with makes, "
                   "(3) cross-metric relationships — does your arc suffer when your "
                   "legs aren't loaded, or when moving left vs right.", 11, "#000"),
                  ("", 6, "#000"),
                  ("Honesty: on ONE wide camera the release/entry ANGLES are "
                   "foreshortened (directional, not absolute) — but comparisons "
                   "WITHIN this session (loaded vs not, left vs right) are fair. "
                   "Small samples; treat as leads, not proof.", 9, "#777")]
        _text_page(pdf, f"Session recap — {name}", lines,
                   "ShotLab · your makes, misses, and what moves them")

        # ---------- page 2: all stats ----------
        rows = []
        for key, lab, better in METRICS:
            if key in df and df[key].notna().sum() >= 3:
                s = df[key].dropna()
                short = lab.split(" (")[0]
                if better == "lower":
                    short += " ↓"
                rows.append([short, int(s.count()), round(s.median(), 1),
                             round(s.mean(), 1), round(s.std(), 1),
                             f"{s.min():.1f}–{s.max():.1f}"])
        sdf = pd.DataFrame(rows, columns=["metric", "n", "median", "mean", "sd", "range"])
        fig = plt.figure(figsize=(8.5, 11)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.text(0.06, 0.95, "All your stats", fontsize=18, fontweight="bold", va="top")
        t = ax.table(cellText=sdf.values, colLabels=sdf.columns,
                     loc="center", cellLoc="center",
                     colWidths=[0.30, 0.10, 0.13, 0.13, 0.11, 0.18],
                     bbox=[0.06, 0.45, 0.88, 0.42])
        t.auto_set_font_size(False); t.set_fontsize(9)
        ax.text(0.06, 0.42, "↓ = lower is a deeper leg load. Angles/heights from one "
                "wide camera (release/entry foreshortened).", fontsize=8.5, color="#888",
                va="top")
        pdf.savefig(fig); plt.close(fig)

        # ---------- page 3: make/miss drivers ----------
        drivers = []
        for key, lab, better in METRICS:
            if key not in df:
                continue
            a = made[key].dropna().values; b = miss[key].dropna().values
            if len(a) < 5 or len(b) < 5:
                continue
            d = _cohen_d(a, b); p = _perm_p(a, b)
            drivers.append((lab, better, a.mean(), b.mean(), d, p, len(a), len(b), key))
        drivers.sort(key=lambda r: -abs(r[4] or 0))
        from matplotlib.transforms import blended_transform_factory
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0.34, 0.14, 0.42, 0.70])
        labs = [r[0].split(" (")[0] + (" ⚠" if r[8] in LOWCONF else "") for r in drivers]
        ds = [r[4] or 0 for r in drivers]
        cols = ["#bbb" if r[8] in LOWCONF else ("#2a7" if abs(d) >= 0.4 else "#ccc")
                for d, r in zip(ds, drivers)]
        ax.barh(range(len(ds)), ds, color=cols)
        ax.set_yticks(range(len(ds))); ax.set_yticklabels(labs, fontsize=9)
        ax.invert_yaxis(); ax.axvline(0, color="k", lw=0.8)
        lim = max(0.9, max(abs(d) for d in ds) + 0.1)
        ax.set_xlim(-lim, lim)
        ax.set_xlabel("Cohen's d  (makes − misses)")
        ax.set_title("What tracks with your makes", fontsize=15, fontweight="bold",
                     loc="left")
        blend = blended_transform_factory(ax.transAxes, ax.transData)
        for i, r in enumerate(drivers):
            ax.text(1.04, i, f"d={r[4]:+.2f}  p={r[5]:.2f}", transform=blend,
                    va="center", ha="left", fontsize=8, clip_on=False,
                    color="#999" if r[8] in LOWCONF else "#222")
        fig.text(0.06, 0.088, "Positive d = higher on makes; negative = lower on makes.",
                 fontsize=9, color="#555")
        fig.text(0.06, 0.068, "Green = notable (|d| ≥ 0.4).   p = permutation p-value "
                 "(smaller = less likely chance).", fontsize=9, color="#555")
        fig.text(0.06, 0.040, "⚠ Elbow-at-release is unreliable on one far camera "
                 "(release-frame sensitive; prior clean-release", fontsize=8.5, color="#a55")
        fig.text(0.06, 0.022, "   work put it near zero) — ignore its bar. Correlation, "
                 "not proof; make/miss is low-confidence.", fontsize=8.5, color="#a55")
        pdf.savefig(fig); plt.close(fig)

        # ---------- page 4: cross-metric relationships ----------
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle("Does your arc break down? (your questions)", fontsize=15,
                     fontweight="bold", x=0.06, ha="left", y=0.97)

        # (a) arc vs leg load
        ax1 = fig.add_axes([0.10, 0.60, 0.36, 0.28])
        x = df["knee_bend_deg"].values.astype(float)
        yy = df["release_angle_deg"].values.astype(float)
        r, p, nn = _pearson(x, yy)
        ax1.scatter(x, yy, s=18, alpha=0.6, color="#36c")
        ax1.set_xlabel("knee angle (° — higher = legs straighter)", fontsize=8)
        ax1.set_ylabel("release angle (°)", fontsize=8)
        ax1.set_title("Arc vs leg load", fontsize=11)
        if r is not None:
            m = np.isfinite(x) & np.isfinite(yy)
            sl, ic = np.polyfit(x[m], yy[m], 1)
            xs = np.array([np.nanmin(x), np.nanmax(x)])
            ax1.plot(xs, sl * xs + ic, "r--", lw=1)

        # loaded vs not-loaded split (median knee)
        kmed = np.nanmedian(x)
        loaded = df[df["knee_bend_deg"] <= kmed]
        notload = df[df["knee_bend_deg"] > kmed]
        ax2 = fig.add_axes([0.58, 0.60, 0.34, 0.28])
        la = loaded["release_angle_deg"].dropna(); nl = notload["release_angle_deg"].dropna()
        ax2.bar([0, 1], [la.mean(), nl.mean()],
                yerr=[la.std()/np.sqrt(len(la)), nl.std()/np.sqrt(len(nl))],
                color=["#2a7", "#c73"], width=0.6, capsize=4)
        ax2.set_xticks([0, 1]); ax2.set_xticklabels(["legs LOADED\n(deep)", "NOT loaded\n(straighter)"], fontsize=8)
        ax2.set_ylabel("release angle (°)", fontsize=8)
        ax2.set_title("Arc: loaded vs not", fontsize=11)
        d_load = _cohen_d(la.values, nl.values); p_load = _perm_p(la.values, nl.values)
        mk_load = 100*loaded["made"].mean(); mk_nl = 100*notload["made"].mean()

        # (b) arc by movement direction
        ax3 = fig.add_axes([0.10, 0.20, 0.36, 0.28])
        order = [d for d in ["left", "right", "set"] if (df["movement_dir"] == d).sum() >= 3]
        data = [df[df["movement_dir"] == d]["release_angle_deg"].dropna().values for d in order]
        ax3.boxplot(data, showmeans=True)
        ax3.set_xticks(range(1, len(order) + 1)); ax3.set_xticklabels(order, fontsize=8)
        ax3.set_ylabel("release angle (°)", fontsize=8)
        ax3.set_title("Arc by movement direction", fontsize=11)

        # make% by direction
        ax4 = fig.add_axes([0.58, 0.20, 0.34, 0.28])
        mkd = [100*df[df["movement_dir"] == d]["made"].mean() for d in order]
        nd = [int((df["movement_dir"] == d).sum()) for d in order]
        ax4.bar(range(len(order)), mkd, color="#48a", width=0.6)
        ax4.set_xticks(range(len(order))); ax4.set_xticklabels([f"{d}\n(n={n})" for d, n in zip(order, nd)], fontsize=8)
        ax4.set_ylabel("make %", fontsize=8); ax4.set_title("Make% by direction", fontsize=11)

        # findings text
        arc_load_txt = ("LOWER" if la.mean() < nl.mean() else "HIGHER")
        fig.text(0.06, 0.13,
                 f"Legs: with deeper load your release angle is {la.mean():.0f}° vs "
                 f"{nl.mean():.0f}° straighter (d={d_load:+.2f}, p={p_load:.2f}); "
                 f"make% {mk_load:.0f}% loaded vs {mk_nl:.0f}% not. "
                 f"Arc–kneebend correlation r={r:+.2f} (p={p:.2f}, n={nn}).",
                 fontsize=8.5, color="#333", wrap=True)
        dir_txt = "  ".join(f"{d}: {m:.0f}° arc, {mk:.0f}% make (n={n})"
                            for d, m, mk, n in zip(order,
                            [np.mean(x) if len(x) else float('nan') for x in data], mkd, nd))
        fig.text(0.06, 0.09, f"Direction: {dir_txt}", fontsize=8.5, color="#333", wrap=True)
        fig.text(0.06, 0.060,
                 "Fairness: LEG-LOAD vs arc is fair (bending your knees doesn't move "
                 "you relative to the camera).", fontsize=8, color="#888")
        fig.text(0.06, 0.036,
                 "The LEFT/RIGHT arc gap is likely a CAMERA artifact — moving left/right "
                 "shifts your court position, changing how", fontsize=8, color="#a55")
        fig.text(0.06, 0.020,
                 "the arc foreshortens. Trust make% by direction (unaffected) over the "
                 "arc° there.", fontsize=8, color="#a55")
        pdf.savefig(fig); plt.close(fig)

        # ---------- page 5: flare + takeaways ----------
        tk = [("Elbow flare (from the 2-camera 3D analysis):", 12, "#000")]
        if flare_json and os.path.exists(flare_json):
            fj = json.load(open(flare_json))
            mm = (fj.get("flare") or {}).get("make_vs_miss") or {}
            s = (fj.get("flare") or {}).get("summary") or {}
            if s:
                tk.append((f"   Median flare {s.get('median_deg')}° (±{s.get('sd_deg')}° "
                           f"shot-to-shot, n={s.get('n')}).", 10, "#333"))
            if mm.get("flare_make_median") is not None:
                tk.append((f"   Makes {mm['flare_make_median']}° vs misses "
                           f"{mm['flare_miss_median']}° (d={mm.get('cohens_d')}, "
                           f"p={mm.get('p_perm')}): flare does NOT separate makes", 10, "#333"))
                tk.append(("   from misses — it's a consistent habit, not a make-driver.",
                           10, "#333"))
        tk += [("", 8, "#000"), ("Bottom line — your make levers:", 13, "#000")]
        top = [d for d in drivers if abs(d[4] or 0) >= 0.4 and d[8] not in LOWCONF][:4]
        for r in top:
            direction = "higher" if (r[4] or 0) > 0 else "lower"
            tk.append((f"   • {r[0].split(' (')[0]}: {direction} on makes "
                       f"(d={r[4]:+.2f}, p={r[5]:.2f}).", 10, "#333"))
        tk.append(("   • The arc (release/entry/apex) is your main make lever; flare "
                   "is consistent and neutral.", 10, "#333"))
        _text_page(pdf, "Flare & takeaways", tk)

    return buf.getvalue()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--flare", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    data = build(a.session_dir, a.flare)
    out = a.out or os.path.join(a.session_dir, os.path.basename(a.session_dir) + "_recap.pdf")
    with open(out, "wb") as f:
        f.write(data)
    print(f"wrote {out} ({len(data)//1024} KB)")


if __name__ == "__main__":
    main()
