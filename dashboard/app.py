"""ShotLab local dashboard.

Two views:
  - Per-clip:  the tracked-overlay video next to one clip's shot table, with
               live-tunable shot targets.
  - Session:   the whole-session analytics -- metrics over time (fatigue),
               per-zone breakdown, and make% -- built by build_session.py.

Run:  streamlit run dashboard/app.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

import altair as alt
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from shotlab.config import load_targets, evaluate  # noqa: E402

st.set_page_config(page_title="ShotLab", layout="wide")
OUT_DIR = os.path.join(ROOT, "data", "out")

_ARC_CONF = {"release_angle_deg": "high", "entry_angle_deg": "high",
             "apex_height_ft": "medium"}
_FORM_KEYS = ["elbow_angle_at_release_deg", "knee_bend_deg", "release_vs_apex_s",
              "follow_through_hold_s", "balance_drift_px_per_ht", "squareness_deg"]


# ---------------------------------------------------------------- helpers
def clip_sessions():
    if not os.path.isdir(OUT_DIR):
        return []
    return sorted(d for d in os.listdir(OUT_DIR)
                  if glob.glob(os.path.join(OUT_DIR, d, "*_shots.json")))


def session_dirs():
    if not os.path.isdir(OUT_DIR):
        return []
    return sorted(d for d in os.listdir(OUT_DIR)
                  if os.path.exists(os.path.join(OUT_DIR, d, "session_shots.csv")))


def find_overlay(d):
    for pat in ("*_overlay_h264.mp4", "*_overlay.mp4"):
        hits = glob.glob(os.path.join(d, pat))
        if hits:
            return hits[0]
    return None


def load_shots(d):
    js = glob.glob(os.path.join(d, "*_shots.json"))
    if not js:
        return pd.DataFrame()
    with open(js[0], encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))


def reflag(df, targets):
    arc, form = targets.get("arc", {}), targets.get("form", {})
    flags_col = []
    for _, row in df.iterrows():
        bad = []
        for key, conf in _ARC_CONF.items():
            fl = evaluate(key, row.get(key), arc.get(key, {}), confidence=conf)
            if fl.status in ("low", "high"):
                bad.append(f"{key.replace('_deg','').replace('_ft','')}:{fl.status}")
        for key in _FORM_KEYS:
            if key not in row:
                continue
            conf = row.get(key + "_conf", "medium")
            fl = evaluate(key, row.get(key), form.get(key, {}), confidence=conf)
            if fl.status in ("low", "high"):
                bad.append(f"{key.split('_')[0]}:{fl.status}{'?' if conf=='low' else ''}")
        flags_col.append(", ".join(bad) if bad else "ok")
    out = df.copy()
    out["flags"] = flags_col
    return out


# ---------------------------------------------------------------- per-clip view
def view_clip():
    sessions = clip_sessions()
    if not sessions:
        st.info("No processed clips yet. Run:  `python analyze.py <video> --detector motion`")
        return
    with st.sidebar:
        session = st.selectbox("Processed clip", sessions)
        d = os.path.join(OUT_DIR, session)
        st.header("Shot targets (live)")
        targets = load_targets()
        arc = targets["arc"]
        e_t = st.slider("Entry target °", 30.0, 60.0,
                        float(arc["entry_angle_deg"]["target"]), 0.5)
        e_b = st.slider("Entry band °", 30.0, 60.0,
                        (float(arc["entry_angle_deg"]["band"][0]),
                         float(arc["entry_angle_deg"]["band"][1])), 0.5)
        r_t = st.slider("Release target °", 35.0, 70.0,
                        float(arc["release_angle_deg"]["target"]), 0.5)
        r_b = st.slider("Release band °", 35.0, 70.0,
                        (float(arc["release_angle_deg"]["band"][0]),
                         float(arc["release_angle_deg"]["band"][1])), 0.5)
        targets["arc"]["entry_angle_deg"] = {"target": e_t, "band": list(e_b)}
        targets["arc"]["release_angle_deg"] = {"target": r_t, "band": list(r_b)}

    cv, ct = st.columns([3, 4])
    with cv:
        st.subheader("Tracked overlay")
        ov = find_overlay(d)
        if ov:
            st.video(ov)
            st.caption(os.path.basename(ov))
        else:
            st.warning("No overlay video for this clip.")
    with ct:
        st.subheader("Per-shot metrics")
        df = load_shots(d)
        if df.empty:
            st.warning("No shot table found.")
            return
        df = reflag(df, targets)
        arc_cols = ["shot", "release_angle_deg", "entry_angle_deg",
                    "apex_height_ft", "n_points", "fit_rmse_px", "direction"]
        form_cols = [c for c in _FORM_KEYS if c in df.columns]
        has_form = bool(form_cols)
        view = st.radio("Columns", ["Arc", "Form", "All"] if has_form else ["Arc"],
                        horizontal=True)
        show = {"Arc": arc_cols, "Form": ["shot"] + form_cols,
                "All": arc_cols + form_cols}[view] + ["flags"]
        show = [c for c in show if c in df.columns]
        sty = df[show].style.map(
            lambda v: "color:#d33;font-weight:600" if v != "ok" else "color:#2a2",
            subset=["flags"])
        st.dataframe(sty, width="stretch", hide_index=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Shots", len(df))
        if "entry_angle_deg" in df:
            c2.metric("Avg entry °", f"{df['entry_angle_deg'].mean():.1f}")
        if "release_angle_deg" in df:
            c3.metric("Avg release °", f"{df['release_angle_deg'].mean():.1f}")


# ---------------------------------------------------------------- session view
_METRIC_LABELS = {
    "release_angle_deg": "Release angle °", "entry_angle_deg": "Entry angle °",
    "apex_height_ft": "Apex height (ft)", "knee_bend_deg": "Knee bend °",
}


def _halves_make(df):
    """make% in the first vs second half of the session."""
    m = df[df["made"].isin([True, False])].copy()
    if len(m) < 4:
        return None
    half = m["elapsed_min"].median()
    a = m[m["elapsed_min"] <= half]
    b = m[m["elapsed_min"] > half]
    f = lambda x: 100 * (x["made"] == True).mean() if len(x) else float("nan")
    return f(a), f(b), len(a), len(b)


def view_session():
    sdirs = session_dirs()
    if not sdirs:
        st.info("No session built yet. Run:  "
                "`python build_session.py --clips \"data/raw/Hoops/*.mp4\" --pose`")
        return
    with st.sidebar:
        sd = st.selectbox("Session", sdirs)
    d = os.path.join(OUT_DIR, sd)
    df = pd.read_csv(os.path.join(d, "session_shots.csv"))
    if df.empty:
        st.warning("No shots in this session.")
        return

    # ---- headline KPIs ----
    k = st.columns(4)
    k[0].metric("Shots", len(df))
    k[1].metric("Session length", f"{df['elapsed_min'].max():.0f} min")
    if "made" in df.columns:
        mm = df[df["made"].isin([True, False])]
        if len(mm):
            k[2].metric("Make% (low-conf)", f"{100*(mm['made']==True).mean():.0f}%")
        hv = _halves_make(df)
        if hv:
            f1, f2, n1, n2 = hv
            k[3].metric("2nd-half make%", f"{f2:.0f}%", f"{f2-f1:+.0f}% vs 1st half",
                        delta_color="normal")
    from shotlab.session import volume_stats
    vs = volume_stats(df)
    if vs["attempts"]:
        st.caption(f"🏀 **{vs['makes']}/{vs['attempts']}** makes · "
                   f"🔥 longest make-streak **{vs['longest_make_streak']}** "
                   f"(make% is a low-confidence heuristic)")

    # ---- zone filter ----
    zones = sorted(df["zone"].dropna().unique()) if "zone" in df else []
    with st.sidebar:
        sel = st.multiselect("Zones", zones, default=zones) if zones else []
    view = df[df["zone"].isin(sel)] if sel else df

    # ---- coach review (written feedback) ----
    rpath = os.path.join(d, "review.md")
    if os.path.exists(rpath):
        with open(rpath, encoding="utf-8") as f:
            md = f.read()
        with st.container(border=True):
            st.subheader("📋 Coach review")
            st.markdown(md)

    # ---- make-correlation engine: what tracks with YOUR makes ----
    from shotlab.correlate import correlate_makes, summarize_make_drivers
    def _norm_made(v):
        if v in (True, "True"):
            return True
        if v in (False, "False"):
            return False
        return None
    rows = view.to_dict("records")
    for r in rows:
        r["made"] = _norm_made(r.get("made"))
    assocs = correlate_makes(rows)
    with st.container(border=True):
        st.subheader("🎯 What tracks with your makes")
        st.markdown(summarize_make_drivers(assocs))
        shown = [a.as_row() for a in assocs if a.confidence != "insufficient"]
        if shown:
            with st.expander("All measured associations"):
                cols = ["label", "confidence", "direction", "n_made", "n_miss",
                        "mean_made", "mean_miss", "diff", "cohen_d", "p_perm"]
                st.dataframe(pd.DataFrame(shown)[cols], hide_index=True,
                             width="stretch")

    # ---- interactive metric-over-time chart with trend ----
    st.subheader("Metric over the session (fatigue view)")
    avail = [m for m in _METRIC_LABELS if m in view.columns and view[m].notna().sum() >= 3]
    if avail:
        metric = st.selectbox("Metric", avail,
                              format_func=lambda m: _METRIC_LABELS[m])
        sub = view[["elapsed_min", metric, "zone", "clip", "made"]].dropna(subset=[metric])
        pts = alt.Chart(sub).mark_circle(size=70, opacity=0.75).encode(
            x=alt.X("elapsed_min", title="minutes into session"),
            y=alt.Y(metric, title=_METRIC_LABELS[metric],
                    scale=alt.Scale(zero=False)),
            color=alt.Color("zone", legend=alt.Legend(title="zone")),
            tooltip=["clip", "elapsed_min", metric, "zone", "made"])
        trend = pts.transform_regression("elapsed_min", metric).mark_line(
            color="#c0392b", strokeDash=[6, 4])
        st.altair_chart((pts + trend).interactive(), use_container_width=True)
        st.caption("Red dashed = linear trend. Points colored by court zone. "
                   "Within a zone the foreshortening is constant, so a zone's "
                   "trend over time is meaningful even if the absolute ° is off.")

    # ---- reference-arc overlay: your arc vs the ideal ----
    if {"release_angle_deg", "entry_angle_deg"} <= set(view.columns):
        from shotlab.coach import arc_from_angles, IDEAL_ARC
        rel = view["release_angle_deg"].dropna().mean()
        ent = view["entry_angle_deg"].dropna().mean()
        if rel == rel and ent == ent:
            st.subheader("Your arc vs the ideal")
            ux, uh = arc_from_angles(rel, ent)
            ix, ih = arc_from_angles(*IDEAL_ARC)
            adf = pd.concat([
                pd.DataFrame({"x": ux, "height": uh,
                              "arc": [f"you ({rel:.0f}°→{ent:.0f}°)"] * len(ux)}),
                pd.DataFrame({"x": ix, "height": ih,
                              "arc": [f"ideal ({IDEAL_ARC[0]:.0f}°→{IDEAL_ARC[1]:.0f}°)"] * len(ix)}),
            ])
            arc_ch = alt.Chart(adf).mark_line(strokeWidth=3).encode(
                x=alt.X("x", title="release → rim"),
                y=alt.Y("height", title="(normalized arc)"),
                color=alt.Color("arc", scale=alt.Scale(
                    range=["#2b6cb0", "#1e8449"])))
            st.altair_chart(arc_ch, use_container_width=True)
            st.caption("Reconstructed from your average release & entry angles "
                       "(foreshortened — shape is indicative). A higher, softer arc "
                       "with ~45° entry is the target.")

    # ---- per-zone breakdown ----
    if "zone" in df.columns:
        st.subheader("By court zone")
        agg = (df.groupby("zone")
               .agg(shots=("clip", "count"),
                    avg_release=("release_angle_deg", "mean"),
                    avg_entry=("entry_angle_deg", "mean"),
                    avg_apex_ft=("apex_height_ft", "mean"))
               .round(1).reset_index())
        cc = st.columns([2, 3])
        cc[0].dataframe(agg, hide_index=True, width="stretch")
        bar = alt.Chart(agg).mark_bar().encode(
            x=alt.X("zone", sort="-y"), y="shots",
            tooltip=["zone", "shots", "avg_entry"])
        cc[1].altair_chart(bar, use_container_width=True)

    # ---- by shot type (auto-tagged form + setup) ----
    if "shot_form" in df.columns and df["shot_form"].ne("unknown").any():
        st.subheader("By shot type")
        st.caption("Auto-tagged: form (jumper/layup/floater — mid/far range is a "
                   "confident jumper; near-rim calls are low-confidence) and setup "
                   "(catch-and-shoot / on-the-move / off-dribble). Heuristic.")
        cc = st.columns(2)
        fa = (df[df["shot_form"] != "unknown"].groupby("shot_form")
              .agg(shots=("clip", "count"),
                   make_pct=("made", lambda s: round(100 * (s == True).mean(), 0)))
              .reset_index())
        cc[0].dataframe(fa, hide_index=True, width="stretch")
        if "shot_setup" in df.columns and df["shot_setup"].ne("unknown").any():
            sa = (df[df["shot_setup"] != "unknown"].groupby("shot_setup")
                  .agg(shots=("clip", "count"),
                       make_pct=("made", lambda s: round(100 * (s == True).mean(), 0)))
                  .reset_index())
            cc[1].dataframe(sa, hide_index=True, width="stretch")
        else:
            cc[1].caption("Setup needs movement_dir + ball track from a freshly "
                          "built session.")

    # ---- consistency ----
    cpath = os.path.join(d, "consistency.csv")
    if os.path.exists(cpath):
        st.subheader("Consistency (how repeatable)")
        st.caption("within_zone_std = true shot-to-shot spread from a given spot "
                   "(removes the position confound). first vs second-half shows if "
                   "you get more erratic as you tire.")
        st.dataframe(pd.read_csv(cpath), hide_index=True, width="stretch")

    # ---- by movement direction (moving left vs right into the shot) ----
    if "movement_dir" in df.columns and df["movement_dir"].notna().any():
        sub = df[df["movement_dir"].isin(["left", "right", "set"])]
        if len(sub):
            st.subheader("By movement (moving into the shot)")
            agg = {"shots": ("clip", "count")}
            for m, lab in [("release_angle_deg", "avg_release"),
                           ("entry_angle_deg", "avg_entry"),
                           ("knee_bend_deg", "avg_knee")]:
                if m in sub.columns:
                    agg[lab] = (m, "mean")
            g = sub.groupby("movement_dir").agg(**agg).round(1).reset_index()
            if "made" in sub.columns:
                mk = (sub[sub["made"].isin([True, False])]
                      .groupby("movement_dir")["made"]
                      .apply(lambda s: round(100 * (s == True).mean(), 0)))
                g["make%"] = g["movement_dir"].map(mk)
            st.dataframe(g, hide_index=True, width="stretch")
            st.caption("left/right = which way you were moving as you went up. "
                       "If they read backwards for your camera, tell me and I flip "
                       "them (one-line `LEFT_RIGHT_FLIP`).")

    # ---- best shots (ideal form/arc) ----
    bpath = os.path.join(d, "best_shots.csv")
    if os.path.exists(bpath):
        st.subheader("⭐ Your best shots (ideal form/arc)")
        bdf = pd.read_csv(bpath)
        bcols = [c for c in ["shot_num", "clip", "shot_in_clip", "zone",
                            "release_angle_deg", "entry_angle_deg", "apex_height_ft",
                            "made", "why"] if c in bdf.columns]
        st.dataframe(bdf[bcols], hide_index=True, width="stretch")
        st.caption("Render their clips with `python tools/render_shots.py <clip>` "
                   "then watch them in the **Shot review** tab.")

    # ---- per-shot grades (what went wrong) ----
    gpath = os.path.join(d, "shot_grades.csv")
    if os.path.exists(gpath):
        gdf = pd.read_csv(gpath)
        off = (gdf["grade"] == "off").sum()
        st.subheader(f"Shot-by-shot ({off} of {len(gdf)} flagged 'off')")
        only_off = st.checkbox("Show only flagged shots", value=False)
        show = gdf[gdf["grade"] == "off"] if only_off else gdf
        sty = show.style.map(
            lambda v: "color:#d33;font-weight:600" if v == "off" else "color:#2a2",
            subset=["grade"])
        st.dataframe(sty, hide_index=True, width="stretch")

    with st.expander("All shots (metrics table)"):
        cols = [c for c in ["shot_num", "clip", "elapsed_min", "zone",
                            "release_angle_deg", "entry_angle_deg", "apex_height_ft",
                            "knee_bend_deg", "made", "make_conf"] if c in df.columns]
        st.dataframe(df[cols].round(2), hide_index=True, width="stretch")


def view_shot_review():
    st.subheader("Per-shot review")
    SESS = os.path.join(ROOT, "data", "sessions")
    # source A: freshly-rendered clips in data/out/<clip>/shots/ (with index.json)
    sources = {}
    for p in glob.glob(os.path.join(OUT_DIR, "*", "shots", "index.json")):
        name = os.path.basename(os.path.dirname(os.path.dirname(p)))
        sources[name] = ("index", p)
    # source B: archived review clips in data/sessions/<session>/clips/
    for d in glob.glob(os.path.join(SESS, "*", "clips")):
        mp4s = sorted(glob.glob(os.path.join(d, "*_h264.mp4")))
        if mp4s:
            name = os.path.basename(os.path.dirname(d)) + "  (archived)"
            sources[name] = ("clips", mp4s)
    if not sources:
        st.info("No per-shot clips yet. Render some with "
                "`python tools/render_shots.py <clip>`.")
        return

    clip = st.selectbox("Clip / session", sorted(sources))
    kind, payload = sources[clip]

    if kind == "index":
        with open(payload, encoding="utf-8") as f:
            shots = json.load(f)
        if not shots:
            st.warning("No shots in this clip.")
            return
        labels = {f"Shot {s['shot']} — {s.get('zone','')} "
                  f"(release {s.get('release','?')}°, entry {s.get('entry','?')}°)": s
                  for s in shots}
        pick = st.selectbox("Shot", list(labels))
        s = labels[pick]
        vid = os.path.join(os.path.dirname(payload), s["file"])
        meta = [("Release angle", f"{s.get('release','?')}°"),
                ("Entry angle", f"{s.get('entry','?')}°"),
                ("Zone", s.get("zone", "?"))]
    else:  # archived bare clips
        names = {os.path.basename(m): m for m in payload}
        pick = st.selectbox("Shot clip", sorted(names))
        vid = names[pick]
        meta = [("Clip", pick.replace("_h264.mp4", ""))]

    cv, ci = st.columns([3, 2])
    with cv:
        st.video(vid) if os.path.exists(vid) else st.warning("Clip file missing.")
    with ci:
        for lab, val in meta:
            st.metric(lab, val)
        st.caption("Red dots = tracked ball · yellow = fitted arc.")


def view_compare():
    st.subheader("Compare shots (key-phase stills)")
    comps = sorted(glob.glob(os.path.join(OUT_DIR, "comparisons", "*.png")))
    if not comps:
        st.info("No comparisons yet. Render one with:\n\n"
                "`python tools/compare_shots.py --a <clip> --shot-a N "
                "--b <clip> --shot-b M --labels good weak --out "
                "data/out/comparisons/x.png`")
        return
    pick = st.selectbox("Comparison", [os.path.basename(c) for c in comps])
    st.image(os.path.join(OUT_DIR, "comparisons", pick), width="stretch")
    st.caption("Rows = the two shots; columns = load → rise → release → "
               "follow-through. Red dots = elbow & knee (angles labeled). Angles "
               "are foreshortened, so use the visual pose difference, not the exact °.")


def view_progress():
    from shotlab.session import aggregate_sessions, consistency_progress
    agg = aggregate_sessions(OUT_DIR)
    st.subheader("Progress across sessions")
    if agg.empty or len(agg) < 1:
        st.info("No sessions built yet.")
        return
    if len(agg) == 1:
        st.caption("Only one session so far — build more (different days) to see "
                   "progress trends. Here's the baseline (avg_ = level, "
                   "std_ = spread/consistency):")
    st.dataframe(agg, hide_index=True, width="stretch")

    # consistency over time -- is your shot getting more repeatable?
    cp = consistency_progress(agg)
    if not cp.empty:
        with st.container(border=True):
            st.subheader("📈 Consistency over time (lower spread = better)")
            view = cp.copy()
            view["improving"] = view["improving"].map(
                lambda b: "✅ tighter" if b else "⚠️ wider")
            st.dataframe(view, hide_index=True, width="stretch")
            st.caption("Spread = within-zone std dev, so it isn't confounded by "
                       "shooting from different spots. A negative slope means your "
                       "shot-to-shot scatter is shrinking across sessions. "
                       "⚠️ Only meaningful when the camera setup is consistent "
                       "session-to-session — a different angle changes the "
                       "foreshortening and the absolute spread with it.")

    num = [c for c in agg.columns if c not in ("session", "date")]
    metric = st.selectbox("Track over time", num)
    sub = agg[["date", metric]].dropna()
    better = "lower is better (consistency)" if metric.startswith("std_") else ""
    if better:
        st.caption(better)
    if len(sub) >= 1:
        ch = alt.Chart(sub).mark_line(point=True).encode(
            x=alt.X("date:O", title="session date"),
            y=alt.Y(metric, scale=alt.Scale(zero=False)),
            tooltip=["date", metric])
        st.altair_chart(ch, use_container_width=True)


# ---------------------------------------------------------------- main
st.title("🏀 ShotLab")
mode = st.sidebar.radio("View", ["Per-clip", "Session analytics", "Shot review",
                                 "Compare shots", "Progress"])
if mode == "Per-clip":
    view_clip()
elif mode == "Session analytics":
    view_session()
elif mode == "Shot review":
    view_shot_review()
elif mode == "Compare shots":
    view_compare()
else:
    view_progress()

st.sidebar.divider()
st.sidebar.caption("Side-on angles (release/entry) = HIGH confidence when the "
                   "camera is square. Apex-ft = MEDIUM. Elbow flare/squareness "
                   "= LOW (1 camera). Make% = LOW until rim resolution improves.")
