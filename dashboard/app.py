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
sys.path.insert(0, os.path.join(ROOT, "tools"))   # export_pdf / export_report

from shotlab.config import load_targets, evaluate  # noqa: E402


@st.cache_data(show_spinner="building PDF…")
def _pdf_bytes(d, _mtime):
    from export_pdf import build_session_pdf
    return build_session_pdf(d)

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
    "apex_above_rim_ft": "Arc peak above rim (ft)",
    "release_height_ft": "Release height (ft)", "jump_height_ft": "Jump height (ft)",
    "tempo_dip_to_release_s": "Tempo dip→release (s)",
    "elbow_angle_at_release_deg": "Elbow at release °",
    "follow_through_hold_s": "Follow-through hold (s)",
}


def _norm_made(v):
    if v in (True, "True"):
        return True
    if v in (False, "False"):
        return False
    return None


def court_chart(df):
    """Half-court shot chart (delegates to shotlab.viz.draw_court)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from shotlab.viz import draw_court
    fig, ax = plt.subplots(figsize=(5.2, 5.6))
    draw_court(ax, df)
    fig.tight_layout()
    return fig


def shot_map_chart(df):
    """Per-shot release-point map (delegates to shotlab.viz.draw_shot_map)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from shotlab.viz import draw_shot_map
    fig, ax = plt.subplots(figsize=(5.2, 5.6))
    draw_shot_map(ax, df)
    fig.tight_layout()
    return fig


def _textbook_panel(df):
    """Universal (textbook) targets vs your session average -- shown SEPARATELY
    from your personal norm so the two never blend."""
    from shotlab.textbook import TEXTBOOK, grade
    rows = []
    for metric, spec in TEXTBOOK.items():
        label = metric.replace("_deg", "").replace("_", " ")
        if spec["measurable_now"] and metric in df.columns and df[metric].notna().sum() >= 3:
            avg = float(df[metric].mean())
            g = grade(metric, avg)
            verdict = "✅ on target" if (g and g[0]) else (f"{g[1]:+.0f}° off" if g else "—")
            rows.append({"metric": label, "your avg": f"{avg:.0f}°",
                         "target": f"{spec['target']:.0f}°", "status": verdict,
                         "why it's universal": spec["why"]})
        else:
            rows.append({"metric": label, "your avg": "—",
                         "target": f"{spec['target']:.0f}°",
                         "status": f"needs {spec.get('blocked_by', '2nd camera')}",
                         "why it's universal": spec.get("needs", spec["why"])})
    with st.container(border=True):
        st.subheader("📐 Textbook targets (universal — separate from your own norm)")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("These few numbers are the same for everyone (physics); your "
                   "personal ideal (the mean of your good shots) is what everything "
                   "else compares to. Body-form angles stay personal on purpose.")


def _clean(v):
    try:
        import numpy as _np
        if isinstance(v, (_np.floating,)):
            v = float(v)
        if isinstance(v, (_np.integer,)):
            v = int(v)
    except Exception:
        pass
    if isinstance(v, float) and v != v:
        return None
    return v


def _shot_inspector(view):
    """Pick a shot row -> full metrics + play its rendered clip if one exists."""
    show = [c for c in ["shot_num", "clip", "zone", "shot_form", "shot_setup",
                        "made", "release_angle_deg", "entry_angle_deg",
                        "tempo_dip_to_release_s", "apex_above_rim_ft"]
            if c in view.columns]
    if not show or "shot_in_clip" not in view.columns:
        return
    with st.container(border=True):
        st.subheader("🔎 Shot inspector")
        st.caption("Click a row to see its full metrics + clip.")
        vv = view.reset_index(drop=True)
        ev = st.dataframe(vv[show], hide_index=True, width="stretch",
                          on_select="rerun", selection_mode="single-row",
                          key="shot_inspect")
        rows = []
        try:
            rows = ev.selection.rows
        except Exception:
            rows = (ev.get("selection", {}) or {}).get("rows", []) if ev else []
        if not rows:
            return
        r = vv.iloc[rows[0]]
        cols = st.columns([2, 3])
        cols[0].json({k: _clean(r[k]) for k in show})
        stem = os.path.splitext(str(r.get("clip", "")))[0]
        sic = r.get("shot_in_clip")
        clip_path = None
        if stem and sic == sic:
            cand = os.path.join(OUT_DIR, stem, "shots", f"shot_{int(sic)}_h264.mp4")
            if os.path.exists(cand):
                clip_path = cand
        if clip_path:
            cols[1].video(clip_path)
        else:
            cols[1].caption("No rendered clip for this shot yet — render with "
                            "`python tools/render_shots.py <clip>`.")


def _relationship_explorer(view):
    """Scatter any two metrics against each other, colored by make / feel / zone --
    a visual way to spot what goes with what."""
    numeric = [m for m in _METRIC_LABELS
               if m in view.columns and view[m].notna().sum() >= 4]
    if len(numeric) < 2:
        return
    with st.container(border=True):
        st.subheader("🔬 Explore relationships")
        c = st.columns(3)
        xm = c[0].selectbox("X", numeric, index=0,
                            format_func=lambda m: _METRIC_LABELS[m], key="relx")
        ym = c[1].selectbox("Y", numeric, index=min(1, len(numeric) - 1),
                            format_func=lambda m: _METRIC_LABELS[m], key="rely")
        by = c[2].selectbox("Color by", ["make", "feel", "zone", "none"], key="relc")
        sub = view.copy()
        color_col, enc = None, None
        if by == "make" and "made" in sub.columns:
            sub["make"] = sub["made"].map(
                lambda v: "make" if v in (True, "True")
                else ("miss" if v in (False, "False") else "?"))
            color_col = "make"
            enc = alt.Color("make:N", scale=alt.Scale(
                domain=["make", "miss", "?"], range=["#39d98a", "#ff6b6b", "#888"]))
        elif by == "feel" and "felt_good" in sub.columns:
            sub["feel"] = sub["felt_good"].map(
                lambda v: "good" if v in (True, "True")
                else ("off" if v in (False, "False") else "?"))
            color_col = "feel"
            enc = alt.Color("feel:N", scale=alt.Scale(
                domain=["good", "off", "?"], range=["#39d98a", "#ffb020", "#888"]))
        elif by == "zone" and "zone" in sub.columns:
            color_col = "zone"
            enc = alt.Color("zone:N")
        cols = [xm, ym] + ([color_col] if color_col else [])
        data = sub[cols].dropna(subset=[xm, ym])
        if data.empty:
            st.caption("no overlapping data for those two metrics")
            return
        mark = alt.Chart(data).mark_circle(size=90, opacity=0.75).encode(
            x=alt.X(xm, title=_METRIC_LABELS[xm], scale=alt.Scale(zero=False)),
            y=alt.Y(ym, title=_METRIC_LABELS[ym], scale=alt.Scale(zero=False)),
            tooltip=cols, **({"color": enc} if enc is not None else {}))
        st.altair_chart(mark, use_container_width=True)
        st.caption("Correlation ≠ causation, and single-camera metrics are "
                   "foreshortened — use it to spot patterns, not prove them.")


def _metric_target(metric):
    """Look up a metric's ideal target + band from config/targets.yaml."""
    tg = load_targets()
    for sec in ("arc", "form", "spin"):
        if metric in tg.get(sec, {}):
            return tg[sec][metric]
    return None


def _data_health(df):
    """How trustworthy is this session -- pose resolve rate + make classifiable."""
    total = len(df)
    if not total:
        return
    pose = int(df["elbow_angle_at_release_deg"].notna().sum()) \
        if "elbow_angle_at_release_deg" in df.columns else 0
    made = int(df["made"].isin([True, False, "True", "False"]).sum()) \
        if "made" in df.columns else 0
    with st.container(border=True):
        st.subheader("🩺 Data health")
        c = st.columns(3)
        c[0].metric("Pose resolved", f"{100*pose/total:.0f}%")
        c[1].metric("Make classifiable", f"{100*made/total:.0f}%")
        c[2].metric("Shots", total)
        st.caption("Higher = more to trust. Low pose% means you're small/occluded "
                   "in frame — a closer 2nd camera is the fix.")


def _export_panel(d, sd):
    """Download a PDF / regenerate the HTML report for this session."""
    with st.container(border=True):
        st.subheader("📄 Export")
        c = st.columns(2)
        try:
            csv = os.path.join(d, "session_shots.csv")
            pdf = _pdf_bytes(d, os.path.getmtime(csv))
            c[0].download_button("⬇️ PDF report", pdf, file_name=f"{sd}_report.pdf",
                                 mime="application/pdf", width="stretch")
        except Exception as e:                      # keep the page alive on any error
            c[0].caption(f"PDF unavailable: {e}")
        if c[1].button("Rebuild HTML report", width="stretch"):
            try:
                from export_report import main as export_html
                export_html([d])
                c[1].success("wrote report.html in the session folder")
            except Exception as e:
                c[1].caption(f"HTML error: {e}")


def _feel_tagging(df, view, d):
    """Tag shots good/off by FEEL and persist to felt_good in the session CSV.
    These labels power correlate_feel -> your personal ideal. Shows a live
    feel-drivers preview from whatever's tagged in the editor."""
    from shotlab.correlate import correlate_feel, summarize_feel_drivers
    key = os.path.basename(d)
    with st.expander("✍️ Tag shots by feel — trains your personal ideal"):
        st.caption("Mark how each shot FELT (not make/miss). These labels feed the "
                   "engine that learns YOUR ideal form — stronger than make/miss.")
        feel_of = (lambda v: "good" if v in (True, "True")
                   else ("off" if v in (False, "False") else ""))
        base = [c for c in ["shot_num", "clip", "zone", "release_angle_deg", "made"]
                if c in df.columns]
        tbl = df[base].copy()
        tbl.insert(0, "feel",
                   df["felt_good"].map(feel_of) if "felt_good" in df.columns else "")
        edited = st.data_editor(
            tbl, hide_index=True, width="stretch", key=f"feel_{key}",
            column_config={"feel": st.column_config.SelectboxColumn(
                "feel", options=["", "good", "off"], width="small")},
            disabled=[c for c in tbl.columns if c != "feel"])
        m = {"good": True, "off": False, "": None}
        feels = edited["feel"].tolist()
        if st.button("💾 Save feel tags", key=f"savefeel_{key}"):
            full = pd.read_csv(os.path.join(d, "session_shots.csv"))
            if len(full) == len(feels):
                full["felt_good"] = [m.get(x) for x in feels]
                full.to_csv(os.path.join(d, "session_shots.csv"), index=False)
                st.success(f"Saved {sum(1 for x in feels if x)} feel tags.")
            else:
                st.error("row-count mismatch — reload the session and retry.")
        rows = df.to_dict("records")
        for r, fv in zip(rows, feels):
            r["felt_good"] = m.get(fv)
        n_good = sum(1 for x in feels if x == "good")
        n_off = sum(1 for x in feels if x == "off")
        st.caption(f"tagged so far: **{n_good}** good · **{n_off}** off")
        st.markdown(summarize_feel_drivers(correlate_feel(rows)))


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

    # ---- real-feet + tempo KPIs (rim-scaled; low-conf but concrete) ----
    feet = [("apex_above_rim_ft", "Arc peak above rim", " ft"),
            ("release_height_ft", "Release height", " ft"),
            ("jump_height_ft", "Jump height", " ft"),
            ("tempo_dip_to_release_s", "Tempo dip→release", " s")]
    have = [(c, lbl, u) for c, lbl, u in feet
            if c in df.columns and df[c].notna().sum() >= 3]
    if have:
        cols = st.columns(len(have))
        for col, (c, lbl, u) in zip(cols, have):
            col.metric(lbl, f"{df[c].mean():.2f}{u}")
        st.caption("Real-world estimates. Jump height is honest when the session "
                   "was built with `--shooter-height` (body-scaled); release height "
                   "is depth-limited on one oblique camera; the 2nd camera firms "
                   "these up.")

    _data_health(df)
    _export_panel(d, sd)

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

    # ---- textbook (universal) targets, kept separate from your personal norm ----
    _textbook_panel(view)

    # ---- shot chart (half-court zones by make% + per-shot map) ----
    if {"depth", "side"}.issubset(df.columns):
        with st.container(border=True):
            st.subheader("🗺️ Shot chart")
            cc = st.columns(2)
            cc[0].pyplot(court_chart(view), use_container_width=True)
            cc[1].pyplot(shot_map_chart(view), use_container_width=True)
            st.caption("Left: your 9 zones shaded by make% (green = higher). "
                       "Right: every shot's release point relative to the rim "
                       "(dot = make, X = miss) — image-space, so it's the "
                       "camera's view, not court feet. True court coordinates "
                       "need the calibration clip.")

    # ---- what fades as you tire ----
    from shotlab.session import fatigue_breakdown
    fb = fatigue_breakdown(view)
    if not fb.empty:
        with st.container(border=True):
            st.subheader("😮‍💨 What fades as you tire")
            top = fb.iloc[0]
            lbl = _METRIC_LABELS.get(top["metric"], top["metric"])
            st.markdown(f"Biggest fade: **{lbl}** "
                        f"({top['first_half']} → {top['second_half']}).")
            show = fb.copy()
            show["metric"] = show["metric"].map(lambda m: _METRIC_LABELS.get(m, m))
            st.dataframe(show, hide_index=True, width="stretch")

    # ---- metric relationship explorer ----
    _relationship_explorer(view)

    # ---- shot inspector: click a row -> detail + its rendered clip ----
    _shot_inspector(view)

    # ---- feel tagging (powers the personal-ideal / feel-correlation engine) ----
    _feel_tagging(df, view, d)

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
        layers = [pts, trend]
        goal = ""
        tgt = _metric_target(metric)
        if tgt and tgt.get("target") is not None:
            band = tgt.get("band") or [None, None]
            if band[0] is not None and band[1] is not None:
                layers.insert(0, alt.Chart(pd.DataFrame({"lo": [band[0]], "hi": [band[1]]}))
                              .mark_rect(opacity=0.08, color="#2e7d32")
                              .encode(y="lo", y2="hi"))
            layers.append(alt.Chart(pd.DataFrame({"y": [tgt["target"]]}))
                          .mark_rule(color="#2e7d32", strokeDash=[2, 2], size=2)
                          .encode(y="y"))
            goal = f" Green line = your target ({tgt['target']}); shaded = ideal band."
        st.altair_chart(alt.layer(*layers).interactive(), use_container_width=True)
        st.caption("Red dashed = linear trend. Points colored by court zone. "
                   "Within a zone the foreshortening is constant, so a zone's "
                   "trend over time is meaningful even if the absolute ° is off." + goal)

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


def _personal_bests(agg):
    """Best-ever numbers across your built sessions."""
    if agg.empty:
        return
    items = []
    if "make_pct" in agg.columns and agg["make_pct"].notna().any():
        i = agg["make_pct"].idxmax()
        items.append(("🎯 Best make%", f"{agg.loc[i,'make_pct']:.0f}%", agg.loc[i, "session"]))
    if "longest_streak" in agg.columns and agg["longest_streak"].notna().any():
        i = agg["longest_streak"].idxmax()
        items.append(("🔥 Longest make-streak", int(agg.loc[i, "longest_streak"]),
                      agg.loc[i, "session"]))
    for col in [c for c in agg.columns if c.startswith("std_")]:
        sub = agg[col].dropna()
        if len(sub):
            i = sub.idxmin()
            items.append((f"🎯 Tightest {_METRIC_LABELS.get(col[4:], col[4:])}",
                          round(float(agg.loc[i, col]), 1), agg.loc[i, "session"]))
    if not items:
        return
    with st.container(border=True):
        st.subheader("🏆 Personal bests")
        for lbl, val, sess in items:
            st.markdown(f"- **{lbl}**: {val}  ·  _{sess}_")


def view_compare_sessions():
    from shotlab.session import consistency_stats
    st.subheader("Compare two sessions")
    sdirs = session_dirs()
    if len(sdirs) < 2:
        st.info("Need at least two built sessions to compare.")
        return
    c = st.columns(2)
    a = c[0].selectbox("Session A", sdirs, index=0, key="cmpA")
    b = c[1].selectbox("Session B", sdirs, index=min(1, len(sdirs) - 1), key="cmpB")
    dfa = pd.read_csv(os.path.join(OUT_DIR, a, "session_shots.csv"))
    dfb = pd.read_csv(os.path.join(OUT_DIR, b, "session_shots.csv"))

    def _mkpct(df):
        if "made" not in df.columns:
            return float("nan")
        m = df[df["made"].isin([True, False, "True", "False"])]
        return 100 * m["made"].isin([True, "True"]).mean() if len(m) else float("nan")

    k = st.columns(3)
    k[0].metric("Shots (A → B)", f"{len(dfa)} → {len(dfb)}")
    k[1].metric("Make% A", f"{_mkpct(dfa):.0f}%")
    k[2].metric("Make% B", f"{_mkpct(dfb):.0f}%", f"{_mkpct(dfb)-_mkpct(dfa):+.0f}%")

    ca = consistency_stats(dfa).set_index("metric")
    cb = consistency_stats(dfb).set_index("metric")
    rows = []
    for m in _METRIC_LABELS:
        if (m in dfa.columns and m in dfb.columns
                and dfa[m].notna().any() and dfb[m].notna().any()):
            ma, mb = float(dfa[m].mean()), float(dfb[m].mean())
            sa = float(ca.loc[m, "within_zone_std"]) if m in ca.index else float("nan")
            sb = float(cb.loc[m, "within_zone_std"]) if m in cb.index else float("nan")
            rows.append({"metric": _METRIC_LABELS[m], "mean_A": round(ma, 2),
                         "mean_B": round(mb, 2), "Δmean": round(mb - ma, 2),
                         "std_A": round(sa, 2), "std_B": round(sb, 2),
                         "Δspread": round(sb - sa, 2)})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Δspread negative = more consistent in B. ⚠️ Only comparable if "
                   "the camera setup matched across the two sessions.")
    else:
        st.caption("No shared metrics with data in both sessions.")


def view_progress():
    from shotlab.session import (aggregate_sessions, consistency_progress,
                                 mean_drift, consistency_stats, prescribe_target,
                                 drill_effectiveness)
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
    _personal_bests(agg)

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

    # level drift -- is a metric creeping session to session?
    md = mean_drift(agg)
    if not md.empty:
        drifting = md[md["drifting"]]
        with st.container(border=True):
            st.subheader("↕️ Level drift across sessions")
            if len(drifting):
                for _, r in drifting.iterrows():
                    lbl = _METRIC_LABELS.get(r["metric"], r["metric"])
                    st.markdown(f"- **{lbl}**: {r['first']} → {r['latest']} "
                                f"({r['delta']:+}) — creeping.")
            else:
                st.caption("No metric is drifting much — your levels are holding.")

    # did the thing a session told you to work on actually improve?
    sessions = []
    for sd in session_dirs():
        try:
            sdf = pd.read_csv(os.path.join(OUT_DIR, sd, "session_shots.csv"))
        except Exception:
            continue
        if sdf.empty:
            continue
        cons = consistency_stats(sdf)
        stds = (dict(zip(cons["metric"], cons["within_zone_std"]))
                if not cons.empty else {})
        t = pd.to_datetime(sdf.get("abs_time"), errors="coerce").dropna()
        sessions.append({"name": sd,
                         "date": t.min() if len(t) else pd.Timestamp.min,
                         "target_metric": prescribe_target(sdf).get("target_metric"),
                         "stds": stds})
    sessions.sort(key=lambda s: s["date"])
    de = drill_effectiveness(sessions)
    if not de.empty:
        with st.container(border=True):
            st.subheader("🎯 Did your homework pay off?")
            show = de.copy()
            show["worked_on"] = show["worked_on"].map(lambda m: _METRIC_LABELS.get(m, m))
            show["improved"] = show["improved"].map(lambda b: "✅ tighter" if b else "⚠️ no")
            st.dataframe(show, hide_index=True, width="stretch")
            st.caption("Each session flags one metric to groove; this checks whether "
                       "it got more repeatable the next session.")

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
        # goal line: if this metric has a textbook target, overlay it + say
        # whether you're closing the gap across sessions
        from shotlab.textbook import TEXTBOOK
        base = metric.replace("avg_", "").replace("std_", "")
        tb = TEXTBOOK.get(base)
        if (tb and tb.get("measurable_now") and metric.startswith("avg_")
                and len(sub) >= 1):
            target = tb["target"]
            rule = alt.Chart(pd.DataFrame({"y": [target]})).mark_rule(
                strokeDash=[5, 5], color="#2e7d32").encode(y="y:Q")
            ch = ch + rule
            latest, first = float(sub[metric].iloc[-1]), float(sub[metric].iloc[0])
            gap = latest - target
            trend = ("closing ✅" if abs(latest - target) < abs(first - target) - 1e-9
                     else "holding" if abs(abs(latest - target) - abs(first - target)) < 1e-9
                     else "widening ⚠️")
            st.caption(f"🎯 Textbook target **{target:.0f}°** (green line). Latest "
                       f"**{latest:.0f}°** → gap **{gap:+.0f}°**, {trend} vs your "
                       f"first tracked session.")
        st.altair_chart(ch, use_container_width=True)


def view_film_room():
    st.subheader("🎬 Film room — study your reps")
    st.caption("Tight closeups of each shot's load / release / follow-through, "
               "skeleton drawn, the phase's key joint in gold. ← / → (or the "
               "buttons) to move between shots.")
    sds = session_dirs()
    if not sds:
        st.info("No sessions built yet."); return
    sd = st.selectbox("Session", sds, key="fr_sess")
    which = st.radio("Show", ["makes", "misses", "all"], horizontal=True, key="fr_which")
    only = {"makes": True, "misses": False, "all": None}[which]
    from shotlab.closeups import build_shot_closeups, film_room_html
    with st.spinner("Building closeups (first time per session re-runs pose per "
                    "shot — slow; cached after)…"):
        cl = build_shot_closeups(os.path.join(OUT_DIR, sd), only_made=only)
    if not cl:
        st.info("No shots for that filter yet."); return
    import streamlit.components.v1 as components
    components.html(film_room_html(cl), height=760, scrolling=True)


# ---------------------------------------------------------------- main
st.title("🏀 ShotLab")
mode = st.sidebar.radio("View", ["Per-clip", "Session analytics", "Film room",
                                 "Shot review", "Compare shots",
                                 "Compare sessions", "Progress"])
if mode == "Per-clip":
    view_clip()
elif mode == "Session analytics":
    view_session()
elif mode == "Film room":
    view_film_room()
elif mode == "Shot review":
    view_shot_review()
elif mode == "Compare shots":
    view_compare()
elif mode == "Compare sessions":
    view_compare_sessions()
else:
    view_progress()

st.sidebar.divider()
st.sidebar.caption("Side-on angles (release/entry) = HIGH confidence when the "
                   "camera is square. Apex-ft = MEDIUM. Elbow flare/squareness "
                   "= LOW (1 camera). Make% = LOW until rim resolution improves.")
