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
import numpy as np
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


@st.cache_data(show_spinner="building recap…")
def _recap_bytes(d, _mtime, flare_path, truth_path):
    from session_recap_pdf import build
    return build(d, flare_path, truth_path)

st.set_page_config(page_title="ShotLab", layout="wide")
OUT_DIR = os.path.join(ROOT, "data", "out")

# release/entry angles are FORESHORTENED on one wide camera (advisory, not high --
# the app skiplists them and the profile keeps them in a diagnostic block; audit D16)
_ARC_CONF = {"release_angle_deg": "low", "entry_angle_deg": "low",
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
        c = st.columns(3)
        csv = os.path.join(d, "session_shots.csv")
        try:
            pdf = _pdf_bytes(d, os.path.getmtime(csv))
            c[0].download_button("⬇️ PDF report", pdf, file_name=f"{sd}_report.pdf",
                                 mime="application/pdf", width="stretch")
        except Exception as e:                      # keep the page alive on any error
            c[0].caption(f"PDF unavailable: {e}")
        try:                                        # rich recap: stats + drivers + relationships
            flarep = os.path.join(OUT_DIR, sd + "_3d", "analysis3d.json")
            truthp = os.path.join(d, "make_truth.json")
            has_truth = os.path.exists(truthp)
            sig = os.path.getmtime(csv) + (os.path.getmtime(truthp) if has_truth else 0)
            recap = _recap_bytes(d, sig, flarep if os.path.exists(flarep) else None,
                                 truthp if has_truth else None)
            c[1].download_button(
                "⬇️ Rich recap" + (" ✓verified" if has_truth else ""), recap,
                file_name=f"{sd}_recap.pdf", mime="application/pdf", width="stretch",
                help="all stats + what tracks with makes + cross-metric relationships"
                     + (" · built on your audited make/miss labels" if has_truth else
                        " · run the Make/miss audit to rebuild on verified labels"))
        except Exception as e:
            c[1].caption(f"recap unavailable: {e}")
        if c[2].button("Rebuild HTML report", width="stretch"):
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
        # editor stays on the RAW rows (row-aligned save), but the feel-drivers
        # preview runs on the CURATED set like every other derived surface
        # (2026-07-06 audit: it was including excluded/layup shots).
        from shotlab.curate import apply_excludes
        crows = (apply_excludes(pd.DataFrame(rows), d).to_dict("records")
                 if "shot_num" in df.columns else rows)
        st.markdown(summarize_feel_drivers(correlate_feel(crows)))


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
        sd = st.selectbox("Session", sdirs, format_func=_session_label)
    d = os.path.join(OUT_DIR, sd)
    raw_df = pd.read_csv(os.path.join(d, "session_shots.csv"))
    if raw_df.empty:
        st.warning("No shots in this session.")
        return
    # Analytics run on the CURATED set (exclude.json), same as the report/profile
    # -- so headline KPIs, make%, drivers and charts here match every other
    # surface instead of silently including human-flagged junk (2026-07-05
    # audit). Feel-tagging below still edits the RAW row-aligned CSV.
    from shotlab.curate import apply_excludes, load_excludes
    df = apply_excludes(raw_df, d)
    _ex, _lay = load_excludes(d)
    if _ex or _lay:
        st.caption(f"Showing **{len(df)}** curated shots — "
                   f"{len(raw_df) - len(df)} flagged/layup shots hidden via "
                   f"`exclude.json`.")

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
    # RAW df: the editor writes back to the whole row-aligned CSV (you can still
    # tag a shot that analytics excludes).
    _feel_tagging(raw_df, view, d)

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
        # make% over CLASSIFIED shots only (denominator excludes made=NaN), to
        # match the KPI/movement blocks (2026-07-06 final sweep #17)
        fa = (df[df["shot_form"] != "unknown"].groupby("shot_form")
              .agg(shots=("clip", "count"),
                   make_pct=("made", lambda s: round(100 * (s == True).sum()
                             / max((s.isin([True, False])).sum(), 1), 0)))
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
    from shotlab.curate import apply_excludes
    dfa = apply_excludes(pd.read_csv(os.path.join(OUT_DIR, a, "session_shots.csv")),
                         os.path.join(OUT_DIR, a))
    dfb = apply_excludes(pd.read_csv(os.path.join(OUT_DIR, b, "session_shots.csv")),
                         os.path.join(OUT_DIR, b))

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
    from shotlab.curate import apply_excludes
    sessions = []
    for sd in session_dirs():
        try:
            sdf = apply_excludes(
                pd.read_csv(os.path.join(OUT_DIR, sd, "session_shots.csv")),
                os.path.join(OUT_DIR, sd))
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
    sd = st.selectbox("Session", sds, key="fr_sess", format_func=_session_label)
    c1, c2 = st.columns([2, 3])
    which = c1.radio("Show", ["makes", "misses", "all"], horizontal=True, key="fr_which")
    only = {"makes": True, "misses": False, "all": None}[which]
    overlay = c2.toggle("Overlay ideal form (magenta = your ideal, green = this rep)",
                        key="fr_overlay",
                        help="warps your profile's ideal load/release/follow onto each "
                             "rep so you can see where you fall short")
    from shotlab.closeups import build_shot_closeups, film_room_html
    with st.spinner("Building closeups (first time per session re-runs pose per "
                    "shot — slow; cached after)…"):
        cl = build_shot_closeups(os.path.join(OUT_DIR, sd), only_made=only,
                                 overlay_ideal=overlay)
    if not cl:
        st.info("No shots for that filter yet."); return
    import streamlit.components.v1 as components
    components.html(film_room_html(cl), height=760, scrolling=True)


# ---------------------------------------------------------------- 3D analysis
def analysis3d_dirs():
    if not os.path.isdir(OUT_DIR):
        return []
    return sorted(d for d in os.listdir(OUT_DIR)
                  if os.path.exists(os.path.join(OUT_DIR, d, "analysis3d.json")))


def view_3d():
    st.subheader("🧊 3D analysis (2-camera)")
    st.caption("Metrics one wide camera can't give alone — the ball's true-feet "
               "arc (depth-corrected, gravity-validated) and elbow flare from the "
               "close camera's 3D pose. No calibration board required.")
    dirs = analysis3d_dirs()
    if not dirs:
        st.info("No 3D analysis yet. Build one with:\n\n"
                "`python tools/analyze3d.py --wide <wide.mp4> --close <close.mp4> "
                "--weights runs/detect/ball_finetune/weights/best.pt --out "
                "data/out/<name>_3d`")
        return
    sd = st.selectbox("Session", dirs, key="a3d_sess", format_func=_session_label)
    from shotlab.analysis3d import Analysis3D
    a = Analysis3D.load(os.path.join(OUT_DIR, sd, "analysis3d.json"))

    # ---- elbow flare (headline new metric) ----
    st.markdown("### Elbow flare")
    fl = (a.flare or {}).get("summary")
    if fl:
        c1, c2, c3 = st.columns(3)
        c1.metric("Median flare", f"{fl['median_deg']:+.1f}°",
                  help="Elbow's swing out of the shooting plane at release. "
                       "Magnitude is the coaching signal; sign is setup-dependent.")
        c2.metric("Shot-to-shot spread", f"±{fl['sd_deg']:.1f}°",
                  help="Lower = more repeatable elbow. This is your consistency.")
        c3.metric("Shots measured", f"{fl['n']}")
        st.caption(f"Confidence: **{a.flare.get('confidence','low-med')}** — "
                   f"{a.flare.get('note','')}")
        fdf = pd.DataFrame(a.flare.get("shots", []))
        if not fdf.empty:
            chart = (alt.Chart(fdf).mark_bar(opacity=0.8)
                     .encode(alt.X("flare_deg:Q", bin=alt.Bin(maxbins=18),
                                   title="elbow flare (deg)  ·  0 = tucked/on-line"),
                             alt.Y("count()", title="shots"))
                     .properties(height=200))
            rule = alt.Chart(pd.DataFrame({"x": [fl["median_deg"]]})).mark_rule(
                color="#d62728", size=2).encode(x="x:Q")
            st.altair_chart(chart + rule, use_container_width=True)

        # ---- does flare track make/miss? ----
        mm = (a.flare or {}).get("make_vs_miss")
        if mm and mm.get("flare_make_median") is not None and mm.get("flare_miss_median") is not None:
            st.markdown("#### Does flare track your makes?")
            d, p = mm.get("cohens_d"), mm.get("p_perm")
            k1, k2, k3 = st.columns(3)
            k1.metric("Flare on MAKES", f"{mm['flare_make_median']:+.1f}°", f"{mm['n_make']} shots")
            k2.metric("Flare on MISSES", f"{mm['flare_miss_median']:+.1f}°", f"{mm['n_miss']} shots")
            gap = mm["flare_make_median"] - mm["flare_miss_median"]
            k3.metric("Difference", f"{gap:+.1f}°",
                      f"d={d}" if d is not None else "n/a")
            if d is not None and abs(d) >= 0.4 and (p is None or p < 0.1):
                st.info(f"Makes and misses **do** separate on flare "
                        f"(d={d}, p={p}) — but this is exploratory: flare is "
                        f"monocular/session-relative and make/miss is low-confidence, "
                        f"cross-mapped between the two cameras by audio sync.")
            else:
                st.caption(f"No clear separation (d={d}, p={p}). On this sample your "
                           f"flare doesn't distinguish makes from misses — it's fairly "
                           f"**consistent** either way, so the arc (release/entry) is the "
                           f"stronger make-driver here. Exploratory; low confidence.")

        # ---- stills: SEE the flare ----
        stills = [s for s in (a.flare.get("shots") or []) if s.get("still")]
        if stills:
            st.markdown("#### See it — release stills (elbow winging off the shoulder→wrist line)")
            which = st.radio("Show", ["makes", "misses", "all"], horizontal=True, key="fl_still")
            pick = [s for s in stills
                    if which == "all" or (which == "makes" and s.get("made"))
                    or (which == "misses" and s.get("made") is False)]
            pick = sorted(pick, key=lambda s: abs(s.get("flare_deg", 0)), reverse=True)[:9]
            base = os.path.join(OUT_DIR, sd)
            cols = st.columns(3)
            for i, s in enumerate(pick):
                fp = os.path.join(base, s["still"])
                if os.path.exists(fp):
                    cols[i % 3].image(fp, caption=f"{s['flare_deg']:+.0f}° · "
                                      f"{'make' if s.get('made') else 'miss' if s.get('made') is False else '?'}",
                                      use_container_width=True)
    else:
        st.info("No flare computed (needs the close body-camera clip).")

    # ---- metric ball arc ----
    st.markdown("### Ball arc (true feet, depth-corrected)")
    wide = a.wide or {}
    shots = wide.get("shots", [])
    if shots:
        wdf = pd.DataFrame(shots)
        good = wdf[wdf["trustworthy"]]
        seg = wide.get("segmentation", "gap-split")
        nclips = wide.get("n_clips")
        st.caption(f"{len(wdf)} shots{f' across {nclips} clips' if nclips else ''} · "
                   f"**{len(good)} are clean shot arcs** (a gravity projectile fits "
                   f"the pixels to <5px — robust at the far ball's size, unlike a raw "
                   f"per-frame check). Segmentation: {seg}. Frame-rate "
                   f"{'VARIABLE' if wide.get('is_vfr') else 'constant'} "
                   f"({wide.get('fps')} fps) — handled per-frame via real timestamps.")
        show = good if not st.checkbox("show all (incl. non-arcs)", value=False) else wdf
        cols = ["clip", "first_frame", "n_points", "flight_s", "apex_above_release_ft",
                "lateral_drift_ft", "release_angle_deg", "entry_angle_deg",
                "reproj_px", "trustworthy"]
        st.dataframe(show[[c for c in cols if c in show.columns]]
                     .rename(columns={"apex_above_release_ft": "apex↑ (ft)",
                                      "lateral_drift_ft": "L/R drift (ft)",
                                      "release_angle_deg": "release°",
                                      "entry_angle_deg": "entry°",
                                      "reproj_px": "fit err (px)",
                                      "trustworthy": "clean arc"}),
                     use_container_width=True, hide_index=True)
        if len(good) >= 2:
            st.caption(f"**Trustworthy (focal-free):** apex above release median "
                       f"**{good['apex_above_release_ft'].median():.1f} ft** across "
                       f"{len(good)} clean arcs — the vertical channel is unambiguous. "
                       f"⚠️ On this shooter-facing camera the horizontal channel "
                       f"(release°/entry°, L/R drift) mixes left-right with toward-rim "
                       f"depth, so treat those as directional until the camera tilt "
                       f"(W4) is recovered.")
    else:
        st.info("No arcs computed (needs the wide clip + ball weights).")

    # ---- depth / calibration status ----
    st.markdown("### Depth & true release angle")
    if a.camera_tilt:
        t = a.camera_tilt
        st.success(f"Camera tilt recovered (pitch {t.get('pitch_deg')}°, roll "
                   f"{t.get('roll_deg')}°) → absolute depth and true 3D release "
                   f"angle are unlocked.")
    else:
        st.warning("Absolute **depth** (toward/away drift) and the **true 3D "
                   "release angle** need the wide camera's tilt. Recover it from "
                   "≥2 clean arcs (W4, `ballistic.fit_camera_tilt`) or a rim-PnP "
                   "pass. The vertical + left/right arc above is already trustworthy.")
    st.caption("The true ±2–3° metric elbow flare (stereo) needs the next session "
               "framed so the ball is co-visible in BOTH cameras.")


# ---------------------------------------------------------------- make/miss audit
def view_make_audit():
    st.subheader("✅ Make/miss audit — verify what the tracker called")
    st.caption("Make/miss is the pipeline's weakest signal: a geometric heuristic "
               "(ball continues straight down through the rim = make) on a small, "
               "far ball — most calls are LOW confidence. You filmed these, so YOU "
               "are the ground truth. Confirm or correct each call; we measure how "
               "often the tracker is right and recompute make% from your answers.")
    sds = session_dirs()
    if not sds:
        st.info("No sessions built yet."); return
    sd = st.selectbox("Session", sds, key="audit_sess", format_func=_session_label)
    d = os.path.join(OUT_DIR, sd)
    df = pd.read_csv(os.path.join(d, "session_shots.csv"))
    shots = []
    for _, r in df.iterrows():
        stem = str(r["clip"]).replace(".mp4", "")
        vid = os.path.join(OUT_DIR, stem, "shots", f"shot_{int(r['shot_in_clip'])}_h264.mp4")
        shots.append({"clip": str(r["clip"]), "shot": int(r["shot_in_clip"]),
                      "made": bool(r["made"]) if pd.notna(r["made"]) else None,
                      "conf": r.get("make_conf"), "vid": vid,
                      "form": r.get("shot_form", ""), "setup": r.get("shot_setup", ""),
                      "apex": r.get("apex_height_ft"), "note": r.get("quality_note"),
                      "key": f"{r['clip']}|{int(r['shot_in_clip'])}"})
    def _suspect(s):
        low = s.get("apex") is not None and pd.notna(s["apex"]) and s["apex"] < 2.0
        return bool((low and (s["form"] in ("layup", "floater")
                              or s.get("setup") == "on_the_move"))
                    or (isinstance(s.get("note"), str) and "not a shot" in s["note"]))
    tpath = os.path.join(d, "make_truth.json")
    truth = json.load(open(tpath, encoding="utf-8")) if os.path.exists(tpath) else {}
    ppath = os.path.join(d, "make_pred.json")   # learned visual model (net + ball-below-rim)
    preds = json.load(open(ppath, encoding="utf-8")) if os.path.exists(ppath) else {}
    for s in shots:
        pr = preds.get(s["key"], {})
        s["mpred"] = pr.get("made"); s["mprob"] = pr.get("prob")
    by_key = {s["key"]: s for s in shots}

    n_susp = sum(_suspect(s) for s in shots)
    opts = ["suspected non-shots first", "low-confidence first", "all", "unreviewed only"]
    if preds:
        opts.insert(0, "model-uncertain first")
    filt = st.radio(f"Review order  ({n_susp} auto-suspected non-shots"
                    + (f" · model predicted {len(preds)}" if preds else "") + ")",
                    opts, horizontal=True, key="audit_filt")
    order = list(shots)
    if filt == "model-uncertain first":
        order.sort(key=lambda s: (abs((s.get("mprob") or 0.5) - 0.5), s["key"] in truth))
    elif filt == "suspected non-shots first":
        order.sort(key=lambda s: (not _suspect(s), s["key"] in truth))
    elif filt == "low-confidence first":
        rank = {"low": 0, "medium": 1, "na": 2}
        order.sort(key=lambda s: (rank.get(s["conf"], 3), s["key"] in truth))
    elif filt == "unreviewed only":
        order = [s for s in shots if s["key"] not in truth] or shots

    n = len(order)
    idx = st.session_state.get("audit_idx", 0) % n
    nav = st.columns([1, 1, 3])
    if nav[0].button("← prev", key="audit_prev"):
        st.session_state["audit_idx"] = (idx - 1) % n; st.rerun()
    if nav[1].button("next →", key="audit_next"):
        st.session_state["audit_idx"] = (idx + 1) % n; st.rerun()
    nav[2].caption(f"Shot {idx+1} of {n} in this list")

    s = order[idx]
    cv, ci = st.columns([3, 2])
    with cv:
        if os.path.exists(s["vid"]):
            st.video(s["vid"])
        else:
            st.warning("This shot's clip isn't rendered. Run "
                       f"`python tools/render_shots.py` on {s['clip']}, or watch it in "
                       "the Per-clip overlay.")
    with ci:
        pred = "MAKE" if s["made"] else ("miss" if s["made"] is False else "?")
        st.metric(f"Shot {s['shot']} · {s['clip'][-16:]}", f"predicted: {pred}",
                  f"confidence: {s['conf']}  ·  {s.get('form','')}")
        if s.get("mpred") is not None:
            mp = "MAKE" if s["mpred"] else "miss"
            sure = "unsure" if abs((s["mprob"] or 0.5) - 0.5) < 0.15 else "confident"
            st.info(f"🤖 Visual model: **{mp}** ({s['mprob']:.0%} make · {sure}) "
                    f"— ~87% accurate; confirm or correct below.")
        if _suspect(s):
            st.warning("⚠ Auto-suspected **NOT a shot** (low apex + layup/floater/"
                       "on-the-move). If it's a dribble/retrieve, mark it below.")
        # 'not a shot' is the key addition: the detector fires on dribbles/retrieves
        labels = ["made", "missed", "NOT a shot (dribble/retrieve)", "can't tell"]
        vals = ["make", "miss", "notshot", "unsure"]
        prev_truth = truth.get(s["key"])
        default = vals.index(prev_truth) if prev_truth in vals else \
            (0 if s["made"] else 1 if s["made"] is False else 3)
        pick = st.radio("What actually happened?", labels, index=default,
                        key=f"truth_{s['key']}")
        choice = vals[labels.index(pick)]
        if st.button("💾 Save & next", key="audit_save", type="primary"):
            truth[s["key"]] = choice
            with open(tpath, "w", encoding="utf-8") as f:
                json.dump(truth, f, indent=2)
            st.session_state["audit_idx"] = (idx + 1) % n; st.rerun()
        if prev_truth:
            tag = {"make": "made", "miss": "missed", "notshot": "NOT a shot",
                   "unsure": "unsure"}.get(prev_truth, prev_truth)
            st.caption(f"your last answer: **{tag}**")

    # ---- scoreboard ----
    reviewed = {k: v for k, v in truth.items() if v in ("make", "miss", "notshot") and k in by_key}
    st.divider()
    if reviewed:
        nnot = sum(v == "notshot" for v in reviewed.values())
        real = {k: v for k, v in reviewed.items() if v in ("make", "miss")}
        correct = sum((v == "make") == bool(by_key[k]["made"]) for k, v in real.items())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Reviewed", f"{len(reviewed)}/{len(shots)}")
        m2.metric("NOT real shots", f"{nnot}",
                  f"{100*nnot/len(reviewed):.0f}% of reviewed", delta_color="inverse")
        if real:
            m3.metric("Make/miss accuracy", f"{100*correct/len(real):.0f}%",
                      f"{correct}/{len(real)} on real shots")
            nmk = sum(v == "make" for v in real.values())
            m4.metric("Real make%", f"{100*nmk/len(real):.0f}%",
                      f"was {100*df['made'].mean():.0f}% (uncleaned)")
        # projected clean counts
        if nnot:
            proj = len(shots) * (1 - nnot / len(reviewed))
            st.warning(f"⚠ At this rate, ~{100*nnot/len(reviewed):.0f}% of the "
                       f"'{len(shots)} shots' are actually dribbles/retrieves — the "
                       f"real session is closer to **~{proj:.0f} shots**. Every stat, "
                       f"make%, and driver in the recap is contaminated until these "
                       f"are removed.")
        if preds and real:
            mp = [(k, v) for k, v in real.items() if by_key[k].get("mpred") is not None]
            if mp:
                agree = sum(by_key[k]["mpred"] == (v == "make") for k, v in mp)
                st.caption(f"🤖 The visual make/miss model agreed with you on "
                           f"**{agree}/{len(mp)} = {100*agree/len(mp):.0f}%** of the real "
                           f"shots you've reviewed — so future sessions can be mostly "
                           f"auto-labeled, with you only checking the 'model-uncertain' ones.")
        st.caption("When you've reviewed enough (esp. the low-confidence ones), tell "
                   "me — I'll drop the non-shots, recompute make% + drivers, and "
                   "regenerate the recap on the clean set. Your answers are saved to "
                   "make_truth.json so you can stop and resume.")
    else:
        st.info("No shots reviewed yet — start with the low-confidence ones (most "
                "likely to be non-shots or wrong calls).")


# ---------------------------------------------------------------- shot explorer
import datetime as _dt
import re as _re

# numeric measurables offered as filters/sorts, with friendly labels
_MEASURABLES = [
    ("release_angle_deg", "Release angle °"), ("entry_angle_deg", "Entry angle °"),
    ("apex_height_ft", "Apex height ft"), ("knee_bend_deg", "Knee angle ° (low=deep)"),
    ("elbow_angle_at_release_deg", "Elbow at release °"),
    ("follow_through_hold_s", "Follow-through s"),
    ("balance_drift_px_per_ht", "Balance drift"), ("release_vs_apex_s", "Release vs apex s"),
    ("tempo_dip_to_release_s", "Tempo s"), ("jump_height_ft", "Jump height ft"),
    ("flare_deg", "Elbow flare °"),
]


def _session_date(name):
    m = _re.search(r"(20\d\d)[-_]?(\d\d)[-_]?(\d\d)", name)
    if m:
        try:
            return _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    m = _re.search(r"_(\d\d)(\d\d)(?:_|$)", name)      # session_0710 -> Jul 10
    if m:
        try:
            return _dt.date(2026, int(m.group(1)), int(m.group(2)))
        except ValueError:
            pass
    return None


def _session_label(name):
    d = _session_date(name)
    if not d:
        return name.replace("session_", "")
    tag = name.replace("session_", "").replace("_", " ")
    tag = _re.sub(r"\b\d{4}-\d\d-\d\d\b|\b\d{6}\b|\b\d{4}\b", "", tag)   # drop date/time digits
    tag = _re.sub(r"\s+", " ", tag).strip()
    base = f"{d.strftime('%b')} {d.day}, {d.year}"
    return f"{base} · {tag}" if tag else base


def _shots_sig():
    parts = []
    for sd in session_dirs():
        for fn in ("session_shots.csv", "make_truth.json", "flare_by_shot.json"):
            p = os.path.join(OUT_DIR, sd, fn)
            if os.path.exists(p):
                parts.append(os.path.getmtime(p))
    return tuple(parts)


@st.cache_data(show_spinner="loading shots…")
def _all_shots(_sig):
    rows = []
    for sd in session_dirs():
        d = os.path.join(OUT_DIR, sd)
        try:
            df = pd.read_csv(os.path.join(d, "session_shots.csv"))
        except Exception:
            continue
        tp = os.path.join(d, "make_truth.json")
        truth = json.load(open(tp, encoding="utf-8")) if os.path.exists(tp) else {}
        fp = os.path.join(d, "flare_by_shot.json")
        flare = json.load(open(fp, encoding="utf-8")) if os.path.exists(fp) else {}
        dobj = _session_date(sd)
        for _, r in df.iterrows():
            key = f"{r['clip']}|{int(r['shot_in_clip'])}"
            t = truth.get(key)
            if t == "notshot":
                continue                                   # drop verified non-shots
            made = (t == "make") if t in ("make", "miss") else bool(r.get("made"))
            stem = str(r["clip"]).replace(".mp4", "")
            vid = os.path.join(OUT_DIR, stem, "shots", f"shot_{int(r['shot_in_clip'])}_h264.mp4")
            row = {k: r.get(k) for k in df.columns}
            row.update(session=sd, date_label=_session_label(sd),
                       date=str(dobj) if dobj else "", outcome=("make" if made else "miss"),
                       made=made, verified=(t in ("make", "miss")),
                       flare_deg=flare.get(key), video=vid,
                       has_video=os.path.exists(vid), key=key)
            rows.append(row)
    return pd.DataFrame(rows)


def view_explore():
    st.subheader("🔎 Shot Explorer — filter every shot by any measurable")
    df = _all_shots(_shots_sig())
    if df.empty:
        st.info("No shots yet. Build a session with build_session.py."); return
    metrics = [(k, lab) for k, lab in _MEASURABLES if k in df and df[k].notna().any()]
    mkeys = [k for k, _ in metrics]

    with st.sidebar:
        st.markdown("### Filters")
        dates = sorted(df["date_label"].unique(),
                       key=lambda s: df[df["date_label"] == s]["date"].iloc[0], reverse=True)
        sel = st.multiselect("Session (date)", dates, default=dates,
                             format_func=lambda s: s)
        outc = st.multiselect("Outcome", ["make", "miss"], default=["make", "miss"])
        moves = st.multiselect("Movement into shot",
                               sorted(df["movement_dir"].dropna().unique()))
        zones = st.multiselect("Court zone", sorted(df["zone"].dropna().unique()))
        only_verified = st.checkbox("Verified make/miss only", value=False,
                                    help="only shots whose make/miss you confirmed in the audit")

    f = df[df["date_label"].isin(sel) & df["outcome"].isin(outc)]
    if moves:
        f = f[f["movement_dir"].isin(moves)]
    if zones:
        f = f[f["zone"].isin(zones)]
    if only_verified:
        f = f[f["verified"]]

    c1, c2, c3 = st.columns([2, 1, 2])
    sort_lab = c1.selectbox("Sort by", [lab for _, lab in metrics],
                            index=0 if metrics else None)
    sort_key = dict((lab, k) for k, lab in metrics).get(sort_lab)
    asc = c2.radio("Order", ["high→low", "low→high"], horizontal=False) == "low→high"
    if sort_key and f[sort_key].notna().any():
        lo, hi = float(np.nanmin(f[sort_key])), float(np.nanmax(f[sort_key]))
        if hi > lo:
            rng = c3.slider(f"{sort_lab} range", lo, hi, (lo, hi))
            f = f[f[sort_key].between(*rng) | f[sort_key].isna()]
        f = f.sort_values(sort_key, ascending=asc, na_position="last")
    st.caption(f"**{len(f)}** shots match  ·  {int((f['outcome']=='make').sum())} makes / "
               f"{int((f['outcome']=='miss').sum())} misses  ·  click a row to watch it")

    show = ["date_label", "outcome", "movement_dir", "zone"] + mkeys
    disp = f[show].rename(columns=dict(metrics, date_label="date", movement_dir="movement",
                                       outcome="result"))
    ev = st.dataframe(disp.round(2), hide_index=True, use_container_width=True,
                      on_select="rerun", selection_mode="single-row", height=340)

    picks = ev.selection.rows if ev and ev.selection else []
    if not picks:
        st.info("👆 Click any row to load that shot's video + measurables.")
        return
    row = f.iloc[picks[0]]
    st.markdown(f"### {row['outcome'].upper()}  ·  {row['date_label']}  ·  "
                f"shot {int(row['shot_in_clip'])}  ·  {row.get('zone','')}")
    cv, ci = st.columns([3, 2])
    with cv:
        if row["has_video"]:
            st.video(row["video"])
        else:
            st.warning("This session's shot clips aren't rendered. Run "
                       "`python tools/render_shots.py` on its clips.")
    with ci:
        st.metric("Result", row["outcome"].upper(),
                  "✓ verified" if row["verified"] else "tracker guess")
        cols = st.columns(2)
        for i, (k, lab) in enumerate(metrics):
            v = row.get(k)
            if pd.notna(v):
                cols[i % 2].metric(lab, f"{v:.1f}")


def _shot_detail(row, metrics):
    """Video + all measurables for one selected shot (shared by Explorer/Scatter)."""
    st.markdown(f"### {row['outcome'].upper()}  ·  {row['date_label']}  ·  "
                f"shot {int(row['shot_in_clip'])}  ·  {row.get('zone','')}")
    cv, ci = st.columns([3, 2])
    with cv:
        if row["has_video"]:
            st.video(row["video"])
        else:
            st.warning("This session's shot clips aren't rendered "
                       "(`python tools/render_shots.py`).")
    with ci:
        st.metric("Result", row["outcome"].upper(),
                  "✓ verified" if row["verified"] else "tracker guess")
        cols = st.columns(2)
        for i, (k, lab) in enumerate(metrics):
            v = row.get(k)
            if pd.notna(v):
                cols[i % 2].metric(lab, f"{v:.1f}")


def view_scatter():
    st.subheader("📈 Shot scatter — plot any measurable against another")
    st.caption("Each dot is a shot, colored make/miss. Spot relationships, then "
               "click a dot to watch that shot.")
    df = _all_shots(_shots_sig())
    if df.empty:
        st.info("No shots yet."); return
    metrics = [(k, lab) for k, lab in _MEASURABLES if k in df and df[k].notna().any()]
    labs = [lab for _, lab in metrics]
    k_of = dict((lab, k) for k, lab in metrics)

    with st.sidebar:
        st.markdown("### Axes & filters")
        xlab = st.selectbox("X axis", labs, index=labs.index("Elbow flare °")
                            if "Elbow flare °" in labs else 0)
        ylab = st.selectbox("Y axis", labs, index=1 if len(labs) > 1 else 0)
        dates = sorted(df["date_label"].unique(),
                       key=lambda s: df[df["date_label"] == s]["date"].iloc[0], reverse=True)
        sel = st.multiselect("Session (date)", dates, default=dates[:1] or dates)
        vonly = st.checkbox("Verified make/miss only", value=True)
    xk, yk = k_of[xlab], k_of[ylab]
    f = df[df["date_label"].isin(sel)].copy()
    if vonly:
        f = f[f["verified"]]
    f = f[f[xk].notna() & f[yk].notna()].reset_index(drop=True)
    if f.empty:
        st.info("No shots with both metrics for that filter (try more sessions, or "
                "uncheck 'verified only')."); return

    f["_i"] = f.index
    ch = (alt.Chart(f).mark_circle(size=110, opacity=0.75)
          .encode(x=alt.X(f"{xk}:Q", title=xlab, scale=alt.Scale(zero=False)),
                  y=alt.Y(f"{yk}:Q", title=ylab, scale=alt.Scale(zero=False)),
                  color=alt.Color("outcome:N", scale=alt.Scale(
                      domain=["make", "miss"], range=["#2a9d4a", "#c0392b"]),
                      legend=alt.Legend(title="result")),
                  tooltip=["date_label", "outcome", alt.Tooltip(f"{xk}:Q", title=xlab),
                           alt.Tooltip(f"{yk}:Q", title=ylab)])
          .add_params(alt.selection_point(name="pt", fields=["_i"], on="click"))
          .properties(height=430))
    ev = st.altair_chart(ch, use_container_width=True, on_select="rerun", key="scatterchart")
    # correlation readout
    import numpy as _np
    r = float(_np.corrcoef(f[xk], f[yk])[0, 1]) if len(f) > 2 else float("nan")
    st.caption(f"{len(f)} shots · correlation {xlab} vs {ylab}: **r = {r:+.2f}** "
               f"(near 0 = unrelated). Click a dot to watch.")

    sel_pts = (ev.selection.get("pt") if ev and ev.selection else None) or []
    idxs = [p.get("_i") for p in sel_pts if isinstance(p, dict) and "_i" in p]
    if idxs:
        _shot_detail(f.iloc[int(idxs[0])], metrics)
    else:
        st.info("👆 Click a dot to load that shot.")


@st.cache_data
def _profile():
    p = os.path.join(ROOT, "app", "profile.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


def _cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or len(b) < 3:
        return None
    sp = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else None


def _date_picker(df, label="Session (date)", multi=False, key=None):
    dates = sorted(df["date_label"].unique(),
                   key=lambda s: df[df["date_label"] == s]["date"].iloc[0], reverse=True)
    if multi:
        return st.multiselect(label, dates, default=dates, key=key)
    return st.selectbox(label, dates, key=key)


# ---------------- shot chart
def view_shotchart():
    st.subheader("🎯 Shot chart — where you shoot from")
    df = _all_shots(_shots_sig())
    if df.empty or "rim_dx_px" not in df:
        st.info("No shot-position data yet."); return
    with st.sidebar:
        st.markdown("### Chart")
        sess = _date_picker(df, key="sc_sess")
        vonly = st.checkbox("Verified make/miss only", value=True, key="sc_v")
    f = df[df["date_label"] == sess].copy()
    if vonly:
        f = f[f["verified"]]
    f = f[f["rim_dx_px"].notna() & f["rim_dist_px"].notna()].reset_index(drop=True)
    if f.empty:
        st.info("No positioned shots for that filter."); return
    f["_i"] = f.index
    mk = int((f["outcome"] == "make").sum())
    ch = (alt.Chart(f).mark_point(size=170, filled=True, opacity=0.8, strokeWidth=2)
          .encode(
              x=alt.X("rim_dx_px:Q", title="← left   ·   horizontal offset from rim   ·   right →",
                      scale=alt.Scale(zero=False)),
              y=alt.Y("rim_dist_px:Q", title="distance from rim  (farther up = deeper shot)",
                      scale=alt.Scale(zero=False)),
              color=alt.Color("outcome:N", scale=alt.Scale(domain=["make", "miss"],
                              range=["#2a9d4a", "#c0392b"]), legend=alt.Legend(title="result")),
              shape=alt.Shape("outcome:N", scale=alt.Scale(domain=["make", "miss"],
                              range=["circle", "cross"]), legend=None),
              tooltip=["outcome", "zone", "movement_dir",
                       alt.Tooltip("release_angle_deg:Q", title="release°")])
          .add_params(alt.selection_point(name="pt", fields=["_i"], on="click"))
          .properties(height=470))
    ev = st.altair_chart(ch, use_container_width=True, on_select="rerun", key="scchart")
    st.caption(f"{len(f)} shots · {mk} makes ({100*mk/len(f):.0f}%) · ● make / ✚ miss · "
               f"image-space relative to the rim (one camera, so positions are only "
               f"comparable WITHIN this session). Click a dot to watch.")
    metrics = [(k, lab) for k, lab in _MEASURABLES if k in df and df[k].notna().any()]
    pts = (ev.selection.get("pt") if ev and ev.selection else None) or []
    idxs = [p.get("_i") for p in pts if isinstance(p, dict) and "_i" in p]
    if idxs:
        _shot_detail(f.iloc[int(idxs[0])], metrics)


# ---------------- makes vs misses split
def view_makesplit():
    st.subheader("⚖️ Makes vs misses — what's different when you score")
    df = _all_shots(_shots_sig())
    if df.empty:
        st.info("No shots yet."); return
    with st.sidebar:
        st.markdown("### Compare")
        sel = _date_picker(df, "Sessions (date)", multi=True, key="ms_sess")
        vonly = st.checkbox("Verified make/miss only", value=True, key="ms_v")
    f = df[df["date_label"].isin(sel)].copy()
    if vonly:
        f = f[f["verified"]]
    made, miss = f[f["outcome"] == "make"], f[f["outcome"] == "miss"]
    if len(made) < 3 or len(miss) < 3:
        st.info("Need at least a few verified makes AND misses — audit a session in "
                "Make/miss audit, or widen the filter."); return
    metrics = [(k, lab) for k, lab in _MEASURABLES if k in f and made[k].notna().sum() >= 3
               and miss[k].notna().sum() >= 3]
    rows = []
    for k, lab in metrics:
        a, b = made[k].dropna(), miss[k].dropna()
        rows.append({"metric": lab, "makes": round(a.mean(), 1), "misses": round(b.mean(), 1),
                     "d": _cohen_d(a.values, b.values), "_k": k})
    rd = pd.DataFrame(rows).dropna(subset=["d"]).sort_values("d", key=lambda s: s.abs(),
                                                             ascending=False)
    st.caption(f"{len(made)} makes vs {len(miss)} misses. Cohen's d = how separated "
               f"(|d|≥0.4 notable). Small samples — leads, not proof.")
    bars = (alt.Chart(rd).mark_bar().encode(
                x=alt.X("d:Q", title="Cohen's d  (makes − misses)"),
                y=alt.Y("metric:N", sort=None, title=None),
                color=alt.condition("abs(datum.d) >= 0.4", alt.value("#2a7"), alt.value("#bbb")),
                tooltip=["metric", "makes", "misses", alt.Tooltip("d:Q", format="+.2f")])
            .properties(height=28*len(rd)+20))
    st.altair_chart(bars, use_container_width=True)
    lab = st.selectbox("Look at one metric's distribution", [r["metric"] for _, r in rd.iterrows()])
    k = rd[rd["metric"] == lab]["_k"].iloc[0]
    dd = pd.concat([made[[k]].assign(result="make"), miss[[k]].assign(result="miss")]).dropna()
    hist = (alt.Chart(dd).mark_bar(opacity=0.6).encode(
                x=alt.X(f"{k}:Q", bin=alt.Bin(maxbins=20), title=lab),
                y=alt.Y("count()", title="shots", stack=None),
                color=alt.Color("result:N", scale=alt.Scale(domain=["make", "miss"],
                                range=["#2a9d4a", "#c0392b"])))
            .properties(height=220))
    st.altair_chart(hist, use_container_width=True)


# ---------------- closest-to-ideal leaderboard
def view_ideal_board():
    st.subheader("🏅 Closest-to-ideal reps — study your best form")
    prof = _profile()
    ideal, tol = prof.get("ideal", {}), prof.get("tolerance", {})
    if not ideal:
        st.info("No profile ideals yet (run tools/export_profile.py)."); return
    df = _all_shots(_shots_sig())
    if df.empty:
        st.info("No shots yet."); return
    metrics = [(k, lab) for k, lab in _MEASURABLES if k in df and df[k].notna().any()]
    used = [k for k in ideal if k in df.columns and tol.get(k)]
    with st.sidebar:
        st.markdown("### Leaderboard")
        sel = _date_picker(df, "Sessions (date)", multi=True, key="ib_sess")
        outc = st.multiselect("Outcome", ["make", "miss"], default=["make", "miss"], key="ib_o")
        vonly = st.checkbox("Verified make/miss only", value=False, key="ib_v")
    f = df[df["date_label"].isin(sel) & df["outcome"].isin(outc)].copy()
    if vonly:
        f = f[f["verified"]]
    if f.empty or not used:
        st.info("No shots / ideal metrics for that filter."); return

    def _dist(r):
        ds = [abs(r[k] - ideal[k]) / tol[k] for k in used
              if pd.notna(r.get(k)) and tol[k]]
        return float(np.mean(ds)) if ds else np.nan
    f = f.reset_index(drop=True)
    f["form_gap"] = f.apply(_dist, axis=1)
    f = f[f["form_gap"].notna()].sort_values("form_gap").reset_index(drop=True)
    st.caption(f"Ranked by average distance from your ideal ({', '.join(k for k in used)}), "
               f"in tolerance units — lower = closer to ideal form. Click a row to watch.")
    show = ["form_gap", "outcome", "date_label"] + [k for k, _ in metrics if k in used]
    disp = f[show].rename(columns=dict(metrics, form_gap="ideal gap", date_label="date",
                                       outcome="result")).round(2)
    ev = st.dataframe(disp, hide_index=True, use_container_width=True,
                      on_select="rerun", selection_mode="single-row", height=340)
    picks = ev.selection.rows if ev and ev.selection else []
    if picks:
        _shot_detail(f.iloc[picks[0]], metrics)
    else:
        st.info("👆 Click a row to watch that rep (top rows = closest to your ideal).")


# ---------------------------------------------------------------- main
st.title("🏀 ShotLab")
mode = st.sidebar.radio("View", ["Shot Explorer", "Shot scatter", "Shot chart",
                                 "Makes vs misses", "Closest to ideal",
                                 "Session analytics", "3D analysis", "Make/miss audit",
                                 "Film room", "Shot review", "Per-clip", "Compare shots",
                                 "Compare sessions", "Progress"])
if mode == "Shot Explorer":
    view_explore()
elif mode == "Shot scatter":
    view_scatter()
elif mode == "Shot chart":
    view_shotchart()
elif mode == "Makes vs misses":
    view_makesplit()
elif mode == "Closest to ideal":
    view_ideal_board()
elif mode == "Per-clip":
    view_clip()
elif mode == "Session analytics":
    view_session()
elif mode == "3D analysis":
    view_3d()
elif mode == "Make/miss audit":
    view_make_audit()
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
st.sidebar.caption("Release/entry angles are FORESHORTENED on one wide camera "
                   "(advisory until court-corner / 2-cam calibration). Apex-ft = "
                   "MEDIUM. Elbow flare/squareness = LOW (1 camera). Make% = LOW.")
