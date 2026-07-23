"""Turn per-shot metrics into a reviewable session table + deviation flags.

Confidence levels are attached per metric and honor the camera angle, following
the 2026 pose/detection survey:
  - arc release/entry angles: HIGH on a square side-on camera (in-plane).
  - apex height in feet: MEDIUM (depends on the ball-diameter pixel scale).
  - form metrics get their confidence from phase2 (depth-dependent ones are LOW).
"""

from __future__ import annotations

import json
import os

import pandas as pd

from .config import evaluate, Flag

# Arc metrics -> config key + confidence, keyed by whether the camera is SIDE-ON.
# release/entry angle assume the optical axis is perpendicular to the flight plane;
# on an oblique/behind view they carry an unquantified projection bias, so they are
# image-space diagnostics only (2026-07-23 honesty pass -- were hardcoded "high"
# regardless of the actual camera). apex_height also depends on the rim vs ball
# ruler (which disagree ~1.6x under foreshortening), so never above medium.
_ARC_METRICS_SIDE_ON = {
    "release_angle_deg": ("release_angle_deg", "medium"),
    "entry_angle_deg": ("entry_angle_deg", "medium"),
    "apex_height_ft": ("apex_height_ft", "medium"),
}
_ARC_METRICS_OBLIQUE = {
    "release_angle_deg": ("release_angle_deg", "low"),
    "entry_angle_deg": ("entry_angle_deg", "low"),
    "apex_height_ft": ("apex_height_ft", "low"),
}


def flag_arc_metrics(metric_row: dict, targets: dict) -> list[Flag]:
    arc_targets = targets.get("arc", {})
    side_on = metric_row.get("camera_angle") == "side_on"
    table = _ARC_METRICS_SIDE_ON if side_on else _ARC_METRICS_OBLIQUE
    flags = []
    for key, (cfg_key, conf) in table.items():
        spec = arc_targets.get(cfg_key, {})
        flags.append(evaluate(cfg_key, metric_row.get(key), spec, confidence=conf))
    return flags


# Form metric -> config key under targets["form"].
_FORM_KEYS = [
    "elbow_angle_at_release_deg", "knee_bend_deg", "release_vs_apex_s",
    "follow_through_hold_s", "balance_drift_px_per_ht", "squareness_deg",
]


def flag_form_metrics(shot_form, targets: dict) -> list[Flag]:
    form_targets = targets.get("form", {})
    by_name = {m.name: m for m in shot_form.metrics}
    flags = []
    for key in _FORM_KEYS:
        m = by_name.get(key)
        if m is None:
            continue
        spec = form_targets.get(key, {})
        flags.append(evaluate(key, m.value, spec, confidence=m.confidence,
                              note=m.note))
    return flags


def build_combined_table(arc_metrics: list, forms: list, targets: dict,
                         spins: dict | None = None) -> pd.DataFrame:
    """Merge Phase-1 arc metrics and Phase-2 form metrics into one session table,
    with a single 'flags' column. Low-confidence metrics are flagged but tagged
    with '?' so they read as advisory, never hard errors."""
    forms_by_shot = {f.shot: f for f in (forms or [])}
    rows = []
    for m in arc_metrics:
        row = m.as_row()
        flags = flag_arc_metrics(row, targets)

        sf = forms_by_shot.get(m.shot)
        if sf is not None:
            row["release_frame"] = sf.release_frame
            row["release_conf"] = sf.release_conf
            for fm in sf.metrics:
                row[fm.name] = fm.value
                row[fm.name + "_conf"] = fm.confidence
            flags += flag_form_metrics(sf, targets)

        if spins:
            sp = spins.get(m.shot)
            if sp is not None:
                row.update(sp.as_row())
                spec = targets.get("spin", {}).get("backspin_rpm", {})
                if sp.backspin_rpm is not None:
                    flags.append(evaluate("backspin_rpm", sp.backspin_rpm, spec,
                                          confidence=sp.confidence, note=sp.note))

        bad = []
        for f in flags:
            if f.status in ("low", "high"):
                tag = "?" if f.confidence == "low" else ""
                bad.append(f"{f.metric}:{f.status}{tag}")
        row["flags"] = ", ".join(bad) if bad else "ok"
        rows.append(row)
    return pd.DataFrame(rows)


def build_session_table(metrics: list, targets: dict) -> pd.DataFrame:
    """metrics: list of ShotArcMetrics (phase1). Returns a tidy DataFrame with a
    'flags' column summarizing out-of-band metrics."""
    rows = []
    for m in metrics:
        row = m.as_row()
        flags = flag_arc_metrics(row, targets)
        bad = [f"{f.metric}:{f.status}" for f in flags if f.status in ("low", "high")]
        row["flags"] = ", ".join(bad) if bad else "ok"
        rows.append(row)
    return pd.DataFrame(rows)


def write_outputs(df: pd.DataFrame, out_dir: str, stem: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{stem}_shots.csv")
    json_path = os.path.join(out_dir, f"{stem}_shots.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)
    return {"csv": csv_path, "json": json_path}


def print_table(df: pd.DataFrame) -> None:
    if df.empty:
        print("No shots detected.")
        return
    cols = [c for c in ["shot", "release_angle_deg", "entry_angle_deg",
                        "apex_height_ft", "n_points", "fit_rmse_px", "direction",
                        "flags"] if c in df.columns]
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df[cols].to_string(index=False))
