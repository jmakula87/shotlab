"""Shared plotting helpers (used by both the dashboard and the PDF export)."""

from __future__ import annotations


def _norm_made(v):
    if v in (True, "True"):
        return True
    if v in (False, "False"):
        return False
    return None


def draw_court(ax, df):
    """Draw the schematic half-court onto `ax`: the 9 zones (near/mid/far ×
    left/center/right) shaded by make% (green=high, red=low, grey=no makes),
    annotated with shot counts. Honest to our zone system -- true court
    coordinates need calibration."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle

    depths, sides = ["near", "mid", "far"], ["left", "center", "right"]
    cmap = plt.get_cmap("RdYlGn")
    ax.set_xlim(0, 3); ax.set_ylim(-0.3, 3.7); ax.axis("off")
    ax.add_patch(Circle((1.5, 3.25), 0.12, color="#e8703a", zorder=5))
    ax.text(1.5, 3.46, "RIM", ha="center", va="bottom", fontsize=9, color="#444")
    has_depth = "depth" in df.columns and "side" in df.columns
    for di, depth in enumerate(depths):
        for si, side in enumerate(sides):
            n, mk = 0, float("nan")
            if has_depth:
                sub = df[(df["depth"] == depth) & (df["side"] == side)]
                n = len(sub)
                if n and "made" in sub.columns:
                    mm = sub["made"].map(_norm_made).dropna()
                    if len(mm):
                        mk = float((mm == True).mean())
            y = 2 - di
            color = "#e9edf3" if n == 0 else (cmap(mk) if mk == mk else "#c8d0dc")
            ax.add_patch(Rectangle((si, y), 1, 1, facecolor=color,
                                   edgecolor="white", lw=2))
            txt = "" if n == 0 else (f"{n} shot{'s' if n != 1 else ''}"
                                     + (f"\n{mk*100:.0f}% make" if mk == mk else ""))
            ax.text(si + 0.5, y + 0.5, txt, ha="center", va="center",
                    fontsize=9, color="#1b2430")
    for si, side in enumerate(sides):
        ax.text(si + 0.5, -0.18, side, ha="center", fontsize=8, color="#888")
    for di, depth in enumerate(depths):
        ax.text(-0.05, 2 - di + 0.5, depth, ha="right", va="center",
                fontsize=8, color="#888")


def draw_shot_map(ax, df):
    """Per-shot release points relative to the rim -- an image-space proxy
    (x = pixels beside the rim, y = pixels below it; the camera's view, not
    court feet, and the left/right sign depends on which side the camera sat).
    Make = filled dot, miss = X: the marker SHAPE carries make/miss too, so the
    map still reads for colorblind viewers and in print."""
    if not {"rim_dx_px", "rim_dy_px"}.issubset(df.columns):
        ax.text(0.5, 0.5, "rebuild the session to add per-shot positions",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="#888")
        ax.axis("off")
        return
    sub = df.dropna(subset=["rim_dx_px", "rim_dy_px"])
    if not len(sub):
        ax.text(0.5, 0.5, "no shot positions in this session",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="#888")
        ax.axis("off")
        return
    made = sub["made"].map(_norm_made) if "made" in sub.columns else None
    groups = [("make", made == True, dict(marker="o", s=55, facecolor="#2e7d32",
                                          edgecolor="white", linewidth=1.2)),
              ("miss", made == False, dict(marker="X", s=65, color="#c62828",
                                           edgecolor="white", linewidth=0.6)),
              ("unknown", made.isna(), dict(marker="o", s=45, facecolor="#c8d0dc",
                                            edgecolor="white", linewidth=1.2))] \
        if made is not None else \
        [("shot", sub["rim_dx_px"].notna(), dict(marker="o", s=55,
                                                 facecolor="#5b7fa6",
                                                 edgecolor="white", linewidth=1.2))]
    shown = 0
    for label, mask, style in groups:
        g = sub[mask]
        if len(g):
            ax.scatter(g["rim_dx_px"], g["rim_dy_px"], label=label,
                       zorder=3, **style)
            shown += 1
    ax.scatter([0], [0], s=110, color="#e8703a", zorder=5)
    ax.annotate("RIM", (0, 0), xytext=(0, 12), textcoords="offset points",
                ha="center", fontsize=9, color="#444")
    ax.invert_yaxis()                      # image y grows down: rim on top
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, color="#e9edf3", linewidth=0.8, zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors="#888", labelsize=8)
    ax.set_xlabel("px beside rim (sign = camera side)", fontsize=8, color="#888")
    ax.set_ylabel("px below rim", fontsize=8, color="#888")
    if shown >= 2:
        ax.legend(loc="lower right", fontsize=8, frameon=False)
