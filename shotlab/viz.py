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
