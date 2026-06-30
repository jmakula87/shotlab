"""Generate an explainer diagram: current diagonal camera vs dead-side-on,
and what each does to the ball arc in-frame."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle
import numpy as np

fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.28, wspace=0.18)

# ---------------- TOP: bird's-eye court + camera placement ----------------
ax = fig.add_subplot(gs[0, :])
ax.set_title("Bird's-eye view — where to put the camera", fontsize=14, weight="bold")
# court
ax.add_patch(Rectangle((0, 0), 10, 6, fill=False, ec="#888", lw=2))
ax.plot([5, 5], [0, 6], color="#ddd", lw=1)
# hoop at top baseline
ax.add_patch(Circle((5, 5.7), 0.18, color="#e8731e"))
ax.text(5, 6.15, "HOOP", ha="center", fontsize=10, weight="bold")
# shooter
ax.add_patch(Circle((5, 1.6), 0.16, color="#2b6cb0"))
ax.text(5, 1.1, "you (shooter)", ha="center", fontsize=10)
# shot line
ax.add_patch(FancyArrowPatch((5, 1.8), (5, 5.5), arrowstyle="->",
             mutation_scale=18, color="#2b6cb0", lw=2))
ax.text(5.25, 3.6, "shot travels\nup the court", color="#2b6cb0", fontsize=9)

# Camera A: current diagonal/behind
camA = (8.6, 1.0)
ax.add_patch(Circle(camA, 0.14, color="#c0392b"))
ax.add_patch(FancyArrowPatch(camA, (5.3, 4.8), arrowstyle="->", mutation_scale=12,
             color="#c0392b", lw=1.5, ls="--"))
ax.text(8.7, 0.5, "✗ current (diagonal / behind)\nshot goes INTO the frame\n→ arc foreshortened",
        color="#c0392b", fontsize=9, ha="center")

# Camera B: dead side-on
camB = (0.5, 3.5)
ax.add_patch(Circle(camB, 0.14, color="#1e8449"))
ax.add_patch(FancyArrowPatch(camB, (4.7, 3.5), arrowstyle="->", mutation_scale=14,
             color="#1e8449", lw=2))
ax.text(0.5, 4.05, "✓ DEAD SIDE-ON", color="#1e8449", fontsize=11, weight="bold")
ax.text(0.5, 2.55, "camera 90° to the shot line,\nat ~chest height.\nBall travels ACROSS the frame.",
        color="#1e8449", fontsize=9, ha="left")
# right-angle marker
ax.plot([4.7, 4.7, 5.0], [3.2, 3.5, 3.5], color="#1e8449", lw=1)
ax.text(4.55, 4.4, "90°", color="#1e8449", fontsize=9)

ax.set_xlim(-0.5, 10.5); ax.set_ylim(0, 6.6); ax.set_aspect("equal"); ax.axis("off")

# ---------------- helper to draw a "camera frame" ----------------
def frame(ax, title, ok):
    ax.add_patch(Rectangle((0, 0), 16, 9, fill=True, fc="#f4f6f8", ec="#333", lw=2))
    ax.set_xlim(-0.5, 16.5); ax.set_ylim(-0.5, 9.5); ax.set_aspect("equal"); ax.axis("off")
    col = "#1e8449" if ok else "#c0392b"
    ax.set_title(title, fontsize=12, weight="bold", color=col)

# ---------------- BOTTOM-LEFT: diagonal view (bad) ----------------
ax1 = fig.add_subplot(gs[1, 0])
frame(ax1, "✗ What you get now (diagonal)", ok=False)
# hoop top-left corner
ax1.add_patch(Circle((3.0, 8.0), 0.4, fc="none", ec="#e8731e", lw=3))
# steep foreshortened arc from bottom-center to hoop
t = np.linspace(0, 1, 50)
ax1.plot(7 - 4*t, 1.2 + 6.8*t - 0.6*np.sin(np.pi*t), color="#c0392b", lw=2.5)
ax1.text(8, 4.5, "arc looks steep &\ncompressed; ball moves\nAWAY toward the hoop",
         fontsize=9, color="#c0392b")
ax1.text(8, 1.6, "release/entry angles\nDISTORTED", fontsize=9, color="#c0392b", weight="bold")

# ---------------- BOTTOM-RIGHT: side-on view (good) ----------------
ax2 = fig.add_subplot(gs[1, 1])
frame(ax2, "✓ Dead side-on (what we want)", ok=True)
# hoop on the right at rim height
ax2.add_patch(Circle((13.5, 5.6), 0.45, fc="none", ec="#e8731e", lw=3))
ax2.text(13.5, 6.5, "rim", fontsize=9, ha="center", color="#e8731e")
# clean wide parabola left->right
x = np.linspace(2.5, 13.3, 60)
y = 2.2 + (x - 2.5) * 1.15 - 0.085 * (x - 2.5) ** 2
ax2.plot(x, y, color="#1e8449", lw=2.5)
ax2.add_patch(Circle((2.5, 2.2), 0.32, color="#1e8449"))   # release
# release angle marker
ax2.plot([2.5, 4.5], [2.2, 2.2], color="#333", lw=1, ls=":")
ax2.annotate("release\nangle", (3.4, 3.1), fontsize=9, color="#1e8449")
# entry angle marker at rim
ax2.plot([13.5, 11.8], [5.6, 5.6], color="#333", lw=1, ls=":")
ax2.annotate("entry\nangle", (11.0, 6.6), fontsize=9, color="#1e8449")
ax2.text(2.4, 0.8, "full parabola ACROSS the frame → release, apex & entry all read TRUE",
         fontsize=9, color="#1e8449")

fig.suptitle("Filming for accurate arc metrics: go dead side-on",
             fontsize=15, weight="bold", y=0.98)
fig.savefig("data/out/dead_side_on_guide.png", dpi=120, bbox_inches="tight")
print("saved data/out/dead_side_on_guide.png")
