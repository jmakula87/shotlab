"""Why a single fixed camera can't be side-on for shots from all over the arc:
shot lines radiate from the hoop, so the camera is perpendicular to only some."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np

H = np.array([0.0, 8.0])          # hoop
C = np.array([-10.0, 3.5])        # fixed dead-side-on camera (left sideline)
R = 6.0                            # 3pt arc radius

fig, ax = plt.subplots(figsize=(11, 8.5))

# 3-point arc (lower half around the hoop = the court side)
ang = np.linspace(200, 340, 200)
arc = H[:, None] + R * np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
ax.plot(arc[0], arc[1], color="#bbb", lw=2)
ax.add_patch(Circle(H, 0.22, color="#e8731e")); ax.text(H[0], H[1]+0.4, "HOOP", ha="center", weight="bold")

# camera
ax.add_patch(Circle(C, 0.22, color="#333"))
ax.text(C[0], C[1]-0.7, "fixed\ncamera", ha="center", fontsize=10, weight="bold")
ax.add_patch(FancyArrowPatch(C, H, arrowstyle="->", mutation_scale=14, color="#333", ls=":", lw=1.2))

# several shooting spots around the arc
spots_deg = [205, 235, 265, 295, 325]
labels = ["L corner", "L wing", "top", "R wing", "R corner"]
for a, lab in zip(spots_deg, labels):
    P = H + R * np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
    shot_dir = H - P                       # ball travels toward the hoop
    sight = P - C                          # camera's line of sight to the shooter
    # "accuracy" = how perpendicular the shot is to the sightline (ball moving
    # across the view = good; toward/away from camera = foreshortened)
    cs = abs(np.cross(shot_dir, sight)) / (np.linalg.norm(shot_dir)*np.linalg.norm(sight))
    color = plt.cm.RdYlGn(cs)              # red(bad) -> green(good)
    ax.add_patch(FancyArrowPatch(P, H, arrowstyle="->", mutation_scale=14,
                 color=color, lw=3))
    ax.add_patch(Circle(P, 0.18, color="#2b6cb0"))
    pct = int(round(cs*100))
    ax.text(P[0], P[1]-0.55, f"{lab}\n{pct}% side-on", ha="center", fontsize=9,
            color="#1e8449" if cs>0.8 else ("#b8860b" if cs>0.55 else "#c0392b"))

ax.text(-10.5, 9.6,
        "Each arrow = a shot (ball → hoop).\n"
        "GREEN = travels across the camera → angles accurate.\n"
        "RED = travels toward/away from camera → foreshortened (like now).",
        fontsize=10, va="top",
        bbox=dict(boxstyle="round", fc="#f4f6f8", ec="#999"))

ax.set_title("One fixed camera is side-on for SOME spots, not all\n"
             "(shot lines fan out from the hoop)", fontsize=13, weight="bold")
ax.set_aspect("equal"); ax.axis("off")
ax.set_xlim(-12, 8); ax.set_ylim(-1, 10.5)
fig.savefig("data/out/moving_shooter_geometry.png", dpi=120, bbox_inches="tight")
print("saved data/out/moving_shooter_geometry.png")
