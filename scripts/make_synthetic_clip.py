"""Generate a synthetic 'shooting workout' clip to validate Phase 1 end-to-end
without real footage.

Renders a side-on view: a floor line, a backboard+rim at the right, and an
orange ball launched through several parabolic arcs (each = one 'shot') at known
release angles. Because we KNOW the ground-truth angles, this doubles as an
accuracy check for the whole pipeline (detect -> track -> fit -> angles).

Usage:
    python scripts/make_synthetic_clip.py [out.mp4] [--fps 60] [--shots 5]
"""

import argparse
import math
import os

import cv2
import numpy as np

W, H = 1280, 720
FLOOR_Y = 640
RIM_X, RIM_Y = 1120, 300          # rim center
BALL_R = 17                       # px radius -> ~34px diameter ball


def render(out_path, fps=60, shots=5, seed=7):
    rng = np.random.default_rng(seed)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    # ground-truth release angles for the shots (degrees)
    true_angles = np.linspace(48, 56, shots)
    ground_truth = []

    def draw_court(img):
        img[:] = (40, 40, 45)
        cv2.line(img, (0, FLOOR_Y), (W, FLOOR_Y), (90, 90, 95), 4)
        # backboard + rim
        cv2.line(img, (RIM_X + 60, RIM_Y - 90), (RIM_X + 60, RIM_Y + 40),
                 (200, 200, 205), 6)
        cv2.line(img, (RIM_X - 25, RIM_Y), (RIM_X + 25, RIM_Y), (0, 140, 255), 5)

    # idle frames between shots
    for si in range(shots):
        ang = math.radians(true_angles[si])
        # launch from a shooter position on the left
        x0, y0 = 250.0, FLOOR_Y - 230   # release point ~ head height
        # choose speed so the apex clears and ball lands near the rim x
        dx = RIM_X - x0
        g = 1500.0                      # px/s^2 (arbitrary but consistent)
        # solve for v given angle and that it passes x=dx at the descending rim
        v = math.sqrt(g * dx / max(math.sin(2 * ang), 0.2))
        vx, vy = v * math.cos(ang), v * math.sin(ang)
        t_flight = dx / vx
        n = max(12, int(t_flight * fps))
        for k in range(n):
            t = k / fps
            x = x0 + vx * t
            h = vy * t - 0.5 * g * t * t
            y = (FLOOR_Y - 230) - h
            img = np.zeros((H, W, 3), np.uint8)
            draw_court(img)
            jitter = rng.normal(0, 0.4, 2)   # tiny sub-pixel jitter
            cx, cy = int(round(x + jitter[0])), int(round(y + jitter[1]))
            cv2.circle(img, (cx, cy), BALL_R, (30, 120, 235), -1)   # orange BGR
            cv2.circle(img, (cx, cy), BALL_R, (20, 80, 160), 2)
            vw.write(img)
        ground_truth.append({"shot": si + 1, "release_angle_deg": round(float(true_angles[si]), 2)})
        # gap (no ball) so the segmenter can split shots
        for _ in range(int(0.5 * fps)):
            img = np.zeros((H, W, 3), np.uint8)
            draw_court(img)
            vw.write(img)

    vw.release()
    return ground_truth


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="data/raw/synthetic_side.mp4")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--shots", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    gt = render(args.out, fps=args.fps, shots=args.shots)
    print(f"Wrote {args.out} @ {args.fps}fps, {args.shots} shots")
    print("Ground-truth release angles:")
    for g in gt:
        print(f"  shot {g['shot']}: {g['release_angle_deg']} deg")
