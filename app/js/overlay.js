// Skeleton rendering. The canvas is sized to the video's intrinsic resolution
// and CSS-scaled identically to the <video> (both object-fit:contain in the same
// box), so landmark coords (x*canvasW, y*canvasH) line up with the picture.

const CONNECTIONS = [
  [11, 12], [11, 23], [12, 24], [23, 24],   // torso
  [12, 14], [14, 16],                       // right arm
  [11, 13], [13, 15],                       // left arm
  [24, 26], [26, 28],                       // right leg
  [23, 25], [25, 27],                       // left leg
];
const MIN_VIS = 0.3;

function clear(ctx) { ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); }

export function drawSkeleton(ctx, lm, { color = "#39d98a", width = 4,
                                        dot = 5 } = {}) {
  const W = ctx.canvas.width, H = ctx.canvas.height;
  ctx.lineWidth = width; ctx.strokeStyle = color; ctx.fillStyle = color;
  for (const [a, b] of CONNECTIONS) {
    if ((lm[a].visibility ?? 1) < MIN_VIS || (lm[b].visibility ?? 1) < MIN_VIS) continue;
    ctx.beginPath();
    ctx.moveTo(lm[a].x * W, lm[a].y * H);
    ctx.lineTo(lm[b].x * W, lm[b].y * H);
    ctx.stroke();
  }
  for (const i of [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]) {
    if ((lm[i].visibility ?? 1) < MIN_VIS) continue;
    ctx.beginPath();
    ctx.arc(lm[i].x * W, lm[i].y * H, dot, 0, Math.PI * 2);
    ctx.fill();
  }
}

const mid = (lm, a, b) => ({ x: (lm[a].x + lm[b].x) / 2, y: (lm[a].y + lm[b].y) / 2 });
const dist = (p, q) => Math.hypot(p.x - q.x, p.y - q.y);

// Align an ideal skeleton onto the actual one (translate to the same shoulder
// center, scale by shoulder->hip length) so limb angles compare visually.
export function drawIdealAligned(ctx, idealLm, actualLm, opts = {}) {
  const aS = mid(actualLm, 11, 12), aH = mid(actualLm, 23, 24);
  const iS = mid(idealLm, 11, 12), iH = mid(idealLm, 23, 24);
  const aScale = dist(aS, aH), iScale = dist(iS, iH) || 1e-6;
  const s = aScale / iScale;
  const warped = idealLm.map(p => ({
    x: aS.x + (p.x - iS.x) * s,
    y: aS.y + (p.y - iS.y) * s,
    visibility: p.visibility ?? 1,
  }));
  drawSkeleton(ctx, warped, { color: opts.color || "#ffb020", width: 3, dot: 4 });
}

export function render(ctx, actualLm, idealLm, showIdeal) {
  clear(ctx);
  if (idealLm && showIdeal) drawIdealAligned(ctx, idealLm, actualLm);
  if (actualLm) drawSkeleton(ctx, actualLm);
}

export { clear };
