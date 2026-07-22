"""ROCm-on-Windows BatchNorm bypass (RX 9070 XT / RDNA4 / gfx1201).

Why this exists: on AMD's ROCm-for-Windows (7.2.1 AND 7.13.0 preview), MIOpen
cannot JIT-compile its BatchNorm kernels for gfx1201 -- HIPRTC dies with
`'type_traits' file not found` building MIOpenBatchNormFwd{Train,Infer}Spatial
(a header-packaging bug, ROCm/ROCm #6150). Everything else on the GPU works
(matmul, and crucially CONVOLUTION), so training only breaks when it hits BN.

Fix: replace nn.BatchNorm2d.forward with pure-torch primitives (mean/var/rsqrt,
running-stat updates by hand) for BOTH train and eval, so MIOpen's BN kernels are
never invoked. Conv stays on MIOpen (GPU-accelerated). Measured: yolo11n trains
on the 9070 XT at imgsz 1280 in ~3 min/epoch (steady state) vs ~10 min on the
Ryzen 9 7900X. Numerically equivalent to real BN (reduction-order differences
only); pretrained BN weights + running stats + state-dict keys are preserved.

Call maybe_apply(device) once before building the model. It is a no-op on CPU and
on non-ROCm torch (so the same train script runs unpatched on the CPU env). Remove
this once AMD ships a ROCm-Windows build whose MIOpen BN kernels compile for
gfx1201 -- then native MIOpen BN is faster.
"""

from __future__ import annotations

_applied = False


def maybe_apply(device) -> bool:
    """Patch BatchNorm2d to pure-torch primitives IFF we're on ROCm + a GPU
    device. Returns True if patched. Idempotent."""
    global _applied
    if _applied:
        return True
    import torch
    # only on a real GPU device and only when torch is a ROCm build
    if str(device).lower() in ("cpu", "", "none") or getattr(torch.version, "hip", None) is None:
        return False
    import torch.nn as nn

    def _prim_bn(self, x):
        s = (1, -1, 1, 1)
        if self.training and self.track_running_stats:
            d = (0, 2, 3)
            m = x.mean(d); v = x.var(d, unbiased=False)
            with torch.no_grad():
                self.num_batches_tracked.add_(1)
                f = self.momentum if self.momentum is not None else 1.0 / float(self.num_batches_tracked)
                n = x.numel() // x.shape[1]
                uv = v.detach() * (n / (n - 1) if n > 1 else 1.0)
                self.running_mean.mul_(1 - f).add_(m.detach(), alpha=f)
                self.running_var.mul_(1 - f).add_(uv, alpha=f)
        elif self.training:                       # no running stats -> batch stats
            d = (0, 2, 3)
            m = x.mean(d); v = x.var(d, unbiased=False)
        else:                                     # eval -> running stats
            m = self.running_mean; v = self.running_var
        y = (x - m.view(s)) * torch.rsqrt(v.view(s) + self.eps)
        if self.affine:
            y = y * self.weight.view(s) + self.bias.view(s)
        return y

    nn.BatchNorm2d.forward = _prim_bn
    _applied = True
    print("[rocm_bn_patch] BatchNorm2d -> pure-torch primitives "
          "(MIOpen BN kernels don't compile for gfx1201 on ROCm-Windows)")
    return True
