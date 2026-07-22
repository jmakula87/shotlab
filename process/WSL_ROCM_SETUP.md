# WSL2 + Linux ROCm — correct GPU training for ShotLab (RX 9070 XT / gfx1201)

The reliable route to **correct** GPU training after the two Windows dead-ends
(native ROCm-Windows froze the box; torch-directml silently zeroed the loss — see
`GPU_SETUP.md`). Linux ROCm on WSL2 has mature MIOpen and no silent-op problem.

**Compatibility confirmed (2026-07-22):** ROCm **7.2** officially supports the
RX 9070 XT (gfx1201) on WSL2 via the ROCDXG/`librocdxg` path, on Ubuntu 24.04 /
22.04, with AMD Adrenalin ≥ 26.1.1 for WSL2. This box: driver `32.0.31021.5001`
(2026-06-27, recent enough), 103 GB free on C:, Windows build 26200,
virtualization on. Sources in the session log.

---

## Division of labor
- **You (once):** the elevated install + reboot + Ubuntu account creation. WSL
  isn't installed and I'm not running as admin, so I can't do this part.
- **Me (after):** everything inside WSL — I drive it via `wsl -u root -- bash -c`
  (root needs no password), so no further elevation or interaction needed.

---

## STEP 1 — YOU: install WSL2 + Ubuntu 24.04 (elevated, one reboot)

Open **Windows Terminal / PowerShell as Administrator** (right-click → Run as
administrator) and run:

```powershell
wsl --install -d Ubuntu-24.04
```

Then **reboot** if it asks. After reboot, an **Ubuntu** window opens and asks you
to create a **UNIX username and password** — pick anything you'll remember (this
is your WSL sudo password; I won't need it since I use the root path). Once you're
at the Ubuntu shell prompt, WSL is ready.

Quick sanity check you can run back here (non-elevated) once it's done:
```powershell
wsl -l -v          # should list Ubuntu-24.04, VERSION 2, and its state
```

Tell me when Step 1 is done and I'll take it from here.

---

## STEP 2 — ME: ROCm 7.2 for WSL (verified AMD commands)

Run inside WSL as root (I'll invoke these; listed for the record):
```bash
sudo apt update
wget https://repo.radeon.com/amdgpu-install/7.2/ubuntu/noble/amdgpu-install_7.2.70200-1_all.deb
sudo apt install -y ./amdgpu-install_7.2.70200-1_all.deb
amdgpu-install -y --usecase=wsl,rocm --no-dkms   # --no-dkms: WSL uses the Windows driver, no kernel module
rocminfo | grep -E 'gfx|Name'                    # must show gfx1201
```
`--usecase=wsl` pulls the ROCDXG runtime that bridges to the Windows driver.

## STEP 3 — ME: PyTorch for ROCm + the WSL libhsa swap (the classic gotcha)

Use AMD's **repo.radeon.com** wheels (not the PyTorch Foundation nightlies — AMD
doesn't test those for Radeon-WSL), then replace the wheel's bundled
`libhsa-runtime64.so` with the WSL-compatible one from `/opt/rocm/lib`, or the GPU
won't be found. Exact wheel filenames pulled live at run time from the ROCm-7.2
Radeon-WSL PyTorch index; procedure:
```bash
python3.12 -m venv ~/shotlab-rocm && source ~/shotlab-rocm/bin/activate
pip install --upgrade pip "numpy==1.26.4"
# install AMD ROCm torch + torchvision wheels (URLs resolved from the 7.2 index)
pip install <torch-rocm7.2 wheel> <torchvision-rocm7.2 wheel>
# WSL fix: swap in the WSL-compatible HSA runtime
loc=$(pip show torch | awk '/Location/{print $2}')/torch/lib
rm -f "$loc"/libhsa-runtime64.so*
cp /opt/rocm/lib/libhsa-runtime64.so.1.2 "$loc/libhsa-runtime64.so"
```

## STEP 4 — ME: verify correct GPU compute
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: 2.x+rocm7.2  True  AMD Radeon RX 9070 XT
```

## STEP 5 — ME: train ShotLab on the GPU
The repo is reachable from WSL at `/mnt/c/Users/jmaku/Desktop/ShotLab`.
```bash
cd /mnt/c/Users/jmaku/Desktop/ShotLab
pip install ultralytics opencv-python
python -X utf8 tools/train_ball.py --data dataset_ball_human.yaml \
    --base runs/detect/ball_orange/weights/best.pt \
    --imgsz 1280 --epochs 40 --batch 16 --freeze 10 --device 0 --name ball_gpu_wsl
```
Notes:
- **Start with a SMALL batch (16, not 48)** and confirm one epoch completes +
  writes `results.csv` before scaling up. (Batch 48 is what froze Windows.)
- **BN patch:** `tools/rocm_bn_patch.py` was written for the *Windows* MIOpen BN
  bug. Linux ROCm's MIOpen compiles BN fine, so the patch should be **disabled**
  on WSL (it reimplements BN in slower pure-torch). Verify `maybe_apply` is
  Windows-gated, or add a `platform.system()=='Linux'` bypass before the real run.
- **First validate correctness the same way we caught DirectML:** confirm
  box_loss/dfl_loss are NONZERO in the first steps (DirectML zeroed them). Linux
  ROCm should match CPU.
- I/O over `/mnt/c` is slower than WSL-native FS; if data loading bottlenecks,
  copy `dataset_ball_*` into the WSL home first.
