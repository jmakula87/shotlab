"""Head-to-head: motion detector vs the fine-tuned YOLO on one clip.
Reports per-frame ball coverage and rim-anchored shot count for each."""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.phase1_ball.detect import MotionBallDetector
from shotlab.phase1_ball.detect_yolo import YoloBallDetector
from shotlab.phase1_ball.pipeline import run_phase1
from shotlab.court import auto_calibrate, filter_shots_by_rim
from shotlab.video_io import probe

VID = sys.argv[1] if len(sys.argv) > 1 else "data/raw/Hoops/PXL_20260628_192125174.mp4"
WEIGHTS = "runs/detect/ball_finetune/weights/best.pt"

info = probe(VID)
calib = auto_calibrate(VID, os.path.basename(VID))
print(f"clip {os.path.basename(VID)}  {info.n_frames} frames @ {info.fps:.0f}fps")
print(f"rim @ ({calib.rim_x:.0f},{calib.rim_y:.0f})\n")

for name, det in [
    ("motion", MotionBallDetector()),
    ("yolo_finetuned", YoloBallDetector(weights=WEIGHTS, ball_class=0,
                                        conf=0.25, imgsz=768)),
]:
    res = run_phase1(VID, detector=det)
    shots, rej = filter_shots_by_rim(res.shots, calib)
    cov = len(res.track)
    print(f"{name:16s}  ball tracked in {cov:5d} frames  "
          f"({100*cov/info.n_frames:.1f}%)  |  raw flights {len(res.shots):3d}  "
          f"-> rim shots {len(shots)}")
