"""refine_release_frame must reject bent-arm (gather/pump) frames and snap to the
extended-arm release -- the fix for bogus high flare readings."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.phase2_pose.pose import FramePose, L
from shotlab.analysis3d import refine_release_frame


def _fp(idx, sh, el, wr, nose):
    xy = np.zeros((33, 2)); vis = np.ones(33)
    xy[L["r_shoulder"]] = sh; xy[L["r_elbow"]] = el
    xy[L["r_wrist"]] = wr; xy[L["nose"]] = nose
    return FramePose(idx, xy, vis, np.zeros(33))


NOSE = [100, 100]                       # image y grows down; "above nose" = y<100
_EXT = ([100, 130], [100, 80], [100, 40])      # shoulder/elbow/wrist ~collinear -> ~180
_BENT = ([100, 130], [150, 80], [100, 40])     # elbow winged out -> ~90


def test_rejects_all_bent():
    """A cluster with only bent-arm frames near the candidate -> no clean release."""
    series = {i: _fp(i, *_BENT, NOSE) for i in range(0, 13)}
    f, elb = refine_release_frame(series, 6)
    assert f is None, f"bent-only should be rejected (got frame {f}, elbow {elb})"
    assert elb < 145


def test_snaps_to_extended():
    """Bent early, extended late -> snap to the extended frame, accepted."""
    series = {}
    for i in range(0, 6):
        series[i] = _fp(i, *_BENT, NOSE)
    for i in range(6, 13):
        series[i] = _fp(i, *_EXT, NOSE)
    f, elb = refine_release_frame(series, 5)      # candidate near the bend
    assert f is not None and f >= 6, (f, elb)
    assert elb >= 145


def test_rejects_wrist_below_head():
    """Extended arm but wrist below the nose (not overhead) -> not a release."""
    low = ([100, 130], [100, 160], [100, 190])    # wrist y=190 > nose 100
    series = {i: _fp(i, *low, NOSE) for i in range(0, 13)}
    assert refine_release_frame(series, 6)[0] is None


if __name__ == "__main__":
    test_rejects_all_bent(); test_snaps_to_extended(); test_rejects_wrist_below_head()
    print("ok")
