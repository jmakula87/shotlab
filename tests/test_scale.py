"""Validate rim-based px->feet scale and the height metrics against known values."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shotlab.scale import (px_per_foot_from_rim, apex_above_rim_ft,
                           release_height_ft, jump_height_ft, RIM_WIDTH_FT,
                           px_per_foot_from_body, parse_height,
                           NOSE_TO_ANKLE_FRAC)


def test_ppf_from_rim():
    # rim half-width 30px -> 60px across = 1.5 ft -> 40 px/ft
    assert abs(px_per_foot_from_rim(30) - 40.0) < 1e-9
    assert px_per_foot_from_rim(0) is None
    assert px_per_foot_from_rim(None) is None


def test_apex_above_rim():
    ppf = px_per_foot_from_rim(30)          # 40 px/ft
    # ball peak 100px ABOVE the rim (smaller y) -> 2.5 ft
    assert abs(apex_above_rim_ft(200, 300, ppf) - 2.5) < 1e-9
    # a flat shot peaking BELOW the rim line -> negative
    assert apex_above_rim_ft(340, 300, ppf) < 0


def test_release_and_jump_height():
    ppf = px_per_foot_from_rim(30)          # 40 px/ft
    # ball released 320px above the ankle line -> 8.0 ft release height
    assert abs(release_height_ft(180, 500, ppf) - 8.0) < 1e-9
    # body rises 60px from its lowest (y=500) to its peak (y=440) -> 1.5 ft jump
    assert abs(jump_height_ft(500, 440, ppf) - 1.5) < 1e-9


def test_none_scale_is_safe():
    assert apex_above_rim_ft(200, 300, None) is None
    assert release_height_ft(180, 500, None) is None
    assert RIM_WIDTH_FT == 1.5


def test_ppf_from_body():
    # 5'10" shooter, nose->ankle = 0.875*5.833ft = 5.104ft; at 51 px/ft that
    # span is ~260px. Round-trip: 260px / (5.833*0.875) = ~50.9 px/ft
    height = 70 / 12.0
    span_px = height * NOSE_TO_ANKLE_FRAC * 51.0
    assert abs(px_per_foot_from_body(span_px, height) - 51.0) < 1e-6
    assert px_per_foot_from_body(0, height) is None
    assert px_per_foot_from_body(260, None) is None


def test_body_ruler_is_not_rim_hot():
    # the whole point: at the shooter's plane (denser px/ft) a given pixel
    # height is FEWER feet than the rim ruler would say
    rim_ppf = px_per_foot_from_rim(15)          # far rim: 20 px/ft
    body_ppf = px_per_foot_from_body(260, 70 / 12.0)   # ~51 px/ft
    px = 300.0
    assert release_height_ft(0, px, body_ppf) < release_height_ft(0, px, rim_ppf)


def test_parse_height():
    assert abs(parse_height("5'10\"") - (5 + 10 / 12)) < 1e-9
    assert abs(parse_height("5'10") - (5 + 10 / 12)) < 1e-9
    assert abs(parse_height("5ft10") - (5 + 10 / 12)) < 1e-9
    assert abs(parse_height("5.83") - 5.83) < 1e-9
    assert abs(parse_height("70in") - 70 / 12) < 1e-9
    assert abs(parse_height("70\"") - 70 / 12) < 1e-9
    assert abs(parse_height("6'") - 6.0) < 1e-9
    assert parse_height(None) is None
    assert parse_height("") is None
    assert parse_height("tall") is None


if __name__ == "__main__":
    import traceback
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in funcs:
        try:
            fn(); print(f"PASS {fn.__name__}"); passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(funcs)} passed")
    sys.exit(0 if passed == len(funcs) else 1)
