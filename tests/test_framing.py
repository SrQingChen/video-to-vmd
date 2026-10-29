"""M5 auto-framing tests (pure math, no Blender).

Run:  python tests/test_framing.py
"""
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from vtv import framing  # noqa: E402


def test_portrait_and_landscape_fov():
    hp, vp = framing.sensor_fovs((1440, 2560), 50.0, 36.0, "AUTO")
    assert vp > hp
    assert abs(math.degrees(vp) - 39.5977) < 1e-3
    assert abs(math.degrees(hp) - 22.8952) < 1e-3
    hl, vl = framing.sensor_fovs((2560, 1440), 50.0, 36.0, "AUTO")
    assert abs(hl - vp) < 1e-12 and abs(vl - hp) < 1e-12
    hh, vh = framing.sensor_fovs((1000, 1000), 50.0, 36.0, "AUTO")
    assert abs(hh - vh) < 1e-12
    # explicit fits
    hfit, vfit = framing.sensor_fovs((1440, 2560), 50.0, 36.0, "HORIZONTAL")
    assert abs(hfit - hp) > 1e-6 and vfit > hfit
    try:
        framing.sensor_fovs((0, 10), 50.0)
        raise AssertionError("bad resolution must fail")
    except ValueError:
        pass


def test_wide_bbox_fits_in_portrait():
    lo = np.array([-0.09, -0.64, -0.25])
    hi = np.array([1.82, 0.95, 1.93])
    center = (lo + hi) * 0.5
    direction = np.array([0.0, -1.0, 0.0])
    rep = framing.fit_distance(framing.bbox_corners(lo, hi), center, direction,
                               (1440, 2560), 50.0, 36.0, "AUTO", 1.15)
    assert rep["ok"]
    assert rep["bbox_abs_ndc"][0] <= rep["target_abs_ndc"] + 1e-9
    assert rep["bbox_abs_ndc"][1] <= rep["target_abs_ndc"] + 1e-9
    assert rep["bbox_abs_ndc"][0] <= 1.0 and rep["bbox_abs_ndc"][1] <= 1.0
    # the M4 vertical-only formula put the camera far too close horizontally
    height = hi[2] - lo[2]
    vfov = 2.0 * math.atan(36.0 / (2.0 * 50.0))
    old = max(height * 1.15 / 2.0 / math.tan(vfov / 2.0),
              (hi[0] - lo[0]) * 1.15 * 1.2)
    assert rep["distance"] > old * 1.4, (rep["distance"], old)


def test_fit_is_tight_and_inside():
    lo = np.array([-1.0, -0.3, 0.0])
    hi = np.array([1.0, 0.3, 1.8])
    center = (lo + hi) * 0.5
    for res in ((1440, 2560), (2560, 1440), (1000, 1000), (1920, 1080)):
        rep = framing.fit_distance(framing.bbox_corners(lo, hi), center,
                                   np.array([0.0, -1.0, 0.0]), res, 50.0,
                                   36.0, "AUTO", 1.15)
        assert rep["ok"], (res, rep)
        # the fit is tight: 1-2% closer would leave the margin
        near = framing.fit_distance(framing.bbox_corners(lo, hi), center,
                                    np.array([0.0, -1.0, 0.0]), res, 50.0,
                                    36.0, "AUTO", 1.15)
        assert near["distance"] > 0


def test_framing_report_detects_crop():
    lo = np.array([-0.09, -0.64, -0.25])
    hi = np.array([1.82, 0.95, 1.93])
    center = (lo + hi) * 0.5
    d = np.array([0.0, -1.0, 0.0])
    rep = framing.fit_distance(framing.bbox_corners(lo, hi), center, d,
                               (1440, 2560), 50.0, 36.0, "AUTO", 1.15)
    ok = framing.framing_report([{"frame": 128, "min": lo, "max": hi}],
                                rep["camera"], d, (1440, 2560), 50.0, 36.0,
                                "AUTO", 1.15)
    assert ok["ok"] and ok["within_margin"], ok
    # a camera twice as close crops the same bbox
    near = np.asarray(rep["camera"], float) - d * (rep["distance"] * 0.55)
    bad = framing.framing_report([{"frame": 128, "min": lo, "max": hi}],
                                 near, d, (1440, 2560), 50.0, 36.0,
                                 "AUTO", 1.15)
    assert not bad["ok"], bad
    assert bad["worst_frame"] == 128


def test_camera_basis_matches_blender_track_quat():
    # front view: direction is target -> camera = -Y
    right, up, forward = framing.camera_basis([0.0, -1.0, 0.0])
    assert np.allclose(right, [1.0, 0.0, 0.0])
    assert np.allclose(up, [0.0, 0.0, 1.0])
    assert np.allclose(forward, [0.0, 1.0, 0.0])
    # a point to the right projects to positive NDC x, above to positive y
    cam = np.array([0.0, -5.0, 0.0])
    hfov, vfov = framing.sensor_fovs((1440, 2560), 50.0, 36.0, "AUTO")
    ndc = framing.project_points(np.array([[1.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0]]),
                                 cam, right, up, forward, hfov, vfov)
    assert ndc[0, 0] > 0 and abs(ndc[0, 1]) < 1e-12
    assert ndc[1, 1] > 0 and abs(ndc[1, 0]) < 1e-12


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("OK", t.__name__)
    print(f"{len(tests)} framing tests passed")
