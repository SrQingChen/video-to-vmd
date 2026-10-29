# -*- coding: utf-8 -*-
"""Contact-IK tests: vertical 2-bone IK, trust region, non-contact preservation.

Pure Python; a 3-segment leg is posed from explicit local rotations and the
real ``pose_track`` path is used, so the test exercises MMD FK semantics.
"""
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

from vtv.contact_ik import contact_probability, solve_contact_ik
from vtv.metrics.core import pose_track

HEIGHT = 10.0
L1 = 4.5
L2 = 4.5


def qx(deg):
    h = math.radians(deg) * 0.5
    return (math.sin(h), 0.0, 0.0, math.cos(h))


def leg_model():
    bones = [{"name": "pelvis", "parent": -1, "pos": [0.0, 12.0, 0.0],
              "flags": 0, "layer": 0}]
    for side, sx in (("l", 1.0), ("r", -1.0)):
        i_thigh = len(bones)
        bones.append({"name": f"thigh_{side}", "parent": 0,
                      "pos": [sx, 11.0, 0.0], "flags": 0, "layer": 0})
        i_shin = len(bones)
        bones.append({"name": f"shin_{side}", "parent": i_thigh,
                      "pos": [sx, 11.0 - L1, 0.0], "flags": 0, "layer": 0})
        i_foot = len(bones)
        bones.append({"name": f"foot_{side}", "parent": i_shin,
                      "pos": [sx, 11.0 - L1 - L2, 0.0], "flags": 0, "layer": 0})
        bones.append({"name": f"toe_{side}", "parent": i_foot,
                      "pos": [sx, 11.0 - L1 - L2 - 0.4, -1.0],
                      "flags": 0, "layer": 0})
    smap = {"pelvis": "pelvis"}
    for side in ("l", "r"):
        for b in ("thigh", "shin", "foot", "toe"):
            smap[f"{b}_{side}"] = f"{b}_{side}"
    profile = {"standard_map": smap, "height": HEIGHT}
    return {"bones": bones, "name": "legtest"}, profile


def make_vmd(knee_deg, n=60, foot_flat=True):
    """Left knee flexed by knee_deg (right leg straight); foot kept flat."""
    curve = (20, 20, 107, 107)
    bones = {}
    for side in ("l", "r"):
        knee = knee_deg if side == "l" else 0.0
        foot = -knee if foot_flat else 0.0
        for b, rot in (("thigh", (0, 0, 0, 1)), ("shin", qx(knee)),
                       ("foot", qx(foot)), ("toe", (0, 0, 0, 1))):
            bones[f"{b}_{side}"] = [
                {"frame": f, "pos": (0.0, 0.0, 0.0), "rot": rot,
                 "curves": (curve, curve, curve, curve)} for f in range(n)]
    bones["pelvis"] = [{"frame": f, "pos": (0.0, 0.0, 0.0), "rot": (0, 0, 0, 1),
                        "curves": (curve,) * 4} for f in range(n)]
    return {"model_name": "legtest", "bones": bones, "morphs": {},
            "ik_frames": [], "max_frame": n - 1}


def toe_y(vmd, model, profile, side, frame=None):
    tr = pose_track(model, vmd, profile, solve_ik=False)
    y = tr["world"][f"toe_{side}"][:, 1]
    return float(y[frame]) if frame is not None else y


def test_inside_band_is_untouched():
    model, profile = leg_model()
    vmd = make_vmd(6.0)   # ~0.02 unit lift: inside the 0.02*10 band
    res = solve_contact_ik(vmd, model, profile)
    assert res["report"]["verify"]["ok"], res["report"]["verify"]
    y0 = toe_y(vmd, model, profile, "l")
    y1 = toe_y(res["vmd"], model, profile, "l")
    assert np.abs(y1 - y0).max() < 1e-6, np.abs(y1 - y0).max()


def test_contact_lowers_toe_without_moving_xz():
    model, profile = leg_model()
    vmd = make_vmd(30.0)  # 0.6 unit lift, above the band, inside the exit band
    y0 = toe_y(vmd, model, profile, "l")
    res = solve_contact_ik(vmd, model, profile, band=0.05, target_frac=0.5,
                           max_knee_deg=15.0, max_thigh_deg=25.0, ramp=0)
    rep = res["report"]
    assert rep["verify"]["ok"], rep["verify"]
    s = rep["sides"]["l"]
    assert s["applied"], s
    assert s["frames"] > 30, s["frames"]
    assert s["drop_max"] > 0.05, s["drop_max"]
    tr0 = pose_track(model, vmd, profile, solve_ik=False)
    tr1 = pose_track(model, res["vmd"], profile, solve_ik=False)
    t0, t1 = tr0["world"]["toe_l"], tr1["world"]["toe_l"]
    assert np.abs(t1[:, 1] - t0[:, 1]).max() > 0.05
    assert np.abs(t1[:, [0, 2]] - t0[:, [0, 2]]).max() < 1e-6
    # knee change respects the trust region
    assert s["knee_max_deg"] <= 15.0 + 1e-6, s["knee_max_deg"]
    # bone lengths preserved
    for a, b in (("thigh_l", "shin_l"), ("shin_l", "foot_l")):
        la = np.linalg.norm(tr0["world"][a] - tr0["world"][b], axis=1).mean()
        lb = np.linalg.norm(tr1["world"][a] - tr1["world"][b], axis=1).mean()
        assert abs(la - lb) < 1e-5, (a, b, la, lb)


def test_high_foot_is_not_pressed_down():
    model, profile = leg_model()
    vmd = make_vmd(80.0)  # far above the exit band: a real lift
    y0 = toe_y(vmd, model, profile, "l")
    res = solve_contact_ik(vmd, model, profile)
    y1 = toe_y(res["vmd"], model, profile, "l")
    assert not res["report"]["sides"]["l"]["applied"]
    assert np.abs(y1 - y0).max() < 1e-6
    assert res["report"]["verify"]["ok"]


def test_short_flicker_is_ignored():
    # two contact frames only -> dropped by min_len
    y = np.full(40, 5.0)
    y[10:12] = 0.0
    c = contact_probability(y, ground=0.0, height=HEIGHT, band=0.02,
                            vspeed=0.10, min_len=3, fill_gap=2)
    assert c["mask"].sum() == 0, c["intervals"]


def test_contact_hysteresis_keeps_slow_exit():
    y = np.full(40, 5.0)
    y[10:20] = 0.0
    y[20:24] = 0.25      # still inside exit band (2% * 10 = 0.2 -> 2.0? no: 0.25 > 0.2)
    c = contact_probability(y, ground=0.0, height=HEIGHT, band=0.02,
                            vspeed=0.10, min_len=3, fill_gap=2)
    # 0.25 units above ground is 1.25 band: inside enter (1.5 band)
    assert c["mask"][12] and c["mask"][21], c["mask"].astype(int)


if __name__ == "__main__":
    test_inside_band_is_untouched()
    test_contact_lowers_toe_without_moving_xz()
    test_high_foot_is_not_pressed_down()
    test_short_flicker_is_ignored()
    test_contact_hysteresis_keeps_slow_exit()
    print("test_contact_ik: OK")
