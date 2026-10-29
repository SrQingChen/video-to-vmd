# -*- coding: utf-8 -*-
"""Selective cleanup tests: hemisphere, spike bridge, contact/foot lock.

These are pure-Python (no Blender, no GVHMR).  The foot-lock test uses a
3-bone PMX-like skeleton so the real ``pose_track`` path is exercised.
"""
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

from vtv.cleanup import clean_vmd, contact_intervals, hemisphere


def qaxis(axis, deg):
    h = math.radians(deg) * 0.5
    s = math.sin(h)
    return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(h))


def minimal_model():
    # root is movable (0x0004); two toes are direct children so the evaluated
    # toe world position is exactly root_translation + rest offset.
    bones = [
        {"name": "root", "parent": -1, "pos": [0.0, 0.0, 0.0],
         "flags": 0x0004, "layer": 0},
        {"name": "toe_l", "parent": 0, "pos": [-2.0, 0.0, 0.0],
         "flags": 0, "layer": 0},
        {"name": "toe_r", "parent": 0, "pos": [2.0, 0.0, 0.0],
         "flags": 0, "layer": 0},
    ]
    profile = {"standard_map": {"root": "root", "toe_l": "toe_l", "toe_r": "toe_r"},
               "height": 10.0}
    return {"bones": bones, "name": "test"}, profile


def vmd_with_tracks(n, rot_l, root_x):
    frames = list(range(n))
    curve = (20, 20, 107, 107)
    bones = {
        "root": [{"frame": f, "pos": (root_x[f], 0.0, 0.0), "rot": (0, 0, 0, 1),
                  "curves": (curve,) * 4} for f in frames],
        "toe_l": [{"frame": f, "pos": (0, 0, 0), "rot": rot_l[f],
                   "curves": (curve,) * 4} for f in frames],
        "toe_r": [{"frame": f, "pos": (0, 0, 0), "rot": (0, 0, 0, 1),
                   "curves": (curve,) * 4} for f in frames],
    }
    return {"model_name": "test", "bones": bones, "morphs": {},
            "ik_frames": [], "max_frame": n - 1}


def test_hemisphere_and_spike_bridge():
    n = 40
    rot = [(0.0, 0.0, 0.0, 1.0)] * n
    spike = qaxis((1.0, 0.0, 0.0), 40.0)
    rot[20] = spike
    model, profile = minimal_model()
    vmd = vmd_with_tracks(n, rot, [0.0] * n)
    res = clean_vmd(vmd, model, profile, do_footlock=False, spike_deg=5.0)
    out = res["vmd"]["bones"]["toe_l"]
    # the spike must be bridged back to identity, neighbours untouched
    for f in (0, 10, 19, 21, 30, 39):
        q = out[f]["rot"]
        assert abs(abs(q[3]) - 1.0) < 1e-6, (f, q)
    q20 = out[20]["rot"]
    assert abs(abs(q20[3]) - 1.0) < 1e-4, q20
    assert res["report"]["spike_frames_total"] >= 1
    # hemisphere helper itself
    q = hemisphere(np.array([qaxis((1, 0, 0), 10), (-1, 0, 0, 0)]))
    assert float(np.dot(q[0], q[1])) >= 0.0


def test_foot_lock_removes_slow_root_slide():
    n = 100
    # 3 units of slow slide: speed 0.03 < 0.12, so both toes are "planted";
    # the lock should pull the root back to the interval anchor.
    root_x = [3.0 * f / (n - 1) for f in range(n)]
    rot = [(0.0, 0.0, 0.0, 1.0)] * n
    model, profile = minimal_model()
    vmd = vmd_with_tracks(n, rot, root_x)
    res = clean_vmd(vmd, model, profile, do_spikes=False, do_footlock=True,
                    root_cap=5.0, lock_drift=0.5)
    root = res["vmd"]["bones"]["root"]
    x0 = root[0]["pos"][0]
    x1 = root[-1]["pos"][0]
    assert abs(x1 - x0) < 0.15, (x0, x1)
    assert res["report"]["foot_lock"]["applied"] is True


def test_contact_intervals_require_speed_gate():
    n = 60
    toe = np.zeros((n, 3))
    toe[:, 0] = np.linspace(0.0, 12.0, n)  # ~0.20 units/frame, above 0.12
    toe[:, 1] = 0.0
    c = contact_intervals(toe, height=10.0, margin=0.02, speed_thr=0.12,
                          vspeed_thr=0.08)
    # the toe is in the height band but moving too fast: no planted interval
    assert c["mask"].sum() == 0, c["intervals"]


if __name__ == "__main__":
    test_hemisphere_and_spike_bridge()
    test_foot_lock_removes_slow_root_slide()
    test_contact_intervals_require_speed_gate()
    print("test_cleanup: OK")
