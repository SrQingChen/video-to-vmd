"""Unit tests for the metric primitives (no PMX needed).

Locks the semantics the M0 review asked for: a copy scores exactly zero, a
*jittering* planted foot is not "slide", a constant offset is not "slide" but
does show up in position, uniform noise shows up in jerk ratio (not in the MAD
spike counter), and world orientation catches a pure axial twist that local
rotations and positions cannot see.

Run: .venv/Scripts/python.exe tests/test_metrics_units.py
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from vtv.metrics.core import compute_metrics  # noqa: E402

F = 120
HEIGHT = 22.0


def qid():
    return np.array([0.0, 0.0, 0.0, 1.0])


def axis_q(axis, deg):
    h = np.radians(deg) * 0.5
    s = np.sin(h)
    q = np.zeros(4)
    q["xyz".index(axis)] = s
    q[3] = np.cos(h)
    return q


def make_track(foot_slide=0.0, foot_lift=0.0, noise=0.0, seed=0,
               twist_l=0.0, root_dx=0.0):
    """Synthetic 23-joint track: standing figure, toes near the ground."""
    rng = np.random.default_rng(seed)
    joints = {
        "root": (0.0, 9.0, 0.0), "pelvis": (0.0, 12.5, 0.0),
        "spine": (0.0, 13.5, 0.0), "spine1": (0.0, 14.5, 0.0),
        "chest": (0.0, 15.5, 0.0), "neck": (0.0, 16.8, 0.0), "head": (0.0, 18.0, 0.0),
        "clavicle_l": (0.2, 16.5, 0.0), "upperarm_l": (1.4, 16.0, 0.0),
        "forearm_l": (3.4, 14.5, 0.2), "hand_l": (5.8, 12.8, 0.3),
        "clavicle_r": (-0.2, 16.5, 0.0), "upperarm_r": (-1.4, 16.0, 0.0),
        "forearm_r": (-3.4, 14.5, 0.2), "hand_r": (-5.8, 12.8, 0.3),
        "thigh_l": (1.0, 11.8, 0.0), "shin_l": (1.0, 6.9, 0.0),
        "foot_l": (1.0, 1.9, 0.1), "toe_l": (1.0, 0.19, -1.3),
        "thigh_r": (-1.0, 11.8, 0.0), "shin_r": (-1.0, 6.9, 0.0),
        "foot_r": (-1.0, 1.9, 0.1), "toe_r": (-1.0, 0.19, -1.3),
    }
    world, world_q, local_q = {}, {}, {}
    ramp = np.linspace(0, 1, F)
    for j, p in joints.items():
        arr = np.tile(np.array(p, dtype=float), (F, 1))
        q = np.tile(qid(), (F, 1))
        if j in ("toe_l", "toe_r"):
            # slow sway so the GT itself has a little contact drift
            arr[:, 0] += 0.15 * np.sin(np.linspace(0, 2 * np.pi, F))
            arr[:, 1] += foot_lift
            arr[:, 0] += foot_slide * ramp
        if root_dx:
            # a root translation moves the whole body, like a VMD root key
            arr[:, 0] += root_dx * ramp
        world[j] = arr
        world_q[j] = q
        local_q[j] = q.copy()
    if noise:
        for j in world:
            ang = rng.uniform(-noise, noise, size=(F, 1))
            ax = rng.integers(0, 3, size=(F,))
            qs = np.array([axis_q("xyz"[a], float(d)) for a, d in zip(ax, ang[:, 0])])
            world[j] = world[j] + rng.normal(0, noise * 0.02, size=world[j].shape)
            world_q[j] = qs
            local_q[j] = qs
    if twist_l:
        # pure axial twist of the left arm: no joint moves, only world rotation
        world_q["forearm_l"] = np.tile(axis_q("x", twist_l), (F, 1))
        world_q["hand_l"] = np.tile(axis_q("x", twist_l), (F, 1))
    return {"frames": list(range(F)), "local_q": local_q, "world_q": world_q,
            "world": world, "finger_joints": [], "height": HEIGHT,
            "toe_rest_y": {"l": 0.19, "r": 0.19}, "model_key": "synthetic"}


def m(gt, pred, **kw):
    return compute_metrics(gt, pred, {"height": HEIGHT, "standard_map": {},
                                      "model_key": "synthetic"}, **kw)


def main():
    gt = make_track()
    v = m(gt, gt)["vector"]
    assert all(abs(v[k]) < 1e-6 for k in
               ("angle_mean_deg", "world_angle_mean_deg", "position_pct_height",
                "contact_slide_delta", "contact_penetration_delta")), v
    assert abs(v["jerk_rms_ratio"] - 1.0) < 1e-6
    assert v["contact_recall"] == 1.0 and v["contact_precision"] == 1.0
    assert v["contact_fraction_gt"] > 0.5, v

    # uniform 2 deg noise: tempo blows up, contact slide stays far below a real
    # slide.  (Independent per-frame jitter is the worst case for a net-drift
    # metric, so the assertion is relative, not absolute.)
    noise_v = m(gt, make_track(noise=2.0, seed=3))["vector"]
    assert noise_v["jerk_rms_ratio"] > 5, noise_v
    assert noise_v["angle_mean_deg"] < 3.0, noise_v

    # a real slide: planted toe travels horizontally
    slide_v = m(gt, make_track(foot_slide=3.0))["vector"]
    assert slide_v["contact_slide_delta"] > 2.5, slide_v
    assert slide_v["contact_height_mae_pct_height"] < 0.1, slide_v
    assert (slide_v["contact_slide_delta"]
            > 5.0 * max(noise_v["contact_slide_delta"], 0.1)), (slide_v, noise_v)

    # a constant lift: height error + recall, no slide
    v = m(gt, make_track(foot_lift=2.0))["vector"]
    assert v["contact_height_mae_pct_height"] > 5, v
    assert v["contact_recall"] < 0.2, v

    # pure axial twist is invisible to local angles/positions but not to world
    v = m(gt, make_track(twist_l=30.0))["vector"]
    assert v["angle_mean_deg"] < 1e-6, v
    assert v["position_pct_height"] < 1e-6, v
    assert v["world_angle_mean_deg"] > 2.0, v

    # root drift is a root component, not an angle component
    v = m(gt, make_track(root_dx=2.0))["vector"]
    assert v["root_drift_pct_height"] > 5, v
    assert v["angle_mean_deg"] < 1e-6, v

    # penetration is measured against the floor, independent of the GT's float
    gt2 = make_track(foot_lift=1.0)
    v = m(gt2, make_track())["vector"]
    assert v["contact_penetration_delta"] <= 1e-9, v
    v = m(gt2, make_track(foot_lift=-1.0))["vector"]
    assert abs(v["contact_penetration_delta"] - 1.0) < 0.2, v

    print("test_metrics_units: OK")


if __name__ == "__main__":
    main()
