# -*- coding: utf-8 -*-
"""M2.4 arm tests: palm hinge frame, swing/roll decomposition, guards.

The hinge-projection test is a *pure rotation* test: projecting the elbow local
axis onto the palm hinge line with ``keep_flexion=True`` must preserve the
local rotation angle exactly (the cap is disabled in the test).  The
decomposition test pins the known front numbers so a future change that
silently swaps swing and roll fails loudly.  The constraint tests check both
sides of the guard: a clean view is a no-op, a deep-penetration view is
rejected by the trust region instead of being forced.
"""
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

from vtv.adapter import q_from_axis_angle, qrot
from vtv.arm_prior import (_q_axis_angle, _q_dist, _shoulder_delta,
                           capsule_penetrations, compare_arm,
                           palm_frame, project_hinge,
                           resolve_self_intersections, verify)
from vtv.cleanup import dense_tracks
from vtv.metrics.core import pose_track
from vtv.mmd.pmx import parse_pmx
from vtv.mmd.vmd import parse_vmd

import json

CFG = os.path.join(ROOT, "configs", "models.json")
MODEL = "si"
FRONT = os.path.join(ROOT, "outputs", "m2", "vmd", "front_clean_ik.vmd")
SIDE = os.path.join(ROOT, "outputs", "m2", "vmd", "side_clean_ik.vmd")
GT = os.path.join(ROOT, "outputs", "gt", "si_R2_safe.vmd")


def paths():
    for p in (FRONT, SIDE, GT):
        assert os.path.exists(p), f"missing test fixture: {p}"
    cfg = json.load(open(CFG, encoding="utf-8"))["models"][MODEL]
    prof = json.load(open(os.path.join(ROOT, "profiles", f"{MODEL}.json"),
                          encoding="utf-8"))
    return parse_pmx(cfg["pmx"]), prof


def test_palm_frame_is_a_hinge_line_and_mirrors():
    model, prof = paths()
    for side in ("l", "r"):
        pf = palm_frame(model, prof, side)
        n = np.asarray(pf["hinge_line"])
        up = np.asarray(pf["upperarm_rest_dir"])
        assert abs(float(np.linalg.norm(n)) - 1.0) < 1e-9
        # elbow hinge must be roughly perpendicular to the upper arm
        assert abs(float(n @ up)) < 0.2, (side, float(n @ up))
    nl = np.asarray(palm_frame(model, prof, "l")["hinge_line"])
    nr = np.asarray(palm_frame(model, prof, "r")["hinge_line"])
    mirror = np.array([-nl[0], nl[1], nl[2]])   # mirror across the YZ plane
    assert abs(float(mirror @ nr)) > 0.95, (mirror, nr)


def test_compare_arm_self_is_zero():
    model, prof = paths()
    tr = pose_track(model, parse_vmd(GT), prof)
    cmp = compare_arm(tr, tr, model, prof)
    assert cmp["arm_world_mean_deg"] < 1e-6
    assert cmp["arm_swing_mean_deg"] < 1e-6
    assert cmp["arm_roll_mean_deg"] < 1e-6


def test_front_swing_roll_decomposition_matches_known():
    model, prof = paths()
    gt = pose_track(model, parse_vmd(GT), prof)
    pr = pose_track(model, parse_vmd(FRONT), prof, frames=gt["frames"])
    cmp = compare_arm(gt, pr, model, prof)
    # M2.4 diagnostic (2026-10-26, si/R2): 52.77 total, 35.35 swing, 33.19 roll
    assert abs(cmp["arm_world_mean_deg"] - 52.77) < 1.5, cmp["arm_world_mean_deg"]
    assert abs(cmp["arm_swing_mean_deg"] - 35.35) < 1.5, cmp["arm_swing_mean_deg"]
    assert abs(cmp["arm_roll_mean_deg"] - 33.19) < 1.5, cmp["arm_roll_mean_deg"]


def test_hinge_projection_preserves_local_flexion():
    model, prof = paths()
    vmd = parse_vmd(FRONT)
    res = project_hinge(vmd, model, prof, alpha=1.0, cap_deg=180.0,
                        keep_flexion=True)
    _, tin = dense_tracks(vmd)
    _, tout = dense_tracks(res["vmd"])
    for side in ("l", "r"):
        eye = prof["standard_map"][f"forearm_{side}"]
        assert eye in tin and eye in tout
        for a, b in zip(tin[eye]["rot"], tout[eye]["rot"]):
            assert abs(_q_axis_angle(a)[1] - _q_axis_angle(b)[1]) < 1e-6
        for other in (prof["standard_map"][f"upperarm_{side}"],
                      prof["standard_map"][f"hand_{side}"]):
            assert np.max(np.abs(tin[other]["rot"] - tout[other]["rot"])) < 1e-9
        assert _q_dist(tin[eye]["rot"][0], tout[eye]["rot"][0]) > 0.0


def test_clean_view_is_untouched():
    model, prof = paths()
    vmd = parse_vmd(SIDE)
    res = resolve_self_intersections(vmd, model, prof, rounds=3)
    rep = res["report"]
    ver = verify(vmd, res["vmd"], model, prof, rep)
    assert ver["ok"], ver
    _, tin = dense_tracks(vmd)
    _, tout = dense_tracks(res["vmd"])
    for bn in tin:
        assert np.max(np.abs(tin[bn]["rot"] - tout[bn]["rot"])) < 1e-9
        assert np.max(np.abs(tin[bn]["pos"] - tout[bn]["pos"])) < 1e-9


def test_deep_penetration_is_rejected_not_forced():
    model, prof = paths()
    vmd = parse_vmd(FRONT)
    res = resolve_self_intersections(vmd, model, prof, rounds=2,
                                     max_deg=8.0, max_disp=0.3,
                                     arm_arm=False)
    ver = verify(vmd, res["vmd"], model, prof, res["report"])
    assert ver["ok"] is False
    tools = ver["checks"]
    assert (tools["arm_torso_improved"]["ok"] is False
            or tools["arm_arm_not_worse"]["ok"] is False
            or tools["correction_cap"]["ok"] is False), tools


def test_shoulder_delta_moves_the_point_along_push():
    shoulder = np.array([0.0, 10.0, 0.0])
    point = np.array([0.0, 10.0, -2.5])
    track = {"world": {"upperarm_l": [shoulder]}}
    push = np.array([1.0, 0.0, 0.0])
    omega, deg = _shoulder_delta(track, 0, "l", point, push, 0.1)
    assert deg > 0.0
    moved = np.asarray(qrot(q_from_axis_angle(*omega), tuple(point - shoulder)))
    want = push * 0.1
    assert float(np.linalg.norm(moved - (point - shoulder) - want)) < 5e-3


def test_front_has_real_target_space_violations():
    model, prof = paths()
    from vtv.cleanup import eval_tracks
    tr = eval_tracks(model, prof, dense_tracks(parse_vmd(FRONT))[1])
    pen = capsule_penetrations(tr, model, prof, arm_arm=True)
    assert pen["max_arm_torso"] > 0.5, pen["summary"]
    assert pen["max_arm_arm"] > 0.5, pen["summary"]


if __name__ == "__main__":
    test_palm_frame_is_a_hinge_line_and_mirrors()
    test_compare_arm_self_is_zero()
    test_front_swing_roll_decomposition_matches_known()
    test_hinge_projection_preserves_local_flexion()
    test_clean_view_is_untouched()
    test_deep_penetration_is_rejected_not_forced()
    test_shoulder_delta_moves_the_point_along_push()
    test_front_has_real_target_space_violations()
    print("test_arm_prior: OK")
