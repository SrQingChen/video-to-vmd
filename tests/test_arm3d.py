# -*- coding: utf-8 -*-
"""M2.4b multi-view 3D arm direction prior tests.

The triangulation test is a pure geometry test (three synthetic cameras, one
known 3D segment).  The application tests use the real front VMD fixture and
pin the properties the stage must preserve: no-op when the target is the
current direction, target reached when it is not, cap respected, non-arm
tracks untouched, and the torso-relative metric invariant to a global rotation
(the check that catches a world-only "win").
"""
import json
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

from vtv.adapter import q_from_axis_angle, qmul, qrot
from vtv.arm3d import (apply_direction_prior, body_relative_errors,
                       current_directions, penetration,
                       targets_from_result, triangulate_directions, verify)
from vtv.cleanup import dense_tracks
from vtv.metrics.core import pose_track
from vtv.mmd.pmx import parse_pmx
from vtv.mmd.vmd import parse_vmd

CFG = os.path.join(ROOT, "configs", "models.json")
MODEL = "si"
FRONT = os.path.join(ROOT, "outputs", "m2", "vmd", "front_clean_ik.vmd")
GT = os.path.join(ROOT, "outputs", "gt", "si_R2_safe.vmd")


def fixture():
    for p in (FRONT, GT):
        assert os.path.exists(p), f"missing test fixture: {p}"
    cfg = json.load(open(CFG, encoding="utf-8"))["models"][MODEL]
    prof = json.load(open(os.path.join(ROOT, "profiles", f"{MODEL}.json"),
                          encoding="utf-8"))
    return parse_pmx(cfg["pmx"]), prof


def _look_at(C, target, up=(0.0, 1.0, 0.0)):
    C = np.asarray(C, dtype=np.float64)
    z = np.asarray(target, dtype=np.float64) - C
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, dtype=np.float64))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)  # camera -> world


def synth_cameras():
    cams = {}
    for i, az in enumerate((0.0, 60.0, 120.0)):
        a = math.radians(az)
        C = np.array([4.0 * math.cos(a), 0.8, 4.0 * math.sin(a)])
        cams[f"v{i}"] = {
            "R": _look_at(C, (0.0, 0.0, 0.0)),
            "C": C, "fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0,
            "width": 640, "height": 480,
        }
    return cams


def _project(cam, P):
    pc = np.asarray(cam["R"]).T @ (np.asarray(P) - cam["C"])
    return np.array([cam["fx"] * pc[0] / pc[2] + cam["cx"],
                     cam["fy"] * pc[1] / pc[2] + cam["cy"]])


def test_triangulation_recovers_known_direction():
    cams = synth_cameras()
    d = np.array([0.3, 0.2, 0.9])
    d /= np.linalg.norm(d)
    P0 = np.array([0.0, 0.0, 0.0])
    P1 = 0.7 * d
    kps = {}
    for name, cam in cams.items():
        kp = np.zeros((1, 17, 3), dtype=np.float64)
        kp[0, :, 2] = 1.0
        kp[0, 0, :2] = _project(cam, P0)
        kp[0, 1, :2] = _project(cam, P1)
        kps[name] = kp
    out = triangulate_directions(kps, cams, pairs={"test": (0, 1)},
                                 smooth=0, max_step_deg=0.0)
    rec = out["test"]["dir"][0]
    # The cameras are in the scene frame; the returned target is in the PMX
    # frame by construction (scene (x,y,z) -> PMX (x,z,y)).
    want = np.asarray([d[0], d[2], d[1]])
    err = math.degrees(math.acos(min(1.0, abs(float(np.dot(rec, want))))))
    assert err < 0.5, (err, rec, want)
    assert out["test"]["cond"][0] < 0.2, out["test"]["cond"][0]


def test_targets_from_result_marks_degenerate_frames():
    tri = {"x": {"dir": np.array([[1.0, 0, 0], [0, 1.0, 0]]),
                 "cond": np.array([0.01, 0.999]),
                 "weight": np.array([3.0, 3.0])}}
    dirs, valid = targets_from_result(tri)
    assert valid["x"].tolist() == [True, False]
    assert dirs["x"].shape == (2, 3)


def test_noop_apply_preserves_everything():
    model, prof = fixture()
    vmd = parse_vmd(FRONT)
    prior = current_directions(vmd, model, prof)
    targets = {k: v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
               for k, v in prior.items()}
    res = apply_direction_prior(vmd, model, prof, targets, pen_guard=False)
    fin, tin = dense_tracks(vmd)
    fout, tout = dense_tracks(res["vmd"])
    assert fin == fout
    for bn in tin:
        assert np.max(np.abs(tin[bn]["rot"] - tout[bn]["rot"])) < 1e-6, bn
        assert np.max(np.abs(tin[bn]["pos"] - tout[bn]["pos"])) < 1e-9, bn
    chk = verify(vmd, res["vmd"], model, prof, res["report"],
                 targets=targets, applied=res["applied"])
    assert chk["ok"], [k for k, v in chk["checks"].items()
                       if isinstance(v, dict) and not v.get("ok", True)]


def test_target_reached_and_cap():
    model, prof = fixture()
    vmd = parse_vmd(FRONT)
    prior = current_directions(vmd, model, prof)
    targets = {k: v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
               for k, v in prior.items()}
    u = targets["upper_l"]
    g = q_from_axis_angle(0.0, math.radians(20.0), 0.0)
    target = np.stack([np.asarray(qrot(g, tuple(v))) for v in u])
    targets["upper_l"] = target / np.linalg.norm(target, axis=1, keepdims=True)
    res = apply_direction_prior(vmd, model, prof, targets, cap_deg=60.0,
                                pen_guard=False)
    tr = pose_track(model, res["vmd"], prof)
    d = tr["world"]["forearm_l"] - tr["world"]["upperarm_l"]
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    err = np.degrees(np.arccos(np.clip((d * targets["upper_l"]).sum(1), -1, 1)))
    assert float(err.mean()) < 0.6, float(err.mean())
    assert res["report"]["sides"]["l"]["frames_upper"] > 0

    # Cap: ask for 120 deg but allow 10; the achieved swing is <= 10.25 deg.
    g2 = q_from_axis_angle(0.0, math.radians(120.0), 0.0)
    target2 = np.stack([np.asarray(qrot(g2, tuple(v))) for v in u])
    targets["upper_l"] = target2 / np.linalg.norm(target2, axis=1, keepdims=True)
    res2 = apply_direction_prior(vmd, model, prof, targets, cap_deg=10.0,
                                 pen_guard=False)
    tr2 = pose_track(model, res2["vmd"], prof)
    d2 = tr2["world"]["forearm_l"] - tr2["world"]["upperarm_l"]
    d2 = d2 / np.maximum(np.linalg.norm(d2, axis=1, keepdims=True), 1e-12)
    swing = np.degrees(np.arccos(np.clip((d2 * u).sum(1), -1, 1)))
    assert float(swing.mean()) <= 10.25, float(swing.mean())
    chk = verify(vmd, res2["vmd"], model, prof, res2["report"],
                 targets=targets, applied=res2["applied"])
    assert chk["checks"]["correction_cap"]["ok"]


def test_body_relative_is_invariant_to_global_rotation():
    model, prof = fixture()
    gt = pose_track(model, parse_vmd(GT), prof)
    pr = pose_track(model, parse_vmd(FRONT), prof, frames=gt["frames"])
    b0 = body_relative_errors(gt, pr, model, prof)
    g = q_from_axis_angle(0.0, math.radians(37.0), 0.0)

    def rot_track(tr):
        out = dict(tr)
        out["world_q"] = {j: np.stack([np.asarray(qmul(tuple(g), tuple(q)))
                                       for q in qs])
                          for j, qs in tr["world_q"].items()}
        out["world"] = {j: np.stack([np.asarray(qrot(tuple(g), tuple(p)))
                                     for p in ps])
                        for j, ps in tr["world"].items()}
        return out

    b1 = body_relative_errors(rot_track(gt), rot_track(pr), model, prof)
    assert abs(b1["arm_body_mean_deg"] - b0["arm_body_mean_deg"]) < 1e-6
    for side in ("l", "r"):
        for bone in b0["sides"][side]:
            assert abs(b1["sides"][side][bone] - b0["sides"][side][bone]) < 1e-6


def test_verify_rejects_non_arm_change():
    model, prof = fixture()
    vmd = parse_vmd(FRONT)
    prior = current_directions(vmd, model, prof)
    targets = {k: v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
               for k, v in prior.items()}
    res = apply_direction_prior(vmd, model, prof, targets, pen_guard=False)
    out = json.loads(json.dumps(res["vmd"]))
    bn = prof["standard_map"]["chest"]
    q = np.asarray(out["bones"][bn][0]["rot"], dtype=np.float64)
    g = np.asarray(q_from_axis_angle(math.radians(8.0), 0.0, 0.0))
    out["bones"][bn][0]["rot"] = tuple(qmul(tuple(g), tuple(q)))
    chk = verify(vmd, out, model, prof, res["report"],
                 targets=targets, applied=res["applied"])
    assert not chk["checks"]["non_arm_unchanged"]["ok"]


def test_penetration_guard_rejects_inward_target():
    model, prof = fixture()
    vmd = parse_vmd(FRONT)
    prior = current_directions(vmd, model, prof)
    targets = {k: v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
               for k, v in prior.items()}
    tr = pose_track(model, vmd, prof)
    # Point the left upper arm at the pelvis: a deliberately inward target.
    inward = np.asarray(tr["world"]["pelvis"]) - np.asarray(tr["world"]["upperarm_l"])
    inward = inward / np.maximum(np.linalg.norm(inward, axis=1, keepdims=True), 1e-12)
    targets["upper_l"] = inward
    res = apply_direction_prior(vmd, model, prof, targets, cap_deg=60.0,
                                pen_guard=True, pen_tol=0.01)
    assert res["report"]["sides"]["l"]["pen_guard_rejected"] > 0
    pin = penetration(vmd, model, prof)
    pout = penetration(res["vmd"], model, prof)
    assert pout["arm_torso"]["max"] <= pin["arm_torso"]["max"] + 0.02


def test_signed_target_is_not_flipped():
    """A signed prior target may be opposite the current arm.

    Triangulated directions are lines and are flipped toward the prediction.
    A learned torso-relative prior outputs a signed direction; flipping it
    injects a 180 deg jump when the target is near perpendicular.  The
    ``signed=True`` path must apply the target, while ``signed=False`` keeps
    the old line behaviour.
    """
    model, prof = fixture()
    vmd = parse_vmd(FRONT)
    prior = current_directions(vmd, model, prof)
    targets = {k: v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
               for k, v in prior.items()}
    u = targets["upper_l"].copy()
    targets["upper_l"] = -u
    res = apply_direction_prior(vmd, model, prof, targets, cap_deg=180.0,
                                pen_guard=False, signed=True)
    tr = pose_track(model, res["vmd"], prof)
    d = tr["world"]["forearm_l"] - tr["world"]["upperarm_l"]
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    err_signed = np.degrees(np.arccos(np.clip((d * (-u)).sum(1), -1.0, 1.0)))
    assert float(err_signed.mean()) < 2.0, float(err_signed.mean())

    res2 = apply_direction_prior(vmd, model, prof, targets, cap_deg=180.0,
                                 pen_guard=False, signed=False)
    tr2 = pose_track(model, res2["vmd"], prof)
    d2 = tr2["world"]["forearm_l"] - tr2["world"]["upperarm_l"]
    d2 = d2 / np.maximum(np.linalg.norm(d2, axis=1, keepdims=True), 1e-12)
    err_line = np.degrees(np.arccos(np.clip((d2 * (-u)).sum(1), -1.0, 1.0)))
    assert float(err_line.mean()) > 90.0, float(err_line.mean())


if __name__ == "__main__":
    test_triangulation_recovers_known_direction()
    test_targets_from_result_marks_degenerate_frames()
    test_noop_apply_preserves_everything()
    test_target_reached_and_cap()
    test_body_relative_is_invariant_to_global_rotation()
    test_verify_rejects_non_arm_change()
    test_penetration_guard_rejects_inward_target()
    test_signed_target_is_not_flipped()
    print("test_arm3d: OK")
