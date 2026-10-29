# -*- coding: utf-8 -*-
"""M2.4c target-PMX multi-view BA tests.

These pin the properties the stage must preserve, not a particular optimizer
result:

* the differentiable FK agrees with ``adapter.fk_positions`` at zero delta;
* the PMX projection agrees with a direct numpy projection;
* ``merge_arm_tracks`` copies only the arm chain, so the root/torso/legs stay
  byte-identical;
* ``verify_ba`` rejects a changed non-arm track and accepts a no-op merge
  under the GT-budget contact rule.
"""
import copy
import json
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

from vtv.adapter import canonical_from_pmx, fk_positions, q_from_axis_angle, qmul
from vtv.ba3d import (PmxBodyBA, merge_arm_tracks, project_torch, verify_ba)
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


def test_fk_matches_adapter():
    import torch
    model, prof = fixture()
    vmd = parse_vmd(FRONT)
    motion = canonical_from_pmx(model, prof, vmd)
    body = PmxBodyBA(motion)
    F = len(motion["frames"])
    with torch.no_grad():
        G, J = body.forward({}, torch.zeros((F, 3), dtype=torch.float64))
    Jref = fk_positions(motion["world_rot"], motion["rest_joints"],
                        motion["root_pos"])
    worst = 0.0
    for j, p in J.items():
        if j in Jref:
            worst = max(worst, float(np.max(np.abs(
                p.detach().cpu().numpy() - Jref[j]))))
    assert worst < 1e-9, worst


def test_projection_matches_numpy():
    import torch
    view = {"R": np.eye(3), "C": np.zeros(3), "fx": 100.0, "fy": 100.0,
            "cx": 50.0, "cy": 40.0}
    P = torch.tensor([[[0.0, 0.0, 2.0], [0.1, 0.2, 3.0]]],
                     dtype=torch.float64)
    uv = project_torch(P, view).detach().cpu().numpy()
    want = np.array([[[50.0, 40.0],
                      [100.0 * 0.1 / 3.0 + 50.0, 100.0 * 0.2 / 3.0 + 40.0]]])
    assert np.max(np.abs(uv - want)) < 1e-9, (uv, want)


def test_merge_copies_only_arm_chain():
    model, prof = fixture()
    base = parse_vmd(FRONT)
    ba = copy.deepcopy(base)
    smap = prof["standard_map"]
    # change one arm bone and one non-arm bone in the BA input
    for j, axis_deg in (("upperarm_l", 12.0), ("chest", 9.0)):
        bn = smap[j]
        q = np.asarray(ba["bones"][bn][0]["rot"], dtype=np.float64)
        g = np.asarray(q_from_axis_angle(math.radians(axis_deg), 0.0, 0.0))
        ba["bones"][bn][0]["rot"] = tuple(qmul(tuple(g), tuple(q)))
    merged, meta = merge_arm_tracks(base, ba, model, prof)
    assert smap["upperarm_l"] in meta["copied"]
    assert merged["bones"][smap["upperarm_l"]] != base["bones"][smap["upperarm_l"]]
    # the non-arm chest edit must NOT be copied
    assert merged["bones"][smap["chest"]] == base["bones"][smap["chest"]]
    assert merged["ik_frames"] == base["ik_frames"]


def test_verify_accepts_noop_and_rejects_non_arm():
    model, prof = fixture()
    base = parse_vmd(FRONT)
    gt = parse_vmd(GT)
    same = copy.deepcopy(base)
    ok = verify_ba(base, same, model, prof, gt_vmd=gt)
    assert ok["ok"], [k for k, v in ok["checks"].items()
                      if not v.get("ok", True) and k != "penetration_pairs_strict"]
    out = copy.deepcopy(base)
    bn = prof["standard_map"]["chest"]
    q = np.asarray(out["bones"][bn][0]["rot"], dtype=np.float64)
    g = np.asarray(q_from_axis_angle(math.radians(5.0), 0.0, 0.0))
    out["bones"][bn][0]["rot"] = tuple(qmul(tuple(g), tuple(q)))
    bad = verify_ba(base, out, model, prof, gt_vmd=gt)
    assert not bad["checks"]["non_arm_unchanged"]["ok"]
    assert not bad["ok"]


if __name__ == "__main__":
    test_fk_matches_adapter()
    test_projection_matches_numpy()
    test_merge_copies_only_arm_chain()
    test_verify_accepts_noop_and_rejects_non_arm()
    print("test_ba3d: OK")
