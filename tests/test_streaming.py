# -*- coding: utf-8 -*-
"""M2.5 v0 streaming/gap-completion tests.

Pure Python (no GT, no Blender, no learned model).  The synthetic clips have a
smoothly rotating arm and a translating root, so a fill that freezes or jumps
is visible in the numbers.
"""
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

from vtv.cleanup import dense_tracks
from vtv.mmd.vmd import parse_vmd
from vtv.streaming import (StreamingConfig, _carry_contact, _q_angle_deg,
                           _q_conj, _q_exp, _q_log, _q_mul, _q_normalize,
                           _q_slerp, _squad, _visible_tracks, estimate_contact,
                           fill_tracks, fill_vmd, normalize_confidence,
                           summarize)


def qaxis(axis, deg):
    h = math.radians(deg) * 0.5
    s = math.sin(h)
    return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(h))


def make_vmd(n=80, root_step=0.02, arm_deg=35.0):
    curve = (20, 20, 107, 107)
    bones = {
        "root": [{"frame": f, "pos": (root_step * f, 0.0, 0.0),
                  "rot": (0, 0, 0, 1), "curves": (curve,) * 4}
                 for f in range(n)],
        "arm_l": [{"frame": f, "pos": (0, 0, 0),
                   "rot": qaxis((0, 1, 0), arm_deg * math.sin(2 * math.pi * f / n)),
                   "curves": (curve,) * 4} for f in range(n)],
        "arm_r": [{"frame": f, "pos": (0, 0, 0),
                   "rot": qaxis((0, 1, 0), -arm_deg * math.sin(2 * math.pi * f / n)),
                   "curves": (curve,) * 4} for f in range(n)],
        "toe_l": [{"frame": f, "pos": (0, 0, 0), "rot": (0, 0, 0, 1),
                   "curves": (curve,) * 4} for f in range(n)],
        "toe_r": [{"frame": f, "pos": (0, 0, 0), "rot": (0, 0, 0, 1),
                   "curves": (curve,) * 4} for f in range(n)],
    }
    return {"model_name": "streamtest", "bones": bones, "morphs": {},
            "ik_frames": [], "max_frame": n - 1}


def minimal_model():
    bones = [
        {"name": "root", "parent": -1, "pos": [0.0, 0.0, 0.0],
         "flags": 0x0004, "layer": 0},
        {"name": "toe_l", "parent": 0, "pos": [-2.0, 0.0, 0.0],
         "flags": 0, "layer": 0},
        {"name": "toe_r", "parent": 0, "pos": [2.0, 0.0, 0.0],
         "flags": 0, "layer": 0},
    ]
    profile = {"standard_map": {"root": "root", "toe_l": "toe_l",
                                "toe_r": "toe_r"}, "height": 10.0}
    return {"bones": bones, "name": "test"}, profile


def max_step(vmd_name, rot):
    return max(_q_angle_deg(rot[i], rot[i - 1]) for i in range(1, len(rot)))


def test_quat_math():
    q = _q_normalize(_q_exp(np.array([0.3, -0.2, 0.5])))
    r = _q_log(q)
    q2 = _q_exp(r)
    assert _q_angle_deg(q, q2) < 1e-6
    q0 = _q_normalize(np.array([0.1, 0.2, 0.3, 0.9]))
    q1 = _q_normalize(np.array([-0.4, 0.1, 0.2, 0.8]))
    assert _q_angle_deg(_squad(q0, q1, q0, q1, 0.0), q0) < 1e-9
    assert _q_angle_deg(_squad(q0, q1, q0, q1, 1.0), q1) < 1e-9
    assert _q_angle_deg(_q_mul(q0, _q_conj(q0)), np.array([0, 0, 0, 1.0])) < 1e-9


def test_confidence_shapes():
    vmd = make_vmd(20)
    _, tracks = dense_tracks(vmd)
    names = sorted(tracks)
    c1 = normalize_confidence(None, tracks, 20)
    assert all(np.all(c1[n] == 1.0) for n in names)
    c2 = normalize_confidence(np.linspace(0, 1, 20), tracks, 20)
    assert np.allclose(c2[names[0]], np.linspace(0, 1, 20))
    mat = np.tile(np.linspace(0, 1, 20)[:, None], (1, len(names)))
    c3 = normalize_confidence(mat, tracks, 20)
    assert np.allclose(c3[names[-1]], np.linspace(0, 1, 20))
    c4 = normalize_confidence({names[0]: [0.5] * 20}, tracks, 20)
    assert np.allclose(c4[names[0]], 0.5) and np.all(c4[names[1]] == 1.0)


def test_no_gap_identity():
    vmd = make_vmd(60)
    res = fill_vmd(vmd, None, mode="offline")
    assert res["report"]["observed_changed_max_deg"] < 1e-5
    assert res["report"]["observed_pos_changed_max"] < 1e-9
    assert res["report"]["gaps_total"] == 0


def test_one_frame_gap_holds():
    vmd = make_vmd(40)
    conf = np.ones(40)
    conf[20] = 0.0
    res = fill_vmd(vmd, conf, mode="offline")
    tr = res["tracks"]
    for name, rec in tr.items():
        src = vmd["bones"][name]
        q_prev = np.asarray(src[19]["rot"], dtype=np.float64)
        assert _q_angle_deg(rec["rot"][20], q_prev) < 1e-6, name
    assert res["report"]["observed_changed_max_deg"] < 1e-5


def test_long_gap_moves_and_closes():
    vmd = make_vmd(80)
    conf = np.ones(80)
    s, e = 30, 41
    conf[s:e + 1] = 0.0
    res = fill_vmd(vmd, conf, mode="offline",
                   cfg=StreamingConfig(smooth_gaps=True))
    rot = res["tracks"]["arm_l"]["rot"]
    # not frozen: the gap has real motion
    gap_step = max(_q_angle_deg(rot[t], rot[t - 1])
                   for t in range(s + 1, e + 1))
    assert gap_step > 0.05, gap_step
    # no boundary jump: the first/last gap frames are close to the observations
    assert _q_angle_deg(rot[s], rot[s - 1]) < 10.0
    assert _q_angle_deg(rot[e], rot[e + 1]) < 10.0
    assert res["report"]["observed_changed_max_deg"] < 1e-5
    gaps = [g for g in res["report"]["gaps"] if g["bone"] == "arm_l"]
    assert gaps and gaps[0]["planned"] == "squad", gaps


def test_stream_long_gap_uses_lookahead():
    vmd = make_vmd(100)
    conf = np.ones(100)
    s, e = 30, 59
    conf[s:e + 1] = 0.0
    cfg = StreamingConfig(lookahead_s=0.4, max_rollout_s=1.5)
    res = fill_vmd(vmd, conf, mode="stream", cfg=cfg)
    gaps = [g for g in res["report"]["gaps"] if g["bone"] == "arm_l"]
    assert gaps and gaps[0]["planned"] == "rollout_then_bridge", gaps
    rot = res["tracks"]["arm_l"]["rot"]
    assert max(_q_angle_deg(rot[t], rot[t - 1])
               for t in range(s + 1, e + 1)) > 0.02
    assert _q_angle_deg(rot[e], rot[e + 1]) < 15.0
    assert res["report"]["observed_changed_max_deg"] < 1e-5


def test_mid_confidence_blend_is_not_observation():
    vmd = make_vmd(50)
    conf = np.ones(50)
    conf[25] = 0.5
    res = fill_vmd(vmd, conf, mode="offline")
    q = res["tracks"]["arm_l"]["rot"][25]
    q_obs = np.asarray(vmd["bones"]["arm_l"][25]["rot"], dtype=np.float64)
    assert _q_angle_deg(q, q_obs) > 1e-4
    assert np.all(np.isfinite(q))


def test_contact_carry():
    obs = np.zeros(40, dtype=bool)
    obs[5:10] = True
    conf = np.ones(40)
    conf[10:] = 0.0
    carried = _carry_contact(obs, conf, high=0.75, hold_frames=10)
    assert carried[5:10].all()
    assert carried[10:19].all()
    assert not carried[20:].any()


def test_estimate_contact_minimal():
    model, profile = minimal_model()
    vmd = make_vmd(80, root_step=0.01)
    _, tracks = dense_tracks(vmd)
    conf = {n: np.ones(80) for n in tracks}
    contact = estimate_contact(model, profile, tracks, conf,
                               StreamingConfig())
    assert contact is not None
    assert contact["sides"]["l"]["observed_fraction"] > 0.8
    assert contact["sides"]["r"]["observed_fraction"] > 0.8
    # fast root motion: the toe slides, so it is not planted
    vmd2 = make_vmd(80, root_step=0.3)
    _, tracks2 = dense_tracks(vmd2)
    conf2 = {n: np.ones(80) for n in tracks2}
    contact2 = estimate_contact(model, profile, tracks2, conf2,
                                StreamingConfig())
    assert contact2["sides"]["l"]["observed_fraction"] < 0.2


def test_visible_tracks_holds_gap():
    vmd = make_vmd(30)
    _, tracks = dense_tracks(vmd)
    conf = {n: np.ones(30) for n in tracks}
    for n in conf:
        conf[n][10:15] = 0.0
    vis = _visible_tracks(tracks, conf, StreamingConfig())
    for name in tracks:
        for t in range(10, 15):
            assert _q_angle_deg(vis[name]["rot"][t],
                                tracks[name]["rot"][9]) < 1e-9
    assert _q_angle_deg(vis["arm_l"]["rot"][20],
                        tracks["arm_l"]["rot"][20]) < 1e-9


def test_fill_tracks_direct_and_summary():
    vmd = make_vmd(40)
    _, tracks = dense_tracks(vmd)
    conf = {n: np.ones(40) for n in tracks}
    conf["arm_l"][20] = 0.0
    res = fill_tracks(tracks, conf, mode="offline")
    assert res["report"]["observed_changed_max_deg"] < 1e-5
    summ = summarize(res["report"])
    assert summ["gaps"] >= 1 and summ["gap_frames"] >= 1


def make_leg_model(height=30.0, hip_x=1.0):
    """A minimal two-leg PMX-like skeleton for the v0.1 stance IK tests."""
    h = float(height)
    bones = [
        {"name": "root", "parent": -1, "pos": [0.0, 0.0, 0.0],
         "flags": 0x0004, "layer": 0},
        {"name": "pelvis", "parent": 0, "pos": [0.0, h, 0.0],
         "flags": 0, "layer": 0},
    ]
    smap = {"root": "root", "pelvis": "pelvis"}
    for side, sx in (("l", 1.0), ("r", -1.0)):
        base = len(bones)
        bones += [
            {"name": f"thigh_{side}", "parent": 1,
             "pos": [sx * hip_x, h, 0.0], "flags": 0, "layer": 0},
            {"name": f"shin_{side}", "parent": base,
             "pos": [sx * hip_x, h * 0.5, 0.0], "flags": 0, "layer": 0},
            {"name": f"foot_{side}", "parent": base + 1,
             "pos": [sx * hip_x, 0.0, 0.0], "flags": 0, "layer": 0},
            {"name": f"toe_{side}", "parent": base + 2,
             "pos": [sx * hip_x, 0.0, -1.0], "flags": 0, "layer": 0},
        ]
        for j in ("thigh", "shin", "foot", "toe"):
            smap[f"{j}_{side}"] = f"{j}_{side}"
    profile = {"standard_map": smap, "height": h, "model_key": "legtest"}
    return {"bones": bones, "name": "legtest"}, profile


def make_leg_vmd(n=80, root_step=0.05, height=30.0, hip_x=1.0):
    curve = (20, 20, 107, 107)
    bones = {}

    def add(name, pos_fn, rot=(0, 0, 0, 1)):
        bones[name] = [{"frame": f, "pos": pos_fn(f), "rot": rot,
                        "curves": (curve,) * 4} for f in range(n)]

    add("root", lambda f: (root_step * f, 0.0, 0.0))
    for name in ("pelvis",):
        add(name, lambda f: (0.0, 0.0, 0.0))
    for side, sx in (("l", 1.0), ("r", -1.0)):
        for j in ("thigh", "shin", "foot", "toe"):
            add(f"{j}_{side}", lambda f: (0.0, 0.0, 0.0))
    return {"model_name": "legtest", "bones": bones, "morphs": {},
            "ik_frames": [], "max_frame": n - 1}


def synthetic_contact(n, gap, sides=("l", "r"), thr=100.0):
    mask = np.ones(n, dtype=bool)
    mask[gap[0]:gap[1] + 1] = False
    out = {"sides": {}, "both": np.ones(n, dtype=bool), "height": 30.0,
           "frames": n}
    for side in ("l", "r"):
        out["sides"][side] = {
            "observed": mask.copy(), "carried": mask.copy(),
            "thr": thr, "observed_fraction": float(mask.mean()),
            "carried_fraction": float(mask.mean()), "intervals": [],
        }
    for side in ("l", "r"):
        if side not in sides:
            out["sides"][side]["observed"][:] = False
            out["sides"][side]["carried"][:] = False
            # defeat the band-slack fallback: this side is genuinely lifted
            out["sides"][side]["thr"] = -1.0
    return out


def _eval(model, profile, tracks):
    from vtv.cleanup import eval_tracks
    return eval_tracks(model, profile, tracks)


def test_stance_ik_anchors_toe():
    from vtv.stance_ik import StanceConfig, apply_stance_ik
    model, profile = make_leg_model(30.0)
    vmd = make_leg_vmd(80, root_step=0.1, height=30.0)
    _, tracks = dense_tracks(vmd)
    conf = {n: np.ones(80) for n in tracks}
    conf = {n: v.copy() for n, v in conf.items()}
    for n in conf:
        conf[n][20:50] = 0.0
    contact = synthetic_contact(80, (20, 49))
    cfg = StanceConfig(release_mode="budget", anchor_max_spread_frac=0.02)
    res = apply_stance_ik(tracks, conf, contact, model, profile, cfg)
    rep = res["report"]
    assert rep["gaps_anchored"] >= 1, rep
    assert rep["frames_edited"] > 0
    # the anchored toe stays put (the v0/base moves 3 units over the gap)
    tr0 = _eval(model, profile, tracks)
    tr1 = _eval(model, profile, res["tracks"])
    anchors = {p["side"]: np.asarray(p["anchor_toe"])
               for p in rep["plans"] if p.get("applied")}
    assert set(anchors) == {"l", "r"}, anchors
    for side in ("l", "r"):
        anchor = anchors[side]
        drift = np.linalg.norm(
            (tr1["world"][f"toe_{side}"][20:50] - anchor)[:, [0, 2]], axis=1)
        assert float(drift.max()) <= 0.07, (side, float(drift.max()))
        before = np.linalg.norm(
            (tr0["world"][f"toe_{side}"][20:50] - anchor)[:, [0, 2]], axis=1)
        assert float(before.max()) > 2.0, (side, float(before.max()))
    # observed frames are untouched
    for name in tracks:
        ok = conf[name] >= 0.75
        assert np.allclose(res["tracks"][name]["rot"][ok],
                           tracks[name]["rot"][ok], atol=0.0)
        assert np.allclose(res["tracks"][name]["pos"][ok],
                           tracks[name]["pos"][ok], atol=0.0)
    assert rep["observed_changed_max_deg"] < 1e-6
    # bone lengths unchanged
    for p in rep["plans"]:
        if p.get("applied"):
            assert p["verify"]["bone_length_max_err"] < 1e-6


def test_stance_ik_single_foot_leaves_other_leg():
    from vtv.stance_ik import StanceConfig, apply_stance_ik
    model, profile = make_leg_model(30.0)
    vmd = make_leg_vmd(70, root_step=0.1, height=30.0)
    _, tracks = dense_tracks(vmd)
    conf = {n: np.ones(70) for n in tracks}
    for n in conf:
        conf[n][15:45] = 0.0
    contact = synthetic_contact(70, (15, 44), sides=("l",))
    res = apply_stance_ik(tracks, conf, contact, model, profile,
                          StanceConfig(release_mode="budget",
                                       anchor_max_spread_frac=0.02))
    rep = res["report"]
    assert rep["sides_anchored"] == ["l"], rep["sides_anchored"]
    for name in ("thigh_r", "shin_r", "foot_r"):
        assert np.array_equal(res["tracks"][name]["rot"], tracks[name]["rot"])
    assert not np.array_equal(res["tracks"]["thigh_l"]["rot"],
                              tracks["thigh_l"]["rot"])


def test_stance_ik_unreachable_releases():
    from vtv.stance_ik import StanceConfig, apply_stance_ik
    model, profile = make_leg_model(30.0, hip_x=0.4)
    # Shrink the legs: move the knee/ankle rest points up so l1+l2 is 3.
    for b in model["bones"]:
        if b["name"] in ("shin_l", "shin_r"):
            b["pos"][1] = 28.5
        if b["name"] in ("foot_l", "foot_r"):
            b["pos"][1] = 27.0
        if b["name"] in ("toe_l", "toe_r"):
            b["pos"][1] = 27.0
    vmd = make_leg_vmd(80, root_step=1.0, height=30.0)
    _, tracks = dense_tracks(vmd)
    conf = {n: np.ones(80) for n in tracks}
    for n in conf:
        conf[n][20:60] = 0.0
    contact = synthetic_contact(80, (20, 59))
    cfg = StanceConfig(release_mode="budget", anchor_max_spread_frac=0.1,
                       max_lift_frac=0.0)
    res = apply_stance_ik(tracks, conf, contact, model, profile, cfg)
    rep = res["report"]
    assert rep["reach_limited_frames"] > 0, rep
    assert rep["weight_mean"] < 0.9, rep
    for p in rep["plans"]:
        if p.get("applied"):
            assert p["weight_min"] <= 1.0
            assert p["verify"]["bone_length_max_err"] < 1e-6
    for name in res["tracks"]:
        assert np.all(np.isfinite(res["tracks"][name]["rot"]))
    assert rep["observed_changed_max_deg"] < 1e-6


def test_stance_ik_no_long_gap():
    from vtv.stance_ik import StanceConfig, apply_stance_ik
    model, profile = make_leg_model(30.0)
    vmd = make_leg_vmd(20, root_step=0.02, height=30.0)
    _, tracks = dense_tracks(vmd)
    conf = {n: np.ones(20) for n in tracks}
    contact = synthetic_contact(20, (10, 10))
    res = apply_stance_ik(tracks, conf, contact, model, profile,
                          StanceConfig())
    rep = res["report"]
    assert rep["gaps_anchored"] == 0 and rep["plans"] == []
    assert rep["toe_xz_drift_max"] == 0.0


def test_fill_vmd_anchor_stance_end_to_end():
    model, profile = make_leg_model(30.0)
    vmd = make_leg_vmd(70, root_step=0.08, height=30.0)
    conf = np.ones(70)
    conf[20:45] = 0.0
    res = fill_vmd(vmd, conf, model=model, profile=profile, mode="offline",
                   anchor_stance=True,
                   stance={"release_mode": "budget",
                           "anchor_max_spread_frac": 0.02})
    st = res["report"]["stance_ik"]
    assert st and st["enabled"] and st["gaps_anchored"] >= 1, st
    assert res["report"]["observed_changed_max_deg"] < 1e-5
    # a v0 fill without stance is still available and untouched by the flag
    res0 = fill_vmd(vmd, conf, model=model, profile=profile, mode="offline")
    assert res0["report"]["stance_ik"] is None
    assert res0["report"]["observed_changed_max_deg"] < 1e-5


if __name__ == "__main__":
    test_quat_math()
    test_confidence_shapes()
    test_no_gap_identity()
    test_one_frame_gap_holds()
    test_long_gap_moves_and_closes()
    test_stream_long_gap_uses_lookahead()
    test_mid_confidence_blend_is_not_observation()
    test_contact_carry()
    test_estimate_contact_minimal()
    test_visible_tracks_holds_gap()
    test_fill_tracks_direct_and_summary()
    test_stance_ik_anchors_toe()
    test_stance_ik_single_foot_leaves_other_leg()
    test_stance_ik_unreachable_releases()
    test_stance_ik_no_long_gap()
    test_fill_vmd_anchor_stance_end_to_end()
    print("test_streaming: OK")
