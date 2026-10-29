"""Unit tests for src/vtv/hands.py (M3).

Run:  python tests/test_hands.py
"""
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from vtv.hands import (  # noqa: E402
    JOINT_INDEX, JOINT_KEYS, LEVEL_L0, LEVEL_L1, LEVEL_L2, LEVEL_L3,
    MP_NEUTRAL_SPREAD, MP_NEUTRAL_THUMB, HandRig, HandTrackConfig,
    _swing_q, angles_to_local_quats, angles_to_targets, assign_levels,
    build_angle_track, fit_l1_gain, flatten_angles, hand_plausibility,
    mp_joint_angles, vector_to_measured,
)
from vtv.mmd.mathutil import qaxis, qmul, qnorm, qrot  # noqa: E402

PROFILES = {k: json.load(open(os.path.join(ROOT, "profiles", f"{k}.json"),
                               encoding="utf-8"))
            for k in ("si", "luoxi", "tiflos")}
IDENT = (0.0, 0.0, 0.0, 1.0)


def test_rig_geometry():
    for key, prof in PROFILES.items():
        for side in ("l", "r"):
            rig = HandRig.from_profile(prof, side)
            assert len(rig.finger_names()) == 15, (key, side, len(rig.finger_names()))
            assert len(set(rig.finger_names())) == 15
            u, l, n = rig.hand_basis
            assert abs(np.linalg.norm(u) - 1) < 1e-6
            assert abs(np.linalg.norm(l) - 1) < 1e-6
            assert abs(np.linalg.norm(n) - 1) < 1e-6
            assert abs(np.dot(u, l)) < 1e-6
            assert abs(np.dot(u, n)) < 1e-6
            # palm normal must point to the thumb-base side
            by_name = {b["name"]: b for b in prof["bones"]}
            thumb = np.asarray(by_name[prof["finger_map"][f"thumb0_{side}"]]["pos"]) \
                - np.asarray(by_name[prof["standard_map"][f"hand_{side}"]]["pos"])
            assert float(np.dot(thumb, n)) > 0, (key, side)
            # rest spread must fan from index (positive) to pinky (negative)
            sp = [rig.rest_angles[f]["spread"] for f in ("index", "middle", "ring", "pinky")]
            assert sp[0] > sp[3], (key, side, sp)
            for finger, chain in rig.bones.items():
                for b in chain:
                    assert b.length > 1e-6, (key, side, finger, b.name)


def _neutral_measurement():
    meas = {f: {"spread": MP_NEUTRAL_SPREAD[f], "mcp": 0.0, "pip": 0.0, "dip": 0.0}
            for f in ("index", "middle", "ring", "pinky")}
    meas["thumb"] = {"cmc": MP_NEUTRAL_THUMB, "mcp": 0.0, "ip": 0.0}
    return meas


def test_identity_pose_is_rest():
    for key, prof in PROFILES.items():
        for side in ("l", "r"):
            rig = HandRig.from_profile(prof, side)
            qs = angles_to_local_quats(rig, _neutral_measurement())
            assert len(qs) == 15
            for name, q in qs.items():
                ang = 2 * math.degrees(math.acos(min(1.0, abs(q[3]))))
                assert ang < 1e-5, (key, side, name, ang)


def test_flexion_goes_to_palm_and_never_backwards():
    for key, prof in PROFILES.items():
        for side in ("l", "r"):
            rig = HandRig.from_profile(prof, side)
            u, l, n = rig.hand_basis
            for finger in ("index", "middle", "ring", "pinky"):
                for mcp in (10.0, 40.0, 80.0):
                    meas = _neutral_measurement()
                    meas[finger]["mcp"] = mcp
                    t = angles_to_targets(rig, meas)
                    d0 = rig.rest_dirs[finger][0]
                    d1 = t[(finger, 0)]
                    ang = math.degrees(math.acos(float(np.clip(np.dot(d0, d1), -1, 1))))
                    assert abs(ang - mcp) < 0.5, (key, side, finger, mcp, ang)
                    assert float(np.dot(d1, n)) > 0 or mcp == 0, (key, side, finger, mcp)
            # thumb flexion also bends toward the palm
            meas = _neutral_measurement()
            meas["thumb"]["mcp"] = 40.0
            t = angles_to_targets(rig, meas)
            d0 = rig.rest_dirs["thumb"][0]
            assert float(np.dot(t[("thumb", 1)], n)) > 0, (key, side)


def test_spread_moves_toward_index():
    for key, prof in PROFILES.items():
        for side in ("l", "r"):
            rig = HandRig.from_profile(prof, side)
            u, l, n = rig.hand_basis
            d0 = rig.rest_dirs["ring"][0]
            meas = _neutral_measurement()
            meas["ring"]["spread"] = MP_NEUTRAL_SPREAD["ring"] + 15.0
            t = angles_to_targets(rig, meas)
            assert float(np.dot(t[("ring", 0)], l)) > float(np.dot(d0, l)), (key, side)


def test_limits_and_negative_flexion():
    prof = PROFILES["si"]
    rig = HandRig.from_profile(prof, "l")
    meas = _neutral_measurement()
    meas["index"].update({"mcp": 300.0, "pip": 300.0, "dip": 300.0, "spread": 90.0})
    meas["thumb"].update({"cmc": 200.0, "mcp": 300.0, "ip": 300.0})
    t = angles_to_targets(rig, meas)
    for finger in ("index",):
        angles = []
        d_prev = rig.rest_dirs[finger][0]
        for k in range(3):
            d = t[(finger, k)]
            angles.append(math.degrees(math.acos(float(np.clip(np.dot(d_prev, d), -1, 1)))))
            d_prev = d
        # mcp <= 85, pip <= 105, dip <= 70 (allow the rest bend)
        assert angles[0] <= 86, angles
        assert angles[1] <= 106, angles
        assert angles[2] <= 71, angles
    # negative flexion clamps to 0 (no hyperextension)
    meas2 = _neutral_measurement()
    meas2["index"]["mcp"] = -40.0
    t2 = angles_to_targets(rig, meas2)
    d0 = rig.rest_dirs["index"][0]
    ang = math.degrees(math.acos(float(np.clip(np.dot(d0, t2[("index", 0)]), -1, 1))))
    assert ang < 0.6, ang


def test_wrist_rotation_is_irrelevant_to_local_quats():
    prof = PROFILES["si"]
    rig = HandRig.from_profile(prof, "r")
    ax = np.array([0.2, -0.4, 0.9])
    ax /= np.linalg.norm(ax)
    Q = qaxis(tuple(ax), math.radians(73))
    m = _neutral_measurement()
    m["index"]["mcp"] = 35.0
    q0 = angles_to_local_quats(rig, m, None)
    q1 = angles_to_local_quats(rig, m, Q)
    for k in q0:
        assert max(abs(a - b) for a, b in zip(q0[k], q1[k])) < 1e-9, k
    # and fingers follow the wrist in world space
    qs = angles_to_local_quats(rig, m)
    q = qs[rig.chains["index"][0]]
    world = qrot(qmul(Q, q), rig.rest_dirs["index"][0])
    target_world = qrot(Q, angles_to_targets(rig, m)[("index", 0)])
    assert math.degrees(math.acos(float(np.clip(np.dot(world, target_world), -1, 1)))) < 1e-5


def test_mp_angles_on_rest_hand():
    # Build a flat synthetic hand: wrist at origin, fingers along +X, index on +Z.
    pts = np.zeros((21, 3))
    pts[0] = [0, 0, 0]
    lateral = {"index": 0.55, "middle": 0.0, "ring": -0.55, "pinky": -1.0}
    ids = {"thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
           "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
    for f, idx in ids.items():
        if f == "thumb":
            pts[1] = [0.2, 0, 0.5]
            pts[2] = [0.5, 0, 0.7]
            pts[3] = [0.8, 0, 0.8]
            pts[4] = [1.0, 0, 0.85]
        else:
            z = lateral[f]
            mcp = idx[0]
            pts[mcp] = [1.0, 0, z]
            pts[mcp + 1] = [2.0, 0, z]
            pts[mcp + 2] = [2.7, 0, z]
            pts[mcp + 3] = [3.2, 0, z]
    a = mp_joint_angles(pts)
    # fingers are straight, so PIP/DIP are 0; the MCP angle equals the splay
    # between the metacarpal (wrist->MCP) and the phalanx (+X)
    for f in ("index", "middle", "ring", "pinky"):
        assert a[f]["pip"] < 0.5 and a[f]["dip"] < 0.5, (f, a[f])
        expected = math.degrees(math.atan2(abs(lateral[f]), 1.0))
        assert abs(a[f]["mcp"] - expected) < 0.8, (f, a[f], expected)
    assert 30 < a["thumb"]["cmc"] < 60, a["thumb"]


def test_hand_plausibility_accepts_hand_rejects_garbage():
    pts = np.zeros((21, 2))
    pts[0] = [0, 0]
    for base, z in [(5, 15), (9, 0), (13, -12), (17, -20)]:
        pts[base] = [40, z]
        pts[base + 1] = [70, z]
        pts[base + 2] = [95, z]
        pts[base + 3] = [115, z]
    pts[1] = [10, 20]
    pts[2] = [25, 35]
    pts[3] = [35, 45]
    pts[4] = [45, 50]
    world = np.zeros((21, 3))
    world[:, 0] = pts[:, 0] / 400.0
    world[:, 2] = pts[:, 1] / 800.0
    world[:, 1] = np.linspace(0, 0.05, 21)
    ok, fails, m = hand_plausibility(pts, world, (0, 0), 40.0)
    assert ok, (fails, m)
    # collapsed fingers (zero-length chains) = not a hand
    bad = pts.copy()
    for base in (5, 9, 13, 17):
        for k in (1, 2, 3):
            bad[base + k] = bad[base]
    ok2, fails2, _ = hand_plausibility(bad, world, (0, 0), 40.0)
    assert not ok2, fails2
    # a tiny palm is rejected outright
    tiny = pts / 10.0
    ok3, fails3, _ = hand_plausibility(tiny, world, (0, 0), 4.0)
    assert not ok3, fails3


def _rec(angles, valid=True, score=0.9, palm=30.0, world_ok=True):
    a = dict(angles)
    return {"valid": valid, "score": score, "palm_px": palm,
            "angles3d": a, "angles2d": a, "world_ok": world_ok}


def _angles(mcp=20.0, pip=20.0):
    d = {}
    for f in ("index", "middle", "ring", "pinky"):
        d[f] = {"spread": 0.0, "mcp": mcp, "pip": pip, "dip": 10.0}
    d["thumb"] = {"cmc": 45.0, "mcp": 10.0, "ip": 5.0}
    return d


def test_levels_and_smoothing():
    cfg = HandTrackConfig()
    n = 20
    recs = [None] * n
    for i in range(5, 12):          # 7-frame stable core
        recs[i] = _rec(_angles())
    recs[16] = _rec(_angles())      # isolated frame: must stay fallback
    conf = [0.9] * n
    levels, segs, diag = assign_levels(recs, conf, cfg)
    assert levels[5] == LEVEL_L0 and levels[11] == LEVEL_L0
    assert levels[16] >= LEVEL_L2
    assert levels[0] >= LEVEL_L2
    angles, target = build_angle_track(recs, levels, cfg)
    assert angles.shape == (n, len(JOINT_KEYS))
    assert np.isfinite(angles).all()
    # per-frame step limits
    for i in range(1, n):
        d = np.abs(angles[i] - angles[i - 1])
        assert d[JOINT_INDEX["index.mcp"]] <= cfg.max_step_mcp + 1e-6
        assert d[JOINT_INDEX["index.pip"]] <= cfg.max_step_pip + 1e-6
        assert d[JOINT_INDEX["index.dip"]] <= cfg.max_step_dip + 1e-6
        assert d[JOINT_INDEX["index.spread"]] <= cfg.max_step_spread + 1e-6
    # fallback values are the L3 pose when the wrist is invisible
    levels3, _, _ = assign_levels(recs, [0.1] * n, cfg)
    assert levels3[0] == LEVEL_L3
    assert levels3[16] == LEVEL_L3
    # L2 with a visible wrist
    levels2, _, _ = assign_levels(recs, [0.8] * n, cfg)
    assert levels2[0] == LEVEL_L2


def test_noisy_segment_is_rejected():
    cfg = HandTrackConfig()
    n = 10
    recs = []
    for i in range(n):
        a = _angles(mcp=10.0 + (i % 2) * 60.0, pip=10.0 + (i % 2) * 50.0)
        recs.append(_rec(a))
    levels, _, diag = assign_levels(recs, [0.9] * n, cfg)
    assert all(l >= LEVEL_L2 for l in levels), levels
    assert diag["rejected"] > 0


def test_sparse_segment_is_rejected():
    cfg = HandTrackConfig()
    recs = [None] * 8
    recs[3] = _rec(_angles())
    recs[4] = _rec(_angles())
    levels, _, _ = assign_levels(recs, [0.9] * 8, cfg)
    assert all(l >= LEVEL_L2 for l in levels), levels


def test_agreement_gate_rejects_3d_2d_mismatch():
    cfg = HandTrackConfig()
    recs = []
    for i in range(6):
        a3 = _angles(mcp=80.0, pip=80.0)
        a2 = _angles(mcp=0.0, pip=0.0)
        recs.append({"valid": True, "score": 0.9, "palm_px": 30.0,
                     "angles3d": a3, "angles2d": a2, "world_ok": True})
    levels, _, _ = assign_levels(recs, [0.9] * 6, cfg)
    assert all(l >= LEVEL_L2 for l in levels), levels


def test_fit_l1_gain():
    recs = []
    for i in range(10):
        a2 = _angles(mcp=30.0, pip=40.0)
        a3 = _angles(mcp=45.0, pip=60.0)
        recs.append({"valid": True, "score": 0.9, "palm_px": 30.0,
                     "angles3d": a3, "angles2d": a2, "world_ok": True})
    gain, info = fit_l1_gain(recs, default=1.3)
    assert 1.4 < gain <= 1.5, (gain, info)
    assert info["source"] == "fit"


def test_vector_roundtrip():
    from vtv.hands import _fallback_angles
    v = _fallback_angles(LEVEL_L2)
    m = vector_to_measured(v)
    assert set(m) == {"thumb", "index", "middle", "ring", "pinky"}
    for f, vals in m.items():
        for k, x in vals.items():
            assert isinstance(x, float)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"test_hands: OK ({len(tests)} tests)")
