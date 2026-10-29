"""Unit tests for the M4 feedback application and QC core.

Run:  python tests/test_feedback_apply.py
"""
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from vtv import feedback_apply as FA  # noqa: E402
from vtv import feedback_data as FD  # noqa: E402
from vtv import qc  # noqa: E402
from vtv.metrics.core import pose_track  # noqa: E402
from vtv.mmd.mathutil import QID, qaxis  # noqa: E402
from vtv.mmd.pmx import parse_pmx  # noqa: E402

PROFILE = json.load(open(os.path.join(ROOT, "profiles", "si.json"),
                         encoding="utf-8"))
CFG = json.load(open(os.path.join(ROOT, "configs", "models.json"),
                     encoding="utf-8"))["models"]
BODY_JOINTS = ("root", "pelvis", "spine", "spine1", "chest", "neck", "head",
               "clavicle_l", "upperarm_l", "forearm_l", "hand_l",
               "clavicle_r", "upperarm_r", "forearm_r", "hand_r",
               "thigh_l", "shin_l", "foot_l", "toe_l",
               "thigh_r", "shin_r", "foot_r", "toe_r")
IK_OFF = [{"frame": 0, "ik": {PROFILE["ik"]["l"]["leg"]["bone"]: False,
                              PROFILE["ik"]["l"]["toe"]["bone"]: False,
                              PROFILE["ik"]["r"]["leg"]["bone"]: False,
                              PROFILE["ik"]["r"]["toe"]["bone"]: False}}]


def _body_vmd(n=24, chest_rot=0.0):
    smap = PROFILE["standard_map"]
    bones = {}
    q_chest = qaxis((1.0, 0.0, 0.0), math.radians(chest_rot)) if chest_rot else QID
    for j in BODY_JOINTS:
        name = smap[j]
        q = q_chest if j == "chest" else QID
        bones[name] = [{"frame": f, "pos": (0.0, 0.0, 0.0), "rot": q,
                        "curves": (FD.DEFAULT_CURVE,) * 4} for f in range(n)]
    return {"model_name": "", "max_frame": n - 1, "bones": bones,
            "morphs": {}, "ik_frames": [dict(x) for x in IK_OFF]}


def _assembled(n=24, chest_rot=0.0):
    return FD.assemble_vmd(_body_vmd(n, chest_rot), PROFILE,
                           np.zeros((n, 2, 19), np.float32),
                           np.zeros((n, 12), np.float32), n_frames=n)


def _track(n=120, spike_at=None, penetration_at=None):
    t = np.arange(n, dtype=float)
    world = {
        "root": np.stack([t * 0.01, np.zeros(n), np.zeros(n)], axis=1),
        "pelvis": np.stack([t * 0.01, np.zeros(n), np.zeros(n)], axis=1),
        "chest": np.stack([np.cos(t * 0.1), np.sin(t * 0.1), t * 0.002], axis=1),
        "spine1": np.stack([np.cos(t * 0.1), np.sin(t * 0.1), t * 0.001], axis=1),
        "upperarm_l": np.stack([np.cos(t * 0.2), np.zeros(n), np.sin(t * 0.2)], axis=1),
        "upperarm_r": np.stack([-np.cos(t * 0.2), np.zeros(n), np.sin(t * 0.2)], axis=1),
        "forearm_l": np.stack([np.cos(t * 0.2) * 0.5, np.zeros(n), np.sin(t * 0.2)], axis=1),
        "forearm_r": np.stack([-np.cos(t * 0.2) * 0.5, np.zeros(n), np.sin(t * 0.2)], axis=1),
        "hand_l": np.stack([np.cos(t * 0.2) * 0.4, np.zeros(n), np.sin(t * 0.2)], axis=1),
        "hand_r": np.stack([-np.cos(t * 0.2) * 0.4, np.zeros(n), np.sin(t * 0.2)], axis=1),
        "toe_l": np.stack([t * 0.005, np.zeros(n), np.zeros(n)], axis=1),
        "toe_r": np.stack([-t * 0.005, np.zeros(n), np.zeros(n)], axis=1),
    }
    if spike_at is not None:
        world["chest"][spike_at] += np.array([0.30, 0.0, 0.0])
    if penetration_at is not None:
        world["toe_l"][penetration_at, 1] = -0.5
    return {"world": world}


# ---------------------------------------------------------------------------
def test_certificate():
    v = _assembled()
    cert = qc.vmd_certificate(v, PROFILE)
    assert cert["ok"], cert
    assert cert["bone_tracks"] == 53 and cert["finger_tracks"] == 30
    assert cert["ik_off"] and cert["finger_translation_max"] == 0.0
    # a missing finger track must fail
    finger = next(iter(PROFILE["finger_map"].values()))
    broken = dict(v)
    broken["bones"] = {k: x for k, x in v["bones"].items() if k != finger}
    assert not qc.vmd_certificate(broken, PROFILE)["ok"]
    # a D-chain track must be rejected even though it is a valid PMX bone
    dname = PROFILE["d_chain"]["l"][0]["name"]
    extra = dict(v)
    extra["bones"] = dict(v["bones"])
    extra["bones"][dname] = [{"frame": 0, "pos": (0, 0, 0), "rot": QID,
                              "curves": (FD.DEFAULT_CURVE,) * 4}]
    cert = qc.vmd_certificate(extra, PROFILE)
    assert not cert["ok"] and dname in cert["forbidden_tracks"], cert
    # finger translation must be zero
    moved = dict(v)
    moved["bones"] = {k: [dict(x) for x in keys] for k, keys in v["bones"].items()}
    moved["bones"][finger][3]["pos"] = (0.1, 0.0, 0.0)
    assert not qc.vmd_certificate(moved, PROFILE)["ok"]


def test_shape_residual_policy():
    n = 8
    raw = np.ones((n, FD.RESIDUAL_DIM), np.float32)
    ck = {"hand_supported": np.ones(2, bool), "face_supported": np.ones(12, bool)}
    lv = np.zeros((n, 2), np.int8)
    fl = np.zeros(n, np.int8)
    res = FA.shape_residual(raw, lv, fl, PROFILE, ck,
                            FA.resolve_config("default"))
    assert not res[:, FD.ROOT_SLICE].any()
    assert not res[:, FD.HAND_SLICE].any()
    assert not res[:, FD.FACE_SLICE].any()
    for i, bone in enumerate(FD.BODY_FACTOR_NAMES):
        is_arm = bone.startswith(("clavicle", "upperarm", "forearm"))
        want = FA.resolve_config("default")["arm_gain" if is_arm else "torso_gain"]
        assert np.allclose(res[:, 2 + 3 * i:5 + 3 * i], want), (bone, want)
    # L2 hand frames are gated to zero even when hands are requested
    cfg = FA.resolve_config("full")
    raw2 = np.ones((n, FD.RESIDUAL_DIM), np.float32)
    lv2 = np.full((n, 2), 2, np.int8)
    fl2 = np.full(n, 2, np.int8)
    res2 = FA.shape_residual(raw2, lv2, fl2, PROFILE, ck, cfg)
    assert not res2[:, FD.HAND_SLICE].any()
    assert not res2[:, FD.FACE_SLICE].any()


def test_off_mode_is_identity():
    n = 16
    body = _assembled(n)
    cfg = FA.resolve_config("off")
    raw = np.ones((n, FD.RESIDUAL_DIM), np.float32)
    ck = {"hand_supported": np.ones(2, bool), "face_supported": np.ones(12, bool)}
    res = FA.shape_residual(raw, np.zeros((n, 2), np.int8),
                            np.zeros(n, np.int8), PROFILE, ck, cfg)
    assert not res.any()
    corr = FD.apply_body_residual(body, parse_pmx(CFG["si"]["pmx"]), PROFILE, res)
    audit = qc.unchanged_audit(body, corr, PROFILE)
    assert audit["ok"] and not audit["changed_tracks"], audit


def test_default_preserves_untouched_tracks():
    n = 24
    body = _body_vmd(n)
    model = parse_pmx(CFG["si"]["pmx"])
    fb = FA.load_feedback_model(os.path.join(ROOT, "outputs/m35/net/m35_best.pt"))
    ev = {
        "kp2d": np.zeros((n, 17, 3), np.float32),
        "bbx_xys": np.tile(np.array([0.5, 0.5, 1.0], np.float32), (n, 1)),
        "img_wh": (1080, 1920), "adapter_scale": 1.0,
        "hand": {"angles": np.zeros((n, 2, 19), np.float32),
                 "level": np.zeros((n, 2), np.int8),
                 "score": np.zeros((n, 2), np.float32),
                 "detected": np.zeros((n, 2), bool)},
        "face": {"weights": np.zeros((n, 12), np.float32),
                 "level": np.zeros(n, np.int8),
                 "detected": np.zeros(n, bool)},
    }
    out = FA.apply_feedback(body, model, PROFILE, fb, ev,
                            FA.resolve_config("default"), {"enabled": False})
    cert = qc.vmd_certificate(out["vmd"], PROFILE)
    assert cert["ok"], cert
    audit = qc.unchanged_audit(body, FD.assemble_vmd(
        body, PROFILE, ev["hand"]["angles"], ev["face"]["weights"], n_frames=n),
        PROFILE)
    assert audit["ok"], audit
    # compare against the M3 baseline (same hands/face, no body correction)
    baseline = FD.assemble_vmd(body, PROFILE, ev["hand"]["angles"],
                               ev["face"]["weights"], n_frames=n)
    # the 53-track baseline is compared to the feedback output
    audit2 = qc.unchanged_audit(baseline, out["vmd"], PROFILE)
    assert audit2["ok"], audit2
    assert audit2["root_unchanged"]
    assert not out["residual"][:, FD.ROOT_SLICE].any()
    assert not out["residual"][:, FD.HAND_SLICE].any()
    assert not out["residual"][:, FD.FACE_SLICE].any()
    assert out["vmd"]["max_frame"] == n - 1


def test_fallback_window_detection():
    n = 120
    before = _track(n)
    # a single-frame jerk spike belongs to the 30..59 window
    after = _track(n, spike_at=45)
    bad = qc.failing_windows(before, after, n, height=20.0, window=30)
    assert any(w["start"] == 30 and "jerk" in w["reasons"] for w in bad), bad
    # a significance floor suppresses a quiet-window relative ratio
    assert not qc.failing_windows(before, after, n, height=20.0, window=30,
                                  jerk_floor=1e18)
    # new foot penetration is always reported
    pen = _track(n, penetration_at=70)
    bad = qc.failing_windows(before, pen, n, height=20.0, window=30)
    assert any(w["start"] == 60 and "penetration" in w["reasons"] for w in bad), bad


def test_resample_evidence_matches_adapter():
    n_src, n_out, slow = 968, 242, 0.25
    kp = np.arange(n_src, dtype=np.float32)[:, None].repeat(17, 1)[:, :, None]
    kp = np.tile(kp, (1, 1, 3))
    ev = {"kp2d": kp, "bbx_xys": np.zeros((n_src, 3), np.float32),
          "img_wh": (1080, 1920), "n_src": n_src,
          "hand": {"angles": np.zeros((n_src, 2, 19), np.float32),
                   "level": np.zeros((n_src, 2), np.int8),
                   "score": np.zeros((n_src, 2), np.float32),
                   "detected": np.zeros((n_src, 2), bool)},
          "face": {"weights": np.zeros((n_src, 12), np.float32),
                   "level": np.zeros(n_src, np.int8),
                   "detected": np.zeros(n_src, bool)}}
    out = FA.resample_evidence(ev, n_out, slow)
    idx = np.clip(np.round(np.arange(n_out) / slow).astype(int), 0, n_src - 1)
    assert out["kp2d"].shape == (n_out, 17, 3)
    assert np.array_equal(out["kp2d"][:, 0, 0], idx.astype(np.float32))
    assert len(out["hand"]["level"]) == n_out
    assert len(out["face"]["weights"]) == n_out


def test_gt_scores_improve_when_after_equals_gt():
    n = 40
    model = parse_pmx(CFG["si"]["pmx"])
    before_vmd = _body_vmd(n)
    gt_vmd = _body_vmd(n, chest_rot=12.0)
    after_vmd = _body_vmd(n, chest_rot=12.0)
    frames = list(range(n))
    bt = pose_track(model, before_vmd, PROFILE, frames=frames)
    gt = pose_track(model, gt_vmd, PROFILE, frames=frames)
    at = pose_track(model, after_vmd, PROFILE, frames=frames)
    rec = qc.gt_scores(gt, bt, at, PROFILE)
    assert rec["main_after"] < rec["main_before"], rec
    assert rec["checks"]["body_relative_main_improves"]
    sec = qc.gt_secondary(gt, bt, at, PROFILE)
    assert sec["available"]
    assert sec["checks"]["position_not_worse"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("OK", t.__name__)
    print(f"{len(tests)} feedback-apply tests passed")
