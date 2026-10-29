"""Unit tests for src/vtv/feedback_data.py (M3.5).

Run:  python tests/test_feedback_data.py
"""
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from vtv import feedback_data as FD  # noqa: E402
from vtv.mmd.mathutil import qmul, qnorm  # noqa: E402
from vtv.mmd.pmx import parse_pmx  # noqa: E402
from vtv.mmd.vmd import parse_vmd  # noqa: E402
from vtv.metrics.core import pose_track  # noqa: E402

PROFILES = {k: json.load(open(os.path.join(ROOT, "profiles", f"{k}.json"),
                              encoding="utf-8"))
            for k in ("si", "luoxi", "tiflos")}


def _models():
    cfg = json.load(open(os.path.join(ROOT, "configs", "models.json"),
                         encoding="utf-8"))["models"]
    return {k: parse_pmx(cfg[k]["pmx"]) for k in PROFILES}


def test_layout():
    lim = FD.residual_limits()
    assert lim.shape == (FD.RESIDUAL_DIM,)
    assert (lim > 0).all()
    groups = FD.residual_groups()
    covered = np.zeros(FD.RESIDUAL_DIM, int)
    for sl in groups.values():
        covered[sl] += 1
    assert (covered == 1).all(), covered
    assert FD.ROOT_SLICE.stop == 2
    assert FD.HAND_SLICE.stop - FD.HAND_SLICE.start == 38
    assert FD.FACE_SLICE.stop - FD.FACE_SLICE.start == 12


def test_splits():
    counts = {"train": 0, "val": 0, "test": 0}
    recs = []
    for m in ("R1", "R2", "R3"):
        for mo in ("si", "luoxi", "tiflos"):
            for v in ("front", "three_quarter", "side"):
                s = FD.assign_split(m, mo, v)
                counts[s] += 1
                recs.append({"clip": FD.clip_id(m, mo, v), "motion": m,
                             "model": mo, "view": v, "split": s})
    assert counts == {"train": 8, "val": 10, "test": 9}, counts
    rep = FD.leakage_report(recs)
    assert rep["ok"], rep
    assert "tiflos" not in rep["models"]["train"]
    assert "side" not in rep["views"]["train"]
    assert "R3" not in rep["motions"]["train"]


def _mini_vmd():
    return {"model_name": "", "max_frame": 19,
            "ik_frames": [],
            "bones": {},
            "morphs": {
                "まばたき": [(0, 0.0), (10, 1.0), (19, 0.0)],
                "ワ": [(0, 0.0), (5, 0.4), (19, 0.0)],
                "にやり２": [(0, 0.0), (9, 0.6), (19, 0.0)],
                "存在しない": [(0, 1.0), (19, 1.0)],
            }}


def test_morph_mapping():
    names = {"blink": "まばたき", "mouth_a": "あ", "smile": "笑い"}
    w, src = FD.gt_semantic_weights(_mini_vmd(), 20, names)
    assert w.shape == (20, 12)
    i = FD.SEMANTIC.index("blink")
    assert abs(w[10, i] - 1.0) < 1e-6
    assert abs(w[0, i]) < 1e-6
    i = FD.SEMANTIC.index("mouth_a")
    assert abs(w[5, i] - 0.4) < 1e-6          # ワ -> mouth_a
    i = FD.SEMANTIC.index("smile")
    assert abs(w[9, i] - 0.6 * 0.8) < 1e-6    # にやり２ -> smile at 0.8
    assert "blink" in src and "mouth_a" in src
    assert "surprise" not in src


def test_finger_remap_identity():
    """si -> si must reproduce the GT hand *pose* (twist is allowed to differ).

    Local finger quats carry an arbitrary twist around the phalanx; what the
    remap guarantees is every bone direction, joint position and the 19
    anatomical angles.
    """
    prof = PROFILES["si"]
    model = _models()["si"]
    vmd = parse_vmd(os.path.join(ROOT, "outputs", "gt", "si_R2_safe.vmd"))
    frames = [0, 30, 90]
    tracks = FD.remap_fingers(model, prof, vmd, model, prof, frames=frames)
    out = {"model_name": vmd.get("model_name", ""), "max_frame": vmd["max_frame"],
           "ik_frames": vmd.get("ik_frames", []), "morphs": {},
           "bones": {k: [dict(x) for x in v] for k, v in vmd["bones"].items()}}
    out["bones"].update(tracks)
    a = pose_track(model, vmd, prof, frames=frames)
    b = pose_track(model, out, prof, frames=frames)
    for j in a["finger_joints"]:
        d = np.abs(np.asarray(a["world"][j]) - np.asarray(b["world"][j])).max()
        assert d < 1e-5, (j, d)
    ga = FD.gt_hand_angles(model, prof, vmd, frames)
    gb = FD.gt_hand_angles(model, prof, out, frames)
    assert np.abs(ga - gb).max() < 1e-3


def test_strip_twist_local_tracks():
    """Twist helpers are removed and child local tracks are untouched.

    The helper carries the wrist offset in this PMD rig, so world positions do
    move; the guarantee used by the diagnostic metric is only that every child
    *local* rotation (what the teacher stores) is unchanged.
    """
    prof = PROFILES["si"]
    model = _models()["si"]
    vmd = parse_vmd(os.path.join(ROOT, "outputs", "gt", "si_R2_safe.vmd"))
    strip = FD.strip_twist_vmd(vmd, prof)
    smap = prof["standard_map"]
    for key in FD.TWIST_STD:
        assert smap[key] not in strip["bones"]
    frames = [0, 40, 120]
    a = pose_track(model, vmd, prof, frames=frames)
    b = pose_track(model, strip, prof, frames=frames)
    for bone in FD.BODY_FACTOR_NAMES:
        assert np.abs(np.asarray(a["local_q"][bone])
                      - np.asarray(b["local_q"][bone])).max() < 1e-12, bone


def test_residual_roundtrip():
    prof = PROFILES["si"]
    model = _models()["si"]
    gt = parse_vmd(os.path.join(ROOT, "outputs", "gt", "si_R2_safe.vmd"))
    frames = list(range(0, 120))
    gt_t = pose_track(model, FD.strip_twist_vmd(gt, prof), prof, frames=frames)
    # perturb local rotations on every factor bone and the root XZ
    # the real prediction has no arm twist tracks; strip them from the
    # perturbed copy so the target is representable
    pred = FD.strip_twist_vmd(gt, prof)
    smap = prof["standard_map"]
    for bone in FD.BODY_FACTOR_NAMES:
        name = smap[bone]
        if name not in pred["bones"]:
            continue
        for k in pred["bones"][name]:
            d = (math.radians(3.0), math.radians(-2.0), math.radians(1.5), 1.0)
            q = qnorm(tuple(np.array(d) / np.linalg.norm(d)))
            k["rot"] = qmul(q, k["rot"])
    root = smap["root"]
    for k in pred["bones"][root]:
        ramp = 0.002 * int(k["frame"])
        k["pos"] = (k["pos"][0] + ramp * 0.5, k["pos"][1],
                    k["pos"][2] - ramp)
    pred_t = pose_track(model, pred, prof, frames=frames)
    res = FD.extract_body_residual(gt_t, pred_t)
    full = FD.combine_residuals(res["target"], np.zeros((len(frames), 38), np.float32),
                                np.zeros((len(frames), 12), np.float32))
    corr = FD.apply_body_residual(pred, model, prof, full, frames=frames)
    ct = pose_track(model, corr, prof, frames=frames)
    for bone in FD.BODY_FACTOR_NAMES:
        a = np.asarray(gt_t["local_q"][bone])
        b = np.asarray(ct["local_q"][bone])
        dot = np.abs(np.sum(a * b, axis=1))
        err = np.degrees(2 * np.arccos(np.clip(dot, -1, 1)))
        assert err.max() < 1e-3, (bone, err.max())
    before = FD.body_relative_errors(gt_t, pred_t)["body_relative_main_deg"]
    after = FD.body_relative_errors(gt_t, ct)["body_relative_main_deg"]
    assert before > 1.0
    # chest-relative arm/forearm are exact; spine_pelvis keeps the uncorrected
    # pelvis sibling rotation, so the group mean is allowed a small floor
    assert after < 1.5, (before, after)
    g = np.asarray(gt_t["world"]["root"])[:, [0, 2]]
    p = np.asarray(ct["world"]["root"])[:, [0, 2]]
    root_err = np.linalg.norm((g - g[0]) - (p - p[0]), axis=1).max()
    assert root_err < 2e-3, root_err


def test_hand_face_masks():
    n = 8
    gt = np.full((n, 2, 19), 40.0, np.float32)
    pred = np.zeros_like(gt)
    level = np.array([[2, 2], [1, 2], [2, 0], [3, 3], [0, 0], [2, 2], [2, 2], [2, 2]])
    r = FD.extract_hand_residual(gt, pred, level)
    assert r["mask"].shape == (n, 38)
    for f in range(n):
        s0 = r["mask"][f, :19].mean()
        s1 = r["mask"][f, 19:].mean()
        assert s0 == (1.0 if level[f, 0] <= 1 else 0.0)
        assert s1 == (1.0 if level[f, 1] <= 1 else 0.0)
    assert (np.abs(r["target"].reshape(n, 2, -1))
            <= FD.HAND_BOUND_VEC.reshape(2, -1)[None]).all()

    gtf = np.full((n, 12), 1.0, np.float32)
    pf = np.zeros_like(gtf)
    fl = np.array([0, 1, 2, 3, 0, 1, 2, 3], np.int8)
    ch = np.zeros(12, np.float32); ch[0] = 1.0
    fr = FD.extract_face_residual(gtf, pf, fl, ch)
    assert fr["mask"].shape == (n, 12)
    assert (fr["mask"][:, 0] == (fl <= 1)).all()
    assert (fr["mask"][:, 1:] == 0).all()
    assert (np.abs(fr["target"]) <= FD.FACE_BOUND).all()
    # at L1 only blink is supervised; at L0 every existing channel is
    allch = np.ones(12, np.float32)
    m = FD.face_channel_mask(np.array([0, 1, 2, 3]), allch)
    assert m[0].sum() == 12
    assert m[1].sum() == 1 and m[1, 0] == 1
    assert m[2].sum() == 0 and m[3].sum() == 0


def test_features_evidence_only():
    prof = PROFILES["si"]
    model = _models()["si"]
    vmd = parse_vmd(os.path.join(ROOT, "outputs", "gt", "si_R2_safe.vmd"))
    frames = list(range(0, 60))
    track = pose_track(model, vmd, prof, frames=frames)
    n = len(frames)
    kp = np.random.default_rng(1).uniform(0, 1, (n, 17, 3)).astype(np.float32)
    bbx = np.tile(np.array([[0.5, 0.5, 0.4]], np.float32), (n, 1))
    hand = np.zeros((n, 2, 19), np.float32)
    level = np.full((n, 2), 2, np.int8)
    score = np.zeros((n, 2), np.float32)
    face = np.zeros((n, 12), np.float32)
    fl = np.full(n, 2, np.int8)
    feats = FD.build_features(track, prof["height"], kp, bbx, hand, level, score,
                              np.zeros((n, 2)), face, fl, np.zeros(n), prof,
                              1.0, (1080, 1920))
    names = FD.feature_names(prof)
    assert feats.shape == (n, len(names)), (feats.shape, len(names))
    assert np.isfinite(feats).all()
    feats2 = FD.build_features(track, prof["height"], kp, bbx, hand, level, score,
                               np.zeros((n, 2)), face, fl, np.zeros(n), prof,
                               1.0, (1080, 1920))
    assert np.abs(feats - feats2).max() == 0.0
    import inspect
    params = set(inspect.signature(FD.build_features).parameters)
    assert not any("gt" in p.lower() for p in params), params


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"test_feedback_data: OK ({len(tests)} tests)")
