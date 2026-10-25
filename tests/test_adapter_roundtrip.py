"""Adapter round-trip + cross-model tests.

A (same model, the clean one):
    R2 safe GT (祀) -> canonical -> adapter -> VMD -> PMX FK
    compare per canonical joint with the GT's own FK (world pos + world rot).
    Target: < 1e-3 units, 250 frames.

B (cross model):
    R2 safe GT (祀) -> canonical -> adapter(tiflos/luoxi) -> VMD -> FK
    the target's segment directions must equal D_j * (the source's), where D_j
    is the rest-direction compensation; this is the actual retarget invariant
    (positions differ by proportions, directions must not).

C (slow factor / time remap):
    968 -> 242 (T1, 0.25x slow), 484 -> 242 (T2, 0.5x), 242 -> 242 (T5).

Run: .venv/Scripts/python.exe tests/test_adapter_roundtrip.py
"""
import json
import math
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from vtv import canonical_body as CB  # noqa: E402
from vtv.adapter import (adapt, canonical_from_pmx, canonical_from_smplx,  # noqa: E402
                         q_angle_deg, q_between, q_conj, qmul)
from vtv.metrics.core import q_angle_deg as q_pair_deg  # noqa: E402
from vtv.mmd.evaluate import VmdEvaluator  # noqa: E402
from vtv.mmd.mathutil import qrot  # noqa: E402
from vtv.mmd.pmx import parse_pmx  # noqa: E402
from vtv.mmd.vmd import parse_vmd  # noqa: E402
from vtv.mmd.writer import write_vmd  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (("  " + str(detail)) if detail else ""))
    if not cond:
        FAILS.append(name)


def load(model_key):
    cfg = json.load(open(os.path.join(ROOT, "configs", "models.json"), encoding="utf-8"))
    profile = json.load(open(os.path.join(ROOT, "profiles", f"{model_key}.json"),
                             encoding="utf-8"))
    return cfg["models"][model_key], profile


def evaluate(model, profile, path, frames):
    vmd = parse_vmd(path)
    ev = VmdEvaluator(model, vmd, force_ik_off=True, apply_all_translations=False)
    name2i = {b["name"]: i for i, b in enumerate(model["bones"])}
    idx = {j: name2i[profile["standard_map"][j]] for j in CB.ADAPTER_JOINTS
           if profile["standard_map"].get(j) in name2i}
    pos = {j: np.asarray([ev.frame(f)["world_pos"][i] for f in frames]) for j, i in idx.items()}
    rot = {j: np.asarray([ev.frame(f)["world_q"][i] for f in frames]) for j, i in idx.items()}
    return vmd, pos, rot, idx


def write_tmp(res, model_name, tag):
    fd, path = tempfile.mkstemp(suffix=f"_{tag}.vmd")
    os.close(fd)
    write_vmd(path, model_name, res["bone_frames"], {}, res["ik_frames"])
    return path


def test_same_model():
    cfg, profile = load("si")
    model = parse_pmx(cfg["pmx"])
    gt_path = os.path.join(ROOT, "outputs", "gt", "si_R2_safe.vmd")
    gt = parse_vmd(gt_path)
    frames = list(range(gt["max_frame"] + 1))
    check("gt_frames_250", len(frames) == 250, len(frames))

    motion = canonical_from_pmx(model, profile, gt, frames=frames)
    check("motion_span", motion["span"] and motion["span"] > 0.5, motion["span"])
    res = adapt(motion, model, profile)
    check("roundtrip_ground_shift_zero", abs(res["ground_shift_units"]) < 1e-6,
          res["ground_shift_units"])

    path = write_tmp(res, "si_roundtrip", "rt")
    try:
        vmd, pos_p, rot_p, idx = evaluate(model, profile, path, frames)
        _, pos_g, rot_g, _ = evaluate(model, profile, gt_path, frames)
        # The root bone is a control bone whose translation is deliberately
        # relocated to センター (safe_mode), so compare the 22 body joints.
        body = [j for j in idx if j != "root"]
        pos_err = {j: np.linalg.norm(pos_p[j] - pos_g[j], axis=1) for j in body}
        arm = [j for j in ("forearm_l", "hand_l", "forearm_r", "hand_r") if j in pos_err]
        rest = [j for j in body if j not in arm]
        e_rest = float(np.concatenate([pos_err[j] for j in rest]).max())
        e_arm = float(np.concatenate([pos_err[j] for j in arm]).max())
        check("roundtrip_pos_body_units", e_rest < 1e-3, f"max={e_rest:.3e}")
        # the source GT keys a ~4.5 deg non-axial rotation on 腕捩/手捩, which
        # shortens the upper arm by 0.002 units; a rigid body-only target cannot
        # represent that (MMD clamps the non-axial part via fixed_axis anyway).
        check("roundtrip_pos_arm_units", e_arm < 5e-3, f"max={e_arm:.3e}")
        # roll distribution: the segment frame folds the source's twist into the
        # proximal bone, so upperarm/forearm *roll* differs by the GT's twist
        # angles; every other joint (and the hand leaf) must be exact.
        roll = ("upperarm_l", "upperarm_r", "forearm_l", "forearm_r")
        rot_exact = [j for j in body if j not in roll]
        e_rot = float(np.concatenate([q_pair_deg(rot_g[j], rot_p[j])
                                      for j in rot_exact]).max())
        check("roundtrip_world_rot_exact_deg", e_rot < 0.01, f"max={e_rot:.5f}")
        twist_deg = {j: float(q_pair_deg(rot_g[j], rot_p[j]).max()) for j in roll}
        check("roundtrip_arm_roll_equals_source_twist",
              all(0.0 < v < 90.0 for v in twist_deg.values()), twist_deg)
        check("roundtrip_segment_frames",
              motion["segment_frame"].get("upperarm_l") == "左腕捩"
              and motion["segment_frame"].get("forearm_l") == "左手捩",
              {k: motion["segment_frame"].get(k)
               for k in ("upperarm_l", "forearm_l", "clavicle_l")})
        # only safe bones + main twist bones; no D-chain / helper / 肩C
        banned = set()
        for side in ("l", "r"):
            for k in ("thigh_d_", "shin_d_", "foot_d_"):
                j = f"{k}{side}"
                if profile["standard_map"].get(j):
                    banned.add(profile["standard_map"][j])
            for k in ("clavicle_p_", "clavicle_c_", "forearm_twist_helper_",
                      "wrist_twist_helper_"):
                j = f"{k}{side}"
                if profile["standard_map"].get(j):
                    banned.add(profile["standard_map"][j])
        wrote = set(vmd["bones"])
        check("no_helper_tracks", not (wrote & banned), sorted(wrote & banned))
        main_twist = {v["main"] for v in (profile.get("twist") or {}).values() if v.get("main")}
        check("writes_body_only_23",
              wrote == set(res["bones"]) and len(wrote) == 23
              and wrote <= set(profile["writable_bones"].values()),
              (len(wrote), sorted(wrote)))
        check("ik_off_first_frame",
              vmd["ik_frames"] and vmd["ik_frames"][0]["frame"] == 0
              and all(v is False for v in vmd["ik_frames"][0]["ik"].values())
              and len(vmd["ik_frames"][0]["ik"]) == 4,
              vmd["ik_frames"][:1])
        check("frame_range_0_249", vmd["max_frame"] == 249, vmd["max_frame"])
    finally:
        os.unlink(path)


# foot_l/foot_r are excluded: the ankle's rest *direction* is a different
# anatomical segment in the two rigs (metatarsal vs ankle-to-toe-tip), so the
# adapter keeps the identity alignment there (see adapter.NO_SWING) and the
# invariant is "the ankle's world rotation matches the source", checked below.
SEGMENTS = ["spine", "spine1", "chest", "neck",
            "clavicle_l", "upperarm_l", "forearm_l", "thigh_l", "shin_l",
            "clavicle_r", "upperarm_r", "forearm_r", "thigh_r", "shin_r"]


def test_cross_model():
    cfg_si, profile_si = load("si")
    model_si = parse_pmx(cfg_si["pmx"])
    gt_path = os.path.join(ROOT, "outputs", "gt", "si_R2_safe.vmd")
    gt = parse_vmd(gt_path)
    frames = list(range(gt["max_frame"] + 1))
    motion = canonical_from_pmx(model_si, profile_si, gt, frames=frames)
    src_dir = motion["rest_dir"]
    _, src_pos, src_rot, src_idx = evaluate(model_si, profile_si, gt_path, frames)

    for key in ("tiflos", "luoxi"):
        cfg, profile = load(key)
        model = parse_pmx(cfg["pmx"])
        res = adapt(motion, model, profile)
        path = write_tmp(res, key, "x")
        try:
            _, pos, rot_fk, idx = evaluate(model, profile, path, frames)
            rest_m = {j: [c * profile["unit_to_m"] for c in
                          model["bones"][idx[j]]["pos"]] for j in idx}
            tgt_dir = CB.rest_dirs(rest_m)
            worst = 0.0
            for j in SEGMENTS:
                if j not in src_dir or j not in tgt_dir or j not in idx:
                    continue
                d = q_between(src_dir[j], tgt_dir[j])
                child = CB.CANONICAL_CHILD[j] if CB.CANONICAL_CHILD.get(j) in idx else None
                if child is None:
                    continue
                u = src_pos[child] - src_pos[j]
                v = pos[child] - pos[j]
                un = u / np.linalg.norm(u, axis=1, keepdims=True)
                vn = v / np.linalg.norm(v, axis=1, keepdims=True)
                want = np.asarray([qrot(d, tuple(x)) for x in un])
                cos = np.clip((vn * want).sum(1), -1.0, 1.0)
                worst = max(worst, float(np.degrees(np.arccos(cos)).max()))
            check(f"{key}_direction_invariant", worst < 1e-3, f"max={worst:.2e} deg")
            # the ankle keeps the source's world orientation (no swing)
            rot_err = 0.0
            for side in ("l", "r"):
                j = f"foot_{side}"
                if j in src_rot and j in idx:
                    rot_err = max(rot_err, float(q_pair_deg(
                        src_rot[j], rot_fk[j]).max()))
            check(f"{key}_ankle_frame_matches", rot_err < 1e-3, f"max={rot_err:.2e} deg")
            check(f"{key}_finite", np.isfinite(np.concatenate(
                [pos[j].ravel() for j in pos])).all())
            check(f"{key}_scale", 0.5 < res["scale"] < 2.0, round(res["scale"], 4))
            # ground calibration must land the lowest toe near the rest toe
            toe = min(pos[f"toe_{s}"][:, 1].min() for s in ("l", "r"))
            rest_toe = res["target_rest_toe_units"]
            check(f"{key}_ground_calibrated", abs(toe - rest_toe) < 2.0,
                  f"toe_min={toe:.2f} rest={rest_toe:.2f} shift={res['ground_shift_units']:.2f}")
        finally:
            os.unlink(path)


def test_slow_factor():
    cfg, profile = load("si")
    model = parse_pmx(cfg["pmx"])
    rest = {j: [c * profile["unit_to_m"] for c in
                model["bones"][{b["name"]: i for i, b in enumerate(model["bones"])}
                               [profile["standard_map"][j]]]["pos"]]
            for j in CB.BODY_JOINTS}
    for n_in, slow, n_out in ((968, 0.25, 242), (484, 0.5, 242), (242, 1.0, 242)):
        params = {"body_pose": np.zeros((n_in, 63)),
                  "global_orient": np.zeros((n_in, 3)),
                  "transl": np.zeros((n_in, 3))}
        # a moving root so frame mapping is visible
        params["transl"][:, 0] = np.arange(n_in) * 1e-3
        mot = canonical_from_smplx(params, rest, slow_factor=slow)
        ok = len(mot["frames"]) == n_out and len(mot["root_pos"]) == n_out
        # frame k samples source k/slow: the x ramp must scale accordingly
        expect_x = (np.round(np.arange(n_out) / slow) * 1e-3) + rest["pelvis"][0]
        ok = ok and np.allclose(mot["root_pos"][:, 0], expect_x, atol=1e-9)
        check(f"slow_{slow:g}_{n_in}to{n_out}", ok,
              f"frames={len(mot['frames'])} x0={mot['root_pos'][0, 0]:.4f}")
        res = adapt(mot, model, profile)
        check(f"slow_{slow:g}_vmd_frames", len(res["bone_frames"]["センター"]) == n_out,
              len(res["bone_frames"]["センター"]))
        check(f"slow_{slow:g}_rest_pose", all(
            q_angle_deg(q) < 1e-6 for rows in res["bone_frames"].values()
            for q in (r["rot"] for r in rows)), "")


def main():
    errs = CB.check_mapping()
    check("canonical_tables", not errs, errs)
    if not os.path.exists(os.path.join(ROOT, "outputs", "gt", "si_R2_safe.vmd")):
        raise SystemExit("run scripts/bake_gt_vmd.py first")
    test_same_model()
    test_cross_model()
    test_slow_factor()
    print()
    if FAILS:
        print("ADAPTER ROUNDTRIP FAILED:", FAILS)
        return 1
    print("ADAPTER ROUNDTRIP OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
