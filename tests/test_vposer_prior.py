# -*- coding: utf-8 -*-
"""M2.4d tests: PMX <-> SMPL-X body_pose rig mapping + VPoser prior wiring.

Run:
  PYTHONPATH=src python tests/test_vposer_prior.py

The VPoser checkpoint is optional; the rig-mapping tests run without it and
the prior/gradient tests are skipped when ``models/vposer`` is absent.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from vtv import canonical_body as CB
from vtv.adapter import adapt, canonical_from_pmx, canonical_from_smplx
from vtv.mmd.pmx import parse_pmx
from vtv.mmd.vmd import parse_vmd
from vtv.mmd.writer import write_vmd
from vtv.vposer_body import (VPoserBodyPrior, body_pose_from_motion,
                             canonical_from_body_pose, canonical_smplx_world,
                             canonical_to_body_pose_t, frame_compensation,
                             smplx_rest_dirs)

CKPT = os.path.join(ROOT, "models", "vposer", "V02_05", "snapshots",
                    "V02_05_epoch=13_val_loss=0.03.ckpt")


def _load_model():
    cfg = json.load(open(os.path.join(ROOT, "configs", "models.json"),
                         encoding="utf-8"))["models"]["si"]
    prof = json.load(open(os.path.join(ROOT, "profiles", "si.json"),
                          encoding="utf-8"))
    return parse_pmx(cfg["pmx"]), prof


def _rest_smplx():
    import smplx
    import torch

    bm = smplx.create(os.path.join(ROOT, "body_models"), model_type="smplx",
                      gender="NEUTRAL", use_pca=False, num_betas=10)
    with torch.no_grad():
        out = bm(betas=torch.zeros(1, 10), global_orient=torch.zeros(1, 3),
                 body_pose=torch.zeros(1, 63), transl=torch.zeros(1, 3))
    J = out.joints[0].numpy()
    return {j: J[i] for j, i in CB.SMPLX_INDEX.items()}


def _random_body_pose(F=6, scale=0.15, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, scale, (F, 21, 3))


def test_roundtrip_smplx_to_pmx_to_body_pose():
    """body_pose -> canonical -> adapt -> VMD -> canonical -> body_pose.

    The adapter's retargeting is per-joint ``D`` conjugation, so this is the
    test that the inverse mapping is on the correct side.  Feet are excluded:
    ``adapter.NO_SWING`` deliberately keeps the identity for them, so a
    foot body_pose recovered from the PMX is not the SMPL-X source value.
    """
    model, prof = _load_model()
    F = 6
    bp = _random_body_pose(F)
    rest = _rest_smplx()
    params = {"body_pose": bp.reshape(F, 63),
              "global_orient": np.zeros((F, 3)),
              "transl": np.zeros((F, 3))}
    motion = canonical_from_smplx(params, rest)
    ad = adapt(motion, model, prof, ground_calibrate=True, stage_yaw_deg=180.0)
    path = os.path.join(ROOT, "outputs", "tmp_test_vposer.vmd")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_vmd(path, model.get("name", ""), ad["bone_frames"], {},
              ad["ik_frames"])
    try:
        read = canonical_from_pmx(model, prof, parse_vmd(path))
    finally:
        if os.path.exists(path):
            os.remove(path)
    rd = smplx_rest_dirs(betas=np.zeros(10))
    D = frame_compensation(read, rd)
    rec = body_pose_from_motion(read, D, 180.0).reshape(F, 21, 3)
    from scipy.spatial.transform import Rotation as Rot

    qa = Rot.from_rotvec(bp.reshape(-1, 3)).as_quat()
    qb = Rot.from_rotvec(rec.reshape(-1, 3)).as_quat()
    geo = np.degrees(2.0 * np.arccos(np.clip(
        np.abs(np.sum(qa * qb, -1)), 0.0, 1.0))).reshape(F, 21)
    skip = [CB.BODY_POSE_INDEX[j] for j in ("foot_l", "foot_r", "toe_l", "toe_r")]
    keep = [i for i in range(21) if i not in skip]
    err = float(geo[:, keep].max())
    print(f"  body_pose roundtrip max error (torso+arms+legs, no feet): "
          f"{err:.6f} deg")
    assert err < 1e-4, f"body_pose roundtrip error {err} deg"


def test_torch_matches_numpy():
    """The differentiable conversion must agree with the numpy conversion."""
    model, prof = _load_model()
    base_vmd = parse_vmd(os.path.join(ROOT, "outputs", "m2", "vmd",
                                      "front_clean_ik.vmd"))
    motion = canonical_from_pmx(model, prof, base_vmd, frames=list(range(5)))
    rd = smplx_rest_dirs(betas=np.zeros(10))
    D = frame_compensation(motion, rd)
    bp_np = body_pose_from_motion(motion, D)
    import torch

    G = {j: torch.as_tensor(np.asarray(motion["world_rot"][j][:5]),
                            dtype=torch.float64)
         for j in motion["world_rot"]}
    bp_t = canonical_to_body_pose_t(G, D).detach().numpy()
    diff = np.abs(bp_np[:5] - bp_t).max()
    print(f"  torch/numpy body_pose max abs diff: {diff:.3e}")
    assert diff < 1e-9


def test_reconstruct_world_roundtrip():
    """body_pose -> PMX world -> body_pose with the pelvis fixed."""
    model, prof = _load_model()
    base_vmd = parse_vmd(os.path.join(ROOT, "outputs", "m2", "vmd",
                                      "front_clean_ik.vmd"))
    motion = canonical_from_pmx(model, prof, base_vmd, frames=list(range(5)))
    rd = smplx_rest_dirs(betas=np.zeros(10))
    D = frame_compensation(motion, rd)
    bp = body_pose_from_motion(motion, D)
    G = canonical_from_body_pose(bp, np.asarray(motion["world_rot"]["pelvis"][:5]),
                                 D)
    for j in CB.BODY_JOINTS:
        a = np.asarray(G[j])
        b = np.asarray(motion["world_rot"][j][:5])
        d = np.degrees(2.0 * np.arccos(np.clip(np.abs((a * b).sum(-1)), 0, 1)))
        assert float(d.max()) < 1e-5, f"{j}: {d.max()} deg"
    print("  body_pose -> PMX world roundtrip OK")


def test_vposer_energy_and_gradient():
    if not os.path.exists(CKPT):
        print("  SKIP: VPoser checkpoint not present")
        return
    model, prof = _load_model()
    base_vmd = parse_vmd(os.path.join(ROOT, "outputs", "m2", "vmd",
                                      "front_clean_ik.vmd"))
    motion = canonical_from_pmx(model, prof, base_vmd, frames=list(range(6)))
    rd = smplx_rest_dirs(betas=np.zeros(10))
    D = frame_compensation(motion, rd)
    prior = VPoserBodyPrior(motion, vposer=None, betas=np.zeros(10),
                            source_body_pose=None)
    import torch

    G = {j: torch.tensor(np.asarray(motion["world_rot"][j][:6]),
                         dtype=torch.float64, requires_grad=True)
         for j in motion["world_rot"]}
    joints = ("spine", "spine1", "chest", "neck", "clavicle_l", "clavicle_r",
              "upperarm_l", "upperarm_r", "forearm_l", "forearm_r")
    base, mask = prior.freeze_context(joints)
    e = prior.energy(G, base=base, mask=mask)
    assert np.isfinite(float(e)), "VPoser energy is not finite"
    e.backward()
    grads = [float(G[j].grad.abs().max()) for j in joints
             if G[j].grad is not None]
    assert grads and max(grads) > 0.0, "no gradient reached the pose"
    # frozen torso joints must not receive gradient when only the arms are free
    base2, mask2 = prior.freeze_context(("upperarm_l", "upperarm_r",
                                         "forearm_l", "forearm_r"))
    G2 = {j: torch.tensor(np.asarray(motion["world_rot"][j][:6]),
                          dtype=torch.float64, requires_grad=True)
          for j in motion["world_rot"]}
    prior.energy(G2, base=base2, mask=mask2).backward()
    assert G2["chest"].grad is None or float(G2["chest"].grad.abs().max()) == 0.0
    print(f"  VPoser energy {float(e):.3f}, max |grad| {max(grads):.3f}, "
          f"frozen chest grad 0")


def main():
    print("test_vposer_prior:")
    test_torch_matches_numpy()
    test_reconstruct_world_roundtrip()
    test_roundtrip_smplx_to_pmx_to_body_pose()
    test_vposer_energy_and_gradient()
    print("test_vposer_prior: OK")


if __name__ == "__main__":
    main()
