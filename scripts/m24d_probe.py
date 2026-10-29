"""M2.4d VPoser v2 probe: rig conversion, prior direction, projection.

What this script answers, before any BA sweep:

1. Does the PMX -> SMPL-X body_pose conversion recover the GVHMR source pose?
   (It must, otherwise the prior is scoring the wrong skeleton.)
2. Is the prior informative on this clip?  VPoser KL of the prediction vs the
   GT, and group swaps (GT arms / torso / legs).
3. Which way does the prior gradient push the arms?  A prior-only optimization
   (no 2D) measures the direction and the body-relative cost.
4. Does the VPoser autoencoder projection produce a usable torso-relative
   target?  Decode(encode(x)) is evaluated as a candidate target.

Everything is GT-read-only scoring.  No network is trained; no GT enters the
solver.  Nothing here is a pipeline stage or a candidate unless the gates in
the M2.4d report pass.
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np

from vtv.adapter import canonical_from_pmx
from vtv.arm3d import body_relative_errors
from vtv.ba3d import (TORSO_ARM_CHAIN, merge_pose_tracks, optimize_multiview,
                      verify_ba, write_ba_vmd)
from vtv.cleanup import dense_tracks, eval_tracks
from vtv.mmd.pmx import parse_pmx
from vtv.mmd.vmd import parse_vmd
from vtv.mmd.writer import write_vmd
from vtv.vposer_body import VPoserBodyPrior, body_pose_from_motion
from vtv.vposer_prior import load_vposer_v2

from refine_ba3d import body_scores, temporal_steps, view_px


def pair_budget(pred_vmd, base_vmd, gt_vmd, model, profile, pair_tol=0.02):
    """Global + per-pair capsule budget (max(baseline, GT) + tol)."""
    from vtv.arm_prior import capsule_penetrations
    from vtv.arm3d import penetration as pen_vmd

    pb = pen_vmd(base_vmd, model, profile)
    pp = pen_vmd(pred_vmd, model, profile)
    pg = pen_vmd(gt_vmd, model, profile)

    def pmap(p):
        out = {}
        for g in ("arm_torso", "arm_arm"):
            for k, rec in (p[g].get("pairs_detail") or {}).items():
                out[(g, k)] = float(rec.get("max", 0.0))
        return out

    b, p, g = pmap(pb), pmap(pp), pmap(pg)
    failed = []
    for k, a in sorted(p.items()):
        allowed = max(b.get(k, 0.0), g.get(k, 0.0)) + pair_tol
        if a > allowed + 1e-9:
            failed.append({"pair": f"{k[0]} {k[1]}", "after": round(a, 4),
                           "budget": round(allowed, 4),
                           "baseline": round(b.get(k, 0.0), 4),
                           "gt": round(g.get(k, 0.0), 4)})
    return {
        "global": {
            "baseline": {"arm_torso": pb["arm_torso"]["max"],
                         "arm_arm": pb["arm_arm"]["max"]},
            "after": {"arm_torso": pp["arm_torso"]["max"],
                      "arm_arm": pp["arm_arm"]["max"]},
            "gt": {"arm_torso": pg["arm_torso"]["max"],
                   "arm_arm": pg["arm_arm"]["max"]},
            "ok": (pp["arm_torso"]["max"] <= pb["arm_torso"]["max"] + 1e-9
                   and pp["arm_arm"]["max"] <= pb["arm_arm"]["max"] + 0.02),
        },
        "pairs_budget": {"ok": not failed, "failed": failed[:12],
                         "tolerance": pair_tol},
    }


def metrics(vmd, base_vmd, gt_vmd, views, model, profile, opt_joints):
    scores = body_scores(vmd, gt_vmd, model, profile)
    return {
        "upper_fore_body_deg": scores["upper_fore_body_mean_deg"],
        "full_body_deg": scores["full_body_mean_deg"],
        "hand_body_deg": scores["hand_body_mean_deg"],
        "torso_body_relative": scores["torso_body_relative"],
        "chest_world_deg": scores["body"]["chest_world_mean_deg"],
        "position_3d": scores["position_3d"],
        "temporal": temporal_steps(vmd, model, profile),
        "penetration": pair_budget(vmd, base_vmd, gt_vmd, model, profile),
        "view_error": view_px(vmd, views, model, profile),
        "verify": verify_ba(base_vmd, vmd, model, profile, gt_vmd=gt_vmd,
                            opt_joints=opt_joints),
    }


def write_candidate(motion, res, base_vmd, model, profile, opt_joints, path):
    out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    tmp = path + ".full.vmd"
    write_ba_vmd(motion, res, model, profile, tmp)
    ba = parse_vmd(tmp)
    os.remove(tmp)
    merged, meta = merge_pose_tracks(base_vmd, ba, model, profile,
                                     joints=opt_joints)
    write_vmd(path, merged.get("model_name", ""), merged["bones"],
              merged.get("morphs", {}), merged.get("ik_frames", []))
    return parse_vmd(path), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "models.json"))
    ap.add_argument("--model", default="si")
    ap.add_argument("--vmd", default=os.path.join(
        ROOT, "outputs", "m2", "vmd", "front_clean_ik.vmd"))
    ap.add_argument("--gt", default=os.path.join(
        ROOT, "outputs", "gt", "si_R2_safe.vmd"))
    ap.add_argument("--source-npz", default=os.path.join(
        ROOT, "outputs", "gvhmr", "R2fix_front", "gvhmr_result.npz"))
    ap.add_argument("--view-meta", nargs="+", default=None)
    ap.add_argument("--view-kp", nargs="+", default=None)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs", "m2", "vposer"))
    ap.add_argument("--prior-iters", type=int, default=120)
    ap.add_argument("--weights", nargs="+", type=float,
                    default=[0.003, 0.01, 0.03, 0.1])
    ap.add_argument("--stage-yaw", type=float, default=180.0)
    ap.add_argument("--skip-prior-sweep", action="store_true")
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8"))["models"][args.model]
    profile = json.load(open(os.path.join(ROOT, "profiles", f"{args.model}.json"),
                             encoding="utf-8"))
    model = parse_pmx(cfg["pmx"])
    base_vmd = parse_vmd(args.vmd)
    gt_vmd = parse_vmd(args.gt)
    motion = canonical_from_pmx(model, profile, base_vmd)
    gt_motion = canonical_from_pmx(model, profile, gt_vmd)
    d = np.load(args.source_npz)
    betas = np.asarray(d["smpl_params_global/betas"][0])
    source_bp = np.asarray(d["smpl_params_global/body_pose"])
    vp = load_vposer_v2()
    prior = VPoserBodyPrior(motion, vposer=vp, betas=betas,
                            stage_yaw_deg=args.stage_yaw,
                            source_body_pose=source_bp)

    # ---- 1. rig conversion audit -------------------------------------
    bp_pred = prior.body_pose_np(motion).reshape(len(motion["frames"]), 63)
    bp_gt = body_pose_from_motion(gt_motion, prior.D, args.stage_yaw).reshape(
        len(gt_motion["frames"]), 63)
    n = min(len(bp_pred), len(source_bp))
    from scipy.spatial.transform import Rotation as Rot

    def geo_per_joint(a, b):
        qa = Rot.from_rotvec(a.reshape(-1, 3)).as_quat().reshape(-1, 21, 4)
        qb = Rot.from_rotvec(b.reshape(-1, 3)).as_quat().reshape(-1, 21, 4)
        return np.degrees(2.0 * np.arccos(np.clip(
            np.abs(np.sum(qa * qb, -1)), 0.0, 1.0)))

    rec = geo_per_joint(bp_pred[:n], source_bp[:n])
    audit = {
        "betas": [round(float(x), 4) for x in betas],
        "stage_yaw_deg": args.stage_yaw,
        "recover_vs_gvhmr_deg": {
            "arms": round(float(rec[:, [15, 16, 17, 18, 19, 20]].mean()), 3),
            "torso": round(float(rec[:, [2, 5, 8, 11]].mean()), 3),
            "all": round(float(rec.mean()), 3),
        },
        "D_deg": {j: round(float(np.degrees(2.0 * np.arccos(min(
            1.0, abs(float(prior.D[j][3])))))), 2)
            for j in ("spine1", "chest", "clavicle_l", "upperarm_l",
                      "forearm_l", "upperarm_r", "forearm_r")},
        "foot_D_identity": bool(prior.D["foot_l"][3] == 1.0),
    }

    # ---- 2. KL diagnostics -------------------------------------------
    import torch

    def kl_of(x):
        with torch.no_grad():
            return vp.kl(vp.encode(torch.as_tensor(x, dtype=torch.float32))).numpy()

    kp, kg = kl_of(bp_pred), kl_of(bp_gt)
    groups = {"legs": (0, 1, 3, 4, 6, 7, 9, 10),
              "torso": (2, 5, 8, 11),
              "arms": (12, 13, 15, 16, 17, 18, 19, 20),
              "head": (14,)}
    swaps = {}
    for name, idx in groups.items():
        mix = bp_pred.copy()
        for i in idx:
            mix[:, i * 3:i * 3 + 3] = bp_gt[:, i * 3:i * 3 + 3]
        swaps[name] = round(float(kl_of(mix).mean()), 3)
    kl = {
        "pred_mean": round(float(kp.mean()), 3),
        "pred_median": round(float(np.median(kp)), 3),
        "gt_mean": round(float(kg.mean()), 3),
        "gt_median": round(float(np.median(kg)), 3),
        "pred_below_gt_frac": round(float((kp < kg).mean()), 4),
        "pred_with_gt_group": swaps,
        "note": ("A prior that rates the GT less likely than the prediction "
                 "cannot point at the GT; on this clip it can only regularize."),
    }

    # ---- 3. projection (autoencoder target) --------------------------
    x_proj = prior.project_body_pose(bp_pred).reshape(-1, 21, 3)
    proj_delta = geo_per_joint(bp_pred, x_proj.reshape(len(bp_pred), 63))
    arm_idx = [15, 16, 17, 18, 19, 20]
    torso_idx = [2, 5, 8, 11]
    projections = {}
    views = {}
    if args.view_meta and args.view_kp:
        from vtv.ba3d import load_view

        for m, k in zip(args.view_meta, args.view_kp):
            name, mp = m.split("=", 1)
            name2, kp_path = k.split("=", 1)
            views[name] = load_view(os.path.abspath(mp), os.path.abspath(kp_path))
    for tag, joints in (("full", None), ("arms", "arms"), ("torso", "torso")):
        mix = bp_pred.copy()
        if joints == "arms":
            for i in arm_idx:
                mix[:, i * 3:i * 3 + 3] = x_proj.reshape(len(mix), 63)[:, i * 3:i * 3 + 3]
        elif joints == "torso":
            for i in torso_idx:
                mix[:, i * 3:i * 3 + 3] = x_proj.reshape(len(mix), 63)[:, i * 3:i * 3 + 3]
        else:
            mix = x_proj.reshape(len(bp_pred), 63)
        G = prior.reconstructed_world(mix, motion)
        res = {"G": G, "root_pos": np.asarray(motion["root_pos"]),
               "root_rot": np.zeros((len(mix), len(views), 3)),
               "root_trans": np.zeros((len(mix), len(views), 3)),
               "hist": [], "view_error": {}, "names": list(views),
               "opt_joints": list(TORSO_ARM_CHAIN)}
        out = os.path.join(args.out_dir, f"proj_{tag}.vmd")
        vmd, meta = write_candidate(motion, res, base_vmd, model, profile,
                                    TORSO_ARM_CHAIN, out)
        m = metrics(vmd, base_vmd, gt_vmd, views, model, profile,
                    TORSO_ARM_CHAIN)
        projections[tag] = {
            "out": os.path.abspath(out),
            "merge": meta,
            "projection_delta_arm_deg": round(float(proj_delta[:, arm_idx].mean()), 3),
            "projection_delta_torso_deg": round(float(proj_delta[:, torso_idx].mean()), 3),
            "metrics": m,
        }

    # ---- 4. prior-only optimization ----------------------------------
    prior_only = {}
    if not args.skip_prior_sweep:
        if not views:
            raise SystemExit("--view-meta/--view-kp are required for the prior sweep")
        for w in args.weights:
            res = optimize_multiview(
                motion, views, opt_joints=TORSO_ARM_CHAIN,
                iters=args.prior_iters, lr=0.03, w_2d=0.0,
                prior_temporal=1.5, prior_collision=0.0,
                prior_vposer=float(w), vposer=prior, vposer_mode="kl",
                freeze_shared_translation=True, verbose=False)
            out = os.path.join(args.out_dir, f"prior_w{w:g}.vmd")
            vmd, meta = write_candidate(motion, res, base_vmd, model, profile,
                                        TORSO_ARM_CHAIN, out)
            m = metrics(vmd, base_vmd, gt_vmd, views, model, profile,
                        TORSO_ARM_CHAIN)
            prior_only[str(w)] = {"out": os.path.abspath(out),
                                  "vposer": res.get("vposer"),
                                  "metrics": m}

    report = {
        "mode": "M2.4d VPoser v2 probe (rig audit + prior direction + projection)",
        "init": os.path.abspath(args.vmd),
        "gt": os.path.abspath(args.gt),
        "source_npz": os.path.abspath(args.source_npz),
        "rig_audit": audit,
        "kl": kl,
        "projection": projections,
        "prior_only": prior_only,
        "baseline": {
            "upper_fore_body_deg": body_scores(base_vmd, gt_vmd, model, profile)[
                "upper_fore_body_mean_deg"],
            "full_body_deg": body_scores(base_vmd, gt_vmd, model, profile)[
                "full_body_mean_deg"],
            "hand_body_deg": body_scores(base_vmd, gt_vmd, model, profile)[
                "hand_body_mean_deg"],
            "torso_body_relative": body_scores(base_vmd, gt_vmd, model, profile)[
                "torso_body_relative"],
            "chest_world_deg": body_scores(base_vmd, gt_vmd, model, profile)[
                "body"]["chest_world_mean_deg"],
            "temporal": temporal_steps(base_vmd, model, profile),
        },
        "note": ("GT is read-only scoring.  Projection/prior-only outputs are "
                 "diagnostics, not pipeline candidates.  No network is trained."),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "probe.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1)[:6000])
    print("WROTE", path)


if __name__ == "__main__":
    main()
