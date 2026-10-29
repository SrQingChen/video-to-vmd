"""M2.4c multi-view target-PMX body bundle adjustment (multi-view upper bound).

Example (three calibrated synthetic cameras; GT is scoring only):

  python scripts/refine_ba3d.py --model si \
      --vmd outputs/m2/vmd/front_clean_ik.vmd \
      --view-meta front=outputs/synth/si_R2_fixed/front/meta.json \
                  three_quarter=outputs/synth/si_R2_fixed/three_quarter/meta.json \
                  side=outputs/synth/si_R2_fixed/side/meta.json \
      --view-kp front=outputs/gvhmr/R2fix_front/preprocess/vitpose.pt \
                three_quarter=outputs/gvhmr/R2fix_three_quarter/preprocess/vitpose.pt \
                side=outputs/gvhmr/R2fix_side/preprocess/vitpose.pt \
      --gt outputs/gt/si_R2_safe.vmd \
      --out outputs/m2/ba3d/front_ba3d.vmd \
      --report outputs/m2/ba3d/front_ba3d.json

The solver estimates one shared body pose plus a small per-view root correction
from the 2D joint positions.  The written VMD uses only the shared pose and
copies **only the arm chain** (clavicle/upperarm/forearm/hand) onto the init
VMD; root, torso, legs and IK are byte-identical.  ``--gt`` is read-only
scoring, never part of the loss.

This is not a pipeline stage: the default pipeline remains ``cleanup+ik``.
The three-view BA is a multi-view upper bound; T5 (single view) must not depend
on it.
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

from vtv.adapter import canonical_from_pmx, fk_positions
from vtv.arm3d import body_relative_errors, penetration
from vtv.arm_prior import compare_arm
from vtv import canonical_body as CB
from vtv.ba3d import (OPT_PRESETS, TORSO_ARM_CHAIN, joint_observations,
                      load_view, merge_arm_tracks, optimize_multiview,
                      verify_ba, write_ba_vmd, VIEW_JOINTS)

# Which joints the VPoser prior sees as *free*.  The rest are held at the
# source body_pose, so the prior measures only the torso/arm change this stage
# makes.  ``opt`` means "the BA's optimized set".
VJOINT_PRESETS = {
    "opt": None,
    "torso": ("spine", "spine1", "chest", "neck"),
    "arms": ("clavicle_l", "clavicle_r", "upperarm_l", "upperarm_r",
             "forearm_l", "forearm_r"),
    "upperarm_forearm": ("upperarm_l", "upperarm_r",
                         "forearm_l", "forearm_r"),
    "torso_arms": TORSO_ARM_CHAIN,
    "all": tuple(CB.BODY_JOINTS),
}
from vtv.cleanup import dense_tracks, eval_tracks
from vtv.mmd.pmx import parse_pmx
from vtv.mmd.vmd import parse_vmd
from vtv.mmd.writer import write_vmd


def load_cfg(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["models"]


def parse_view_args(items):
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"expected view=path, got {item!r}")
        name, path = item.split("=", 1)
        out[name] = os.path.abspath(path)
    return out


def _rel_rot_deg(qg, qp, pg, pp):
    """Geodesic of the joint rotation expressed in its parent's frame."""
    from vtv.adapter import q_conj, qmul

    vals = []
    for f in range(min(len(qg), len(qp))):
        ag = qmul(q_conj(tuple(pg[f])), tuple(qg[f]))
        ap = qmul(q_conj(tuple(pp[f])), tuple(qp[f]))
        d = abs(float(np.dot(ag, ap)))
        vals.append(np.degrees(2.0 * np.arccos(min(1.0, d))))
    return round(float(np.mean(vals)), 3) if vals else None


def torso_body_relative(trg, trp):
    """Torso joints in the pelvis frame (chest body-relative, GT read-only)."""
    out = {}
    for j in ("spine", "spine1", "chest", "neck"):
        if j in trg["world_q"] and j in trp["world_q"]:
            out[j] = _rel_rot_deg(trg["world_q"][j], trp["world_q"][j],
                                  trg["world_q"]["pelvis"],
                                  trp["world_q"]["pelvis"])
    vals = [v for v in out.values() if v is not None]
    out["mean_deg"] = round(float(np.mean(vals)), 3) if vals else None
    return out


def body_scores(vmd, gt_vmd, model, profile):
    fg, tg = dense_tracks(gt_vmd)
    fp, tp = dense_tracks(vmd)
    trg = eval_tracks(model, profile, tg)
    trp = eval_tracks(model, profile, tp)
    br = body_relative_errors(trg, trp, model, profile)
    upf = float(np.mean([br["sides"][s][b] for s in ("l", "r")
                         for b in ("upperarm", "forearm")]))
    full = float(br["arm_body_mean_deg"])
    hand = float(np.mean([br["sides"][s]["hand"] for s in ("l", "r")]))
    # 3D elbow/wrist error in % of the PMX bounding-box height, the same
    # normalisation as closed_loop.position_pct_height.
    ys = [b["pos"][1] for b in model["bones"]]
    height = max(ys) - min(ys)
    pos = {}
    for j in ("forearm_l", "forearm_r", "hand_l", "hand_r",
              "upperarm_l", "upperarm_r", "chest"):
        d = np.linalg.norm(np.asarray(trp["world"][j]) - np.asarray(trg["world"][j]),
                           axis=1) / height * 100.0
        pos[j] = {"mean_pct_height": round(float(d.mean()), 3),
                  "p99_pct_height": round(float(np.percentile(d, 99)), 3)}
    world = compare_arm(trg, trp, model, profile)
    return {"body": br, "upper_fore_body_mean_deg": round(upf, 3),
            "hand_body_mean_deg": round(hand, 3),
            "full_body_mean_deg": round(full, 3),
            "torso_body_relative": torso_body_relative(trg, trp),
            # world/geodesic/swing/roll are **reported only**; the M2.4 gates
            # are body-relative + 3D position + penetration + temporal.
            "world_diag": {
                "arm_world_mean_deg": world.get("arm_world_mean_deg"),
                "arm_swing_mean_deg": world.get("arm_swing_mean_deg"),
                "arm_roll_mean_deg": world.get("arm_roll_mean_deg"),
                "sides": world.get("sides"),
            },
            "position_3d": pos}


def temporal_steps(vmd, model, profile):
    fp, tp = dense_tracks(vmd)
    tr = eval_tracks(model, profile, tp)
    vals = []
    for side in ("l", "r"):
        for a, b in (("upperarm", "forearm"), ("forearm", "hand")):
            j, c = f"{a}_{side}", f"{b}_{side}"
            d = np.asarray(tr["world"][c]) - tr["world"][j]
            d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
            vals.append(np.degrees(np.arccos(np.clip(
                (d[1:] * d[:-1]).sum(1), -1, 1.0))))
    s = np.concatenate(vals)
    return {"max_deg": round(float(s.max()), 3),
            "p99_deg": round(float(np.percentile(s, 99)), 3)}


def view_px(vmd, views, model, profile):
    """2D median/mean/p90 of the written VMD in each view (pred, not the fit)."""
    motion = canonical_from_pmx(model, profile, vmd)
    J = fk_positions(motion["world_rot"], motion["rest_joints"],
                     motion["root_pos"])
    names = [j for j in VIEW_JOINTS if j in J]
    out = {}
    for v, view in views.items():
        F = min(len(J[names[0]]), len(view["kp"]))
        P = np.stack([np.asarray(J[j][:F]) for j in names], axis=1)
        o, w = joint_observations(view, F)
        O = np.stack([o[j].cpu().numpy() for j in names], axis=1)
        W = np.stack([w[j].cpu().numpy() for j in names], axis=1)
        R, C = view["R"], view["C"]
        Pc = np.einsum("ji,fkj->fki", R, P - C)
        uv = np.stack([view["fx"] * Pc[..., 0] / Pc[..., 2] + view["cx"],
                       view["fy"] * Pc[..., 1] / Pc[..., 2] + view["cy"]],
                      axis=-1)
        d = np.linalg.norm(uv - O, axis=-1)
        mask = W > 0
        out[v] = {
            "median_px": round(float(np.median(d[mask])), 2),
            "mean_px": round(float(np.mean(d[mask])), 2),
            "p90_px": round(float(np.percentile(d[mask], 90)), 2)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "models.json"))
    ap.add_argument("--model", default="si")
    ap.add_argument("--vmd", required=True)
    ap.add_argument("--view-meta", nargs="+", required=True)
    ap.add_argument("--view-kp", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--full-out", default=None,
                    help="also write the full BA VMD (diagnostic; root/torso "
                         "come from adapt and are not the safe merge)")
    ap.add_argument("--gt", default=None)
    ap.add_argument("--opt-joints", choices=tuple(OPT_PRESETS), default="arms")
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--temporal", type=float, default=6.0)
    ap.add_argument("--collision", type=float, default=20.0,
                    help="baseline-relative capsule proxy weight; 0 disables")
    ap.add_argument("--collision-margin", type=float, default=0.01,
                    help="metres of allowed closeness before the proxy fires")
    ap.add_argument("--vposer-weight", type=float, default=0.0,
                    help="VPoser v2 body_pose prior weight (0 disables)")
    ap.add_argument("--vposer-joints", choices=tuple(VJOINT_PRESETS),
                    default="opt",
                    help="joints the prior treats as free; the rest are held "
                         "at the source body_pose")
    ap.add_argument("--vposer-mode", choices=("kl", "recon"), default="kl")
    ap.add_argument("--vposer-ckpt", default=None,
                    help="override the local VPoser v2 checkpoint")
    ap.add_argument("--source-npz", default=None,
                    help="GVHMR npz supplying betas + the frozen body_pose "
                         "context for the VPoser prior")
    ap.add_argument("--stage-yaw", type=float, default=180.0,
                    help="adapter stage yaw used to make the VMD (SMPL-X "
                         "sources: 180)")
    ap.add_argument("--smplx-gender", default="NEUTRAL")
    ap.add_argument("--no-freeze-root", action="store_true",
                    help="also optimize the shared root translation (default: "
                         "frozen so the root track is unchanged)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)[args.model]
    profile = json.load(open(os.path.join(ROOT, "profiles", f"{args.model}.json"),
                             encoding="utf-8"))
    model = parse_pmx(cfg["pmx"])
    vmd = parse_vmd(args.vmd)
    motion = canonical_from_pmx(model, profile, vmd)
    metas = parse_view_args(args.view_meta)
    kps = parse_view_args(args.view_kp)
    if set(metas) != set(kps):
        raise SystemExit("view names in --view-meta and --view-kp differ")
    views = {v: load_view(metas[v], kps[v]) for v in metas}
    opt_joints = OPT_PRESETS[args.opt_joints]
    vpose = None
    source_bp = None
    betas = None
    vposer_joints = VJOINT_PRESETS[args.vposer_joints]
    if args.vposer_weight > 0.0:
        from vtv.vposer_body import VPoserBodyPrior
        from vtv.vposer_prior import load_vposer_v2

        if args.source_npz:
            d = np.load(args.source_npz)
            source_bp = np.asarray(d["smpl_params_global/body_pose"])
            betas = np.asarray(d["smpl_params_global/betas"][0])
        vp = load_vposer_v2(ckpt=args.vposer_ckpt)
        vpose = VPoserBodyPrior(motion, vposer=vp, betas=betas,
                                gender=args.smplx_gender,
                                stage_yaw_deg=args.stage_yaw,
                                source_body_pose=source_bp)
        import torch

        vj = list(vposer_joints or opt_joints)
        base_vp, mask_vp = vpose.freeze_context(vj)
        G0 = {j: torch.as_tensor(np.asarray(q), dtype=torch.float64)
              for j, q in motion["world_rot"].items()}
        print(f"VPoser prior: joints={vj} KL(init, masked) = "
              f"{float(vpose.energy(G0, base=base_vp, mask=mask_vp)):.3f}")
    res = optimize_multiview(
        motion, views, opt_joints=opt_joints, iters=args.iters, lr=args.lr,
        prior_temporal=args.temporal, prior_collision=args.collision,
        collision_margin=args.collision_margin,
        prior_vposer=args.vposer_weight, vposer=vpose,
        vposer_joints=vposer_joints, vposer_mode=args.vposer_mode,
        freeze_shared_translation=not args.no_freeze_root, verbose=True)
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if args.full_out:
        write_ba_vmd(motion, res, model, profile, args.full_out)
    ba_vmd = None
    if args.full_out:
        ba_vmd = parse_vmd(args.full_out)
    else:
        tmp = args.out + ".ba.vmd"
        write_ba_vmd(motion, res, model, profile, tmp)
        ba_vmd = parse_vmd(tmp)
        if os.path.exists(tmp):
            os.remove(tmp)
    merged, merge_meta = merge_arm_tracks(vmd, ba_vmd, model, profile,
                                          joints=opt_joints)
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_vmd(out, merged.get("model_name", ""), merged["bones"],
              merged.get("morphs", {}), merged.get("ik_frames", []))

    report = {
        "mode": ("M2.4d multi-view torso+arms BA with VPoser v2 prior"
                 if args.vposer_weight > 0.0
                 else "M2.4c multi-view target-PMX body BA (upper bound)"),
        "model": args.model,
        "init": os.path.abspath(args.vmd),
        "out": out,
        "full_out": os.path.abspath(args.full_out) if args.full_out else None,
        "gt": os.path.abspath(args.gt) if args.gt else None,
        "views": {v: {"meta": metas[v], "kp": kps[v]} for v in metas},
        "opt_joints": list(opt_joints),
        "iters": args.iters,
        "lr": args.lr,
        "temporal_prior": args.temporal,
        "collision_prior": args.collision,
        "collision_margin": args.collision_margin,
        "vposer": {
            "weight": args.vposer_weight,
            "mode": args.vposer_mode,
            "stage_yaw_deg": args.stage_yaw,
            "source_npz": os.path.abspath(args.source_npz)
            if args.source_npz else None,
            "stats": res.get("vposer"),
        },
        "freeze_shared_root": not args.no_freeze_root,
        "merge": merge_meta,
        "view_error_fit": res["view_error"],
        "view_error_written": view_px(parse_vmd(out), views, model, profile),
        "root_fit": {
            "root_rot_max_deg": round(float(np.degrees(np.abs(
                res["root_rot"]).max())), 3),
            "root_trans_max_m": round(float(np.abs(res["root_trans"]).max()), 4)},
        "history": res["hist"][-1] if res["hist"] else None,
        "default_pipeline_unchanged": True,
        "note": ("The BA is a multi-view upper bound; T5/single-view must not "
                 "rely on it. The output only changes the optimized chain; "
                 "root, legs, IK and any non-optimized locals are copied from "
                 "the init. Hands are excluded from the M2.4d default set and "
                 "reported separately; upper/fore is the joint-moving swing. "
                 "Contact pairs are reported against both the baseline and the "
                 "GT budget."),
    }
    if args.gt:
        gt_vmd = parse_vmd(args.gt)
        before = body_scores(vmd, gt_vmd, model, profile)
        after = body_scores(parse_vmd(out), gt_vmd, model, profile)
        pb = penetration(vmd, model, profile)
        pa = penetration(parse_vmd(out), model, profile)
        vb = verify_ba(vmd, parse_vmd(out), model, profile, gt_vmd=gt_vmd,
                       opt_joints=opt_joints)
        report["body"] = {
            "before_full_deg": before["full_body_mean_deg"],
            "after_full_deg": after["full_body_mean_deg"],
            "delta_full_deg": round(after["full_body_mean_deg"]
                                    - before["full_body_mean_deg"], 3),
            "before_upper_fore_deg": before["upper_fore_body_mean_deg"],
            "after_upper_fore_deg": after["upper_fore_body_mean_deg"],
            "delta_upper_fore_deg": round(
                after["upper_fore_body_mean_deg"]
                - before["upper_fore_body_mean_deg"], 3),
            "before_hand_deg": before["hand_body_mean_deg"],
            "after_hand_deg": after["hand_body_mean_deg"],
            "delta_hand_deg": round(after["hand_body_mean_deg"]
                                    - before["hand_body_mean_deg"], 3),
            "before_torso_rel": before["torso_body_relative"],
            "after_torso_rel": after["torso_body_relative"],
            "chest_world_before_deg": before["body"]["chest_world_mean_deg"],
            "chest_world_after_deg": after["body"]["chest_world_mean_deg"],
            "world_diag_before": before["world_diag"],
            "world_diag_after": after["world_diag"],
            "before": before["body"], "after": after["body"],
            "position_3d_before": before["position_3d"],
            "position_3d_after": after["position_3d"],
        }
        report["penetration"] = {
            "before": {"arm_torso": pb["arm_torso"]["max"],
                       "arm_arm": pb["arm_arm"]["max"]},
            "after": {"arm_torso": pa["arm_torso"]["max"],
                      "arm_arm": pa["arm_arm"]["max"]},
            "pairs_before": pb["arm_torso"].get("pairs_detail", {}),
            "pairs_after": pa["arm_torso"].get("pairs_detail", {}),
        }
        report["temporal"] = {"before": temporal_steps(vmd, model, profile),
                              "after": temporal_steps(parse_vmd(out), model, profile)}
        report["verify"] = vb
        # M2.4d acceptance: the 10 deg target is on the body-relative
        # upper/forearm swing.  The full-6 number includes the hand (a leaf
        # with no direct 2D evidence) and is reported separately, not used to
        # accept.  Every gate in ``verify`` still has to pass.
        report["accepted"] = bool(
            vb["ok"] and report["body"]["delta_upper_fore_deg"] <= -10.0)
        report["accepted_full_6_only"] = bool(
            vb["ok"] and report["body"]["delta_full_deg"] <= -10.0
            and report["body"]["delta_upper_fore_deg"] > -10.0)
    print(json.dumps({k: report[k] for k in
                      ("out", "view_error_written", "root_fit")},
                     ensure_ascii=False, indent=1))
    if "body" in report:
        print("body full", report["body"]["before_full_deg"], "->",
              report["body"]["after_full_deg"],
              f"({report['body']['delta_full_deg']:+.2f})",
              "| upper/fore", report["body"]["before_upper_fore_deg"], "->",
              report["body"]["after_upper_fore_deg"],
              f"({report['body']['delta_upper_fore_deg']:+.2f})",
              "| hand", report["body"]["before_hand_deg"], "->",
              report["body"]["after_hand_deg"],
              f"({report['body']['delta_hand_deg']:+.2f})")
        print("torso body-rel", report["body"]["before_torso_rel"],
              "->", report["body"]["after_torso_rel"],
              "| chest world", report["body"]["chest_world_before_deg"], "->",
              report["body"]["chest_world_after_deg"])
        print("pen", report["penetration"]["before"], "->",
              report["penetration"]["after"])
        print("temporal", report["temporal"]["before"], "->",
              report["temporal"]["after"])
        print("verify ok", report["verify"]["ok"], "accepted",
              report["accepted"])
        if report.get("vposer", {}).get("stats"):
            print("vposer", report["vposer"]["stats"])
        for k, c in report["verify"]["checks"].items():
            if not c.get("ok", True):
                print("  FAIL", k, json.dumps(
                    {kk: vv for kk, vv in c.items() if kk != "ok"},
                    ensure_ascii=False)[:400])
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)),
                    exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
        print("WROTE", args.report)


if __name__ == "__main__":
    main()
