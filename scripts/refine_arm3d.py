"""M2.4b multi-view 3D arm direction prior (experiment).

Triangulates the 3D upper-arm / forearm directions from the calibrated
synthetic cameras + ViTPose keypoints, applies them as a world-space swing to
the arm chain, and reports both the world-space and the **torso-relative**
arm errors.  The latter is the acceptance gate: a world-space win that only
cancels the per-view root/chest error is not an arm-pose improvement.

  python scripts/refine_arm3d.py --model si \
    --vmd outputs/m2/vmd/front_clean_ik.vmd \
    --view-meta front=outputs/synth/si_R2_fixed/front/meta.json \
                three_quarter=outputs/synth/si_R2_fixed/three_quarter/meta.json \
                side=outputs/synth/si_R2_fixed/side/meta.json \
    --view-kp front=outputs/gvhmr/R2fix_front/preprocess/vitpose.pt \
              three_quarter=outputs/gvhmr/R2fix_three_quarter/preprocess/vitpose.pt \
              side=outputs/gvhmr/R2fix_side/preprocess/vitpose.pt \
    --gt outputs/gt/si_R2_safe.vmd \
    --out outputs/m2/arm3d/front_arm3d.vmd \
    --report outputs/m2/arm3d/front_arm3d.json

The solver never reads ``--gt``; it is used only to score the result.
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

from vtv.arm3d import (apply_direction_prior, body_relative_errors,
                       current_directions, load_blender_camera, load_keypoints,
                       penetration, targets_from_result,
                       triangulate_directions, verify)
from vtv.arm_prior import compare_arm
from vtv.metrics.core import pose_track
from vtv.mmd.pmx import parse_pmx
from vtv.mmd.vmd import parse_vmd
from vtv.mmd.writer import write_vmd


def pairs(xs):
    out = {}
    for x in xs or []:
        k, v = x.split("=", 1)
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "models.json"))
    ap.add_argument("--model", default="si")
    ap.add_argument("--vmd", required=True, help="input VMD (one view)")
    ap.add_argument("--view-meta", nargs="+", required=True,
                    help="view=path to a synthetic meta.json (calibrated camera)")
    ap.add_argument("--view-kp", nargs="+", required=True,
                    help="view=path to a ViTPose (F,17,3) tensor")
    ap.add_argument("--cap", type=float, default=60.0,
                    help="per-joint world correction cap (deg)")
    ap.add_argument("--conf", type=float, default=0.30,
                    help="minimum keypoint confidence for a 2D direction")
    ap.add_argument("--smooth", type=int, default=3,
                    help="temporal box-average window for the target direction")
    ap.add_argument("--max-step", type=float, default=0.0,
                    help="maximum target direction change per frame (deg; 0=off)")
    ap.add_argument("--median", type=int, default=0,
                    help="temporal median window for the target direction (0=off)")
    ap.add_argument("--lam", type=float, default=0.0,
                    help="blend weight of the current prediction (0 = pure triangulation)")
    ap.add_argument("--no-pen-guard", action="store_true",
                    help="disable the per-frame capsule guard (experiment only)")
    ap.add_argument("--pen-tol", type=float, default=0.01,
                    help="allowed per-pair penetration increase in the guard")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--gt", default=None,
                    help="GT VMD, used only to score (never by the solver)")
    ap.add_argument("--frames", default=None)
    ap.add_argument("--sides", default="l,r",
                    help="comma list of sides to correct (l/r)")
    ap.add_argument("--segments", default="upper,fore",
                    help="comma list of segments to correct (upper/fore)")
    args = ap.parse_args()
    sides = tuple(s.strip() for s in args.sides.split(",") if s.strip())
    segments = {s.strip() for s in args.segments.split(",") if s.strip()}

    cfg = json.load(open(args.config, encoding="utf-8"))["models"][args.model]
    profile = json.load(open(os.path.join(ROOT, "profiles", f"{args.model}.json"),
                             encoding="utf-8"))
    model = parse_pmx(cfg["pmx"])
    vmd = parse_vmd(args.vmd)
    frames = None
    if args.frames:
        a, b = [int(x) for x in args.frames.split(":")]
        frames = list(range(a, b + 1))

    metas = pairs(args.view_meta)
    kps = {v: load_keypoints(p) for v, p in pairs(args.view_kp).items()}
    cams = {v: load_blender_camera(p) for v, p in metas.items()}
    prior = current_directions(vmd, model, profile)
    tri = triangulate_directions(kps, cams, conf_min=args.conf,
                                 smooth=args.smooth, prior=prior,
                                 lam=args.lam, max_step_deg=args.max_step,
                                 median=args.median)
    targets, valid = targets_from_result(tri)
    for side in ("l", "r"):
        for seg in ("upper", "fore"):
            name = f"{seg}_{side}"
            if side not in sides:
                targets[name] = np.zeros_like(targets[name])
                valid[name] = np.zeros(len(targets[name]), dtype=bool)
                continue
            if seg not in segments:
                if seg == "upper":
                    # Keep the upper arm where it is; this lets a
                    # forearm-only variant run (a zero target would skip the
                    # whole frame in the chain update).
                    targets[name] = np.asarray(prior[name], dtype=np.float64)
                    valid[name] = np.ones(len(targets[name]), dtype=bool)
                else:
                    targets[name] = np.zeros_like(targets[name])
                    valid[name] = np.zeros(len(targets[name]), dtype=bool)
    res = apply_direction_prior(vmd, model, profile, targets, cap_deg=args.cap,
                                valid=valid, sides=sides,
                                pen_guard=not args.no_pen_guard,
                                pen_tol=args.pen_tol)
    rep = dict(res["report"])
    rep.update({
        "input": os.path.abspath(args.vmd),
        "out": os.path.abspath(args.out),
        "model": args.model,
        "views": sorted(cams),
        "sides": list(sides),
        "segments": sorted(segments),
        "triangulation": {
            name: {"cond_mean": round(float(np.mean(rec["cond"])), 4),
                   "cond_p95": round(float(np.percentile(rec["cond"], 95)), 4),
                   "valid_frames": int(valid[name].sum()),
                   "frames": int(len(valid[name]))}
            for name, rec in tri.items()},
    })
    # Geometry self-check (GT-free).
    chk = verify(vmd, res["vmd"], model, profile, rep, targets=targets,
                 applied=res.get("applied"),
                 max_target_err_deg=float(args.cap) + 0.5)
    rep["verify"] = chk
    rep["verify_ok"] = bool(chk["ok"])

    if args.gt:
        gt_vmd = parse_vmd(args.gt)
        gt = pose_track(model, gt_vmd, profile)
        gt_dirs = current_directions(gt_vmd, model, profile)
        pr_in = pose_track(model, vmd, profile, frames=gt["frames"])
        pr_out = pose_track(model, res["vmd"], profile, frames=gt["frames"])
        c_in = compare_arm(gt, pr_in, model, profile)
        c_out = compare_arm(gt, pr_out, model, profile)
        b_in = body_relative_errors(gt, pr_in, model, profile)
        b_out = body_relative_errors(gt, pr_out, model, profile)
        p_in = penetration(vmd, model, profile)
        p_out = penetration(res["vmd"], model, profile)
        rep["before"] = {"vs_gt": c_in, "body_relative": b_in,
                         "penetration": p_in}
        rep["after"] = {"vs_gt": c_out, "body_relative": b_out,
                        "penetration": p_out}
        d_world = round(c_out["arm_world_mean_deg"] - c_in["arm_world_mean_deg"], 3)
        d_swing = round(c_out["arm_swing_mean_deg"] - c_in["arm_swing_mean_deg"], 3)
        d_body = round(b_out["arm_body_mean_deg"] - b_in["arm_body_mean_deg"], 3)
        rep["delta"] = {"arm_world_mean_deg": d_world,
                        "arm_swing_mean_deg": d_swing,
                        "arm_body_mean_deg": d_body}
        # Triangulated-target quality against GT (the information bound of the
        # multi-view 2D evidence; the solver never uses this).
        tv = {}
        for name, tgt in targets.items():
            if name not in gt_dirs:
                continue
            t = np.asarray(tgt, dtype=np.float64)
            g = np.asarray(gt_dirs[name], dtype=np.float64)
            n = np.minimum(len(t), len(g))
            tn = t[:n] / np.maximum(np.linalg.norm(t[:n], axis=1, keepdims=True), 1e-12)
            gn = g[:n] / np.maximum(np.linalg.norm(g[:n], axis=1, keepdims=True), 1e-12)
            ok = np.linalg.norm(t[:n], axis=1) > 1e-9
            if not np.any(ok):
                continue
            ang = np.degrees(np.arccos(np.clip(
                np.abs((tn[ok] * gn[ok]).sum(1)), -1.0, 1.0)))
            tv[name] = {"mean_deg": round(float(ang.mean()), 3),
                        "median_deg": round(float(np.median(ang)), 3),
                        "p90_deg": round(float(np.percentile(ang, 90)), 3),
                        "frames": int(ok.sum())}
        rep["target_vs_gt"] = tv
        # The gate deliberately requires the torso-relative error to improve:
        # a world-only win can be a root-orientation compensation.
        pen_ok = (p_out["arm_torso"]["max"] <= p_in["arm_torso"]["max"] + 1e-9
                  and p_out["arm_arm"]["max"] <= p_in["arm_arm"]["max"] + 0.02)
        body_ok = d_body < -1e-6
        swing_ok = d_swing < -1e-6
        rep["gate"] = {
            "verify": bool(chk["ok"]),
            "penetration_ok": bool(pen_ok),
            "world_swing_improved": bool(swing_ok),
            "body_relative_improved": bool(body_ok),
            "accepted": bool(chk["ok"] and pen_ok and swing_ok and body_ok),
            "note": ("world-only wins are rejected: the 3/4 chest/root error "
                     "is 51 deg, so a world target can move the arm away from "
                     "the torso."),
        }
        rep["accepted"] = rep["gate"]["accepted"]
        print(f"{args.vmd}")
        print(f"  world  {c_in['arm_world_mean_deg']:.2f} -> "
              f"{c_out['arm_world_mean_deg']:.2f} ({d_world:+.2f})")
        print(f"  swing  {c_in['arm_swing_mean_deg']:.2f} -> "
              f"{c_out['arm_swing_mean_deg']:.2f} ({d_swing:+.2f})")
        print(f"  body   {b_in['arm_body_mean_deg']:.2f} -> "
              f"{b_out['arm_body_mean_deg']:.2f} ({d_body:+.2f})  "
              f"chest {b_in['chest_world_mean_deg']:.1f}/"
              f"{b_out['chest_world_mean_deg']:.1f}")
        print(f"  penT   {p_in['arm_torso']['max']:.3f} -> "
              f"{p_out['arm_torso']['max']:.3f}  penA "
              f"{p_in['arm_arm']['max']:.3f} -> {p_out['arm_arm']['max']:.3f}")
        print(f"  verify {chk['ok']}  accepted {rep['accepted']}")
    else:
        rep["accepted"] = bool(chk["ok"])
        print(f"verify {chk['ok']}  (no --gt: no accuracy claim)")

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_vmd(out, res["vmd"].get("model_name", ""), res["vmd"]["bones"],
              res["vmd"].get("morphs", {}), res["vmd"].get("ik_frames", []))
    if args.report:
        rp = os.path.abspath(args.report)
        os.makedirs(os.path.dirname(rp), exist_ok=True)
        with open(rp, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=1)
        print("WROTE", rp)
    print("WROTE", out)


if __name__ == "__main__":
    main()
