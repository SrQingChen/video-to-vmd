"""M2.4 arm stage: target-space constraints + anatomical arm audit.

  python scripts/refine_arm.py --vmd outputs/m2/vmd/front_clean_ik.vmd \
      --model si --out outputs/m2/vmd/front_clean_ik_arm.vmd \
      --mode constraints --rounds 3 --pen-tol 0.25 --max-deg 12

Modes
-----
audit       only measure; write the input unchanged.
constraints  resolve deep arm-torso capsule penetration (shoulder-only,
             2-3 rounds, trust region).  ``--arm-arm`` also separates the two
             arms (opt-in: the GT itself keeps them close and the measured
             direction was worse).
hinge       project the elbow local axis toward the palm hinge line (M2.4
             experiment; it lowers the geodesic metric by changing roll and
             worsens the actual forearm swing on R2).
hinge+constraints  both, in that order.

``--gt`` is read-only: it only scores the result (closed-loop evaluation), it is
never used by the solver.  The output is a normal 23-track IK-off VMD; no MMD
IK track and no D-chain helper is written.
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from vtv.arm_prior import (DEFAULT_HINGE_CAP, DEFAULT_MAX_DEG,
                           DEFAULT_MAX_DISP, DEFAULT_PEN_TOL, DEFAULT_ROUNDS,
                           DEFAULT_SMOOTH, arm_structure,
                           capsule_penetrations, compare_arm, project_hinge,
                           resolve_self_intersections, verify)
from vtv.cleanup import dense_tracks, eval_tracks
from vtv.metrics.core import pose_track
from vtv.mmd.pmx import parse_pmx
from vtv.mmd.vmd import parse_vmd
from vtv.mmd.writer import write_vmd


def load_cfg(path=os.path.join(ROOT, "configs", "models.json")):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "models.json"))
    ap.add_argument("--model", default="si")
    ap.add_argument("--vmd", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt", default=None,
                    help="optional synthetic GT, read-only evaluation")
    ap.add_argument("--mode", default="constraints",
                    choices=("audit", "constraints", "hinge", "hinge+constraints"))
    ap.add_argument("--pen-tol", type=float, default=DEFAULT_PEN_TOL)
    ap.add_argument("--max-deg", type=float, default=DEFAULT_MAX_DEG)
    ap.add_argument("--max-disp", type=float, default=DEFAULT_MAX_DISP)
    ap.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    ap.add_argument("--smooth", type=int, default=DEFAULT_SMOOTH)
    ap.add_argument("--arm-arm", action="store_true",
                    help="also resolve left/right arm capsule penetration")
    ap.add_argument("--hinge-alpha", type=float, default=1.0)
    ap.add_argument("--hinge-cap", type=float, default=DEFAULT_HINGE_CAP)
    ap.add_argument("--report", default=None)
    ap.add_argument("--force", action="store_true",
                    help="write the edited VMD even if the self-check fails "
                         "(experiments only; the report keeps accepted=false)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.config)["models"][args.model]
    profile = json.load(open(os.path.join(ROOT, "profiles", f"{args.model}.json"),
                             encoding="utf-8"))
    model = parse_pmx(cfg["pmx"])
    vmd = parse_vmd(args.vmd)
    gt_vmd = parse_vmd(args.gt) if args.gt else None

    tr_in = eval_tracks(model, profile, dense_tracks(vmd)[1])
    pen_in = capsule_penetrations(tr_in, model, profile, arm_arm=True)
    report = {
        "input": os.path.abspath(args.vmd),
        "model": args.model,
        "mode": args.mode,
        "frames": len(vmd["bones"][next(iter(vmd["bones"]))]) if vmd["bones"] else 0,
        "before": {
            "structure": arm_structure(tr_in, model, profile),
            "penetration": pen_in["summary"],
        },
    }
    if gt_vmd is not None:
        tr_gt = pose_track(model, gt_vmd, profile,
                           frames=list(range(vmd["max_frame"] + 1)))
        report["before"]["vs_gt"] = compare_arm(tr_gt, tr_in, model, profile)

    out_vmd = vmd
    step_reports = []
    if args.mode in ("hinge", "hinge+constraints"):
        res = project_hinge(out_vmd, model, profile, alpha=args.hinge_alpha,
                            cap_deg=args.hinge_cap)
        out_vmd = res["vmd"]
        step_reports.append(res["report"])
    if args.mode in ("constraints", "hinge+constraints"):
        res = resolve_self_intersections(
            out_vmd, model, profile, pen_tol=args.pen_tol,
            max_deg=args.max_deg, max_disp=args.max_disp,
            rounds=args.rounds, smooth=args.smooth, arm_arm=bool(args.arm_arm))
        out_vmd = res["vmd"]
        step_reports.append(res["report"])

    tr_out = eval_tracks(model, profile, dense_tracks(out_vmd)[1])
    pen_out = capsule_penetrations(tr_out, model, profile, arm_arm=True)
    report["steps"] = step_reports
    report["after"] = {
        "structure": arm_structure(tr_out, model, profile),
        "penetration": pen_out["summary"],
    }
    if gt_vmd is not None:
        tr_gt = pose_track(model, gt_vmd, profile,
                           frames=list(range(vmd["max_frame"] + 1)))
        report["after"]["vs_gt"] = compare_arm(tr_gt, tr_out, model, profile)

    ver = verify(vmd, out_vmd, model, profile,
                 {"max_deg": max(args.max_deg, args.hinge_cap) if args.mode.startswith("hinge")
                  else args.max_deg,
                  "arm_arm": bool(args.arm_arm),
                  "mode": args.mode})
    report["verify"] = ver
    report["accepted"] = bool(ver["ok"])
    if not ver["ok"] and not args.force:
        out_vmd = vmd
        report["note"] = "self-check failed; output reverted to the input"
    elif not ver["ok"] and args.force:
        report["note"] = "self-check failed; experimental output written (--force)"
    elif args.mode == "hinge" or args.mode == "hinge+constraints":
        report["note"] = ("hinge projection is an experiment: it changes roll, "
                          "not the observable swing; see before/after vs_gt")
    if not args.dry_run:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        write_vmd(args.out, out_vmd.get("model_name", ""), out_vmd["bones"],
                  out_vmd.get("morphs", {}), out_vmd.get("ik_frames", []))
        report["out"] = os.path.abspath(args.out)
        print("WROTE", args.out)
    rp = args.report or (args.out + ".json")
    os.makedirs(os.path.dirname(os.path.abspath(rp)), exist_ok=True)
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("WROTE", rp)

    b = report["before"]["penetration"]
    a = report["after"]["penetration"]
    print(f"penetration arm-torso max {b['arm_torso']['max']:.3f} -> "
          f"{a['arm_torso']['max']:.3f}; arm-arm max {b['arm_arm']['max']:.3f} -> "
          f"{a['arm_arm']['max']:.3f}")
    if "vs_gt" in report["after"]:
        g = report["after"]["vs_gt"]
        print(f"arm world {g['arm_world_mean_deg']:.2f}  swing "
              f"{g['arm_swing_mean_deg']:.2f}  roll {g['arm_roll_mean_deg']:.2f} "
              f"(GT-scored only)")
        print(f"before arm world {report['before']['vs_gt']['arm_world_mean_deg']:.2f}  "
              f"swing {report['before']['vs_gt']['arm_swing_mean_deg']:.2f}  "
              f"roll {report['before']['vs_gt']['arm_roll_mean_deg']:.2f}")
    print("verify", "OK" if ver["ok"] else "FAILED",
          "| accepted", report["accepted"])


if __name__ == "__main__":
    main()
