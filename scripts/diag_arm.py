"""Arm error decomposition + target-space audit (GT read-only).

  python scripts/diag_arm.py --model si \
      --gt outputs/gt/si_R2_safe.vmd \
      --pred outputs/m2/vmd/front_clean_ik.vmd \
      --out outputs/m2/arm/front_diag.json

Prints per-bone geodesic world angle, its swing (bone-direction) and roll
(rotation about the bone axis) parts, the elbow hinge axis/stability, wrist
bend and the PMX rigid-body capsule penetration.  GT is only used to score the
prediction; the solver (`refine_arm.py`) never reads it.

Why this exists: the geodesic world-angle gate conflates two different errors.
A forearm can have a 25 deg direction error and a 43 deg roll residual; a
correction that only changes roll lowers the geodesic number without fixing the
observable pose.  The report keeps swing and roll separate so a candidate can
never win on roll alone.
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from vtv.arm_prior import (arm_structure, capsule_penetrations, compare_arm,
                           palm_frame)
from vtv.cleanup import dense_tracks, eval_tracks
from vtv.metrics.core import pose_track
from vtv.mmd.pmx import parse_pmx
from vtv.mmd.vmd import parse_vmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "models.json"))
    ap.add_argument("--model", default="si")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--frames", default=None)
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8"))["models"][args.model]
    profile = json.load(open(os.path.join(ROOT, "profiles", f"{args.model}.json"),
                             encoding="utf-8"))
    model = parse_pmx(cfg["pmx"])
    gt = parse_vmd(args.gt)
    pred = parse_vmd(args.pred)
    frames = None
    if args.frames:
        a, b = [int(x) for x in args.frames.split(":")]
        frames = list(range(a, b + 1))
    tr_gt = pose_track(model, gt, profile, frames=frames)
    tr_pr = pose_track(model, pred, profile, frames=tr_gt["frames"])
    cmp = compare_arm(tr_gt, tr_pr, model, profile)
    structure = arm_structure(tr_pr, model, profile)
    pen = capsule_penetrations(
        eval_tracks(model, profile, dense_tracks(pred)[1]), model, profile,
        arm_arm=True)
    report = {
        "model": args.model,
        "gt": os.path.abspath(args.gt),
        "pred": os.path.abspath(args.pred),
        "frames": len(tr_gt["frames"]),
        "vs_gt": cmp,
        "structure": structure,
        "penetration": pen["summary"],
        "note": ("swing = bone direction; roll = residual about the bone axis. "
                 "Only swing is observable from joint positions; a candidate "
                 "that lowers world_total but raises swing is rejected."),
    }
    print(f"arm world {cmp['arm_world_mean_deg']:.2f}  "
          f"swing {cmp['arm_swing_mean_deg']:.2f}  roll {cmp['arm_roll_mean_deg']:.2f}")
    for side, rows in cmp["sides"].items():
        for j, r in rows.items():
            print(f"  {j:10s} top {r['world_total_deg']:6.2f}  swing {r['swing_deg']:6.2f}"
                  f"  roll {r['roll_deg']:6.2f}"
                  + (f"  pos {r['pos_child_units']:.3f}" if r['pos_child_units'] is not None else ""))
        s = structure["sides"][side]
        print(f"    hinge scatter {s['elbow_hinge']['axis_scatter_deg_mean']:.2f} deg, "
              f"vs palm {s['hinge_vs_palm_deg']:.2f} deg, "
              f"wrist local {s['wrist_local_deg_mean']:.2f} deg")
    print("  penetration arm-torso max "
          f"{pen['summary']['arm_torso']['max']:.3f} (frames "
          f"{pen['summary']['arm_torso']['frames']}), arm-arm max "
          f"{pen['summary']['arm_arm']['max']:.3f} (frames "
          f"{pen['summary']['arm_arm']['frames']})")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
        print("WROTE", args.out)


if __name__ == "__main__":
    main()
