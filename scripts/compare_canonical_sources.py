"""Split adapter error from feed-forward error (canonical-pose comparison).

Compares GVHMR's canonical pose (SMPL-X) with a GT VMD's canonical pose using
pelvis-relative joint rotations, so the global yaw and the SMPL-X/PMX frame
difference cancel.  If this is large, the residual is in the estimator, not in
the adapter (whose same-model round-trip is exact to float precision).

  python scripts/compare_canonical_sources.py \
      --npz outputs/gvhmr/R2fix_front/gvhmr_result.npz \
      --gt outputs/gt/si_R2_safe.vmd --model si
"""
import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from vtv.adapter import canonical_from_pmx, canonical_from_smplx  # noqa: E402
from vtv.canonical_body import BODY_JOINTS  # noqa: E402
from vtv.mmd.mathutil import qinv, qmul  # noqa: E402
from vtv.mmd.pmx import parse_pmx  # noqa: E402
from vtv.mmd.vmd import parse_vmd  # noqa: E402


def ang_deg(qa, qb):
    d = abs(float(np.dot(qa, qb)))
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, d))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="si")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from run_adapter import smplx_rest_joints  # heavy imports only when needed

    cfg = json.load(open(os.path.join(ROOT, "configs", "models.json"),
                        encoding="utf-8"))["models"][args.model]
    profile = json.load(open(os.path.join(ROOT, "profiles", f"{args.model}.json"),
                             encoding="utf-8"))
    model = parse_pmx(cfg["pmx"])
    gt = canonical_from_pmx(model, profile, parse_vmd(args.gt))

    d = np.load(args.npz)
    params = {k: d[f"smpl_params_global/{k}"] for k in
              ("body_pose", "global_orient", "transl")}
    rest, _ = smplx_rest_joints(args.npz, betas=d["smpl_params_global/betas"][:1])
    src = canonical_from_smplx(params, rest, slow_factor=1.0)

    joints = [j for j in BODY_JOINTS if j != "pelvis"]
    n = min(len(gt["frames"]), len(src["frames"]))
    per_joint = {j: [] for j in joints}
    per_frame = []
    for f in range(n):
        qp_g = tuple(gt["world_rot"]["pelvis"][f])
        qp_s = tuple(src["world_rot"]["pelvis"][f])
        row = {}
        for j in joints:
            rg = qmul(qinv(qp_g), tuple(gt["world_rot"][j][f]))
            rs = qmul(qinv(qp_s), tuple(src["world_rot"][j][f]))
            a = ang_deg(rg, rs)
            per_joint[j].append(a)
            row[j] = a
        per_frame.append(row)

    allv = np.array([a for j in joints for a in per_joint[j]])
    worst = sorted(((float(np.mean(v)), j) for j, v in per_joint.items()),
                   reverse=True)
    report = {
        "npz": os.path.abspath(args.npz),
        "gt": os.path.abspath(args.gt),
        "frames": n,
        "overall_mean_deg": round(float(allv.mean()), 2),
        "median_deg": round(float(np.median(allv)), 2),
        "p90_deg": round(float(np.percentile(allv, 90)), 2),
        "per_joint_mean_deg": {j: round(float(np.mean(v)), 2)
                               for j, v in per_joint.items()},
        "worst_joints": [{"joint": j, "mean_deg": round(m, 2),
                          "p95_deg": round(float(np.percentile(per_joint[j], 95)), 2)}
                         for m, j in worst[:10]],
        "per_frame_mean_deg": [round(float(np.mean(list(r.values()))), 2)
                               for r in per_frame],
        "note": ("pelvis-relative rotations: global yaw and the SMPL-X/PMX "
                 "frame difference cancel. This is feed-forward error, not "
                 "adapter error."),
    }
    print(json.dumps({k: report[k] for k in
                      ("frames", "overall_mean_deg", "median_deg", "p90_deg",
                       "worst_joints")}, ensure_ascii=False, indent=1))
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "canonical_source_diff.json"), "w",
                  encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
