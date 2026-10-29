"""Diagnose whether a predicted VMD is time-shifted (or yaw-shifted) vs GT.

For every predicted frame it finds the GT frame whose *body-relative* limb
orientations match best.  Relative rotations (each joint vs the pelvis) cancel
the global yaw, so a constant offset means a temporal shift while a high
residual even at the best frame means the pose itself is wrong.

  python scripts/diagnose_alignment.py --gt outputs/gt/si_R2_safe.vmd \
      --pred outputs/adapter/R2fix_front.vmd --out outputs/closed_loop/m1_5/front
"""
import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from vtv.metrics.core import pose_track  # noqa: E402
from vtv.mmd.mathutil import qinv, qmul  # noqa: E402
from vtv.mmd.pmx import parse_pmx  # noqa: E402
from vtv.mmd.vmd import parse_vmd  # noqa: E402

JOINTS = ["upperarm_l", "forearm_l", "upperarm_r", "forearm_r",
          "thigh_l", "shin_l", "thigh_r", "shin_r", "head"]


def rel_rotations(track, joints, ref="pelvis"):
    """q_ref^-1 * q_joint per frame -> (n_frames, n_joints, 4)."""
    n = len(track["frames"])
    out = np.zeros((n, len(joints), 4))
    for fi in range(n):
        q0 = tuple(track["world_q"][ref][fi])
        qi = qinv(q0)
        for ji, j in enumerate(joints):
            if j not in track["world_q"]:
                out[fi, ji] = (0, 0, 0, 1)
                continue
            out[fi, ji] = qmul(qi, tuple(track["world_q"][j][fi]))
    return out


def ang_deg(qa, qb):
    d = abs(float(np.dot(qa, qb)))
    d = max(-1.0, min(1.0, d))
    return math.degrees(2.0 * math.acos(d))


def dist_matrix(a, b):
    """Mean angular distance between every pair (frame of a, frame of b)."""
    na, nb = a.shape[0], b.shape[0]
    out = np.zeros((na, nb))
    for i in range(na):
        dots = np.abs(np.einsum("jc,kjc->kj", a[i], b))  # (nb, joints)
        dots = np.clip(dots, -1.0, 1.0)
        out[i] = np.degrees(2.0 * np.arccos(dots)).mean(axis=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="si")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--frames", default=None, help="a:b GT frame subset")
    args = ap.parse_args()
    cfg = json.load(open(os.path.join(ROOT, "configs", "models.json"),
                        encoding="utf-8"))["models"][args.model]
    profile = json.load(open(os.path.join(ROOT, "profiles", f"{args.model}.json"),
                             encoding="utf-8"))
    model = parse_pmx(cfg["pmx"])
    gt = parse_vmd(args.gt)
    pred = parse_vmd(args.pred)
    frames = None
    if args.frames:
        a, b = [int(x) for x in args.frames.split(":")]
        frames = list(range(a, b + 1))
    gt_t = pose_track(model, gt, profile, frames=frames)
    pred_t = pose_track(model, pred, profile, frames=gt_t["frames"])

    joints = [j for j in JOINTS if j in gt_t["world_q"] and j in pred_t["world_q"]]
    A = rel_rotations(gt_t, joints)
    B = rel_rotations(pred_t, joints)
    D = dist_matrix(B, A)  # (n_pred, n_gt)
    best = D.argmin(axis=1)
    diag = np.array([D[i, min(i, D.shape[1] - 1)] for i in range(D.shape[0])])
    best_err = D[np.arange(D.shape[0]), best]
    offsets = best - np.arange(len(best))

    rows = []
    for i in range(0, len(best), max(1, len(best) // 10)):
        rows.append({"pred_frame": int(pred_t["frames"][i]),
                     "best_gt_frame": int(gt_t["frames"][best[i]]),
                     "offset": int(offsets[i]),
                     "err_at_best": round(float(best_err[i]), 2),
                     "err_at_same_frame": round(float(diag[i]), 2)})
    summary = {
        "joints": joints,
        "gt_frames": len(A),
        "pred_frames": len(B),
        "median_offset": int(np.median(offsets)),
        "mean_err_same_frame_deg": round(float(diag.mean()), 2),
        "mean_err_best_frame_deg": round(float(best_err.mean()), 2),
        "mean_offset_abs": round(float(np.abs(offsets).mean()), 2),
        "rows": rows,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "alignment_diag.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
