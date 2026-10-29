"""M2.4 information bound probe (GT read-only, not a stage).

Answers three questions with numbers:

1. How much of the arm direction error can the 2D keypoints see?
   (2D upper-arm / forearm direction residual vs the 3D direction error.)
2. If the humerus roll were free and an oracle chose it per frame, how much of
   the forearm direction error could be removed?  (This is the ceiling for any
   single-view roll/arm-plane prior.)
3. What is the upper-arm direction depth error?  (3D error minus the 2D-visible
   part -- the quantity no target-space constraint can observe.)

  python scripts/probe_arm_info.py --model si --views front three_quarter
"""
import argparse
import json
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "shim"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "GVHMR"))

import numpy as np

from vtv.metrics.core import pose_track
from vtv.mmd.pmx import parse_pmx
from vtv.mmd.vmd import parse_vmd

COCO = {"ls": 5, "rs": 6, "le": 7, "re": 8, "lw": 9, "rw": 10}


def _ang(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    c = (a * b).sum(-1) / np.maximum(na * nb, 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def _joints(npz_path):
    import torch
    from hmr4d.utils.smplx_utils import make_smplx
    d = np.load(npz_path)
    body = torch.tensor(d["smpl_params_incam/body_pose"], dtype=torch.float32)
    go = torch.tensor(d["smpl_params_incam/global_orient"], dtype=torch.float32)
    tr = torch.tensor(d["smpl_params_incam/transl"], dtype=torch.float32)
    betas = torch.tensor(d["smpl_params_incam/betas"], dtype=torch.float32)
    K = np.asarray(d["K_fullimg"], dtype=np.float64)
    model = make_smplx("supermotion_coco17").eval()
    with torch.no_grad():
        J = model(body_pose=body, betas=betas, global_orient=go, transl=tr).numpy()
    uv = np.einsum("fij,fkj->fki", K, J)
    uv = uv[..., :2] / uv[..., 2:3]
    if "kp2d" in d.files:
        kp = np.asarray(d["kp2d"], dtype=np.float32)
    else:
        kp = torch.load(os.path.join(os.path.dirname(npz_path),
                                     "preprocess", "vitpose.pt"),
                        map_location="cpu").numpy()
    return J, uv, np.asarray(kp), d.files


def _roll_about(axis, vec, deg):
    th = math.radians(deg)
    a = axis / max(np.linalg.norm(axis), 1e-12)
    # Rodrigues
    return (vec * math.cos(th) + np.cross(a, vec) * math.sin(th)
            + a * (a @ vec) * (1.0 - math.cos(th)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="si")
    ap.add_argument("--views", nargs="+", default=["front", "three_quarter"])
    ap.add_argument("--gt", default=os.path.join(ROOT, "outputs", "gt", "si_R2_safe.vmd"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "m2", "arm", "info_bound.json"))
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(ROOT, "configs", "models.json"),
                         encoding="utf-8"))["models"][args.model]
    prof = json.load(open(os.path.join(ROOT, "profiles", f"{args.model}.json"),
                          encoding="utf-8"))
    model = parse_pmx(cfg["pmx"])
    gt = pose_track(model, parse_vmd(args.gt), prof)
    out = {"model": args.model, "views": {}}
    for view in args.views:
        npz = os.path.join(ROOT, "outputs", "gvhmr", f"R2fix_{view}", "gvhmr_result.npz")
        pred_vmd = os.path.join(ROOT, "outputs", "m2", "vmd",
                                f"{view}_clean_ik.vmd")
        pr = pose_track(model, parse_vmd(pred_vmd), prof, frames=gt["frames"])
        J, uv, kp, _ = _joints(npz)
        v = {"2d": {}, "3d": {}, "oracle_roll": {}}
        for side, (s, e, w) in {"l": (COCO["ls"], COCO["le"], COCO["lw"]),
                                "r": (COCO["rs"], COCO["re"], COCO["rw"])}.items():
            e2u = _ang(uv[:, e] - uv[:, s], kp[:, e, :2] - kp[:, s, :2])
            e2f = _ang(uv[:, w] - uv[:, e], kp[:, w, :2] - kp[:, e, :2])
            up, fo, ha = (f"{p}_{side}" for p in ("upperarm", "forearm", "hand"))
            du3 = _unit(pr["world"][fo] - pr["world"][up])
            df3 = _unit(pr["world"][ha] - pr["world"][fo])
            gu3 = _unit(gt["world"][fo] - gt["world"][up])
            gf3 = _unit(gt["world"][ha] - gt["world"][fo])
            e3u = _ang(du3, gu3)
            e3f = _ang(df3, gf3)
            # depth part of the 3D direction error: sqrt(err^2 - 2d^2), clipped
            d3u = math.sqrt(max(0.0, float(np.mean(e3u)) ** 2
                                - float(np.median(e2u)) ** 2))
            d3f = math.sqrt(max(0.0, float(np.mean(e3f)) ** 2
                                - float(np.median(e2f)) ** 2))
            # oracle humerus roll: roll the forearm about the predicted upper
            # arm axis, pick the angle closest to the GT forearm direction
            resid, deltas = [], []
            for k in range(len(du3)):
                u = du3[k]
                vp = df3[k] - (df3[k] @ u) * u
                nv = np.linalg.norm(vp)
                tg = gf3[k] - (gf3[k] @ u) * u
                ntg = np.linalg.norm(tg)
                if nv < 1e-8 or ntg < 1e-8:
                    resid.append(float(_ang(df3[k], gf3[k])))
                    continue
                e1 = vp / nv
                e2 = np.cross(u, e1)
                c, s2 = float(e1 @ (tg / ntg)), float(e2 @ (tg / ntg))
                delta = math.degrees(math.atan2(s2, c))
                deltas.append(delta)
                v2 = _roll_about(u, df3[k], delta)
                resid.append(float(_ang(v2, gf3[k])))
            v["2d"][side] = {"upperarm_deg_median": round(float(np.median(e2u)), 2),
                             "upperarm_deg_p90": round(float(np.percentile(e2u, 90)), 2),
                             "forearm_deg_median": round(float(np.median(e2f)), 2),
                             "forearm_deg_p90": round(float(np.percentile(e2f, 90)), 2)}
            v["3d"][side] = {"upperarm_deg_mean": round(float(np.mean(e3u)), 2),
                             "forearm_deg_mean": round(float(np.mean(e3f)), 2),
                             "upperarm_depth_est_deg": round(d3u, 2),
                             "forearm_depth_est_deg": round(d3f, 2)}
            v["oracle_roll"][side] = {
                "before_deg_mean": round(float(np.mean(e3f)), 2),
                "after_deg_mean": round(float(np.mean(resid)), 2),
                "gain_deg": round(float(np.mean(e3f) - np.mean(resid)), 2),
                "delta_deg_median": round(float(np.median(deltas)) if deltas else 0.0, 2),
                "delta_deg_p5": round(float(np.percentile(deltas, 5)) if deltas else 0.0, 2),
                "delta_deg_p95": round(float(np.percentile(deltas, 95)) if deltas else 0.0, 2),
            }
        out["views"][view] = v
        print(f"\n== {view}")
        for side in ("l", "r"):
            print(f"  {side}: 2D upper/fore {v['2d'][side]['upperarm_deg_median']:.1f}/"
                  f"{v['2d'][side]['forearm_deg_median']:.1f} deg | 3D upper/fore "
                  f"{v['3d'][side]['upperarm_deg_mean']:.1f}/{v['3d'][side]['forearm_deg_mean']:.1f} "
                  f"| depth est {v['3d'][side]['upperarm_depth_est_deg']:.1f}/"
                  f"{v['3d'][side]['forearm_depth_est_deg']:.1f} | oracle-roll gain "
                  f"{v['oracle_roll'][side]['gain_deg']:.1f} (residual "
                  f"{v['oracle_roll'][side]['after_deg_mean']:.1f})")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nWROTE", args.out)


if __name__ == "__main__":
    main()
