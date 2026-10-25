"""Sanity-check a GVHMR result (scale / root trajectory / joints / NaNs).

Usage:
  .venv/Scripts/python.exe scripts/check_gvhmr.py --npz outputs/gvhmr/T5/gvhmr_result.npz
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "shim"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "GVHMR"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    out = args.out or os.path.join(os.path.dirname(args.npz), "gvhmr_check.json")
    d = np.load(args.npz)
    keys = set(d.files)
    report = {"npz": os.path.abspath(args.npz), "keys": sorted(keys)}

    def get(prefix, name):
        return d[f"{prefix}/{name}"]

    for prefix in ("smpl_params_global", "smpl_params_incam"):
        if f"{prefix}/transl" not in keys:
            continue
        transl = get(prefix, "transl")
        pose = get(prefix, "body_pose")
        orient = get(prefix, "global_orient")
        betas = get(prefix, "betas")
        bad = {k: int((~np.isfinite(v)).sum()) for k, v in
               {"transl": transl, "body_pose": pose, "global_orient": orient,
                "betas": betas}.items()}
        report[prefix] = {
            "frames": int(transl.shape[0]),
            "nan_counts": bad,
            "transl_min": transl.min(0).round(4).tolist(),
            "transl_max": transl.max(0).round(4).tolist(),
            "transl_std": transl.std(0).round(4).tolist(),
            "root_path_length_m": round(float(np.linalg.norm(
                np.diff(transl[:, [0, 2]], axis=0), axis=1).sum()), 4),
            "pose_angle_deg": {
                "mean": round(float(np.degrees(np.linalg.norm(
                    pose.reshape(pose.shape[0], -1, 3), axis=2)).mean()), 3),
                "p99": round(float(np.degrees(np.linalg.norm(
                    pose.reshape(pose.shape[0], -1, 3), axis=2)).max()), 3),
            },
            "orient_angle_deg": {
                "mean": round(float(np.degrees(np.linalg.norm(orient, axis=1)).mean()), 3),
                "max": round(float(np.degrees(np.linalg.norm(orient, axis=1)).max()), 3),
            },
            "betas_mean": betas.mean(0).round(4).tolist(),
        }

    # SMPL-X joints (optional; needs body_models + smplx)
    joints = None
    try:
        import torch
        from hmr4d.utils.smplx_utils import make_smplx
        model = make_smplx("supermotion", num_betas=10)
        prefix = "smpl_params_global"
        params = {
            "betas": torch.tensor(get(prefix, "betas"), dtype=torch.float32),
            "body_pose": torch.tensor(get(prefix, "body_pose"), dtype=torch.float32),
            "global_orient": torch.tensor(get(prefix, "global_orient"), dtype=torch.float32),
            "transl": torch.tensor(get(prefix, "transl"), dtype=torch.float32),
        }
        with torch.no_grad():
            outm = model(**params)
        joints = outm.joints[:, :22].numpy()
        report["smplx_joints"] = {
            "shape": list(joints.shape),
            "bbox_min": joints.reshape(-1, 3).min(0).round(3).tolist(),
            "bbox_max": joints.reshape(-1, 3).max(0).round(3).tolist(),
            "height_mean_m": round(float((joints[:, :, 1].max(1)
                                          - joints[:, :, 1].min(1)).mean()), 4),
            "nan": int((~np.isfinite(joints)).sum()),
        }
        # bone-length consistency for a few SMPL-X edges
        edges = [(0, 1), (1, 4), (4, 7), (0, 2), (2, 5), (5, 8),
                 (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
                 (9, 13), (13, 16), (9, 14), (14, 17)]
        lengths = np.stack([np.linalg.norm(joints[:, a] - joints[:, b], axis=1)
                            for a, b in edges], axis=1)
        report["smplx_bone_length_cv"] = [
            round(float(lengths[:, i].std() / max(lengths[:, i].mean(), 1e-9)), 5)
            for i in range(lengths.shape[1])]
    except Exception as e:
        report["smplx_joints_error"] = repr(e)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print("WROTE", out)

    if joints is not None and not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            root = joints[:, 0]
            ax[0].plot(root[:, 0], root[:, 2])
            ax[0].set_title("root XZ")
            ax[0].axis("equal")
            for j in (0, 9, 12, 15, 16, 17):
                ax[1].plot(joints[:, j, 1], label=str(j))
            ax[1].set_title("joint Y")
            ax[1].legend(fontsize=6)
            png = os.path.join(os.path.dirname(out), "gvhmr_check.png")
            fig.tight_layout()
            fig.savefig(png, dpi=120)
            print("WROTE", png)
        except Exception as e:
            print("plot skipped:", repr(e))


if __name__ == "__main__":
    main()
