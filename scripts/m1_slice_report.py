"""M1 vertical-slice report: absolute stability stats for the exported VMDs.

There is no GT for T1/T2/T5, so this does not score accuracy.  It reports the
quantities the M1 gate asks for, measured on the exported VMD through the same
MMD evaluator the closed loop uses:

  * frame count / duration / written bones / IK-off present
  * contact fraction, slide (net drift while planted), penetration, toe heights
  * jerk RMS / p99 / MAD spike count, bone-length CV
  * root path length, root vertical range, per-frame finiteness
  * slow-factor timing consistency across clips (all must be 242 frames)

Usage:
  python scripts/m1_slice_report.py --vmd outputs/t5_m1/t5_242.vmd --model si \
      --out outputs/t5_m1/report.json
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from vtv.metrics.core import compute_metrics, pose_track  # noqa: E402
from vtv.mmd.pmx import parse_pmx  # noqa: E402
from vtv.mmd.vmd import parse_vmd  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "models.json"))
    ap.add_argument("--model", default="si")
    ap.add_argument("--vmd", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--contact-margin", type=float, default=0.01)
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8"))["models"][args.model]
    profile = json.load(open(os.path.join(ROOT, "profiles", f"{args.model}.json"),
                             encoding="utf-8"))
    model = parse_pmx(cfg["pmx"])
    vmd = parse_vmd(args.vmd)
    frames = list(range(vmd["max_frame"] + 1))
    tr = pose_track(model, vmd, profile, frames=frames)
    # self comparison: the *difference* components are zero by construction,
    # the absolute foot/jerk/bone-length blocks are what we report.
    m = compute_metrics(tr, tr, profile, contact_margin=args.contact_margin)
    foot = m["foot"]
    height = m["height"]

    root = tr["world"]["root"]
    root_path = float(np.linalg.norm(np.diff(root[:, [0, 2]], axis=0), axis=1).sum())
    finite = all(np.isfinite(v).all() for v in tr["world"].values())
    toe_min = {s: v["min_height_units"] for s, v in foot.items()}
    toe_max = {s: v["max_height_units"] for s, v in foot.items()}
    bl = m["bone_lengths_pred"]
    cv_max = max((v["cv"] or 0.0) for v in bl.values()) if bl else None

    ik_names = [n for n in (vmd.get("ik_frames") or [{}])[0].get("ik", {})] \
        if vmd.get("ik_frames") else []
    ik_off = bool(vmd.get("ik_frames")) and all(
        v is False for v in vmd["ik_frames"][0]["ik"].values())

    report = {
        "vmd": os.path.abspath(args.vmd),
        "model": args.model,
        "height_units": round(height, 3),
        "frames": len(frames),
        "duration_s": round(len(frames) / 30.0, 3),
        "tracks": len(vmd["bones"]),
        "bones": sorted(vmd["bones"]),
        "ik_off_frame0": ik_off,
        "ik_bones": ik_names,
        "finite": bool(finite),
        "contact": {
            "margin_pct_height": 100.0 * args.contact_margin,
            "fraction": {s: v["contact_fraction_gt"] for s, v in foot.items()},
            "thr_units": {s: v["contact_thr_units"] for s, v in foot.items()},
            "intervals": {s: v["contact_intervals"] for s, v in foot.items()},
            "slide_units_per_s": {s: v["slide_pred_units_per_s"]
                                  for s, v in foot.items()},
            "height_mae_note": "GT is the same track here; use slide/penetration",
        },
        "penetration_units": {s: v["penetration_units"] for s, v in foot.items()},
        "penetration_pct_height": round(100.0 * max(
            (v["penetration_units"] for v in foot.values()), default=0.0) / height, 4),
        "toe_min_units": toe_min,
        "toe_max_units": toe_max,
        "jerk": m["jerk_pred"],
        "bone_length_cv_max": cv_max,
        "root": {
            "path_length_units": round(root_path, 3),
            "path_pct_height": round(100.0 * root_path / height, 3),
            "y_min": round(float(root[:, 1].min()), 3),
            "y_max": round(float(root[:, 1].max()), 3),
            "y_range_units": round(float(root[:, 1].max() - root[:, 1].min()), 3),
        },
        "per_bone": m["per_bone"],
        "summary_S_note": "not computed for a self comparison (all diffs are 0)",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: report[k] for k in
                      ("frames", "duration_s", "tracks", "ik_off_frame0",
                       "contact", "penetration_pct_height", "toe_min_units",
                       "jerk", "bone_length_cv_max", "root")},
                     ensure_ascii=False, indent=1))
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
