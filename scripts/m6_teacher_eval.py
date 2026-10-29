"""M6 teacher reachability: v1 vs v2 shoulder-chain bounds on held-out R3.

This is an oracle check (GT is used to build the teacher), not a model result.
It answers whether widening the clavicle/upperarm/forearm bounds actually
removes the ~24 deg shoulder chain residual that M3.5 reported, and how much
of the target is still clipped.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vtv import feedback_data as FD  # noqa: E402
from vtv.metrics.core import pose_track  # noqa: E402
from vtv.mmd.pmx import parse_pmx  # noqa: E402
from vtv.mmd.vmd import parse_vmd  # noqa: E402


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def run_clip(cid, variant):
    motion, model_key = cid.split("_")[:2]
    cfg = load_json(ROOT / "configs/models.json")["models"][model_key]
    profile = load_json(ROOT / f"profiles/{model_key}.json")
    model = parse_pmx(cfg["pmx"])
    gt = parse_vmd(str(ROOT / f"outputs/m35/gt/{motion}/{model_key}.vmd"))
    pred = parse_vmd(str(ROOT / f"outputs/m35/pred/{cid}/ik.vmd"))
    n = min(int(gt["max_frame"]), int(pred["max_frame"])) + 1
    frames = list(range(n))
    gt_t = pose_track(model, gt, profile, frames=frames)
    pr_t = pose_track(model, pred, profile, frames=frames)
    bounds = FD.ROT_BOUND_V2 if variant == "v2" else FD.ROT_BOUND
    res = FD.extract_body_residual(gt_t, pr_t, bounds=bounds)
    corr = FD.apply_body_residual(pred, model, profile, res["target"])
    co_t = pose_track(model, corr, profile, frames=frames)
    b = FD.body_relative_errors(gt_t, pr_t)
    c = FD.body_relative_errors(gt_t, co_t)
    # how much of the teacher was clipped (per bone)
    raw = FD.extract_body_residual(gt_t, pr_t, bounds={k: 1e9 for k in bounds})
    clipped = (np.abs(raw["target"][:, FD.ROT_SLICE])
               > np.tile(FD.residual_limits(variant)[FD.ROT_SLICE],
                         (n, 1)))
    per_bone = {}
    for i, bone in enumerate(FD.BODY_FACTOR_NAMES):
        per_bone[bone] = float(clipped[:, 3 * i:3 * i + 3].mean())
    return {"clip": cid, "variant": variant, "frames": n,
            "before": {k: b.get(k) for k in
                       ("body_relative_main_deg", "group_chest_deg",
                        "group_upperarm_deg", "group_forearm_deg")},
            "teacher_after": {k: c.get(k) for k in
                              ("body_relative_main_deg", "group_chest_deg",
                               "group_upperarm_deg", "group_forearm_deg")},
            "clipped_fraction": per_bone}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default=None)
    ap.add_argument("--out", default=str(ROOT / "outputs/m6/teacher"))
    args = ap.parse_args()
    if args.clips:
        clips = [c for c in args.clips.split(",") if c]
    else:
        clips = [f"R3_{m}_{v}" for m in FD.ALL_MODELS for v in FD.ALL_VIEWS]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = []
    for cid in clips:
        r1 = run_clip(cid, "v1")
        r2 = run_clip(cid, "v2")
        recs.append({"clip": cid, "v1": r1, "v2": r2})
        print(f"{cid}: main v1 {r1['teacher_after']['body_relative_main_deg']:.2f} "
              f"v2 {r2['teacher_after']['body_relative_main_deg']:.2f} "
              f"(before {r1['before']['body_relative_main_deg']:.2f})",
              flush=True)
    agg = {}
    for k in ("body_relative_main_deg", "group_chest_deg",
              "group_upperarm_deg", "group_forearm_deg"):
        agg[k] = {}
        for v in ("v1", "v2"):
            agg[k][v] = float(np.mean([r[v]["teacher_after"][k] for r in recs]))
    (out / "report.json").write_text(
        json.dumps({"clips": recs, "aggregate": agg}, ensure_ascii=False,
                   indent=1), encoding="utf-8")
    print("AGG", json.dumps(agg, indent=1))
    print("WROTE", out / "report.json")


if __name__ == "__main__":
    main()
