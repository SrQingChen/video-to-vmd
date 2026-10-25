"""Bake a VMD into the portable safe GT used by the synthetic closed loop.

Default semantics: MMD (translation only on movable bones) + IK off.  The
result replays identically in MMD and in Blender (verified by re-evaluating
and by MMD Tools import).

Usage:
  python scripts/bake_gt_vmd.py --model si --vmd "testdata/R2_王牌竞速/动作＋表情.vmd" \
      --out outputs/gt/si_R2_safe.vmd
"""
import argparse
import json
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from vtv.mmd.bake import bake_evaluated_vmd, select_bake_bones
from vtv.mmd.evaluate import VmdEvaluator
from vtv.mmd.pmx import parse_pmx
from vtv.mmd.vmd import parse_vmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "models.json"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--vmd", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--solve-ik", action="store_true")
    ap.add_argument("--blender-semantics", action="store_true",
                    help="apply translation on non-movable bones too")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)["models"][args.model]
    with open(os.path.join(ROOT, "profiles", f"{args.model}.json"),
              encoding="utf-8") as f:
        profile = json.load(f)
    model = parse_pmx(cfg["pmx"])
    vmd = parse_vmd(args.vmd)
    out = args.out or os.path.join(ROOT, "outputs", "gt",
                                   f"{args.model}_{os.path.basename(args.vmd).split('.')[0]}_safe.vmd")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    res = bake_evaluated_vmd(
        model, vmd, profile, out,
        solve_ik=args.solve_ik,
        apply_all_translations=args.blender_semantics,
    )
    print("BAKE", json.dumps(res, ensure_ascii=False))

    # replay check: original (same semantics) vs baked
    ev1 = VmdEvaluator(model, vmd, solve_ik=args.solve_ik,
                       force_ik_off=not args.solve_ik,
                       apply_all_translations=args.blender_semantics)
    baked = parse_vmd(out)
    ev2 = VmdEvaluator(model, baked, force_ik_off=True,
                       apply_all_translations=args.blender_semantics)
    name2i = {b["name"]: i for i, b in enumerate(model["bones"])}
    errs = []
    for j, bn in profile["standard_map"].items():
        if bn not in name2i:
            continue
        i = name2i[bn]
        for f in range(min(vmd["max_frame"], baked["max_frame"]) + 1):
            errs.append(math.dist(ev1.frame(f)["world_pos"][i],
                                  ev2.frame(f)["world_pos"][i]))
    print("REPLAY max_err_units", max(errs) if errs else None,
          "bones", len(res["bones"]), "frames", res["frames"])


if __name__ == "__main__":
    main()
