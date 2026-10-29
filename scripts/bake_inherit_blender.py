"""Bake MMD inherit bones onto a Blender rig and verify against the PMX evaluator.

Run headless:
  blender --background <scene.blend> --python scripts/bake_inherit_blender.py -- \
    --model si --vmd outputs/t5_m1/t5_242.vmd --frames 1:242 \
    --report outputs/t5_m1/inherit_bake.json

What it proves:
  * every 付与 bone (足D/ひざD/足首D, 肩C, 捩1, eye-aim) gets explicit pose keys;
  * after the bake Blender's deformation matrices equal the PMX evaluator's
    (position < 1e-3 unit, rotation < 1e-3 deg);
  * the D chain flexion measured on the Blender rig is non-zero where the VMD
    says so (the leg really bends).

The VMD is only read, never written: MMD applies 付与 itself, so adding D-bone
tracks to the MMD asset would double-apply the inheritance.
"""
import argparse
import json
import os
import sys

import bpy
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mmd_blender  # noqa: E402


def parse_args():
    argv = sys.argv
    rest = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "models.json"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--vmd", required=True)
    ap.add_argument("--frames", default=None,
                    help="Blender frames a:b (default: action range)")
    ap.add_argument("--verify-frames", type=int, default=8)
    ap.add_argument("--report", default=None)
    ap.add_argument("--save-blend", default=None)
    ap.add_argument("--no-bake", action="store_true",
                    help="only measure (shows the pre-fix behaviour)")
    return ap.parse_args(rest)


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)["models"][args.model]
    arm = bpy.data.objects[cfg["armature"]]

    act = mmd_blender.import_vmd(arm, args.vmd)
    model, vmd = mmd_blender.load_model_and_vmd(cfg, args.vmd)
    print("IMPORTED", act.name if act else None, "range",
          list(act.frame_range) if act else None)

    if args.frames:
        start, end = [int(x) for x in args.frames.split(":")]
    else:
        start, end = int(act.frame_range[0]), int(act.frame_range[1])
    bpy.context.scene.frame_start, bpy.context.scene.frame_end = start, end
    frames = list(range(start, end + 1))
    sample = sorted({int(round(x)) for x in
                     np.linspace(start, end, min(8, end - start + 1))})

    report = {
        "model": args.model,
        "vmd": os.path.abspath(args.vmd),
        "blend": bpy.data.filepath,
        "action": act.name if act else None,
        "frames": [start, end],
        "measure_before": mmd_blender.measure_knees(arm, sample),
    }

    if not args.no_bake:
        report["bake"] = mmd_blender.bake_inherit(arm, model, vmd, frames)
        report["d_chain_vs_fk"] = mmd_blender.d_chain_vs_fk(model, vmd, sample)
        n = args.verify_frames
        vs = sample if n <= 0 else sorted(
            {int(round(x)) for x in np.linspace(start, end, n)})
        report["verify"] = mmd_blender.verify_deformations(
            arm, model, vmd, vs)
        report["measure_after"] = mmd_blender.measure_knees(arm, sample)
        if args.save_blend:
            bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.save_blend))

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
    print("BAKE_REPORT", json.dumps(report, ensure_ascii=False))


main()
