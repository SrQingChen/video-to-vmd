"""Canonical motion -> target PMX VMD (M1 adapter entry point).

Two sources:

  * GVHMR npz (the real pipeline):
        run_adapter.py --npz outputs/gvhmr/T5/gvhmr_result.npz --model si \
            --slow-factor 1.0 --out outputs/t5_m1/t5_242.vmd

  * An existing VMD on a PMX (round-trip / cross-model GT):
        run_adapter.py --vmd outputs/gt/si_R2_safe.vmd --model tiflos \
            --out outputs/adapter/si_R2_to_tiflos.vmd

Outputs the VMD plus ``<out>.json`` with the scale, ground shift, per-joint
rest compensation table, root delta range and frame mapping, so a bad fit can
be diagnosed to a bone / rest direction instead of "looks wrong".

``--slow-factor`` is the video's slow-motion factor: T1 is 0.25x (968 -> 242),
T2 0.5x (484 -> 242), T5 1.0 (242 -> 242).  The VMD always runs at 30 fps.
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402

from vtv import canonical_body as CB  # noqa: E402
from vtv.adapter import (adapt, canonical_from_pmx, canonical_from_smplx)  # noqa: E402
from vtv.mmd.pmx import parse_pmx  # noqa: E402
from vtv.mmd.vmd import parse_vmd  # noqa: E402
from vtv.mmd.writer import write_vmd  # noqa: E402


def smplx_rest_joints(npz_path, betas=None):
    """Rest joints (zero pose, clip betas) in the SMPL-X world frame, metres."""
    sys.path.insert(0, os.path.join(ROOT, "shim"))
    sys.path.insert(0, os.path.join(ROOT, "third_party", "GVHMR"))
    import torch
    from hmr4d.utils.smplx_utils import make_smplx

    d = np.load(npz_path)
    if betas is None:
        betas = d["smpl_params_global/betas"][:1]
    model = make_smplx("supermotion", num_betas=10)
    with torch.no_grad():
        out = model(betas=torch.tensor(np.asarray(betas), dtype=torch.float32),
                    body_pose=torch.zeros(1, 63), global_orient=torch.zeros(1, 3),
                    transl=torch.zeros(1, 3))
    J = out.joints[0, :22].numpy()
    rest = {j: [float(c) for c in J[i]] for j, i in CB.SMPLX_INDEX.items()}
    return rest, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "models.json"))
    ap.add_argument("--model", default="si")
    ap.add_argument("--npz", default=None, help="GVHMR result npz (source)")
    ap.add_argument("--vmd", default=None, help="source VMD on a PMX (source)")
    ap.add_argument("--source-model", default=None,
                    help="model key for --vmd (default: --model)")
    ap.add_argument("--side", default="global", choices=["global", "incam"],
                    help="SMPL-X world parameters to use (global = MMD-like world)")
    ap.add_argument("--slow-factor", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", default=None, help="a:b source frames (inclusive)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--twist-split", action="store_true",
                    help="experimental: write main twist bones (see adapter docstring)")
    ap.add_argument("--no-ground", action="store_true")
    ap.add_argument("--stage-yaw", type=float, default=None,
                    help="yaw applied to the whole character; default 180 for "
                         "SMPL-X sources (their rest template faces +Z while the "
                         "PMX models face -Z) and 0 for VMD sources")
    args = ap.parse_args()
    if not args.npz and not args.vmd:
        raise SystemExit("need --npz or --vmd")

    cfg = json.load(open(args.config, encoding="utf-8"))["models"]
    profile = json.load(open(os.path.join(ROOT, "profiles", f"{args.model}.json"),
                             encoding="utf-8"))
    model = parse_pmx(cfg[args.model]["pmx"])
    frames = None
    if args.frames:
        a, b = [int(x) for x in args.frames.split(":")]
        frames = list(range(a, b + 1))

    if args.vmd:
        src_key = args.source_model or args.model
        src_cfg = cfg[src_key]
        src_profile = json.load(open(
            os.path.join(ROOT, "profiles", f"{src_key}.json"), encoding="utf-8"))
        src_model = parse_pmx(src_cfg["pmx"])
        src_vmd = parse_vmd(args.vmd)
        motion = canonical_from_pmx(src_model, src_profile, src_vmd, frames=frames)
        source_info = {"kind": "vmd", "model": src_key, "vmd": os.path.abspath(args.vmd)}
    else:
        d = np.load(args.npz)
        prefix = f"smpl_params_{args.side}"
        params = {k: d[f"{prefix}/{k}"] for k in
                  ("body_pose", "global_orient", "transl")}
        if frames:
            params = {k: v[frames[0]:frames[-1] + 1] for k, v in params.items()}
        rest, _ = smplx_rest_joints(args.npz, betas=d["smpl_params_global/betas"][:1])
        motion = canonical_from_smplx(params, rest, slow_factor=args.slow_factor)
        source_info = {"kind": "gvhmr", "npz": os.path.abspath(args.npz),
                       "side": args.side, "slow_factor": args.slow_factor}

    if args.stage_yaw is None:
        stage_yaw = 0.0 if args.vmd else 180.0
    else:
        stage_yaw = args.stage_yaw
    res = adapt(motion, model, profile,
                twist_split=args.twist_split,
                ground_calibrate=not args.no_ground,
                stage_yaw_deg=stage_yaw,
                max_frames=args.max_frames)

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_vmd(out, model.get("name", ""), res["bone_frames"], {}, res["ik_frames"])

    root_rows = res["bone_frames"][profile["standard_map"]["root"]]
    xs = np.asarray([r["pos"] for r in root_rows])
    report = {
        "out": out,
        "model": args.model,
        "source": source_info,
        "frames": res["meta"]["frames"],
        "frame_range": res["meta"]["frame_range"],
        "written_bones": res["meta"]["written_bones"],
        "bones": res["bones"],
        "twist_bones": res["twist_bones"],
        "scale": round(res["scale"], 6),
        "src_span_m": None if res["meta"]["src_span_m"] is None
        else round(res["meta"]["src_span_m"], 4),
        "tgt_span_m": None if res["meta"]["tgt_span_m"] is None
        else round(res["meta"]["tgt_span_m"], 4),
        "ground_shift_units": round(res["ground_shift_units"], 5),
        "toe_min_units": None if res["toe_min_units"] is None
        else round(res["toe_min_units"], 4),
        "rest_toe_units": None if res["target_rest_toe_units"] is None
        else round(res["target_rest_toe_units"], 4),
        "root_delta_units": {
            "mean": [round(float(v), 4) for v in xs.mean(0)],
            "min": [round(float(v), 4) for v in xs.min(0)],
            "max": [round(float(v), 4) for v in xs.max(0)],
            "start": [round(float(v), 4) for v in xs[0]],
            "end": [round(float(v), 4) for v in xs[-1]],
        },
        "rest_compensation_deg": res["rest_compensation_deg"],
        "ik_frames": res["ik_frames"],
        "source_meta": motion.get("meta"),
        "source_segment_frame": motion.get("segment_frame"),
        "rest_pelvis_m": [round(float(c), 4) for c in motion["rest_pelvis"]],
        "stage_yaw_deg": stage_yaw,
    }
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("WROTE", out)
    print(json.dumps({k: report[k] for k in
                      ("frames", "written_bones", "scale", "ground_shift_units",
                       "root_delta_units")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
