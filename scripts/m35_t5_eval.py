"""T5 no-GT non-regression check for the M3.5 feedback network.

T5 has no GT, so this is a *visual/stability* gate only: apply the network to
the M3 deliverable, verify the VMD certificate (53 tracks, IK off, morph names),
and compare GT-free stability (jerk, foot slide/penetration, finger and morph
step sizes) before vs after.  Optionally render a short before/after clip at a
fixed camera for visual inspection.

Usage:
  python scripts/m35_t5_eval.py --ckpt outputs/m35/net/m35_best.pt \
      --render --models si
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vtv import feedback_data as FD  # noqa: E402
from vtv.eval_helpers import (  # noqa: E402
    gate_residual, load_net, predict, smooth_residual, stability,
)
from vtv.metrics.core import pose_track  # noqa: E402
from vtv.mmd.pmx import parse_pmx  # noqa: E402
from vtv.mmd.vmd import parse_vmd  # noqa: E402
from vtv.mmd.writer import write_vmd  # noqa: E402

BODY = {
    "si": "outputs/m25/t5_v01_offline.vmd",
    "luoxi": "outputs/m3/t5_body_luoxi.vmd",
    "tiflos": "outputs/m3/t5_body_tiflos.vmd",
}
BEFORE = {
    "si": "outputs/m3/t5_si.vmd",
    "luoxi": "outputs/m3/t5_luoxi.vmd",
    "tiflos": "outputs/m3/t5_tiflos.vmd",
}
BLENDER = "D:/blender/blender.exe"
BLEND = "E:/zcodeAI项目/AIblender/三人_王牌竞速_20261024.blend"
FFMPEG = "D:/ffmpeg/bin/ffmpeg.exe"


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "outputs/m35/net/m35_best.pt"))
    ap.add_argument("--out", default=str(ROOT / "outputs/m35/t5"))
    ap.add_argument("--models", default="si,luoxi,tiflos")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--frames", default="70:150")
    ap.add_argument("--res", default="720x1280")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--apply-root", action="store_true")
    ap.add_argument("--smooth", type=int, default=3)
    ap.add_argument("--gain", type=float, default=0.85,
                    help="torso residual gain (chosen by the held-out sweep)")
    ap.add_argument("--arm-gain", type=float, default=0.2,
                    help="arm residual gain (the T5-safe default)")
    ap.add_argument("--with-hands", action="store_true",
                    help="apply hand residual too (fails the T5 step gate)")
    ap.add_argument("--with-face", action="store_true",
                    help="apply face residual too (fails the T5 morph-step gate)")
    ap.add_argument("--no-arms", action="store_true")
    args = ap.parse_args()

    net, ck = load_net(args.ckpt)
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    g = np.load(ROOT / "outputs/gvhmr/T5/gvhmr_result.npz")
    kp = g["kp2d"] if "kp2d" in g else torch.load(
        ROOT / "outputs/gvhmr/T5/preprocess/vitpose.pt", map_location="cpu").numpy()
    bbx = g["bbx_xys"]
    meta = load_json(ROOT / "outputs/gvhmr/T5/gvhmr_meta.json")
    wh = (int(meta["width"]), int(meta["height"]))
    hand = np.load(ROOT / "outputs/m3/t5/hand.npz")
    face = np.load(ROOT / "outputs/m3/t5/face.npz")

    report = {"ckpt": str(args.ckpt), "apply_root": bool(args.apply_root),
              "models": {}}
    for model_key in args.models.split(","):
        cfg = load_json(ROOT / "configs/models.json")["models"][model_key]
        profile = load_json(ROOT / f"profiles/{model_key}.json")
        model = parse_pmx(cfg["pmx"])
        body = parse_vmd(str(ROOT / BODY[model_key]))
        before = parse_vmd(str(ROOT / BEFORE[model_key]))
        n = int(body["max_frame"]) + 1
        frames = list(range(n))
        tr = pose_track(model, body, profile, frames=frames)
        feats = FD.build_features(
            tr, float(profile["height"]), kp2d=kp[:n], bbx_xys=bbx[:n],
            hand_angles=hand["angles"][:n], hand_level=hand["level"][:n],
            hand_score=hand["score"][:n], hand_detect=hand["detected"][:n],
            face_weights=face["weights"][:n], face_level=face["level"][:n],
            face_detect=face["detected"][:n], profile=profile,
            adapter_scale=1.0, img_wh=wh)
        res = predict(net, ck, feats)
        if not args.apply_root:
            res[:, FD.ROOT_SLICE] = 0.0
        for i, bone in enumerate(FD.BODY_FACTOR_NAMES):
            is_arm = bone.startswith(("clavicle", "upperarm", "forearm"))
            res[:, 2 + 3 * i:5 + 3 * i] *= float(
                args.arm_gain if is_arm else args.gain)
        if not args.with_hands:
            res[:, FD.HAND_SLICE] = 0.0
        if not args.with_face:
            res[:, FD.FACE_SLICE] = 0.0
        if args.no_arms:
            for i, bone in enumerate(FD.BODY_FACTOR_NAMES):
                if bone.startswith(("clavicle", "upperarm", "forearm")):
                    res[:, 2 + 3 * i:5 + 3 * i] = 0.0
        sups = (ck.get("hand_supported"), ck.get("face_supported"))
        res = gate_residual(res, hand["level"][:n], face["level"][:n],
                            FD.face_target_names(profile), *sups)
        res = smooth_residual(res, hand["level"][:n], face["level"][:n],
                              win=args.smooth)
        res = gate_residual(res, hand["level"][:n], face["level"][:n],
                            FD.face_target_names(profile), *sups)
        corr_body = FD.apply_body_residual(body, model, profile, res)
        ha = FD.apply_hand_residual(hand["angles"][:n], res[:, FD.HAND_SLICE])
        fw = FD.apply_face_residual(face["weights"][:n], res[:, FD.FACE_SLICE])
        vmd = FD.assemble_vmd(corr_body, profile, ha, fw, n_frames=n)
        after_p = out_dir / f"t5_{model_key}_fb.vmd"
        write_vmd(str(after_p), vmd.get("model_name", ""), vmd["bones"],
                  vmd["morphs"], vmd["ik_frames"])
        written = parse_vmd(str(after_p))

        cert = {
            "bone_tracks": len(written["bones"]),
            "finger_tracks": sum(1 for k in profile["finger_map"].values()
                                 if k in written["bones"]),
            "morph_tracks": sorted(written["morphs"]),
            "ik_off": bool(written["ik_frames"]) and all(
                not any(fr["ik"].values()) for fr in written["ik_frames"]),
            "morphs_in_profile": all(nm in {m["name"] for m in profile["morphs"]}
                                     for nm in written["morphs"]),
            "quat_norm_max_err": max(abs(math.sqrt(sum(c * c for c in k["rot"])) - 1.0)
                                     for ks in written["bones"].values() for k in ks),
        }
        cert["ok"] = (cert["bone_tracks"] == 53 and cert["finger_tracks"] == 30
                      and cert["ik_off"] and cert["morphs_in_profile"]
                      and cert["quat_norm_max_err"] < 1e-6)
        sb = stability(model, profile, before, n)
        sa = stability(model, profile, written, n)
        hb = hand["level"][:n] <= 1
        fb = face["level"][:n] <= 1
        rec = {
            "vmd": str(after_p.relative_to(ROOT)),
            "cert": cert,
            "before": sb, "after": sa,
            "hand_frames_modified": int(hb.sum()),
            "face_frames_modified": int(fb.sum()),
            "hand_residual_mean_abs": float(np.abs(
                res[:, FD.HAND_SLICE]).mean()),
            "face_residual_mean_abs": float(np.abs(
                res[:, FD.FACE_SLICE]).mean()),
            "body_residual_mean_abs": float(np.abs(
                res[:, FD.ROT_SLICE]).mean()),
            "root_residual_mean_abs": float(np.abs(
                res[:, FD.ROOT_SLICE]).mean()),
        }
        checks = {
            "jerk_not_worse": sa["jerk_rms"] <= sb["jerk_rms"] * 1.05 + 1e-6,
            "slide_not_worse": sa["foot_slide"] <= sb["foot_slide"] * 1.05 + 1e-3,
            "penetration_not_worse": sa["foot_penetration"]
            <= sb["foot_penetration"] + 0.02,
            "finger_step_not_worse": sa["finger_step_p95"]
            <= sb["finger_step_p95"] * 1.10 + 0.1,
            "morph_step_not_worse": sa["morph_step_max"]
            <= sb["morph_step_max"] + 0.05,
        }
        rec["checks"] = checks
        rec["ok"] = bool(cert["ok"] and all(checks.values()))
        report["models"][model_key] = rec
        print(f"{model_key}: cert={cert['ok']} jerk {sb['jerk_rms']:.0f}->{sa['jerk_rms']:.0f} "
              f"slide {sb['foot_slide']:.3f}->{sa['foot_slide']:.3f} "
              f"pen {sb['foot_penetration']:.3f}->{sa['foot_penetration']:.3f} "
              f"finger p95 {sb['finger_step_p95']:.2f}->{sa['finger_step_p95']:.2f} "
              f"morph {sb['morph_step_max']:.2f}->{sa['morph_step_max']:.2f} "
              f"checks={checks}")

    report["ok"] = all(v["ok"] for v in report["models"].values())
    (out_dir / "t5_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    if args.render:
        a, b = [int(x) for x in args.frames.split(":")]
        for tag, vmd in (("before", BEFORE["si"]), ("after", out_dir / "t5_si_fb.vmd")):
            dst = out_dir / "render" / tag
            dst.mkdir(parents=True, exist_ok=True)
            cmd = [BLENDER, "--background", BLEND, "--python",
                   str(ROOT / "scripts/render_synth.py"), "--",
                   "--model", "si", "--vmd", str(ROOT / vmd if not Path(vmd).is_absolute() else vmd),
                   "--view", "front", "--out", str(dst),
                   "--frames", args.frames, "--res", args.res,
                   "--samples", str(args.samples)]
            before_meta = out_dir / "render" / "before" / "meta.json"
            if tag == "after" and before_meta.exists():
                cmd += ["--camera-from", str(before_meta)]
            with open(dst / "render.log", "w", encoding="utf-8") as f:
                r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
            if r.returncode != 0:
                raise SystemExit(f"render failed: {dst}")
            subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", "30",
                            "-start_number", str(a),
                            "-i", str(dst / "frames" / "frame_%04d.png"),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
                            str(dst / f"t5_{tag}_{a}-{b}.mp4")], check=True)
        print("RENDERED", out_dir / "render")
    print("T5_OK", report["ok"], "->", out_dir / "t5_report.json")


if __name__ == "__main__":
    main()
