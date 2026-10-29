"""M6 T5 no-GT gate for the v2 feedback (one-shot or two-pass).

Uses the M4 production T5 artifacts (per-model body/evidence/camera), applies
the v2 checkpoint(s) with the same policy as the pipeline, writes the final
VMD, and compares GT-free stability before/after: jerk, foot slide,
penetration, finger step p95 and morph step.  T5 is never used for training or
model selection.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vtv import feedback_apply as FA  # noqa: E402
from vtv import feedback_apply_v2 as FAV2  # noqa: E402
from vtv import feedback_data as FD  # noqa: E402
from vtv.eval_helpers import stability  # noqa: E402
from vtv.mmd.pmx import parse_pmx  # noqa: E402
from vtv.mmd.vmd import parse_vmd  # noqa: E402
from vtv.mmd.writer import write_vmd  # noqa: E402


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def fit(a, n):
    out = np.zeros((n,) + np.asarray(a).shape[1:], np.asarray(a).dtype)
    m = min(n, len(a))
    out[:m] = np.asarray(a)[:m]
    return out


def evidence(n, model_key):
    base = ROOT / "outputs/m4/t5"
    g = np.load(base / "video/gvhmr/gvhmr_result.npz")
    gm = load_json(base / "video/gvhmr/gvhmr_meta.json")
    hand = np.load(base / "evidence/hand/hand.npz")
    face = np.load(base / "evidence/face/face.npz")
    am = load_json(base / f"models/{model_key}/adapter.vmd.json")
    return {
        "kp2d": fit(g["kp2d"], n), "bbx_xys": fit(g["bbx_xys"], n),
        "img_wh": (int(gm["width"]), int(gm["height"])),
        "adapter_scale": float(am.get("scale", 1.0)),
        "hand": {"angles": fit(hand["angles"], n), "level": fit(hand["level"], n),
                 "score": fit(hand["score"], n),
                 "detected": fit(hand["detected"], n)},
        "face": {"weights": fit(face["weights"], n), "level": fit(face["level"], n),
                 "detected": fit(face["detected"], n)},
    }


def cert(vmd, profile):
    c = {
        "bone_tracks": len(vmd["bones"]),
        "finger_tracks": sum(1 for k in profile["finger_map"].values()
                             if k in vmd["bones"]),
        "morph_tracks": sorted(vmd["morphs"]),
        "ik_off": bool(vmd["ik_frames"]) and all(
            not any(fr["ik"].values()) for fr in vmd["ik_frames"]),
        "morphs_in_profile": all(
            nm in {m["name"] for m in profile["morphs"]} for nm in vmd["morphs"]),
        "quat_norm_max_err": max(
            abs(math.sqrt(sum(c * c for c in k["rot"])) - 1.0)
            for ks in vmd["bones"].values() for k in ks),
    }
    c["ok"] = (c["bone_tracks"] == 53 and c["finger_tracks"] == 30
               and c["ik_off"] and c["morphs_in_profile"]
               and c["quat_norm_max_err"] < 1e-6)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt2", default=None, help="second residual pass")
    ap.add_argument("--out", default=str(ROOT / "outputs/m6/t5"))
    ap.add_argument("--models", default="si,luoxi,tiflos")
    ap.add_argument("--mode", default="default")
    ap.add_argument("--overrides", default=None, help="JSON policy overrides")
    ap.add_argument("--no-arm-torso-check", action="store_true")
    ap.add_argument("--jerk-tol", type=float, default=None)
    args = ap.parse_args()
    cfg = FA.resolve_config(args.mode, overrides=(
        json.loads(args.overrides) if args.overrides else None))
    fallback = dict(FA.DEFAULT_FALLBACK)
    if args.no_arm_torso_check:
        fallback["arm_torso_check"] = False
    if args.jerk_tol is not None:
        fallback["jerk_tol"] = float(args.jerk_tol)
    fb1 = FA.load_feedback_model(args.ckpt)
    fb2 = FA.load_feedback_model(args.ckpt2) if args.ckpt2 else None
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    report = {"ckpt": args.ckpt, "ckpt2": args.ckpt2, "mode": args.mode,
              "overrides": json.loads(args.overrides) if args.overrides else None,
              "models": {}}
    for model_key in args.models.split(","):
        cfg_m = load_json(ROOT / "configs/models.json")["models"][model_key]
        profile = load_json(ROOT / f"profiles/{model_key}.json")
        model = parse_pmx(cfg_m["pmx"])
        base = ROOT / "outputs/m4/t5"
        body = parse_vmd(str(base / f"models/{model_key}/ik.vmd"))
        before = parse_vmd(str(base / f"models/{model_key}/before.vmd"))
        n = int(body["max_frame"]) + 1
        ev = evidence(n, model_key)
        if fb2 is not None:
            res = FAV2.apply_feedback_v2(body, model, profile, fb1, fb2, ev, cfg,
                                          fallback)
        else:
            res = FA.apply_feedback(body, model, profile, fb1, ev, cfg, fallback)
        vmd = res["vmd"]
        p = out_root / f"t5_{model_key}.vmd"
        write_vmd(str(p), vmd.get("model_name", ""), vmd["bones"],
                  vmd["morphs"], vmd["ik_frames"])
        written = parse_vmd(str(p))
        sb = stability(model, profile, before, n)
        sa = stability(model, profile, written, n)
        checks = {
            "jerk": sa["jerk_rms"] <= sb["jerk_rms"] * 1.05 + 1e-6,
            "slide": sa["foot_slide"] <= sb["foot_slide"] * 1.05 + 1e-3,
            "penetration": sa["foot_penetration"]
            <= sb["foot_penetration"] + 0.02,
            "finger_step": sa["finger_step_p95"]
            <= sb["finger_step_p95"] * 1.10 + 0.1,
            "morph_step": sa["morph_step_max"]
            <= sb["morph_step_max"] + 0.05,
        }
        rec = {"vmd": str(p.relative_to(ROOT)), "frames": n,
               "cert": cert(written, profile),
               "before": sb, "after": sa,
               "checks": checks, "ok": all(checks.values()) and cert(written, profile)["ok"],
               "segments_fallback": res["segments_fallback"],
               "whole_clip_fallback": res["whole_clip_fallback"],
               "handface_fallback": res["handface_fallback"],
               "residual_mean_abs": res["residual_mean_abs"]}
        report["models"][model_key] = rec
        print(model_key, json.dumps({
            "jerk": f"{sb['jerk_rms']:.1f}->{sa['jerk_rms']:.1f} "
                    f"({100*(sa['jerk_rms']/max(sb['jerk_rms'],1e-9)-1):+.2f}%)",
            "slide": f"{sb['foot_slide']:.4f}->{sa['foot_slide']:.4f}",
            "finger_p95": f"{sb['finger_step_p95']:.2f}->{sa['finger_step_p95']:.2f}",
            "morph": f"{sb['morph_step_max']:.2f}->{sa['morph_step_max']:.2f}",
            "ok": rec["ok"], "checks": checks}, ensure_ascii=False), flush=True)
    report["ok"] = all(v["ok"] for v in report["models"].values())
    (out_root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("T5_OK", report["ok"], "WROTE", out_root / "report.json")


if __name__ == "__main__":
    main()
