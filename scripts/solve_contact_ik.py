"""Contact-gated 2-bone leg IK for a predicted VMD (M2.2).

Examples:
  python scripts/solve_contact_ik.py --vmd outputs/m2/vmd/front_2d_clean.vmd \
      --model si --out outputs/m2/vmd/front_ik.vmd --gt outputs/gt/si_R2_safe.vmd

The output is a normal 23-track IK-off VMD; ``--gt`` only scores before/after
for the synthetic closed loop and is never read by the solver.  ``--kp2d``
accepts the ViTPose evidence (``preprocess/vitpose.pt`` or an ``(F,17,3)``
``.npy``) to down-weight frames whose ankle keypoint is not visible.
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

from vtv.contact_ik import solve_contact_ik  # noqa: E402
from vtv.metrics.core import _jerk, compute_metrics, pose_track  # noqa: E402
from vtv.mmd.pmx import parse_pmx  # noqa: E402
from vtv.mmd.vmd import parse_vmd  # noqa: E402
from vtv.mmd.writer import write_vmd  # noqa: E402


def load_cfg(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["models"]


def load_kp2d(path):
    if not path:
        return None
    if path.endswith(".npy"):
        return np.load(path)
    import torch  # only needed for the GVHMR preprocess artifact
    t = torch.load(path, map_location="cpu")
    return t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)


def score(model, profile, gt_vmd, pred_vmd, contact_margin):
    gt = pose_track(model, gt_vmd, profile)
    pr = pose_track(model, pred_vmd, profile, frames=gt["frames"])
    return compute_metrics(gt, pr, profile, contact_margin=contact_margin)


def stability(model, profile, before_vmd, after_vmd):
    n = max(before_vmd.get("max_frame", 0), after_vmd.get("max_frame", 0))
    frames = list(range(n + 1))
    a = pose_track(model, before_vmd, profile, frames=frames)
    b = pose_track(model, after_vmd, profile, frames=frames)
    ja, jb = _jerk(a["world"]), _jerk(b["world"])
    ok = (jb["rms"] <= ja["rms"] * 1.02) or (jb["spike_count"] < ja["spike_count"])
    return {"jerk_rms_before": round(ja["rms"], 4),
            "jerk_rms_after": round(jb["rms"], 4),
            "spike_count_before": ja["spike_count"],
            "spike_count_after": jb["spike_count"],
            "ok": bool(ok),
            "note": "world-joint jerk; GT-free, rejects leg-plant flattening"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "models.json"))
    ap.add_argument("--model", default="si")
    ap.add_argument("--vmd", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--kp2d", default=None,
                    help="(F,17,3) ViTPose evidence: .npy or torch .pt")
    ap.add_argument("--ground", type=float, default=None,
                    help="override the estimated floor (units); default auto")
    ap.add_argument("--band", type=float, default=0.02,
                    help="contact band as a fraction of height (scorer: 0.02)")
    ap.add_argument("--ground-pct", type=float, default=1.0)
    ap.add_argument("--mode", choices=("bias", "band"), default="bias",
                    help="bias: one constant per-side offset to the floor; "
                         "band: per-frame pull of frames above the band")
    ap.add_argument("--target-frac", type=float, default=0.0,
                    help="where inside the band the lowest point is aimed "
                         "(bias mode; 0 = the floor)")
    ap.add_argument("--trust-pct", type=float, default=100.0,
                    help="percentile of the per-frame trust cap the constant "
                         "offset must respect")
    ap.add_argument("--gain", type=float, default=1.0,
                    help="scale the constant offset 0..1 (ablation)")
    ap.add_argument("--hover-min-frac", type=float, default=0.15)
    ap.add_argument("--hover-max-frac", type=float, default=1.5)
    ap.add_argument("--enter-frac", type=float, default=1.5)
    ap.add_argument("--exit-frac", type=float, default=2.0)
    ap.add_argument("--vspeed", type=float, default=0.10)
    ap.add_argument("--max-knee-deg", type=float, default=15.0)
    ap.add_argument("--max-thigh-deg", type=float, default=25.0)
    ap.add_argument("--max-drop-pct", type=float, default=2.5,
                    help="per-frame ankle drop cap, percent of height")
    ap.add_argument("--p-contact-min", type=float, default=0.8)
    ap.add_argument("--ramp", type=int, default=3)
    ap.add_argument("--contact-margin", type=float, default=0.02,
                    help="scorer margin for --gt before/after only")
    ap.add_argument("--sides", default="l,r")
    ap.add_argument("--no-self-check", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.config)[args.model]
    profile = json.load(open(os.path.join(ROOT, "profiles", f"{args.model}.json"),
                             encoding="utf-8"))
    model = parse_pmx(cfg["pmx"])
    vmd = parse_vmd(args.vmd)
    kp2d = load_kp2d(args.kp2d)
    res = solve_contact_ik(
        vmd, model, profile,
        mode=args.mode, band=args.band, ground_pct=args.ground_pct,
        target_frac=args.target_frac, enter_frac=args.enter_frac,
        exit_frac=args.exit_frac, vspeed=args.vspeed,
        max_knee_deg=args.max_knee_deg, max_thigh_deg=args.max_thigh_deg,
        max_drop_frac=args.max_drop_pct / 100.0,
        p_contact_min=args.p_contact_min, ramp=args.ramp,
        trust_pct=args.trust_pct, gain=args.gain,
        hover_min_frac=args.hover_min_frac,
        hover_max_frac=args.hover_max_frac,
        kp2d=kp2d, ground=args.ground,
        sides=tuple(x.strip() for x in args.sides.split(",") if x.strip()))

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_vmd(out, res["vmd"].get("model_name", ""), res["vmd"]["bones"],
              res["vmd"].get("morphs", {}), res["vmd"].get("ik_frames", []))
    report = {"in": os.path.abspath(args.vmd), "out": out,
              "model": args.model, "contact_ik": res["report"]}
    print("ground", res["report"]["ground"], "band",
          res["report"]["band_units"], "applied", res["report"]["applied"])
    for side, s in res["report"]["sides"].items():
        if not s.get("applied"):
            print(f"  {side}: not applied ({s.get('reason', 'no correction')})")
            continue
        c = s["contact"]
        print(f"  {side}: hover {s['hover'].get('hover')} -> const "
              f"{s['constant_drop']} ({s['hover'].get('reason')}) | "
              f"contact {c['frames']} frames ({c['fraction']}), "
              f"band top {c['band_top_units']} | drop {s['drop_mean']:.3f}/"
              f"{s['drop_max']:.3f} mean/max over {s['frames']} frames | "
              f"knee<={s['knee_max_deg']} thigh<={s['thigh_max_deg']} "
              f"limits={s['limits']}")
        print(f"      toe y {s['toe_y']['before_min']:.3f}-{s['toe_y']['before_max']:.3f}"
              f" -> {s['toe_y']['after_min']:.3f}-{s['toe_y']['after_max']:.3f}")
    v = res["report"]["verify"]
    print("verify:", "OK" if v["ok"] else "FAIL", v["failures"] or "")
    if not args.no_self_check:
        st = stability(model, profile, vmd, res["vmd"])
        report["stability"] = st
        print("stability: jerk", st["jerk_rms_before"], "->", st["jerk_rms_after"],
              "spikes", st["spike_count_before"], "->", st["spike_count_after"],
              "OK" if st["ok"] else "WARN")
    if args.gt:
        gt = parse_vmd(args.gt)
        before = score(model, profile, gt, vmd, args.contact_margin)
        after = score(model, profile, gt, res["vmd"], args.contact_margin)
        report["metrics"] = {
            "gt": os.path.abspath(args.gt),
            "before": {"vector": before["vector"], "groups": before["groups"],
                       "foot": before["foot"]},
            "after": {"vector": after["vector"], "groups": after["groups"],
                      "foot": after["foot"]},
            "delta": {k: round(after["vector"][k] - before["vector"][k], 4)
                      for k in before["vector"]
                      if isinstance(before["vector"][k], (int, float))
                      and isinstance(after["vector"].get(k), (int, float))},
        }
        keys = ("angle_mean_deg", "angle_p99_deg", "position_pct_height",
                "position_p99_pct_height", "contact_recall", "contact_precision",
                "contact_slide_delta", "contact_penetration_delta",
                "contact_height_mae_pct_height", "jerk_rms_ratio")
        print("before:", {k: before["vector"][k] for k in keys})
        print("after :", {k: after["vector"][k] for k in keys})
    rp = args.report or (out + ".contact_ik.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("WROTE", out)
    print("WROTE", rp)
    if not v["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
