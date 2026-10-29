"""Selective cleanup for a predicted VMD (M2/F).

Examples:
  python scripts/clean_vmd.py --vmd outputs/adapter/R2fix_front.vmd --model si \
      --out outputs/m2/clean/R2fix_front.vmd --gt outputs/gt/si_R2_safe.vmd

The output is a normal 23-track IK-off VMD; the Blender renderer still bakes
the inherited bones from it.  ``--gt`` is only for before/after scoring, the
cleanup itself never reads the GT.
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from vtv.cleanup import clean_vmd  # noqa: E402
from vtv.metrics.core import _jerk, compute_metrics, pose_track  # noqa: E402
from vtv.mmd.pmx import parse_pmx  # noqa: E402
from vtv.mmd.vmd import parse_vmd  # noqa: E402
from vtv.mmd.writer import write_vmd  # noqa: E402


def load_cfg(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["models"]


def score(model, profile, gt_vmd, pred_vmd, contact_margin):
    gt = pose_track(model, gt_vmd, profile)
    pr = pose_track(model, pred_vmd, profile, frames=gt["frames"])
    return compute_metrics(gt, pr, profile, contact_margin=contact_margin)


def stability(model, profile, before_vmd, after_vmd):
    """GT-free sanity check: cleanup must not make world-joint jerk worse."""
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
            "note": "world-joint jerk; GT-free, rejects over-smoothing of accents"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "models.json"))
    ap.add_argument("--model", default="si")
    ap.add_argument("--vmd", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--no-spikes", action="store_true")
    ap.add_argument("--no-footlock", action="store_true")
    ap.add_argument("--vlock", action="store_true",
                    help="also hold the vertical root during contact")
    ap.add_argument("--spike-deg", type=float, default=5.0)
    ap.add_argument("--spike-sigmas", type=float, default=8.0,
                    help="adaptive threshold = max(spike-deg, median + k*MAD)")
    ap.add_argument("--max-run", type=int, default=6)
    ap.add_argument("--max-bridge-deg", type=float, default=120.0)
    ap.add_argument("--contact-margin", type=float, default=0.02)
    ap.add_argument("--contact-speed", type=float, default=0.12)
    ap.add_argument("--contact-vspeed", type=float, default=0.08)
    ap.add_argument("--root-cap", type=float, default=2.0)
    ap.add_argument("--lock-drift", type=float, default=0.5,
                    help="only lock contact intervals with net drift above this")
    ap.add_argument("--lock-contact-fraction", type=float, default=0.8,
                    help="do not foot-lock a side with less predicted contact")
    ap.add_argument("--bones", default=None,
                    help="comma separated bones to spike-clean (default all)")
    ap.add_argument("--no-self-check", action="store_true",
                    help="skip the GT-free world-joint jerk before/after check")
    args = ap.parse_args()

    cfg = load_cfg(args.config)[args.model]
    profile = json.load(open(os.path.join(ROOT, "profiles", f"{args.model}.json"),
                             encoding="utf-8"))
    model = parse_pmx(cfg["pmx"])
    vmd = parse_vmd(args.vmd)
    bones = args.bones.split(",") if args.bones else None
    res = clean_vmd(
        vmd, model, profile,
        spike_deg=args.spike_deg, adaptive_sigmas=args.spike_sigmas,
        max_run=args.max_run, max_bridge_deg=args.max_bridge_deg,
        contact_margin=args.contact_margin,
        contact_speed=args.contact_speed, contact_vspeed=args.contact_vspeed,
        root_cap=args.root_cap, lock_drift=args.lock_drift,
        lock_contact_fraction=args.lock_contact_fraction,
        do_spikes=not args.no_spikes, do_footlock=not args.no_footlock,
        vertical_lock=args.vlock, bones=bones)

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_vmd(out, res["vmd"].get("model_name", ""), res["vmd"]["bones"],
              res["vmd"].get("morphs", {}), res["vmd"].get("ik_frames", []))
    report = {"in": os.path.abspath(args.vmd), "out": out,
              "model": args.model, "cleanup": res["report"]}
    if not args.no_self_check:
        st = stability(model, profile, vmd, res["vmd"])
        report["stability"] = st
        print("stability: jerk", st["jerk_rms_before"], "->", st["jerk_rms_after"],
              "spikes", st["spike_count_before"], "->", st["spike_count_after"],
              "OK" if st["ok"] else "WARN (over-smoothing?)")
    if args.gt:
        before = score(model, profile, parse_vmd(args.gt), vmd, args.contact_margin)
        after = score(model, profile, parse_vmd(args.gt), res["vmd"],
                      args.contact_margin)
        report["metrics"] = {
            "gt": os.path.abspath(args.gt),
            "before": {"vector": before["vector"], "groups": before["groups"]},
            "after": {"vector": after["vector"], "groups": after["groups"]},
            "delta": {k: round(after["vector"][k] - before["vector"][k], 4)
                      for k in before["vector"] if isinstance(before["vector"][k], (int, float))
                      and isinstance(after["vector"].get(k), (int, float))},
        }
        v0, v1 = before["vector"], after["vector"]
        print("before:", {k: v0[k] for k in
                          ("angle_mean_deg", "angle_p99_deg", "position_pct_height",
                           "contact_recall", "contact_slide_delta", "jerk_rms_ratio")})
        print("after :", {k: v1[k] for k in
                          ("angle_mean_deg", "angle_p99_deg", "position_pct_height",
                           "contact_recall", "contact_slide_delta", "jerk_rms_ratio")})
    rp = args.report or (out + ".cleanup.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("WROTE", out)
    print("WROTE", rp)
    print(json.dumps({k: v for k, v in res["report"].items()
                      if k in ("frames", "spike_frames_total")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
