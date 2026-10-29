"""Aggregate the M2.4b multi-view 3D arm direction prior experiments.

Reads ``outputs/m2/arm3d/{view}_{guard,upper}.json`` (written by
``refine_arm3d.py``) and the closed-loop metrics next to them, and prints the
world / swing / torso-relative / penetration / jerk vector.  ``summary_S`` is
never used.

  python scripts/m24b_summary.py
  # -> outputs/m2/arm3d/report.json
"""
import argparse
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VIEWS = ("front", "three_quarter", "side")
VARIANTS = ("guard", "upper")


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def vec(m):
    if not m:
        return None
    v = m["vector"]
    return {
        "angle_mean_deg": v["angle_mean_deg"],
        "angle_p99_deg": v["angle_p99_deg"],
        "position_pct_height": v["position_pct_height"],
        "position_p99_pct_height": v["position_p99_pct_height"],
        "contact_recall": v["contact_recall"],
        "contact_precision": v["contact_precision"],
        "contact_slide_delta": v["contact_slide_delta"],
        "contact_penetration_delta": v["contact_penetration_delta"],
        "contact_height_mae_pct_height": v["contact_height_mae_pct_height"],
        "jerk_rms_ratio": v["jerk_rms_ratio"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "m2",
                                                  "arm3d", "report.json"))
    args = ap.parse_args()
    report = {"views": {}, "mode": "M2.4b multi-view 3D arm direction prior"}
    hdr = (f"{'view':14s} {'variant':6s} {'world':>6s} {'swing':>6s} {'body':>6s} "
           f"{'penT':>6s} {'penA':>6s} {'jerk':>5s} {'pos%':>6s} {'p99':>6s} "
           f"{'rec':>5s} {'slide':>6s} {'ok':>5s}")
    print(hdr)
    for view in VIEWS:
        base = vec(load(os.path.join(ROOT, "outputs", "m2", "closed_loop",
                                     f"{view}_clean_ik", "metrics.json")))
        entry = {"baseline_closed_loop": base, "variants": {}}
        for tag in VARIANTS:
            rep = load(os.path.join(ROOT, "outputs", "m2", "arm3d",
                                    f"{view}_{tag}.json"))
            if not rep:
                continue
            m = vec(load(os.path.join(ROOT, "outputs", "m2", "arm3d",
                                      f"cl_{view}_{tag}", "metrics.json")))
            after = (rep.get("after") or {})
            before = (rep.get("before") or {})
            vs = (after.get("vs_gt") or {})
            bs = (after.get("body_relative") or {})
            b0 = (before.get("body_relative") or {})
            pt = ((after.get("penetration") or {}).get("arm_torso") or {})
            pa = ((after.get("penetration") or {}).get("arm_arm") or {})
            entry["variants"][tag] = {
                "delta": rep.get("delta"),
                "target_vs_gt": rep.get("target_vs_gt"),
                "world_before_after": [
                    (before.get("vs_gt") or {}).get("arm_world_mean_deg"),
                    vs.get("arm_world_mean_deg")],
                "swing_before_after": [
                    (before.get("vs_gt") or {}).get("arm_swing_mean_deg"),
                    vs.get("arm_swing_mean_deg")],
                "body_before_after": [b0.get("arm_body_mean_deg"),
                                      bs.get("arm_body_mean_deg")],
                "chest_world_deg": b0.get("chest_world_mean_deg"),
                "penetration_after": {"arm_torso": pt.get("max"),
                                      "arm_arm": pa.get("max")},
                "closed_loop": m,
                "verify_ok": rep.get("verify_ok"),
                "accepted": rep.get("accepted"),
                "gate": rep.get("gate"),
                "report": f"outputs/m2/arm3d/{view}_{tag}.json",
            }
            print(f"{view:14s} {tag:6s} "
                  f"{vs.get('arm_world_mean_deg', 0):6.2f} "
                  f"{vs.get('arm_swing_mean_deg', 0):6.2f} "
                  f"{bs.get('arm_body_mean_deg', 0):6.2f} "
                  f"{(pt.get('max') or 0):6.3f} {(pa.get('max') or 0):6.3f} "
                  + (f"{m['jerk_rms_ratio']:5.2f} {m['position_pct_height']:6.3f} "
                     f"{m['angle_p99_deg']:6.1f} {m['contact_recall']:5.2f} "
                     f"{m['contact_slide_delta']:6.2f}" if m else " " * 42)
                  + f" {str(rep.get('accepted')):>5s}")
        report["views"][view] = entry
    report["note"] = (
        "world is the mean geodesic over the six arm bones; swing is the "
        "bone-direction part; body is the same direction inside each track's "
        "chest frame (torso-relative). The 3/4 view's chest/root error is "
        "~51 deg, so a world-only drop is not an arm-pose improvement. "
        "upper = no capsule guard (information upper bound); guard = per-frame "
        "capsule guard (safe but still temporally noisy). No variant passes "
        "all of verify / penetration / body-relative / smoothness.")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
