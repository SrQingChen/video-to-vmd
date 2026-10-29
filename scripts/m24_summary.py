"""Aggregate the M2.4 arm audit / constraints / hinge experiments.

Reads the reports written by ``refine_arm.py`` and the closed-loop metrics of
the hinge variant, and prints the arm decomposition next to the gate vector.
``summary_S`` is never used.

  python scripts/m24_summary.py
  # -> outputs/m2/m24_report.json
"""
import argparse
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VIEWS = ("front", "three_quarter", "side")
VARIANTS = ("audit", "constraints", "hinge")


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def arm_from_report(rep, phase="after"):
    if not rep:
        return None
    v = (rep.get(phase) or {}).get("vs_gt")
    if not v:
        return None
    return {"world": v.get("arm_world_mean_deg"),
            "swing": v.get("arm_swing_mean_deg"),
            "roll": v.get("arm_roll_mean_deg")}


def vec(m):
    if not m:
        return None
    v = m["vector"]
    pb = m.get("per_bone") or {}
    arms = ("upperarm_l", "forearm_l", "hand_l",
            "upperarm_r", "forearm_r", "hand_r")
    aw = [pb[a]["rot_world_mean_deg"] for a in arms
          if a in pb and pb[a].get("rot_world_mean_deg") is not None]
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
        "arm_world_mean_deg": (round(sum(aw) / len(aw), 4) if aw else None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "m2", "m24_report.json"))
    args = ap.parse_args()
    report = {"views": {}}
    hdr = (f"{'view':14s} {'variant':11s} {'armW':>6s} {'swing':>6s} {'roll':>6s} "
           f"{'penT':>6s} {'penA':>6s} {'ok':>5s} | {'ang':>6s} {'p99':>6s} "
           f"{'pos%':>6s} {'rec':>5s} {'slide':>6s} {'jerk':>5s}")
    print(hdr)
    for view in VIEWS:
        base_m = load(os.path.join(ROOT, "outputs", "m2", "closed_loop",
                                   f"{view}_clean_ik", "metrics.json"))
        base = vec(base_m)
        entry = {"baseline_closed_loop": base, "variants": {}}
        for variant in VARIANTS:
            rep = load(os.path.join(ROOT, "outputs", "m2", "arm",
                                    f"{view}_{variant}.json"))
            if not rep:
                continue
            arm = arm_from_report(rep, "after")
            b = rep.get("before", {}).get("penetration", {}).get("arm_torso", {})
            a = rep.get("after", {}).get("penetration", {}).get("arm_torso", {})
            entry["variants"][variant] = {
                "arm": arm,
                "penetration_before": b.get("max"),
                "penetration_after": a.get("max"),
                "accepted": rep.get("accepted"),
                "verify_ok": (rep.get("verify") or {}).get("ok"),
                "verify": (rep.get("verify") or {}).get("checks"),
                "report": f"outputs/m2/arm/{view}_{variant}.json",
            }
            cl = None
            if variant == "hinge":
                cl_m = load(os.path.join(ROOT, "outputs", "m2", "arm",
                                         f"cl_{view}_hinge", "metrics.json"))
                cl = vec(cl_m)
                entry["variants"][variant]["closed_loop"] = cl
            print(f"{view:14s} {variant:11s} "
                  f"{(arm or {}).get('world', 0):6.2f} {(arm or {}).get('swing', 0):6.2f} "
                  f"{(arm or {}).get('roll', 0):6.2f} "
                  f"{(b.get('max') or 0):6.3f} {(a.get('max') or 0):6.3f} "
                  f"{str(rep.get('accepted')):>5s} | "
                  + (f"{cl['angle_mean_deg']:6.2f} {cl['angle_p99_deg']:6.2f} "
                     f"{cl['position_pct_height']:6.3f} {cl['contact_recall']:5.2f} "
                     f"{cl['contact_slide_delta']:6.2f} {cl['jerk_rms_ratio']:5.2f}"
                     if cl else " " * 42))
        report["views"][view] = entry
    report["note"] = ("arm world is the mean geodesic over the six arm bones; "
                      "swing is the bone-direction part, roll the residual about "
                      "the bone axis.  Only swing moves joints; a candidate that "
                      "lowers world by lowering roll is rejected.")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
