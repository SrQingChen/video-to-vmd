"""Aggregate the M2.4c prior-availability / oracle / multi-view BA results.

Reads the reports written by ``refine_ba3d.py`` and ``m24c_oracle.py`` plus the
closed-loop vector for the selected BA output.  ``summary_S`` is never used.

  python scripts/m24c_summary.py
  # -> outputs/m2/ba3d/report.json
"""
import argparse
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BA = os.path.join(ROOT, "outputs", "m2", "ba3d")


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def vec(path):
    d = load(path)
    if not d:
        return None
    v = d["vector"]
    return {k: v[k] for k in (
        "angle_mean_deg", "angle_p99_deg", "position_pct_height",
        "position_p99_pct_height", "contact_recall", "contact_precision",
        "contact_slide_delta", "jerk_rms_ratio", "spike_delta")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(BA, "report.json"))
    args = ap.parse_args()
    report = {"mode": "M2.4c prior availability + oracle + multi-view BA",
              "variants": {}, "closed_loop": {}}
    base_cl = vec(os.path.join(ROOT, "outputs", "m2", "closed_loop",
                               "front_clean_ik", "metrics.json"))
    report["closed_loop"]["cleanup+ik"] = base_cl
    for tag, fn in (("ba_w20", "front_ba3d.json"),
                    ("ba_w8", "front_ba3d_w8.json")):
        r = load(os.path.join(BA, fn))
        if not r:
            continue
        report["variants"][tag] = {
            "body_full": [r["body"]["before_full_deg"],
                          r["body"]["after_full_deg"],
                          r["body"]["delta_full_deg"]],
            "body_upper_fore": [r["body"]["before_upper_fore_deg"],
                                r["body"]["after_upper_fore_deg"],
                                r["body"]["delta_upper_fore_deg"]],
            "chest_world_deg": r["body"]["after"]["chest_world_mean_deg"],
            "penetration": r["penetration"],
            "temporal": r["temporal"],
            "view_error_written": r["view_error_written"],
            "view_error_fit": r["view_error_fit"],
            "root_fit": r["root_fit"],
            "verify_ok": r["verify"]["ok"],
            "accepted": r["accepted"],
            "position_3d": r["body"]["position_3d_after"],
        }
    report["closed_loop"]["ba_w20"] = vec(os.path.join(
        BA, "cl_front_ba3d", "metrics.json"))
    report["oracle"] = load(os.path.join(BA, "oracle.json"))
    report["gate_note"] = (
        "M2.4b strict 'no pair may increase vs baseline' is not satisfiable "
        "by GT on R2: GT increases forearm_r x chest, forearm_r x spine1, "
        "hand_r x chest and upperarm_r x chest. Use max(baseline, GT)+0.02 as "
        "the synthetic budget, and report the strict rule as diagnostic. The "
        "primary arm metric is body-relative upper/fore swing + elbow/wrist "
        "3D; the full six-bone mean includes the hand leaf and is reported for "
        "continuity with M2.4b, not as the only gate.")
    report["verdict"] = (
        "No prior-only candidate exists without VPoser/AMASS. The oracle "
        "torso-relative target reaches body upper/fore -12 deg at alpha=0.5 "
        "but increases arm-arm penetration to 1.209 (baseline 1.003), so the "
        "torso must be corrected with the arms. The multi-view target-PMX BA "
        "halves elbow/wrist 3D error and improves world swing -10.4 deg, but "
        "body-relative upper/fore is only -6.18 deg (full six-bone -9.57, "
        "hand-driven). It is a multi-view upper bound, not a T5/single-view "
        "solution; cleanup+ik remains the default.")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"{'variant':14s} {'body_full':>20s} {'body_upf':>20s} "
          f"{'penT':>7s} {'penA':>7s} {'stepmax':>8s} {'p99':>7s} "
          f"{'verify':>7s}")
    for tag, v in report["variants"].items():
        print(f"{tag:14s} "
              f"{v['body_full'][0]:6.2f}->{v['body_full'][1]:6.2f} "
              f"({v['body_full'][2]:+6.2f}) "
              f"{v['body_upper_fore'][0]:6.2f}->{v['body_upper_fore'][1]:6.2f} "
              f"({v['body_upper_fore'][2]:+6.2f}) "
              f"{v['penetration']['after']['arm_torso']:7.3f} "
              f"{v['penetration']['after']['arm_arm']:7.3f} "
              f"{v['temporal']['after']['max_deg']:8.2f} "
              f"{v['temporal']['after']['p99_deg']:7.2f} "
              f"{str(v['verify_ok']):>7s}")
    print("closed-loop cleanup+ik", report["closed_loop"]["cleanup+ik"])
    print("closed-loop ba_w20    ", report["closed_loop"]["ba_w20"])
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
