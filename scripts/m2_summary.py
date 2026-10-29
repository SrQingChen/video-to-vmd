"""Aggregate the M2 cleanup / 2D-reprojection / contact-IK runs.

Reads the closed-loop ``metrics.json`` files written by
``scripts/closed_loop.py`` and prints (and writes) a Pareto-style comparison:

    baseline (M1.5) -> cleanup -> 2D -> 2d+cleanup -> +contact IK

The gate is the vector / per-bone one from v2 §6, not ``summary_S``.  Arm
local angles are reported but are known to be twist-convention sensitive
(M1.5 §4.5); the arm world angle and joint position are the primary arm
signals.  Leg local angles are the primary contact-IK signal: the IK is only
allowed to move the leg when it moves it *toward* GT.
"""
import argparse
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VIEWS = ("front", "three_quarter", "side")
STAGES = {
    "baseline": os.path.join("outputs", "closed_loop", "m1_5", "{view}", "metrics.json"),
    "cleanup": os.path.join("outputs", "m2", "closed_loop_clean", "{view}", "metrics.json"),
    "2d": os.path.join("outputs", "m2", "closed_loop", "{view}_2d", "metrics.json"),
    "2d+cleanup": os.path.join("outputs", "m2", "closed_loop", "{view}_2d_clean", "metrics.json"),
    "cleanup+ik": os.path.join("outputs", "m2", "closed_loop", "{view}_clean_ik", "metrics.json"),
    "2d+cleanup+ik": os.path.join("outputs", "m2", "closed_loop", "{view}_2d_clean_ik", "metrics.json"),
}
KEYS = ("angle_mean_deg", "angle_p99_deg", "world_angle_mean_deg",
        "position_pct_height", "position_p99_pct_height",
        "contact_recall", "contact_precision", "contact_slide_delta",
        "contact_penetration_delta", "contact_height_mae_pct_height",
        "jerk_rms_ratio", "spike_delta")
ARMS = ("upperarm_l", "forearm_l", "hand_l", "upperarm_r", "forearm_r", "hand_r")
LEGS = ("thigh_l", "shin_l", "foot_l", "toe_l", "thigh_r", "shin_r", "foot_r", "toe_r")
IK_STAGES = ("cleanup+ik", "2d+cleanup+ik")


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def group_stats(m, names):
    if not m:
        return None
    pb = m.get("per_bone") or {}
    out = {}
    for k in ("rot_local_mean_deg", "rot_world_mean_deg", "pos_mean_units"):
        vals = [pb[a][k] for a in names if a in pb and pb[a].get(k) is not None]
        out[k] = round(sum(vals) / len(vals), 4) if vals else None
    return out


def gate(base, after, recall_target=None, position_mean_tol=0.02):
    """M2 tau gate vs the M1.5 baseline (+ an IK recall target when given).

    ``position_mean_tol`` is the 0.01-0.05% of height tolerance the M2.2 brief
    asks for on the side view: cleanup moves the root XZ there, and the mean
    position moves by 0.0007% height, which a strict no-worse rule would read
    as a regression even though the foot slide drops 74%.  0.02% leaves the
    check meaningful and stops it from vetoing a Pareto win.
    """
    if not base or not after:
        return None
    b, a = base["vector"], after["vector"]
    checks = {
        "position_mean_not_worse": a["position_pct_height"]
        <= b["position_pct_height"] + position_mean_tol,
        "position_p99_within_0.5pct": a["position_p99_pct_height"] <= b["position_p99_pct_height"] + 0.5,
        "jerk_not_worse": a["jerk_rms_ratio"] <= b["jerk_rms_ratio"] + 1e-6,
        "slide_not_worse_0.05": a["contact_slide_delta"] <= b["contact_slide_delta"] + 0.05,
        "penetration_not_worse_0.01": (a.get("contact_penetration_delta") or 0.0)
        <= (b.get("contact_penetration_delta") or 0.0) + 0.01,
        "recall_not_worse": (a.get("contact_recall") or 0.0)
        >= (b.get("contact_recall") or 0.0) - 1e-9,
        "angle_mean_within_1deg": a["angle_mean_deg"] <= b["angle_mean_deg"] + 1.0,
        "angle_p99_within_1deg": a["angle_p99_deg"] <= b["angle_p99_deg"] + 1.0,
    }
    if recall_target is not None:
        checks[f"recall_ge_{recall_target}"] = (a.get("contact_recall") or 0.0) >= recall_target
    strict = {
        "position_pct_height": a["position_pct_height"] <= b["position_pct_height"] + 1e-6,
        "position_p99_pct_height": a["position_p99_pct_height"] <= b["position_p99_pct_height"] + 1e-6,
        "angle_mean_deg": a["angle_mean_deg"] <= b["angle_mean_deg"] + 1e-6,
        "angle_p99_deg": a["angle_p99_deg"] <= b["angle_p99_deg"] + 1e-6,
        "jerk_rms_ratio": a["jerk_rms_ratio"] <= b["jerk_rms_ratio"] + 1e-6,
        "contact_slide_delta": a["contact_slide_delta"] <= b["contact_slide_delta"] + 1e-6,
        "contact_recall": (a.get("contact_recall") or 0)
        >= (b.get("contact_recall") or 0) - 1e-9,
    }
    return {"checks": checks, "ok": bool(all(checks.values())),
            "slack": [k for k in checks if not checks[k]],
            "strict_no_worse": bool(all(strict.values())),
            "strict_failed": [k for k in strict if not strict[k]]}


def incremental(before, after):
    """What one stage bought, measured against its own input VMD."""
    if not before or not after:
        return None
    b, a = before["vector"], after["vector"]
    keys = ("position_pct_height", "contact_recall", "contact_slide_delta",
            "contact_penetration_delta", "contact_height_mae_pct_height",
            "jerk_rms_ratio", "angle_mean_deg")
    return {k: round((a.get(k) or 0) - (b.get(k) or 0), 4) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "m2", "report.json"))
    ap.add_argument("--recall-target", type=float, default=0.70)
    args = ap.parse_args()
    report = {"stages": {}, "views": {}, "recall_target": args.recall_target}
    for view in VIEWS:
        report["views"][view] = {}
        for stage, pat in STAGES.items():
            m = load(os.path.join(ROOT, pat.format(view=view)))
            if not m:
                continue
            report["views"][view][stage] = {
                "path": pat.format(view=view),
                "vector": {k: m["vector"].get(k) for k in KEYS},
                "groups": m.get("groups"),
                "arms": group_stats(m, ARMS),
                "legs": group_stats(m, LEGS),
            }
        base = report["views"][view].get("baseline")
        for stage in ("cleanup", "2d", "2d+cleanup", "cleanup+ik", "2d+cleanup+ik"):
            rec = report["views"][view].get(stage)
            if not rec or not base:
                continue
            rec["delta_vs_baseline"] = {
                k: round((rec["vector"][k] - base["vector"][k]), 4)
                for k in KEYS if isinstance(rec["vector"].get(k), (int, float))
                and isinstance(base["vector"].get(k), (int, float))}
            rec["gate"] = gate(base, rec,
                               args.recall_target if stage in IK_STAGES else None)
        # stage-to-stage attribution for the IK step
        for stage, prev in (("cleanup+ik", "cleanup"), ("2d+cleanup+ik", "2d+cleanup")):
            rec = report["views"][view].get(stage)
            before = report["views"][view].get(prev)
            if rec and before:
                rec["incremental_vs_" + prev] = incremental(before, rec)

    hdr = (f"{'view':14s} {'stage':14s} {'ang':>7s} {'p99':>7s} {'pos%':>6s} "
           f"{'posP99':>7s} {'rec':>5s} {'prec':>5s} {'slide':>6s} {'pen':>5s} "
           f"{'fH%':>5s} {'jerk':>6s} {'armW':>6s} {'legA':>6s}")
    print(hdr)
    for view in VIEWS:
        for stage in STAGES:
            rec = report["views"][view].get(stage)
            if not rec:
                continue
            v = rec["vector"]
            a, l = rec["arms"] or {}, rec["legs"] or {}
            print(f"{view:14s} {stage:14s} {v['angle_mean_deg']:7.2f} "
                  f"{v['angle_p99_deg']:7.2f} {v['position_pct_height']:6.2f} "
                  f"{v['position_p99_pct_height']:7.2f} "
                  f"{str(v['contact_recall']):>5s} "
                  f"{str(v['contact_precision']):>5s} "
                  f"{v['contact_slide_delta']:6.2f} "
                  f"{(v['contact_penetration_delta'] or 0):5.2f} "
                  f"{v['contact_height_mae_pct_height']:5.2f} "
                  f"{v['jerk_rms_ratio']:6.2f} "
                  f"{(a.get('rot_world_mean_deg') or 0):6.2f} "
                  f"{(l.get('rot_local_mean_deg') or 0):6.2f}")
        print()
        for stage in ("cleanup+ik", "2d+cleanup+ik"):
            rec = report["views"][view].get(stage)
            if not rec:
                continue
            g = rec.get("gate") or {}
            inc = rec.get("incremental_vs_" + ("cleanup" if stage == "cleanup+ik"
                                               else "2d+cleanup")) or {}
            print(f"  {view:14s} {stage:14s} gate "
                  f"{'ok' if g.get('ok') else 'SLACK ' + ','.join(g.get('slack', []))}"
                  f" | strict {'ok' if g.get('strict_no_worse') else 'failed ' + ','.join(g.get('strict_failed', []))}"
                  f" | vs input: recall {inc.get('contact_recall'):+.3f} "
                  f"fH% {inc.get('contact_height_mae_pct_height'):+.2f} "
                  f"jerk {inc.get('jerk_rms_ratio'):+.3f} "
                  f"pos% {inc.get('position_pct_height'):+.3f}")
        print()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
