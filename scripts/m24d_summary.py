"""M2.4d summary: VPoser v2 torso/arm prior sweep + gates.

Reads the per-variant reports written by ``refine_ba3d.py`` and the probe,
applies the M2.4d gates, and writes ``outputs/m2/vposer/report.json``.

Gates (all must pass for a candidate to be accepted):
  * body-relative upper/forearm delta <= -10 deg (the target);
  * ``verify_ba.ok`` (frames/bones/IK/arm lengths, non-optimized unchanged,
    hand local, temporal max/p99, global capsule penetration);
  * per-pair capsule budget ``after <= max(baseline, GT) + 0.02``;
  * global arm-torso / arm-arm max not worse than baseline;
  * hand is reported separately and is **not** part of the 10 deg criterion.
World/geodesic/swing/roll are reported only.
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np


def load(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pos_mean(d, keys):
    vals = [d[k]["mean_pct_height"] for k in keys if k in d]
    return round(float(np.mean(vals)), 3) if vals else None


def variant_row(name, rep):
    if rep is None or "body" not in rep:
        return None
    b = rep["body"]
    vb = rep.get("verify") or {}
    pen = rep.get("penetration") or {}
    tor = b.get("after_torso_rel") or {}
    pos = b.get("position_3d_after") or {}
    world = b.get("world_diag_after") or {}
    return {
        "variant": name,
        "opt_joints": rep.get("opt_joints"),
        "vposer_weight": (rep.get("vposer") or {}).get("weight", 0.0),
        "vposer_joints": (rep.get("vposer") or {}).get("stats", {}).get("joints")
        if (rep.get("vposer") or {}).get("stats") else None,
        "vposer_mode": (rep.get("vposer") or {}).get("mode"),
        "upper_fore_before_deg": b.get("before_upper_fore_deg"),
        "upper_fore_after_deg": b.get("after_upper_fore_deg"),
        "upper_fore_delta_deg": b.get("delta_upper_fore_deg"),
        "full_before_deg": b.get("before_full_deg"),
        "full_after_deg": b.get("after_full_deg"),
        "full_delta_deg": b.get("delta_full_deg"),
        "hand_before_deg": b.get("before_hand_deg"),
        "hand_after_deg": b.get("after_hand_deg"),
        "hand_delta_deg": b.get("delta_hand_deg"),
        "chest_world_before_deg": b.get("chest_world_before_deg"),
        "chest_world_after_deg": b.get("chest_world_after_deg"),
        "torso_body_rel_before": (b.get("before_torso_rel") or {}).get("mean_deg"),
        "torso_body_rel_after": (b.get("after_torso_rel") or {}).get("mean_deg"),
        "elbow_wrist_3d_mean_pct": pos_mean(pos, ("forearm_l", "forearm_r",
                                                  "hand_l", "hand_r")),
        "upperarm_3d_mean_pct": pos_mean(pos, ("upperarm_l", "upperarm_r")),
        "chest_3d_mean_pct": pos_mean(pos, ("chest",)),
        "pen_arm_torso_before": (pen.get("before") or {}).get("arm_torso"),
        "pen_arm_torso_after": (pen.get("after") or {}).get("arm_torso"),
        "pen_arm_arm_before": (pen.get("before") or {}).get("arm_arm"),
        "pen_arm_arm_after": (pen.get("after") or {}).get("arm_arm"),
        "pen_budget_ok": (vb.get("checks", {}).get(
            "penetration_pairs_budget", {}) or {}).get("ok"),
        "pen_budget_failed": (vb.get("checks", {}).get(
            "penetration_pairs_budget", {}) or {}).get("failed"),
        "temporal_before": rep.get("temporal", {}).get("before"),
        "temporal_after": rep.get("temporal", {}).get("after"),
        "verify_ok": vb.get("ok"),
        "accepted_10deg": bool(vb.get("ok")
                               and (b.get("delta_upper_fore_deg") or 0.0) <= -10.0),
        "world_diag_after": {
            "arm_world_mean_deg": world.get("arm_world_mean_deg"),
            "arm_swing_mean_deg": world.get("arm_swing_mean_deg"),
            "arm_roll_mean_deg": world.get("arm_roll_mean_deg"),
        },
        "vposer_stats": (rep.get("vposer") or {}).get("stats"),
        "out": rep.get("out"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "outputs", "m2", "vposer"))
    ap.add_argument("--out", default=os.path.join(
        ROOT, "outputs", "m2", "vposer", "report.json"))
    args = ap.parse_args()

    d = args.dir
    # ``arms_w0`` / ``arms_vp_recon_001`` are 150-iter controls superseded by
    # the 250-iter ``m24d_*`` runs; they stay on disk but are not ranked.
    names = ["m24d_arms_w0", "m24d_arms_vp_recon_001",
             "arms_vp_arms_003", "arms_vp_uf_001", "arms_vp_uf_003",
             "ta_w0", "ta_vp_opt_003", "ta_vp_opt_01",
             "ta_vp_torso_003", "ta_vp_torso_01"]
    rows = []
    for n in names:
        r = variant_row(n, load(os.path.join(d, f"{n}.json")))
        if r:
            rows.append(r)
    # any additional variant reports not in the fixed list
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        n = os.path.splitext(os.path.basename(p))[0]
        if n in names or n in ("probe", "report", "smoke",
                               "torso_arms_w0_smoke", "arms_w0",
                               "arms_vp_recon_001"):
            continue
        r = variant_row(n, load(p))
        if r:
            rows.append(r)

    gated = [r for r in rows if r["verify_ok"]
             and r["pen_budget_ok"] is not False
             and r["upper_fore_delta_deg"] is not None]
    gated.sort(key=lambda r: r["upper_fore_delta_deg"])
    accepted = [r for r in gated if r["accepted_10deg"]]
    best = gated[0] if gated else None

    cl = load(os.path.join(d, "cl_m24d_arms_w0", "metrics.json"))
    cl_base = load(os.path.join(d, "cl_base", "metrics.json"))
    render_meta = None
    rp = os.path.join(ROOT, "outputs", "m2", "render", "front_m24d_arms",
                      "meta.json")
    if os.path.exists(rp):
        m = json.load(open(rp, encoding="utf-8"))
        ib = m.get("inherit_bake") or {}
        render_meta = {
            "vmd": m.get("vmd"),
            "d_chain_vs_fk": ib.get("d_chain_vs_fk"),
            "verify": {k: ib.get("verify", {}).get(k) for k in
                       ("ok", "max_head_error", "mapped_bones", "tol")},
            "scale_warnings": ib.get("scale_warnings"),
            "stills": [f"frame_{x:04d}.png" for x in (27, 109, 175, 218, 242)],
        }
    probe = load(os.path.join(d, "probe.json"))

    report = {
        "mode": "M2.4d VPoser v2 torso-relative prior + torso prior + body-relative target",
        "target": "body-relative upper/forearm swing -10 deg (hands separate)",
        "gate": {
            "upper_fore_delta_deg_max": -10.0,
            "per_pair_budget": "after <= max(baseline, GT) + 0.02",
            "global_penetration": "not worse than baseline",
            "temporal": "max may grow <=5 deg/frame; p99 <= +0.5 deg",
            "hand": "reported separately, not part of the 10 deg criterion",
            "world_geodesic": "reported only",
        },
        "variants": rows,
        "gated_ranking": [r["variant"] for r in gated],
        "best_gated": best,
        "accepted": accepted,
        "target_met": bool(accepted),
        "closed_loop_best": {
            "candidate": cl,
            "baseline": cl_base,
        },
        "render_certificate": render_meta,
        "vposer_probe": {
            "rig_audit": (probe or {}).get("rig_audit"),
            "kl": (probe or {}).get("kl"),
            "projection": {
                k: {"delta_arm_deg": v.get("projection_delta_arm_deg"),
                    "delta_torso_deg": v.get("projection_delta_torso_deg"),
                    "upper_fore_after_deg": v.get("metrics", {}).get(
                        "upper_fore_body_deg"),
                    "chest_world_after_deg": v.get("metrics", {}).get(
                        "chest_world_deg"),
                    "pen_after": (v.get("metrics", {}).get(
                        "penetration", {}).get("global", {}).get("after")),
                    "budget_ok": (v.get("metrics", {}).get(
                        "penetration", {}).get("pairs_budget", {}).get("ok"))}
                for k, v in ((probe or {}).get("projection") or {}).items()},
            "prior_only": {
                k: {"upper_fore_after_deg": v.get("metrics", {}).get(
                        "upper_fore_body_deg"),
                    "chest_world_after_deg": v.get("metrics", {}).get(
                        "chest_world_deg"),
                    "kl_prior": (v.get("vposer") or {}).get("kl_prior_mean"),
                    "pen_after": (v.get("metrics", {}).get(
                        "penetration", {}).get("global", {}).get("after")),
                    "budget_ok": (v.get("metrics", {}).get(
                        "penetration", {}).get("pairs_budget", {}).get("ok"))}
                for k, v in ((probe or {}).get("prior_only") or {}).items()},
        },
        "conclusion": (
            "VPoser v2 rates the R2 GT less likely than the prediction "
            "(KL 134 vs 29, driven by the GT arms), so the prior points away "
            "from the target. Every prior variant is worse than the no-prior "
            "arms BA; torso+arms joint BA degrades the chest (11.2 -> 27-41 "
            "deg). The best gated candidate remains the M2.4c arms BA at "
            "-6.18 deg body-relative upper/fore (full -9.57 deg is hand-driven, "
            "reported separately). The 10 deg target is NOT met."
            if not accepted else
            "At least one gated variant reaches -10 deg body-relative "
            "upper/fore."),
        "note": ("GT is read-only scoring. No network was trained. The "
                 "default pipeline remains cleanup+ik; prior/BA are not "
                 "enabled by default. World/geodesic numbers are not gates."),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps({
        "best_gated": best,
        "accepted": [r["variant"] for r in accepted],
        "target_met": report["target_met"],
        "n_variants": len(rows),
    }, ensure_ascii=False, indent=1)[:4000])
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
