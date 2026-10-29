"""Off-the-shelf 3D body benchmark on the R2 synthetic three-camera rig / T5.

The candidate is an external SMPL-X model (Multi-HMR).  Its per-frame
``body_pose`` is retargeted to the target PMX by ``run_adapter.py``; this script
then either

  * merges only the arm locals onto the existing ``cleanup+ik`` baseline
    (``arms``), so root/legs/IK/hand local are byte-identical and the result is
    directly comparable to the M2.4c arms BA, or
  * merges torso+arms (``torso_arms``) to test whether the candidate's chest
    observation helps, or
  * evaluates the full candidate VMD as a diagnostic.

Gates (same spirit as M2.4c/M2.4d):

  * body-relative upper/forearm + chest (chest in pelvis) vs GT;
  * body-relative elbow/wrist 3D in % of the PMX height;
  * global capsule max and per-pair budget ``after <= max(baseline, GT)+0.02``;
  * temporal arm-direction max/p99 and body-relative jerk;
  * hand reported separately; world/geodesic reported only.

GT is read-only scoring.  No solver here and no network training.
"""
import argparse
import copy
import json
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np

from vtv.adapter import q_conj, qrot
from vtv.arm3d import penetration
from vtv.ba3d import TORSO_ARM_CHAIN, merge_pose_tracks, verify_ba
from vtv.cleanup import dense_tracks, eval_tracks
from vtv.metrics.core import _jerk
from vtv.mmd.pmx import parse_pmx
from vtv.mmd.vmd import parse_vmd
from vtv.mmd.writer import write_vmd
from refine_ba3d import body_scores, temporal_steps

MERGE_ARMS = ("clavicle_l", "clavicle_r",
              "upperarm_l", "upperarm_r",
              "forearm_l", "forearm_r")
MERGE_TORSO_ARMS = TORSO_ARM_CHAIN
POS_JOINTS = ("upperarm_l", "upperarm_r", "forearm_l", "forearm_r",
              "hand_l", "hand_r", "chest")


def parse_pairs(items):
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"expected name=path, got {item!r}")
        k, v = item.split("=", 1)
        out[k] = os.path.abspath(v)
    return out


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _angle(a, b):
    return math.degrees(math.acos(float(np.clip(np.dot(_unit(a), _unit(b)),
                                                  -1.0, 1.0))))


def _tracks(model, profile, vmd):
    _, tr = dense_tracks(vmd)
    return eval_tracks(model, profile, tr)


def body_relative_positions(gt_tr, pr_tr, model):
    """Elbow/wrist/chest positions in the pelvis frame, % of PMX height."""
    ys = [b["pos"][1] for b in model["bones"]]
    height = max(ys) - min(ys)
    out = {}
    for j in POS_JOINTS:
        if j not in gt_tr["world"] or j not in pr_tr["world"]:
            continue
        dg = np.asarray(gt_tr["world"][j]) - gt_tr["world"]["pelvis"]
        dp = np.asarray(pr_tr["world"][j]) - pr_tr["world"]["pelvis"]
        qg = gt_tr["world_q"]["pelvis"]
        qp = pr_tr["world_q"]["pelvis"]
        g = np.stack([np.asarray(qrot(tuple(q_conj(tuple(q))), tuple(v)))
                      for q, v in zip(qg, dg)])
        p = np.stack([np.asarray(qrot(tuple(q_conj(tuple(q))), tuple(v)))
                      for q, v in zip(qp, dp)])
        err = np.linalg.norm(g - p, axis=1) / height * 100.0
        out[j] = {"mean_pct_height": round(float(err.mean()), 3),
                  "p99_pct_height": round(float(np.percentile(err, 99)), 3)}
    return out


def chest_swing(gt_tr, pr_tr):
    """Chest->neck direction error in the pelvis frame (body-relative)."""
    dg = np.asarray(gt_tr["world"]["neck"]) - gt_tr["world"]["chest"]
    dp = np.asarray(pr_tr["world"]["neck"]) - pr_tr["world"]["chest"]
    qg = gt_tr["world_q"]["pelvis"]
    qp = pr_tr["world_q"]["pelvis"]
    vals = []
    for i in range(len(dg)):
        g = qrot(tuple(q_conj(tuple(qg[i]))), tuple(dg[i]))
        p = qrot(tuple(q_conj(tuple(qp[i]))), tuple(dp[i]))
        vals.append(_angle(g, p))
    return round(float(np.mean(vals)), 3) if vals else None


def body_relative_jerk(tr):
    """Jerk RMS of world positions after removing the pelvis translation and
    rotation (so a root/heading difference cannot dominate the number)."""
    qp = tr["world_q"]["pelvis"]
    out = {}
    for j, w in tr["world"].items():
        d = np.asarray(w) - tr["world"]["pelvis"]
        p = np.stack([np.asarray(qrot(tuple(q_conj(tuple(q))), tuple(v)))
                      for q, v in zip(qp, d)])
        out[j] = p
    return _jerk(out)


def pair_map(pen):
    out = {}
    for group in ("arm_torso", "arm_arm"):
        for k, rec in (pen[group].get("pairs_detail") or {}).items():
            out[(group, k)] = float(rec.get("max", 0.0))
    return out


def pair_budget(pen_after, pen_base, pen_gt=None, tol=0.02):
    a = pair_map(pen_after)
    b = pair_map(pen_base)
    g = pair_map(pen_gt) if pen_gt else {}
    bad = []
    for k, v in sorted(a.items()):
        allowed = max(b.get(k, 0.0), g.get(k, 0.0)) + tol
        if v > allowed + 1e-9:
            bad.append({"pair": f"{k[0]} {k[1]}", "after": round(v, 4),
                        "budget": round(allowed, 4),
                        "baseline": round(b.get(k, 0.0), 4),
                        "gt": round(g.get(k, 0.0), 4)})
    return {"ok": not bad, "failed": bad[:12], "tolerance": tol,
            "rule": "after <= max(baseline, GT) + tol"}


def _structural_metrics(vmd, base_vmd, model, profile, merge_joints):
    """Metrics that need no GT: temporal, penetration, structural verify."""
    tr = _tracks(model, profile, vmd)
    base_tr = _tracks(model, profile, base_vmd)
    pen = penetration(vmd, model, profile)
    base_pen = penetration(base_vmd, model, profile)
    jr = body_relative_jerk(tr)
    base_jr = body_relative_jerk(base_tr)
    return {
        "temporal": temporal_steps(vmd, model, profile),
        "base_temporal": temporal_steps(base_vmd, model, profile),
        "penetration": {
            "arm_torso": pen["arm_torso"]["max"],
            "arm_arm": pen["arm_arm"]["max"],
            "global_max": max(pen["arm_torso"]["max"], pen["arm_arm"]["max"]),
        },
        "base_penetration": {
            "arm_torso": base_pen["arm_torso"]["max"],
            "arm_arm": base_pen["arm_arm"]["max"],
            "global_max": max(base_pen["arm_torso"]["max"],
                              base_pen["arm_arm"]["max"]),
        },
        "jerk": {"rms": jr["rms"], "spike_count": jr["spike_count"],
                 "p99": jr["p99"]},
        "base_jerk": {"rms": base_jr["rms"],
                      "spike_count": base_jr["spike_count"],
                      "p99": base_jr["p99"]},
        "jerk_ratio": (round(float(jr["rms"] / base_jr["rms"]), 4)
                       if base_jr["rms"] > 1e-12 else None),
        "_pen": pen,
        "_base_pen": base_pen,
    }


def evaluate(name, vmd, base_vmd, gt_vmd, model, profile,
             merge_joints=None, gt_pen=None):
    """Score one candidate; ``merge_joints`` also runs the structural gates."""
    tr = _tracks(model, profile, vmd)
    base_tr = _tracks(model, profile, base_vmd)
    struct = _structural_metrics(vmd, base_vmd, model, profile, merge_joints)
    out = {
        "name": name,
        "temporal": struct["temporal"],
        "base_temporal": struct["base_temporal"],
        "penetration": struct["penetration"],
        "base_penetration": struct["base_penetration"],
        "jerk": struct["jerk"],
        "base_jerk": struct["base_jerk"],
        "jerk_ratio": struct["jerk_ratio"],
        "budget": pair_budget(struct["_pen"], struct["_base_pen"], gt_pen),
    }
    if gt_vmd is not None:
        bs = body_scores(vmd, gt_vmd, model, profile)
        bbs = body_scores(base_vmd, gt_vmd, model, profile)
        gtr = _tracks(model, profile, gt_vmd)
        out["body"] = {
            "upper_fore_deg": bs["upper_fore_body_mean_deg"],
            "full_deg": bs["full_body_mean_deg"],
            "hand_deg": bs["hand_body_mean_deg"],
            "chest_world_deg": bs["body"]["chest_world_mean_deg"],
            "chest_rel_deg": bs["torso_body_relative"].get("chest"),
            "torso_rel_mean_deg": bs["torso_body_relative"].get("mean_deg"),
            "chest_swing_deg": chest_swing(gtr, tr),
            "world_diag": {
                "arm_world_mean_deg": bs["world_diag"]["arm_world_mean_deg"],
                "arm_swing_mean_deg": bs["world_diag"]["arm_swing_mean_deg"],
                "arm_roll_mean_deg": bs["world_diag"]["arm_roll_mean_deg"],
            },
            "base_world_diag": {
                "arm_world_mean_deg": bbs["world_diag"]["arm_world_mean_deg"],
                "arm_swing_mean_deg": bbs["world_diag"]["arm_swing_mean_deg"],
                "arm_roll_mean_deg": bbs["world_diag"]["arm_roll_mean_deg"],
            },
            "position_3d": body_relative_positions(gtr, tr, model),
            "base_upper_fore_deg": bbs["upper_fore_body_mean_deg"],
            "base_full_deg": bbs["full_body_mean_deg"],
            "base_hand_deg": bbs["hand_body_mean_deg"],
            "base_chest_world_deg": bbs["body"]["chest_world_mean_deg"],
            "base_chest_rel_deg": bbs["torso_body_relative"].get("chest"),
            "base_chest_swing_deg": chest_swing(gtr, base_tr),
            "base_position_3d": body_relative_positions(gtr, base_tr, model),
            "delta_upper_fore_deg": round(
                bs["upper_fore_body_mean_deg"]
                - bbs["upper_fore_body_mean_deg"], 3),
            "delta_full_deg": round(bs["full_body_mean_deg"]
                                    - bbs["full_body_mean_deg"], 3),
            "delta_hand_deg": round(bs["hand_body_mean_deg"]
                                    - bbs["hand_body_mean_deg"], 3),
        }
    if merge_joints:
        vb = verify_ba(base_vmd, vmd, model, profile, gt_vmd=gt_vmd,
                       opt_joints=merge_joints)
        out["verify"] = vb
    return out


def smooth_vmd(vmd, profile, joints, window):
    """Centred median filter on local quaternions (sign aligned)."""
    from vtv.cleanup import dense_tracks, tracks_to_vmd

    smap = profile["standard_map"]
    frames, tracks = dense_tracks(vmd)
    pad = max(1, int(window) // 2)
    for j in joints:
        bn = smap.get(j)
        if not bn or bn not in tracks:
            continue
        q = tracks[bn]["rot"]
        n = len(q)
        out = q.copy()
        for i in range(n):
            lo, hi = max(0, i - pad), min(n, i + pad + 1)
            qs = q[lo:hi]
            q0 = q[i]
            qs = np.where((qs @ q0)[:, None] >= 0.0, qs, -qs)
            med = np.median(qs, axis=0)
            out[i] = med / max(float(np.linalg.norm(med)), 1e-12)
        tracks[bn]["rot"] = out
    return tracks_to_vmd(vmd, frames, tracks)


def fuse_local_tracks(vmds, model, profile, joints, method="mean"):
    """Fuse per-view local rotations for ``joints`` onto the first VMD."""
    from vtv.cleanup import dense_tracks, tracks_to_vmd

    smap = profile["standard_map"]
    frames, tracks = dense_tracks(vmds[0])
    view_tracks = [dense_tracks(v)[1] for v in vmds]
    for j in joints:
        bn = smap.get(j)
        if not bn or any(bn not in t for t in view_tracks):
            continue
        rots = [t[bn]["rot"] for t in view_tracks]
        n = min(len(r) for r in rots)
        fused = np.zeros_like(tracks[bn]["rot"])
        fused[:] = tracks[bn]["rot"]  # fallback for any short tail
        for f in range(n):
            qs = [np.asarray(r[f], dtype=np.float64) for r in rots]
            q0 = qs[0]
            qs = [q if float(np.dot(q, q0)) >= 0.0 else -q for q in qs]
            q = np.median(qs, axis=0) if method == "median" \
                else np.mean(qs, axis=0)
            fused[f] = q / max(float(np.linalg.norm(q)), 1e-12)
        tracks[bn]["rot"] = fused
    return tracks_to_vmd(vmds[0], frames, tracks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "models.json"))
    ap.add_argument("--model", default="si")
    ap.add_argument("--gt", default=None)
    ap.add_argument("--baseline", nargs="+", required=True,
                    help="view=cleanup+ik VMD")
    ap.add_argument("--candidate", nargs="+", required=True,
                    help="view=adapter VMD from the external model")
    ap.add_argument("--reference", default=None,
                    help="M2.4c arms BA VMD for the direct comparison")
    ap.add_argument("--merge-joints", choices=("arms", "torso_arms"),
                    default="arms")
    ap.add_argument("--fuse", choices=("mean", "median"), default="mean")
    ap.add_argument("--smooth", type=int, default=0,
                    help="centred median window on the candidate merge-chain "
                         "locals before merging (0 = raw)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-dir", default=None,
                    help="write merged/fused VMDs for rendering")
    ap.add_argument("--tag", default="external")
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8"))["models"][args.model]
    profile = json.load(open(os.path.join(ROOT, "profiles", f"{args.model}.json"),
                             encoding="utf-8"))
    model = parse_pmx(cfg["pmx"])
    gt_vmd = parse_vmd(args.gt) if args.gt else None
    gt_pen = penetration(gt_vmd, model, profile) if gt_vmd is not None else None

    bases = {k: parse_vmd(v) for k, v in parse_pairs(args.baseline).items()}
    cands = {k: parse_vmd(v) for k, v in parse_pairs(args.candidate).items()}
    if set(bases) != set(cands):
        raise SystemExit(f"baseline/candidate views differ: "
                         f"{sorted(bases)} vs {sorted(cands)}")
    merge_joints = MERGE_TORSO_ARMS if args.merge_joints == "torso_arms" \
        else MERGE_ARMS
    if args.smooth and args.smooth >= 3:
        cands = {k: smooth_vmd(v, profile, merge_joints, args.smooth)
                 for k, v in cands.items()}

    out_dir = os.path.abspath(args.out_dir) if args.out_dir else None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    report = {
        "tag": args.tag,
        "model": args.model,
        "merge_joints": list(merge_joints),
        "gt": os.path.abspath(args.gt) if args.gt else None,
        "views": {},
        "note": ("body-relative upper/fore + chest in pelvis is the primary "
                 "metric; elbow/wrist 3D is in the pelvis frame; hand is "
                 "separate; world/geodesic is not a gate. GT is read-only."),
    }

    merged = {}
    for v in sorted(bases):
        base = bases[v]
        cand = cands[v]
        m, meta = merge_pose_tracks(base, cand, model, profile,
                                    joints=merge_joints)
        merged[v] = m
        if out_dir:
            p = os.path.join(out_dir, f"{v}_{args.merge_joints}_{args.tag}.vmd")
            write_vmd(p, m.get("model_name", ""), m["bones"],
                      m.get("morphs", {}), m.get("ik_frames", []))
        row = evaluate(f"{v}:{args.merge_joints}", m, base, gt_vmd, model,
                       profile, merge_joints=merge_joints, gt_pen=gt_pen)
        row["merge"] = meta
        row["candidate_raw"] = evaluate(
            f"{v}:full", cand, base, gt_vmd, model, profile,
            merge_joints=None, gt_pen=gt_pen)
        report["views"][v] = row

    if len(bases) > 1:
        first = sorted(bases)[0]
        fused = fuse_local_tracks([cands[v] for v in sorted(bases)],
                                  model, profile, merge_joints, args.fuse)
        m, meta = merge_pose_tracks(bases[first], fused, model, profile,
                                    joints=merge_joints)
        if out_dir:
            p = os.path.join(out_dir, f"multiview_{args.merge_joints}_{args.tag}.vmd")
            write_vmd(p, m.get("model_name", ""), m["bones"],
                      m.get("morphs", {}), m.get("ik_frames", []))
        report["multiview"] = evaluate(
            f"multiview:{args.merge_joints}", m, bases[first], gt_vmd, model,
            profile, merge_joints=merge_joints, gt_pen=gt_pen)
        report["multiview"]["merge"] = meta

    if args.reference:
        ref = parse_vmd(args.reference)
        ref_row = evaluate("m24c_arms_w0", ref, bases[sorted(bases)[0]],
                           gt_vmd, model, profile, merge_joints=None,
                           gt_pen=gt_pen)
        report["reference"] = ref_row

    # Concise decision table.  The reference M2.4c arms BA is 17.892 deg on
    # the front view; "clear win" here means at least 2 deg better than that
    # and no penetration/temporal/chest regression.
    ref_up = 17.892
    if "reference" in report and report["reference"].get("body"):
        ref_up = report["reference"]["body"]["upper_fore_deg"]
    decisions = {}
    for key, row in list(report["views"].items()) + (
            [("multiview", report["multiview"])] if "multiview" in report else []):
        if "body" not in row:
            continue
        up = row["body"]["upper_fore_deg"]
        delta = row["body"]["delta_upper_fore_deg"]
        budget_ok = row["budget"]["ok"]
        verify = row.get("verify") or {}
        temp = (verify.get("checks", {}).get("temporal_smoothness", {})
                or {})
        temp_ok = bool(temp.get("ok", True))
        chest_ok = (row["body"]["chest_rel_deg"] is not None
                    and row["body"]["base_chest_rel_deg"] is not None
                    and row["body"]["chest_rel_deg"]
                    <= row["body"]["base_chest_rel_deg"] + 1.0)
        pen_ok = (row["penetration"]["global_max"]
                  <= row["base_penetration"]["global_max"] + 0.02)
        decide = {
            "upper_fore_deg": up,
            "delta_vs_baseline_deg": delta,
            "beats_m24c_by_1deg": up <= ref_up - 1.0,
            "beats_m24c_by_2deg": up <= ref_up - 2.0,
            "meets_10deg": delta <= -10.0,
            "penetration_ok": bool(budget_ok and pen_ok),
            "temporal_ok": bool(temp_ok),
            "chest_ok": bool(chest_ok),
            "verify_ok": bool(verify.get("ok", False)),
            "candidate_ok": bool(budget_ok and pen_ok and temp_ok and chest_ok),
            "recommend_backend_or_init": bool(
                budget_ok and pen_ok and temp_ok and chest_ok
                and up <= ref_up - 1.0),
        }
        decisions[key] = decide
    report["decisions"] = decisions
    report["reference_upper_fore_deg"] = ref_up

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps(decisions, ensure_ascii=False, indent=1))
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
