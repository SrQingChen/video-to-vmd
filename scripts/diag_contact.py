# -*- coding: utf-8 -*-
"""Diagnose foot-height bias vs GT and the reachable leg-IK correction.

GT is read **only for evaluation/reporting** here; nothing in this script is
part of the production solver.  For every view/side it reports

* the per-frame toe height error and how constant that bias is,
* the GT-planted recall of the current prediction,
* the per-foot lowest toe, the shared ground estimate and the resulting
  GT-free target band,
* the reachable ankle drop (how far the 2-bone leg can straighten) and the
  knee rotation an exact vertical IK would need.

Usage:
  python scripts/diag_contact.py --model si --gt outputs/gt/si_R2_safe.vmd \
      --pred outputs/m2/vmd/front_2d_clean.vmd --out outputs/m2/diag_front.json
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

from vtv.metrics.core import pose_track, DEFAULT_CONTACT_MARGIN
from vtv.mmd.pmx import parse_pmx
from vtv.mmd.vmd import parse_vmd


def load_cfg(path=os.path.join(ROOT, "configs", "models.json")):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def angle_between(a, b):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)))


def two_bone(H, K, A, T):
    """Exact 2-bone IK in the current knee plane; returns new K (N,3)."""
    d1 = K - H
    d2 = A - K
    L1 = np.linalg.norm(d1, axis=1)
    L2 = np.linalg.norm(d2, axis=1)
    v = T - H
    dist = np.linalg.norm(v, axis=1)
    dist = np.clip(dist, np.abs(L1 - L2) + 1e-6, L1 + L2 - 1e-6)
    u = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    n = np.cross(d1, d2)
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    # if the current plane is degenerate use any perpendicular
    bad = ~np.isfinite(n).all(axis=1) | (np.linalg.norm(n, axis=1) < 1e-6)
    if bad.any():
        alt = np.cross(u[bad], np.array([0.0, 0.0, 1.0]))
        alt = alt / (np.linalg.norm(alt, axis=1, keepdims=True) + 1e-12)
        n[bad] = alt
    cos_a = (L1 ** 2 + dist ** 2 - L2 ** 2) / (2.0 * L1 * dist + 1e-12)
    a = np.arccos(np.clip(cos_a, -1.0, 1.0))
    perp = np.cross(n, u)
    w = np.cos(a)[:, None] * u + np.sin(a)[:, None] * perp
    # keep the knee on the same side as the current one
    side = np.sign(np.sum((d1 - np.sum(d1 * u, axis=1, keepdims=True) * u) * perp,
                          axis=1))
    side[side == 0] = 1.0
    w = np.cos(a)[:, None] * u + (side * np.sin(a))[:, None] * perp
    return H + L1[:, None] * w, {"L1": L1, "L2": L2, "dist": dist,
                                 "reach": L1 + L2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "models.json"))
    ap.add_argument("--model", default="si")
    ap.add_argument("--gt", default=os.path.join(ROOT, "outputs", "gt", "si_R2_safe.vmd"))
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--margin", type=float, default=DEFAULT_CONTACT_MARGIN)
    ap.add_argument("--band", type=float, default=0.02,
                    help="contact band as a fraction of height (metric uses 0.02)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)["models"][args.model]
    profile = json.load(open(os.path.join(ROOT, "profiles", f"{args.model}.json"),
                             encoding="utf-8"))
    model = parse_pmx(cfg["pmx"])
    gt = parse_vmd(args.gt)
    pred = parse_vmd(args.pred)
    g = pose_track(model, gt, profile, solve_ik=False)
    p = pose_track(model, pred, profile, solve_ik=False)
    height = float(g["height"])

    report = {"model": args.model, "gt": args.gt, "pred": args.pred,
              "height": height, "band_units": args.band * height, "sides": {}}
    gmin = []
    for side in ("l", "r"):
        gt_toe = g["world"][f"toe_{side}"][:, 1]
        pr_toe = p["world"][f"toe_{side}"][:, 1]
        gmin.append(float(gt_toe.min()))
        thr = max(0.0, float(gt_toe.min())) + args.band * height
        mask = gt_toe < thr
        err = pr_toe - gt_toe
        H = p["world"][f"thigh_{side}"]
        K = p["world"][f"shin_{side}"]
        A = p["world"][f"foot_{side}"]
        T = A.copy()
        # oracle constant bias: shift the whole predicted track by -mean(err)
        oracle = pr_toe - float(err[mask].mean())
        rec_oracle = float((oracle < thr).mean())
        # reachable drop: how far the ankle can go straight down
        d1 = K - H
        d2 = A - K
        L1 = np.linalg.norm(d1, axis=1)
        L2 = np.linalg.norm(d2, axis=1)
        reach = L1 + L2
        dist = np.linalg.norm(A - H, axis=1)
        max_drop = reach - dist
        knee_now = angle_between(K - H, A - K)
        rec = {
            "gt_min": round(float(gt_toe.min()), 4),
            "gt_max": round(float(gt_toe.max()), 4),
            "gt_thr": round(thr, 4),
            "gt_contact_fraction": round(float(mask.mean()), 4),
            "pred_min": round(float(pr_toe.min()), 4),
            "pred_max": round(float(pr_toe.max()), 4),
            "pred_below_thr": round(float((pr_toe < thr).mean()), 4),
            "err_mean": round(float(err.mean()), 4),
            "err_std": round(float(err.std()), 4),
            "err_contact_mean": round(float(err[mask].mean()), 4),
            "err_contact_std": round(float(err[mask].std()), 4),
            "oracle_const_bias_recall": round(rec_oracle, 4),
            "leg_L1": round(float(L1.mean()), 4),
            "leg_L2": round(float(L2.mean()), 4),
            "leg_dist_mean": round(float(dist.mean()), 4),
            "leg_dist_max": round(float(dist.max()), 4),
            "max_drop_mean": round(float(max_drop.mean()), 4),
            "max_drop_min": round(float(max_drop.min()), 4),
            "knee_deg_mean": round(float(knee_now.mean()), 4),
        }
        # exact vertical IK at the oracle bias, check needed knee/thigh change
        target_drop = np.maximum(0.0, err - float(err[mask].mean()))
        Ta = A + np.stack([np.zeros_like(target_drop), -target_drop,
                           np.zeros_like(target_drop)], axis=1)
        K2, geo = two_bone(H, K, A, Ta)
        thigh_deg = angle_between(K - H, K2 - H)
        shin_deg = angle_between(A - K, Ta - K2)
        knee2 = angle_between(K2 - H, Ta - K2)
        knee_delta = np.abs(knee2 - knee_now)
        sel = target_drop > 1e-3
        rec.update({
            "ik_frames": int(sel.sum()),
            "ik_drop_max": round(float(target_drop.max()), 4),
            "ik_thigh_deg_max": round(float(thigh_deg[sel].max()), 3) if sel.any() else 0.0,
            "ik_thigh_deg_p95": round(float(np.percentile(thigh_deg[sel], 95)), 3) if sel.any() else 0.0,
            "ik_knee_delta_max": round(float(knee_delta[sel].max()), 3) if sel.any() else 0.0,
            "ik_knee_delta_p95": round(float(np.percentile(knee_delta[sel], 95)), 3) if sel.any() else 0.0,
            "ik_frames_over_15deg": int((knee_delta[sel] > 15.0).sum()) if sel.any() else 0,
            "ik_frames_over_20deg": int((knee_delta[sel] > 20.0).sum()) if sel.any() else 0,
        })
        report["sides"][side] = rec

    report["gt_ground"] = round(min(gmin), 4)
    # GT-free shared ground estimate: lowest toe over the whole clip
    all_toe = np.concatenate([p["world"][f"toe_{s}"][:, 1] for s in ("l", "r")])
    report["pred_ground_estimate"] = round(float(all_toe.min()), 4)
    report["pred_ground_p1"] = round(float(np.percentile(all_toe, 1.0)), 4)
    txt = json.dumps(report, ensure_ascii=False, indent=1)
    print(txt)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(txt)


if __name__ == "__main__":
    main()
