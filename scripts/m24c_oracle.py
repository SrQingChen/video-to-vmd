"""M2.4c GT-only ceiling for a torso-relative arm prior (diagnostic only).

The repo has no VPoser/AMASS weights (VPoser's official download redirects to
a login; AMASS is not present).  This script therefore measures the ceiling of
the *prior-only* path without pretending to have a learned prior:

1. Take the GT torso-relative upper-arm/forearm directions, rotate them into
   the **predicted** chest frame, and blend from the prediction to that oracle
   target in torso-relative space (alpha 0..1).
2. Apply the result with ``apply_direction_prior(signed=True)``.  The signed
   path is required: the triangulation path flips a line toward the prediction,
   which produced 122-179 deg false jerk in M2.4b.
3. Report body-relative full/upper-fore, global and per-pair penetration, and
   temporal steps.
4. Report the GT pose's per-pair penetration vs the baseline.  This is the
   evidence that the M2.4b "no pair may increase vs baseline" rule is not
   satisfiable by GT itself on this clip.

GT is read-only.  Nothing here is a pipeline stage or a candidate.
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

from vtv.adapter import q_conj, qrot
from vtv.arm3d import (apply_direction_prior, body_relative_errors,
                       penetration)
from vtv.cleanup import dense_tracks, eval_tracks
from vtv.mmd.pmx import parse_pmx
from vtv.mmd.vmd import parse_vmd

ALPHAS = (0.0, 0.25, 0.5, 0.6, 0.75, 1.0)


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def pair_map(pen):
    out = {}
    for g in ("arm_torso", "arm_arm"):
        for k, rec in (pen[g].get("pairs_detail") or {}).items():
            out[f"{g}/{k}"] = round(float(rec.get("max", 0.0)), 4)
    return out


def steps(track):
    vals = []
    for side in ("l", "r"):
        for a, b in (("upperarm", "forearm"), ("forearm", "hand")):
            j, c = f"{a}_{side}", f"{b}_{side}"
            d = np.asarray(track["world"][c]) - track["world"][j]
            d = unit(d)
            vals.append(np.degrees(np.arccos(np.clip(
                (d[1:] * d[:-1]).sum(1), -1.0, 1.0))))
    s = np.concatenate(vals)
    return {"max_deg": round(float(s.max()), 3),
            "p99_deg": round(float(np.percentile(s, 99)), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "models.json"))
    ap.add_argument("--model", default="si")
    ap.add_argument("--gt", default=os.path.join(ROOT, "outputs", "gt", "si_R2_safe.vmd"))
    ap.add_argument("--vmd", default=os.path.join(
        ROOT, "outputs", "m2", "vmd", "front_clean_ik.vmd"))
    ap.add_argument("--cap", type=float, default=60.0)
    ap.add_argument("--out", default=os.path.join(
        ROOT, "outputs", "m2", "ba3d", "oracle.json"))
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8"))["models"][args.model]
    prof = json.load(open(os.path.join(ROOT, "profiles", f"{args.model}.json"),
                          encoding="utf-8"))
    model = parse_pmx(cfg["pmx"])
    gt_vmd = parse_vmd(args.gt)
    base_vmd = parse_vmd(args.vmd)
    fg, tg = dense_tracks(gt_vmd)
    fp, tp = dense_tracks(base_vmd)
    trg = eval_tracks(model, prof, tg)
    trp = eval_tracks(model, prof, tp)
    F = min(len(fg), len(fp))
    qcg = trg["world_q"]["chest"][:F]
    qcp = trp["world_q"]["chest"][:F]
    base = {}
    for side in ("l", "r"):
        for seg, j, child in (("upper", f"upperarm_{side}", f"forearm_{side}"),
                              ("fore", f"forearm_{side}", f"hand_{side}")):
            base[f"{seg}_{side}"] = unit(
                np.asarray(trp["world"][child][:F]) - np.asarray(trp["world"][j][:F]))
    gt_rel = {}
    for side in ("l", "r"):
        for seg, j, child in (("upper", f"upperarm_{side}", f"forearm_{side}"),
                              ("fore", f"forearm_{side}", f"hand_{side}")):
            dg = unit(np.asarray(trg["world"][child][:F])
                      - np.asarray(trg["world"][j][:F]))
            gt_rel[f"{seg}_{side}"] = np.asarray([
                qrot(tuple(q_conj(tuple(qcg[f]))), tuple(dg[f]))
                for f in range(F)])

    br0 = body_relative_errors(trg, trp, model, prof)
    p0 = penetration(base_vmd, model, prof)
    out = {
        "mode": "M2.4c oracle torso-relative prior ceiling (GT only)",
        "gt": os.path.abspath(args.gt),
        "init": os.path.abspath(args.vmd),
        "cap_deg": args.cap,
        "alphas": {},
        "baseline": {
            "body_full_deg": br0["arm_body_mean_deg"],
            "body_upper_fore_deg": round(float(np.mean(
                [br0["sides"][s][b] for s in ("l", "r")
                 for b in ("upperarm", "forearm")])), 3),
            "penetration": {"arm_torso": p0["arm_torso"]["max"],
                            "arm_arm": p0["arm_arm"]["max"]},
            "steps": steps(trp)},
    }
    for alpha in ALPHAS:
        targets = {}
        for side in ("l", "r"):
            for seg in ("upper", "fore"):
                k = f"{seg}_{side}"
                cur_rel = np.asarray([
                    qrot(tuple(q_conj(tuple(qcp[f]))), tuple(base[k][f]))
                    for f in range(F)])
                rel = unit((1.0 - alpha) * cur_rel + alpha * gt_rel[k])
                targets[k] = unit(np.asarray([
                    qrot(tuple(qcp[f]), tuple(rel[f])) for f in range(F)]))
        res = apply_direction_prior(base_vmd, model, prof, targets,
                                    cap_deg=args.cap, pen_guard=False,
                                    signed=True)
        vmd = res["vmd"]
        fo, to = dense_tracks(vmd)
        tro = eval_tracks(model, prof, to)
        br = body_relative_errors(trg, tro, model, prof)
        pen = penetration(vmd, model, prof)
        out["alphas"][str(alpha)] = {
            "body_full_deg": br["arm_body_mean_deg"],
            "body_upper_fore_deg": round(float(np.mean(
                [br["sides"][s][b] for s in ("l", "r")
                 for b in ("upperarm", "forearm")])), 3),
            "penetration": {"arm_torso": pen["arm_torso"]["max"],
                            "arm_arm": pen["arm_arm"]["max"]},
            "pairs": pair_map(pen),
            "steps": steps(tro),
        }
    # GT's own per-pair contact vs the baseline: the strict M2.4b rule is
    # impossible for a pose closer to GT on this clip.
    pg = penetration(gt_vmd, model, prof)
    pb = pair_map(p0)
    ga = pair_map(pg)
    out["gt_vs_baseline_pairs"] = {
        k: {"baseline": pb.get(k, 0.0), "gt": ga.get(k, 0.0),
            "strict_increase": bool(ga.get(k, 0.0) > pb.get(k, 0.0) + 0.02)}
        for k in sorted(set(pb) | set(ga))}
    out["note"] = (
        "alpha=1 is the GT torso-relative target applied to the PREDICTED "
        "chest; it is not a candidate. The oracle shows the prior-only ceiling "
        "and the torso coupling. GT itself increases four arm-torso pairs vs "
        "the baseline, so the M2.4b strict per-pair rule is not a valid gate "
        "for this clip.")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1)[:5000])
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
