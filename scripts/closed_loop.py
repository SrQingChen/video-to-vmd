"""Closed-loop harness: score a predicted VMD against a synthetic GT VMD.

Usage:
  python scripts/closed_loop.py --gt outputs/gt/si_R2_safe.vmd --pred <pred.vmd> \
      --model si --out outputs/closed_loop/my_run

Without --pred, runs the perturbation suite.  The suite is the meter's own
calibration (v2 2.9 / M0 review): every variant must move the component it is
built to attack and leave the unrelated ones alone, otherwise the vector is
not trustworthy as an M1/M2 gate.  ``summary_S`` is printed last and is never
used for ranking.
"""
import argparse
import json
import math
import os
import random
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

from vtv.metrics.core import compute_metrics, pose_track
from vtv.mmd.bake import ik_names, select_bake_bones
from vtv.mmd.pmx import parse_pmx
from vtv.mmd.vmd import parse_vmd
from vtv.mmd.writer import identity_vmd, write_vmd


def load_cfg(path=os.path.join(ROOT, "configs", "models.json")):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def axis_quat(axis, deg):
    h = math.radians(deg) * 0.5
    s = math.sin(h)
    return {"x": (s, 0, 0, math.cos(h)),
            "y": (0, s, 0, math.cos(h)),
            "z": (0, 0, s, math.cos(h))}[axis]


def axis_quat_vec(axis, deg):
    ax, ay, az = axis
    n = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
    h = math.radians(deg) * 0.5
    s = math.sin(h)
    return (s * ax / n, s * ay / n, s * az / n, math.cos(h))


def q_unity(q):
    return abs(q[3]) < 1.0 - 1e-9 or abs(q[0]) + abs(q[1]) + abs(q[2]) > 1e-9


def write_variant(gt, out_path, transform, drop_ik=False):
    """transform(name, frame, pos, rot) -> (pos, rot) or None to drop."""
    tracks = {}
    for name, keys in gt["bones"].items():
        rows = []
        for k in keys:
            res = transform(name, k["frame"], k["pos"], k["rot"])
            if res is None:
                continue
            pos, rot = res
            rows.append({"frame": k["frame"], "pos": pos, "rot": rot})
        if rows:
            tracks[name] = rows
    write_vmd(out_path, gt.get("model_name", ""), tracks, gt.get("morphs", {}),
              [] if drop_ik else gt.get("ik_frames", []))
    return tracks


def _rest_dir(model, bone_name):
    """Rest direction of a bone in its (world aligned) local frame."""
    idx = {b["name"]: i for i, b in enumerate(model["bones"])}
    if bone_name not in idx:
        return None
    i = idx[bone_name]
    b = model["bones"][i]
    tip = None
    if b.get("tail_bone") is not None and b["tail_bone"] >= 0:
        tip = model["bones"][b["tail_bone"]]["pos"]
    elif b.get("tail_offset") is not None and any(abs(c) > 1e-9 for c in b["tail_offset"]):
        off = b["tail_offset"]
        tip = [b["pos"][c] + off[c] for c in range(3)]
    elif b.get("fixed_axis"):
        # twist bones have a zero-length tail; their own axis is the fixed axis
        return list(b["fixed_axis"])
    else:
        kids = [x for x in model["bones"] if x.get("parent") == i]
        if kids:
            tip = kids[0]["pos"]
    if tip is None:
        return None
    v = [tip[c] - b["pos"][c] for c in range(3)]
    n = math.sqrt(sum(c * c for c in v))
    return [c / n for c in v] if n > 1e-9 else None


def make_variants(model, profile, gt, out_dir):
    """Build the M0/M1 perturbation suite + the expected component response.

    Each entry: path, the component it is designed to attack, and how it is
    scored.  ``solve_ik=True`` entries are evaluated with IK enabled.
    """
    out = {}
    bake_bones = select_bake_bones(model, profile)
    iks = ik_names(model)
    smap = profile["standard_map"]
    body_bones = [v for k, v in smap.items() if not k.startswith(
        ("all_parent", "groove", "waist", "clavicle_p", "clavicle_c",
         "forearm_twist", "wrist_twist", "thigh_d", "shin_d", "foot_d"))]
    body_bones = sorted(set(body_bones))

    def add(name, fn, **meta):
        path = os.path.join(out_dir, f"pred_{name}.vmd")
        write_variant(gt, path, fn, drop_ik=meta.get("drop_ik", False))
        out[name] = {"path": path, **meta}

    # 1) exact copy must score zero everywhere
    path = os.path.join(out_dir, "pred_copy.vmd")
    write_vmd(path, gt.get("model_name", ""), gt["bones"], gt.get("morphs", {}),
              gt.get("ik_frames", []))
    out["copy"] = {"path": path, "targets": [], "desc": "identical VMD"}

    # 2) bind pose (identity) must blow up angle/position, not be "quiet"
    idv = identity_vmd(bake_bones, ik_names=iks)
    path = os.path.join(out_dir, "pred_identity.vmd")
    write_vmd(path, gt.get("model_name", ""), idv["bone_frames"], {},
              idv["ik_frames"])
    out["identity"] = {"path": path, "targets": ["angle", "position", "world_angle"],
                       "desc": "rest pose, IK off"}

    # 3) independent 2 deg noise on every key: temporal only, angle stays small
    rng = random.Random(7)
    noise = {}
    for name, keys in gt["bones"].items():
        noise[name] = {}
        for k in keys:
            noise[name][k["frame"]] = qmul(
                axis_quat(rng.choice("xyz"), rng.uniform(-2.0, 2.0)), k["rot"])

    def f_noise(n, f, p, r):
        return p, noise.get(n, {}).get(f, r)
    add("noise2deg", f_noise, targets=["temporal"],
        desc="2 deg random rotation on every bone/frame")

    # 4) 3-frame-ish spikes on arms: temporal only
    spike_bones = {"左腕", "右腕", "左ひじ", "右ひじ", "左足", "右足"}
    spikes = {10, 60, 120, 180}

    def f_spike(n, f, p, r):
        if n in spike_bones and f in spikes:
            return p, qmul(axis_quat("x", 15.0), r)
        return p, r
    add("spike15deg", f_spike, targets=["temporal"],
        desc="15 deg single-frame spikes on 6 limb bones")

    # 5) smooth constant bias on both thighs: angle, no temporal
    def f_leg(n, f, p, r):
        if n in ("左足", "右足"):
            return p, qmul(axis_quat("x", 10.0), r)
        return p, r
    add("leg10deg", f_leg, targets=["angle", "position"],
        desc="constant +10 deg X on both thighs")

    # 6) A/T-pose style bias on every body bone: angle/world angle, no temporal
    def f_rest(n, f, p, r):
        if n in body_bones:
            return p, qmul(axis_quat("x", 5.0), r)
        return p, r
    add("rest_bias5deg", f_rest, targets=["angle", "world_angle", "position"],
        desc="constant +5 deg X on all 23 body bones")

    # 7) both feet lifted in the middle of the clip: contact pattern only
    def f_lift(n, f, p, r):
        if n in ("左足", "右足") and 60 <= f <= 180:
            w = 0.5 - 0.5 * math.cos(2.0 * math.pi * (f - 60) / 120.0)
            return p, qmul(axis_quat("x", 35.0 * w), r)
        return p, r
    add("foot_lift", f_lift, targets=["contact"],
        desc="smooth 35 deg thigh bump (+2.3 units toe), frames 60-180")

    # 8) fast ankle yaw in the middle: toe slides at constant height
    def f_slide(n, f, p, r):
        if n in ("左足首", "右足首") and 95 <= f <= 155:
            w = 0.5 - 0.5 * math.cos(2.0 * math.pi * (f - 95) / 60.0)
            return p, qmul(r, axis_quat("y", 45.0 * w))
        return p, r
    add("foot_slide", f_slide, targets=["contact"],
        desc="bone-local 45 deg ankle yaw, frames 95-155, height kept")

    # 9) constant root sink: penetration only (root traj/position are relative)
    def f_pen(n, f, p, r):
        if n == profile["standard_map"]["root"]:
            return (p[0], p[1] - 1.5, p[2]), r
        return p, r
    add("penetration", f_pen, targets=["contact"],
        desc="root -1.5 units in Y for the whole clip")

    # 10) linear root drift: root components (also couples into foot slide)
    def f_drift(n, f, p, r):
        if n == profile["standard_map"]["root"]:
            return (p[0] + 2.0 * f / max(gt["max_frame"], 1), p[1], p[2]), r
        return p, r
    add("root_drift", f_drift, targets=["root"],
        desc="root +2 units in X ramped over the clip")

    # 11) axial twist on the main twist bones: world orientation only
    twist_axes = {}
    for side in ("l", "r"):
        for key in ("forearm", "wrist"):
            rec = (profile.get("twist") or {}).get(f"{key}_{side}") or {}
            main = rec.get("main")
            if not main:
                continue
            d = _rest_dir(model, main)
            if d:
                twist_axes[main] = d

    def f_twist(n, f, p, r):
        if n in twist_axes:
            return p, qmul(axis_quat_vec(twist_axes[n], 30.0), r)
        return p, r
    add("twist30deg", f_twist, targets=["world_angle"],
        desc="30 deg axial twist on 腕捩/手捩 (positions invariant)")

    # 12) IK left on: the legs get dragged toward the resting IK targets
    add("ik_on", lambda n, f, p, r: (p, r), targets=["angle", "position", "contact"],
        desc="same tracks, IK-off records removed, evaluated with IK on",
        drop_ik=True, solve_ik=True)
    return out


COMPONENTS = ["angle_mean_deg", "angle_p99_deg", "world_angle_mean_deg",
              "position_pct_height", "root_traj_pct_height", "root_drift_pct_height",
              "contact_recall", "contact_precision", "contact_slide_delta",
              "contact_penetration_delta", "contact_height_mae_pct_height",
              "jerk_rms_ratio", "spike_delta"]

# Meter calibration (v2 2.9 / M0 review Step 0 gate).  Each rule says which
# component a variant must move ("min") and which must stay quiet ("max"), so
# a broken scorer fails loudly instead of silently producing plausible numbers.
CHECKS = {
    "copy": {"max": {"angle_mean_deg": 1e-6, "world_angle_mean_deg": 1e-6,
                     "position_pct_height": 1e-6, "contact_slide_delta": 1e-6,
                     "contact_penetration_delta": 1e-6}},
    "identity": {"min": {"angle_mean_deg": 20, "position_pct_height": 3,
                         "world_angle_mean_deg": 20}},
    # Uncorrelated 2 deg noise on every bone/frame is a *white* stress test: a
    # 2 deg ankle jitter is a 0.4 unit toe motion, i.e. physically a sliding
    # foot, so only the temporal components are constrained here.
    "noise2deg": {"min": {"jerk_rms_ratio": 5}, "max": {"angle_mean_deg": 3}},
    "spike15deg": {"min": {"spike_delta": 20, "jerk_rms_ratio": 3},
                   "max": {"angle_mean_deg": 1, "position_pct_height": 0.2}},
    "leg10deg": {"min": {"angle_p99_deg": 8, "position_pct_height": 1},
                 "max": {"jerk_rms_ratio": 2, "spike_delta": 5}},
    "rest_bias5deg": {"min": {"angle_mean_deg": 4, "world_angle_mean_deg": 15},
                      "max": {"jerk_rms_ratio": 2}},
    "foot_lift": {"min": {"contact_height_mae_pct_height": 1.5, "contact_slide_delta": 0.2},
                  "max": {"jerk_rms_ratio": 2, "contact_penetration_delta": 0.01}},
    "foot_slide": {"min": {"contact_slide_delta": 0.5},
                   "max": {"contact_height_mae_pct_height": 0.2, "jerk_rms_ratio": 2,
                           "contact_penetration_delta": 0.01}},
    "penetration": {"min": {"contact_penetration_delta": 1.0},
                    "max": {"position_pct_height": 0.1, "root_traj_pct_height": 0.2,
                            "angle_mean_deg": 0.1}},
    "root_drift": {"min": {"root_traj_pct_height": 3, "root_drift_pct_height": 5},
                   "max": {"angle_mean_deg": 0.1, "position_pct_height": 0.1}},
    "twist30deg": {"min": {"world_angle_mean_deg": 2},
                   "max": {"angle_mean_deg": 0.1, "contact_slide_delta": 0.02}},
    "ik_on": {"min": {"angle_mean_deg": 0.5}},
}


def run_checks(results):
    """Evaluate CHECKS and return per-rule pass/fail with the observed values."""
    out = {}
    for name, rules in CHECKS.items():
        if name not in results:
            continue
        v = results[name]["vector"]
        rows, ok = [], True
        for kind, comps in rules.items():
            for comp, bound in comps.items():
                val = v.get(comp)
                if val is None:
                    good = False
                elif kind == "min":
                    good = val >= bound
                else:
                    good = val <= bound
                ok = ok and good
                rows.append({"rule": kind, "component": comp, "bound": bound,
                             "value": val, "ok": bool(good)})
        out[name] = {"ok": bool(ok), "rules": rows}
    return out


def fmt_row(name, m):
    v = m["vector"]
    return (f"{name:15s} ang={v['angle_mean_deg']:6.2f}/{v['angle_p99_deg']:6.2f} "
            f"wAng={v['world_angle_mean_deg']:6.2f} pos%={v['position_pct_height']:6.2f} "
            f"root%={v['root_traj_pct_height']:6.2f}/{v['root_drift_pct_height']:6.2f} "
            f"rec={v['contact_recall'] if v['contact_recall'] is None else round(v['contact_recall'], 2)!s:>4s} "
            f"slideΔ={v['contact_slide_delta']:7.2f} fH%={v['contact_height_mae_pct_height']:5.2f} "
            f"penΔ={v['contact_penetration_delta']:6.2f} "
            f"jerk×={v['jerk_rms_ratio']:7.2f} spikeΔ={v['spike_delta']:+4d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "models.json"))
    ap.add_argument("--model", default="si")
    ap.add_argument("--gt", default=os.path.join(ROOT, "outputs", "gt", "si_R2_safe.vmd"))
    ap.add_argument("--pred", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "closed_loop", "m0"))
    ap.add_argument("--frames", default=None, help="a:b (inclusive) frame subset")
    ap.add_argument("--solve-ik", action="store_true")
    ap.add_argument("--contact-margin", type=float, default=0.02,
                    help="contact tolerance as a fraction of PMX height")
    ap.add_argument("--only", default=None, help="comma separated variant names")
    args = ap.parse_args()

    cfg = load_cfg(args.config)["models"][args.model]
    profile = json.load(open(os.path.join(ROOT, "profiles", f"{args.model}.json"),
                             encoding="utf-8"))
    model = parse_pmx(cfg["pmx"])
    gt = parse_vmd(args.gt)
    frames = None
    if args.frames:
        a, b = [int(x) for x in args.frames.split(":")]
        frames = list(range(a, b + 1))
    os.makedirs(args.out, exist_ok=True)

    gt_track = pose_track(model, gt, profile, frames=frames,
                          solve_ik=args.solve_ik,
                          apply_all_translations=False)

    if args.pred:
        pred = parse_vmd(args.pred)
        pred_track = pose_track(model, pred, profile, frames=gt_track["frames"],
                                solve_ik=args.solve_ik,
                                apply_all_translations=False)
        score = compute_metrics(gt_track, pred_track, profile,
                                contact_margin=args.contact_margin)
        score["name"] = os.path.basename(args.pred)
        path = os.path.join(args.out, "metrics.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(score, f, ensure_ascii=False, indent=1)
        print(json.dumps(score["vector"], ensure_ascii=False, indent=1))
        print("summary_S (not a gate) =", score["summary_S"]["value"], "->", path)
        return

    variants = make_variants(model, profile, gt, args.out)
    keep = set(args.only.split(",")) if args.only else None
    results = {}
    for name, rec in variants.items():
        if keep and name not in keep:
            continue
        pred = parse_vmd(rec["path"])
        solve_ik = bool(rec.get("solve_ik", args.solve_ik))
        tr = pose_track(model, pred, profile, frames=gt_track["frames"],
                        solve_ik=solve_ik, apply_all_translations=False)
        m = compute_metrics(gt_track, tr, profile, contact_margin=args.contact_margin)
        m["name"] = name
        m["desc"] = rec.get("desc")
        m["targets"] = rec.get("targets", [])
        results[name] = m
        results[name]["vmd"] = os.path.relpath(rec["path"], ROOT)
        print(fmt_row(name, m))

    print()
    checks = run_checks(results)
    for name, m in results.items():
        mark = "" if name not in checks else ("  OK" if checks[name]["ok"] else "  FAIL")
        print(f"{name:15s} targets={','.join(m['targets']) or '-':28s} "
              f"summary_S={m['summary_S']['value']:8.3f} (not a gate){mark}")
        if mark == "  FAIL":
            for r in checks[name]["rules"]:
                if not r["ok"]:
                    print(f"    ! {r['rule']} {r['component']} = {r['value']} "
                          f"(need {'>=' if r['rule'] == 'min' else '<='} {r['bound']})")
    n_ok = sum(1 for c in checks.values() if c["ok"])
    print(f"\nmeter calibration: {n_ok}/{len(checks)} variants hit only their target components")
    with open(os.path.join(args.out, "baselines.json"), "w", encoding="utf-8") as f:
        json.dump({"gt": os.path.relpath(args.gt, ROOT),
                   "model": args.model,
                   "contact_margin": args.contact_margin,
                   "components": COMPONENTS,
                   "note": "gate on vector components; summary_S is a report only",
                   "checks": checks,
                   "checks_ok": f"{n_ok}/{len(checks)}",
                   "results": results}, f, ensure_ascii=False, indent=1)
    print("WROTE", os.path.join(args.out, "baselines.json"))


if __name__ == "__main__":
    main()
