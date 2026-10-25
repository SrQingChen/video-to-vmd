# -*- coding: utf-8 -*-
"""Per-bone + spatial + contact + temporal metrics for the closed loop.

Every quantity is reported as its own vector component.  ``summary_S`` is a
weighted sum kept only as a one-line summary of a *report*: it is NOT a gate
and it must not be used to rank runs.  The M0 baselines showed why: a relative
jerk term saturates at 100 for 2 deg of noise and 3-frame spikes, so a smooth
10 deg leg offset scored 3.32 while harmless noise scored 58.54.

Gate on the vector components (with per-bone / p99 / per-group breakdowns):

    rotation     angle_mean_deg / angle_p99_deg / world_angle_*_deg
    position     position_pct_height / pamjpe_pct_height ? position_p99_*
    root         root_traj_pct_height / root_vel_* / root_drift_pct_height
    contact      contact_recall / contact_precision / contact_slide_* /
                 contact_penetration_* / contact_fraction_*
    temporal     jerk_rms_ratio / spike_delta / spike_rate_pred / accel_p99_*

Local rotations are the primary per-bone signal, but they are convention
dependent: a bone whose parent is an intentionally-unwritten helper (twist,
shoulder C) can have identical world motion with different local values.
``world_angle_*`` is convention free, so use it whenever a run changes the
written bone set.

All positions are MMD units; ``*_pct_height`` divides by the PMX bbox height so
numbers are comparable across the three characters.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..mmd.evaluate import VmdEvaluator

BODY_JOINTS = [
    "root", "pelvis", "spine", "spine1", "chest", "neck", "head",
    "clavicle_l", "upperarm_l", "forearm_l", "hand_l",
    "clavicle_r", "upperarm_r", "forearm_r", "hand_r",
    "thigh_l", "shin_l", "foot_l", "toe_l",
    "thigh_r", "shin_r", "foot_r", "toe_r",
]
GROUPS = {
    "spine": ["pelvis", "spine", "spine1", "chest", "neck", "head"],
    "shoulder": ["clavicle_l", "clavicle_r", "upperarm_l", "upperarm_r"],
    "elbow": ["forearm_l", "forearm_r"],
    "wrist": ["hand_l", "hand_r"],
    "hip": ["thigh_l", "thigh_r"],
    "knee": ["shin_l", "shin_r"],
    "ankle": ["foot_l", "foot_r"],
    "toe": ["toe_l", "toe_r"],
}

# A foot counts as planted when its toe is within ``contact_margin * height``
# of the lowest toe height seen in the GT clip (self referencing: R2's toes
# float 0.1-0.56 units above the rest position, so an absolute rest-anchored
# threshold either marks every frame or none).  0.01 * height = 2.2 units for
# 祀 is a 2.2 cm planted band; M0 used 0.05 * height and every GT frame became
# "contact", which made recall/precision meaningless.
DEFAULT_CONTACT_MARGIN = 0.01
MIN_SPIKE_SIGMAS = 6.0


def q_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geodesic angle between quaternions (broadcast over the last axis)."""
    d = np.abs(np.sum(a * b, axis=-1))
    d = np.clip(d, -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(d))


def pose_track(model: Dict, vmd: Dict, profile: Dict,
               frames: Optional[Iterable[int]] = None,
               solve_ik: bool = False,
               apply_all_translations: bool = False) -> Dict:
    """Evaluate a VMD on a PMX skeleton and collect the canonical joints.

    Returns world positions *and* world rotations (rotations make the metric
    independent of how a run splits twists between main and helper bones).
    """
    if frames is None:
        frames = range(vmd["max_frame"] + 1)
    frames = list(frames)
    ev = VmdEvaluator(model, vmd, solve_ik=solve_ik,
                      force_ik_off=not solve_ik,
                      apply_all_translations=apply_all_translations)
    name2i = {b["name"]: i for i, b in enumerate(model["bones"])}
    joints = {}
    for j, bn in profile["standard_map"].items():
        if bn in name2i:
            joints[j] = bn
    local_q = {}
    world_q = {}
    world = {}
    for j, bn in joints.items():
        i = name2i[bn]
        recs = [ev.frame(f) for f in frames]
        local_q[j] = np.asarray([r["local_q"][i] for r in recs], dtype=np.float64)
        world_q[j] = np.asarray([r["world_q"][i] for r in recs], dtype=np.float64)
        world[j] = np.asarray([r["world_pos"][i] for r in recs], dtype=np.float64)
    fingers = {}
    for j, bn in profile.get("finger_map", {}).items():
        if bn not in name2i:
            continue
        i = name2i[bn]
        local_q[j] = np.asarray([ev.frame(f)["local_q"][i] for f in frames],
                                dtype=np.float64)
        world[j] = np.asarray([ev.frame(f)["world_pos"][i] for f in frames],
                              dtype=np.float64)
        fingers[j] = bn
    toe_rest = {}
    for side in ("l", "r"):
        bn = profile["standard_map"].get(f"toe_{side}")
        if bn and bn in name2i:
            toe_rest[side] = float(model["bones"][name2i[bn]]["pos"][1])
    return {
        "frames": frames,
        "local_q": local_q,
        "world_q": world_q,
        "world": world,
        "finger_joints": list(fingers),
        "height": float(profile.get("height") or 1.0),
        "toe_rest_y": toe_rest,
        "model_key": profile.get("model_key"),
    }


def _root_align(world: Dict[str, np.ndarray], ref_joint: str = "pelvis"):
    if ref_joint not in world:
        ref_joint = next(iter(world))
    ref = world[ref_joint]
    return {j: v - ref for j, v in world.items()}, ref


def _pamjpe(gt: Dict[str, np.ndarray], pred: Dict[str, np.ndarray]) -> float:
    """Per-frame similarity-aligned MPJPE (Umeyama), in units."""
    keys = sorted(set(gt) & set(pred))
    if not keys:
        return float("nan")
    errs = []
    for t in range(len(gt[keys[0]])):
        X = np.stack([pred[k][t] for k in keys])
        Y = np.stack([gt[k][t] for k in keys])
        cx, cy = X.mean(0), Y.mean(0)
        X0, Y0 = X - cx, Y - cy
        cov = X0.T @ Y0 / len(keys)
        U, S, Vt = np.linalg.svd(cov)
        d = np.sign(np.linalg.det(U @ Vt))
        D = np.diag([1, 1, d])
        R = U @ D @ Vt
        var = (X0 ** 2).sum() / len(keys)
        scale = S.sum() * (1 if d >= 0 else -1) / var if var > 1e-12 else 1.0
        Xa = (scale * (X0 @ R.T)) + cy
        errs.append(np.linalg.norm(Xa - Y, axis=1).mean())
    return float(np.mean(errs))


def _intervals(mask: np.ndarray) -> List[tuple]:
    """Closed intervals [start, end] of True runs."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    out, start, prev = [], int(idx[0]), int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i == prev + 1:
            prev = i
            continue
        out.append((start, prev))
        start = prev = i
    out.append((start, prev))
    return out


def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    """Centred moving average with edge replication (no zero padding)."""
    if win <= 1 or len(x) < win:
        return x
    pad = win // 2
    xp = np.concatenate([np.repeat(x[:1], pad, axis=0), x,
                         np.repeat(x[-1:], pad, axis=0)], axis=0)
    k = np.ones(win) / win
    return np.stack([np.convolve(xp[:, c], k, mode="valid") for c in range(x.shape[1])],
                    axis=1)


def _slide_drift(points: np.ndarray, intervals: Sequence[tuple],
                 smooth: int = 5) -> float:
    """Net horizontal travel inside each interval (endpoint drift).

    Summing |delta| would count jitter as slide (2 deg of noise moves the toe
    ~0.4 units at 30 fps); the endpoint drift of the *smoothed* trajectory
    measures a foot that is actually being dragged while planted.  The 5-frame
    (0.17 s) average is far below any real slide and above per-frame jitter.
    """
    p = _smooth(points[:, [0, 2]], smooth)
    total = 0.0
    for a, b in intervals:
        if b <= a:
            continue
        d = p[b] - p[a]
        total += float(np.linalg.norm(d))
    return total


def _contact_state_from_gt(gt_world: Dict[str, np.ndarray], height: float,
                           ground_y: float, margin: float) -> Dict[str, Dict]:
    """GT-defined planted intervals + the absolute threshold per side."""
    out = {}
    for side in ("l", "r"):
        toe = gt_world.get(f"toe_{side}")
        if toe is None:
            continue
        y = toe[:, 1]
        thr = max(ground_y, float(y.min())) + margin * height
        mask = y < thr
        out[side] = {
            "thr": thr,
            "mask": mask,
            "intervals": _intervals(mask),
            "fraction": float(mask.mean()),
            "min": float(y.min()),
            "max": float(y.max()),
            "contact_point": toe,
        }
    return out


def _foot_metrics(gt_world: Dict[str, np.ndarray], pred_world: Dict[str, np.ndarray],
                  height: float, ground_y: float, margin: float) -> Dict:
    """Contact/height/slide for one track, gated by the *GT's* planted frames."""
    ref = _contact_state_from_gt(gt_world, height, ground_y, margin)
    out = {}
    for side, r in ref.items():
        toe = f"toe_{side}"
        foot = f"foot_{side}"
        if toe not in pred_world or toe not in gt_world:
            continue
        yp = pred_world[toe][:, 1]
        yg = r["contact_point"][:, 1]
        duration = float(r["mask"].sum()) / 30.0
        slide_gt = _slide_drift(r["contact_point"], r["intervals"])
        slide_pr = _slide_drift(pred_world[toe], r["intervals"])
        mae = float(np.abs(yp[r["mask"]] - yg[r["mask"]]).mean()) if r["mask"].any() else 0.0
        out[side] = {
            "contact_thr_units": round(r["thr"], 4),
            "contact_fraction_gt": round(r["fraction"], 4),
            "contact_intervals": len(r["intervals"]),
            "contact_duration_s": round(duration, 3),
            "contact_height_mae_units": round(mae, 5),
            "slide_gt_units": round(slide_gt, 5),
            "slide_pred_units": round(slide_pr, 5),
            "slide_gt_units_per_s": round(slide_gt / duration, 5) if duration > 0 else 0.0,
            "slide_pred_units_per_s": round(slide_pr / duration, 5) if duration > 0 else 0.0,
            "penetration_units": round(float(np.maximum(0.0, ground_y - yp).max()), 5),
            "min_height_units": round(float(yp.min()), 5),
            "max_height_units": round(float(yp.max()), 5),
            "ground_min_units": round(r["min"], 5),
            "ground_max_units": round(r["max"], 5),
            "contact_joint": toe if toe in pred_world else foot,
        }
    return out


def _contact_pattern(gt_world: Dict[str, np.ndarray], pred_world: Dict[str, np.ndarray],
                     height: float, ground_y: float,
                     margin: float) -> Dict:
    """Recall/precision of the predicted planted frames vs the GT's."""
    ref = _contact_state_from_gt(gt_world, height, ground_y, margin)
    tp = fp = fn = 0
    per_side = {}
    for side, r in ref.items():
        toe = f"toe_{side}"
        if toe not in pred_world:
            continue
        a = r["mask"]
        b = pred_world[toe][:, 1] < r["thr"]
        tp += int((a & b).sum())
        fp += int((~a & b).sum())
        fn += int((a & ~b).sum())
        per_side[side] = {"gt": int(a.sum()), "pred": int(b.sum()),
                          "both": int((a & b).sum()), "thr": round(r["thr"], 4)}
    return {"tp": tp, "fp": fp, "fn": fn, "per_side": per_side}


def _jerk(world: Dict[str, np.ndarray], dt: float = 1.0 / 30.0) -> Dict:
    """Third difference of joint positions (+ MAD spike count per joint)."""
    vals = []
    spikes = 0
    for j, p in world.items():
        if len(p) < 4:
            continue
        acc = np.diff(p, n=3, axis=0) / (dt ** 3)
        mag = np.linalg.norm(acc, axis=1)
        vals.append(mag)
        med = np.median(mag)
        mad = np.median(np.abs(mag - med)) + 1e-9
        spikes += int((mag > med + MIN_SPIKE_SIGMAS * 1.4826 * mad).sum())
    if not vals:
        return {"rms": 0.0, "spike_count": 0, "p99": 0.0, "frames": 0}
    cat = np.concatenate(vals)
    return {"rms": float(np.sqrt(np.mean(cat ** 2))),
            "spike_count": spikes,
            "p99": float(np.percentile(cat, 99)),
            "frames": int(len(cat))}


def _bone_lengths(world: Dict[str, np.ndarray]) -> Dict:
    pairs = [("upperarm", "forearm"), ("forearm", "hand"),
             ("thigh", "shin"), ("shin", "foot")]
    out = {}
    for a, b in pairs:
        for side in ("l", "r"):
            ja, jb = f"{a}_{side}", f"{b}_{side}"
            if ja not in world or jb not in world:
                continue
            L = np.linalg.norm(world[ja] - world[jb], axis=1)
            mean = float(L.mean())
            out[ja] = {
                "mean": round(mean, 5),
                "std": round(float(L.std()), 5),
                "cv": round(float(L.std() / mean), 6) if mean > 1e-9 else None,
            }
    return out


def _fraction(num: int, den: int) -> Optional[float]:
    return None if den == 0 else round(num / den, 4)


def compute_metrics(gt: Dict, pred: Dict, profile: Dict,
                    weights: Optional[Dict[str, float]] = None,
                    ground_y: float = 0.0,
                    contact_margin: float = DEFAULT_CONTACT_MARGIN) -> Dict:
    """Score ``pred`` against ``gt``; returns vector components + breakdowns."""
    weights = dict({"angle": 1.0, "position": 1.0, "root": 1.0,
                    "contact": 1.0, "temporal": 0.5}, **(weights or {}))
    height = float(gt.get("height") or pred.get("height") or 1.0)
    joints = [j for j in BODY_JOINTS if j in gt["world"] and j in pred["world"]]
    gt_w = {j: gt["world"][j] for j in joints}
    pr_w = {j: pred["world"][j] for j in joints}

    # ---- rotation (local, relative to parent) ----
    per_bone_ang = {}
    for j in joints:
        if j not in gt["local_q"] or j not in pred["local_q"]:
            continue
        per_bone_ang[j] = q_angle_deg(gt["local_q"][j], pred["local_q"][j])
    finger_ang = {}
    for j in gt.get("finger_joints", []):
        if j in pred["local_q"]:
            finger_ang[j] = q_angle_deg(gt["local_q"][j], pred["local_q"][j])
    groups = {}
    for g, js in GROUPS.items():
        vals = [per_bone_ang[j] for j in js if j in per_bone_ang]
        if vals:
            v = np.concatenate(vals)
            groups[g] = {"local_mean_deg": round(float(v.mean()), 4),
                         "local_p99_deg": round(float(np.percentile(v, 99)), 4),
                         "world_mean_deg": None}
    all_ang = np.concatenate([v for v in per_bone_ang.values()]) if per_bone_ang else np.array([0.0])
    e_ang = float(all_ang.mean())

    # ---- world orientation (independent of twist/helper convention) ----
    per_bone_world = {}
    for j in joints:
        if j in gt.get("world_q", {}) and j in pred.get("world_q", {}):
            per_bone_world[j] = q_angle_deg(gt["world_q"][j], pred["world_q"][j])
    if per_bone_world:
        all_world = np.concatenate(list(per_bone_world.values()))
        e_world = float(all_world.mean())
        for g, js in GROUPS.items():
            vals = [per_bone_world[j] for j in js if j in per_bone_world]
            if vals and g in groups:
                groups[g]["world_mean_deg"] = round(float(np.concatenate(vals).mean()), 4)
    else:
        all_world = np.array([0.0])
        e_world = 0.0

    # ---- world position (root aligned) ----
    gt_ra, gt_root = _root_align(gt_w)
    pr_ra, pr_root = _root_align(pr_w)
    pos_err = []
    for j in joints:
        pos_err.append(np.linalg.norm(gt_ra[j] - pr_ra[j], axis=1))
    pos_cat = np.concatenate(pos_err) if pos_err else np.array([0.0])
    e_pos_units = float(pos_cat.mean())
    e_pos_pct = 100.0 * e_pos_units / height
    e_pos_p99_pct = 100.0 * float(np.percentile(pos_cat, 99)) / height
    pamjpe = _pamjpe(gt_w, pr_w)
    pamjpe_pct = 100.0 * pamjpe / height

    # ---- root trajectory ----
    gt_d = gt_root - gt_root[0]
    pr_d = pr_root - pr_root[0]
    root_err = np.linalg.norm(gt_d - pr_d, axis=1)
    e_root_pct = 100.0 * float(root_err.mean()) / height
    e_root_drift_pct = 100.0 * float(root_err[-1]) / height
    if len(gt_root) > 2:
        dt = 1.0 / 30.0
        gv = np.diff(gt_root, axis=0) / dt
        pv = np.diff(pr_root, axis=0) / dt
        vel_err = float(np.linalg.norm(gv - pv, axis=1).mean()) / height
    else:
        vel_err = 0.0

    # ---- contact (GT-defined planted frames) ----
    feet = _foot_metrics(gt_w, pr_w, height, ground_y, contact_margin)
    pattern = _contact_pattern(gt_w, pr_w, height, ground_y, contact_margin)
    tp, fp, fn = pattern["tp"], pattern["fp"], pattern["fn"]
    gt_contact_frac = float(np.mean([v["contact_fraction_gt"] for v in feet.values()])) \
        if feet else 0.0
    pr_contact_frac = float(np.mean([
        float((pr_w[f"toe_{s}"][:, 1] < v["contact_thr_units"]).mean())
        for s, v in feet.items()])) if feet else 0.0
    slide_gt = sum(v["slide_gt_units"] for v in feet.values())
    slide_pr = sum(v["slide_pred_units"] for v in feet.values())
    slide_gt_rate = sum(v["slide_gt_units_per_s"] for v in feet.values())
    slide_pr_rate = sum(v["slide_pred_units_per_s"] for v in feet.values())
    pen_pr = max((v["penetration_units"] for v in feet.values()), default=0.0)
    height_mae = max((v["contact_height_mae_units"] for v in feet.values()), default=0.0)
    # the GT's own penetration is reported for reference (0 on all M0 GT clips)
    gt_pen = max((float(np.maximum(0.0, ground_y - gt_w[f"toe_{s}"][:, 1]).max())
                  for s in feet), default=0.0)

    # ---- temporal ----
    gt_jerk = _jerk(gt_w)
    pr_jerk = _jerk(pr_w)
    jerk_ratio = (pr_jerk["rms"] / gt_jerk["rms"]) if gt_jerk["rms"] > 1e-9 else 0.0
    spike_rate = 1000.0 * pr_jerk["spike_count"] / max(pr_jerk["frames"], 1)
    # summary-only scalar (NOT a gate): saturating terms for readability
    e_temp = 100.0 * min(abs(jerk_ratio - 1.0), 1.0) if gt_jerk["rms"] > 1e-9 else 0.0
    e_contact = 100.0 * (abs(slide_pr - slide_gt) / 30.0
                         + max(0.0, pen_pr - gt_pen)
                         + height_mae) / height
    summary = (weights["angle"] * e_ang
               + weights["position"] * e_pos_pct
               + weights["root"] * e_root_pct
               + weights["contact"] * e_contact
               + weights["temporal"] * e_temp)

    per_bone = {}
    for j in joints:
        if j in per_bone_ang:
            per_bone[j] = {
                "rot_local_mean_deg": round(float(per_bone_ang[j].mean()), 4),
                "rot_local_max_deg": round(float(per_bone_ang[j].max()), 4),
                "rot_world_mean_deg": (round(float(per_bone_world[j].mean()), 4)
                                       if j in per_bone_world else None),
                "pos_mean_units": round(float(np.linalg.norm(
                    gt_ra[j] - pr_ra[j], axis=1).mean()), 5),
            }

    return {
        "model": gt.get("model_key") or pred.get("model_key"),
        "frames": len(gt_w[joints[0]]) if joints else 0,
        "height": height,
        "vector": {
            # rotation
            "angle_mean_deg": round(e_ang, 4),
            "angle_p99_deg": round(float(np.percentile(all_ang, 99)), 4),
            "angle_max_deg": round(float(all_ang.max()), 4),
            "world_angle_mean_deg": round(e_world, 4),
            "world_angle_p99_deg": round(float(np.percentile(all_world, 99)), 4),
            # position
            "position_pct_height": round(e_pos_pct, 4),
            "position_p99_pct_height": round(e_pos_p99_pct, 4),
            "pamjpe_pct_height": round(pamjpe_pct, 4),
            # root
            "root_traj_pct_height": round(e_root_pct, 4),
            "root_drift_pct_height": round(e_root_drift_pct, 4),
            "root_vel_pct_height_per_s": round(100.0 * vel_err, 4),
            # contact
            "contact_margin_pct_height": round(100.0 * contact_margin, 3),
            "contact_fraction_gt": round(gt_contact_frac, 4),
            "contact_fraction_pred": round(pr_contact_frac, 4),
            "contact_recall": _fraction(tp, tp + fn),
            "contact_precision": _fraction(tp, tp + fp),
            "contact_slide_gt": round(slide_gt, 5),
            "contact_slide_pred": round(slide_pr, 5),
            "contact_slide_delta": round(slide_pr - slide_gt, 5),
            "contact_slide_gt_per_s": round(slide_gt_rate, 5),
            "contact_slide_pred_per_s": round(slide_pr_rate, 5),
            "contact_penetration_gt": round(gt_pen, 5),
            "contact_penetration_pred": round(pen_pr, 5),
            "contact_penetration_delta": round(pen_pr - gt_pen, 5),
            "contact_height_mae_pct_height": round(100.0 * height_mae / height, 4),
            # temporal
            "jerk_rms_gt": round(gt_jerk["rms"], 4),
            "jerk_rms_pred": round(pr_jerk["rms"], 4),
            "jerk_rms_ratio": round(float(jerk_ratio), 4),
            "jerk_p99_gt": round(gt_jerk["p99"], 4),
            "jerk_p99_pred": round(pr_jerk["p99"], 4),
            "spike_count_gt": gt_jerk["spike_count"],
            "spike_count_pred": pr_jerk["spike_count"],
            "spike_delta": pr_jerk["spike_count"] - gt_jerk["spike_count"],
            "spike_rate_pred": round(spike_rate, 4),
        },
        "summary_S": {
            "value": round(float(summary), 4),
            "is_gate": False,
            "note": "weighted summary of saturating terms; never rank or gate on this",
            "weights": weights,
        },
        "groups": groups,
        "per_bone": per_bone,
        "fingers": {
            "count": len(finger_ang),
            "mean_deg": round(float(np.concatenate(list(finger_ang.values())).mean()), 4)
            if finger_ang else None,
            "max_deg": round(float(np.concatenate(list(finger_ang.values())).max()), 4)
            if finger_ang else None,
        },
        "contact": {
            "thresholds": pattern["per_side"],
            "tp": tp, "fp": fp, "fn": fn,
            "note": "planted frames are defined on the GT track, "
                    "thr = gt_min_toe_y + margin * height",
        },
        "foot": feet,
        "contact_fraction_gt": round(gt_contact_frac, 4),
        "contact_fraction_pred": round(pr_contact_frac, 4),
        "jerk_gt": {"rms": round(gt_jerk["rms"], 3), "spikes": gt_jerk["spike_count"]},
        "jerk_pred": {"rms": round(pr_jerk["rms"], 3), "spikes": pr_jerk["spike_count"]},
        "bone_lengths_pred": _bone_lengths(pr_w),
        "bone_lengths_gt": _bone_lengths(gt_w),
    }
