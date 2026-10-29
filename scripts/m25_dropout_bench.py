"""M2.5 v0 dropout / occlusion benchmark on synthetic R2/R3.

What it does
------------
1. Takes a *complete* pipeline VMD (the prediction at M2.4e, ``cleanup+ik``)
   and a synthetic GT VMD.
2. Hides frames with a confidence mask:
   * random dropout 10/20/30%;
   * continuous occlusion segments 0.3-1.5 s (9-45 frames at 30 fps).
3. Builds three fills:
   * ``hold``    - repeat the last observed frame (the required baseline);
   * ``stream``  - v0 with a 0.3-0.5 s lookahead;
   * ``offline`` - v0 with both gap ends (final-delivery refinement).
4. Scores every fill against GT:
   * gap-frame local-angle and root-aligned 3D position error;
   * freeze (motion energy inside a gap vs its observed context);
   * boundary jump (max step at the gap edges vs observed p99);
   * recovery time;
   * contact recall / precision / slide / penetration;
   * jerk and observed-frame identity.

The filler never reads GT.  ``--gt`` is only the scorer.  The output JSON has
per-seed details and per-scenario aggregates; the printed table is the gate
view.
"""
import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

from vtv.cleanup import dense_tracks, tracks_to_vmd
from vtv.metrics.core import (BODY_JOINTS, _jerk, compute_metrics, pose_track,
                              q_angle_deg)
from vtv.mmd.pmx import parse_pmx
from vtv.mmd.vmd import parse_vmd
from vtv.streaming import (StreamingConfig, fill_vmd, summarize,
                           _q_conj, _q_mul)

FPS = 30.0
BODY = [j for j in BODY_JOINTS if j != "root"]


# ---------------------------------------------------------------- masks

def _fill_random(mask, rng, target, guard=3):
    """Add single-frame holes until ``target`` frames are missing."""
    free = np.where(mask)[0]
    free = free[(free >= guard) & (free < len(mask) - guard)]
    need = max(0, int(target) - int((~mask).sum()))
    if need and len(free):
        pick = rng.choice(free, size=min(need, len(free)), replace=False)
        mask[pick] = False
    return mask


def make_mask(n, kind, p, seed):
    rng = np.random.default_rng(int(seed))
    mask = np.ones(n, dtype=bool)
    if kind == "random":
        mask = rng.random(n) >= float(p)
    elif kind == "occlusion":
        target = int(round(float(p) * n))
        cursor = 3 + int(rng.integers(0, 8))
        while int((~mask).sum()) < target and cursor < n - 3:
            length = int(rng.integers(9, 46))       # 0.3-1.5 s
            if cursor + length > n - 3:
                break
            mask[cursor:cursor + length] = False
            cursor += length + int(rng.integers(5, 18))
        _fill_random(mask, rng, target)
    elif kind == "mixed":
        target = int(round(float(p) * n))
        half = target // 2
        cursor = 3 + int(rng.integers(0, 8))
        while int((~mask).sum()) < half and cursor < n - 3:
            length = int(rng.integers(9, 31))
            if cursor + length > n - 3:
                break
            mask[cursor:cursor + length] = False
            cursor += length + int(rng.integers(6, 18))
        _fill_random(mask, rng, target)
    else:
        raise ValueError(f"unknown mask kind: {kind}")
    mask[:3] = True
    mask[-3:] = True
    return mask


def hold_tracks(tracks, mask):
    out = {n: {"rot": np.asarray(v["rot"], dtype=np.float64).copy(),
               "pos": np.asarray(v["pos"], dtype=np.float64).copy()}
           for n, v in tracks.items()}
    obs = np.where(mask)[0]
    if len(obs) == 0:
        return out
    for name, rec in out.items():
        last = int(obs[0])
        for t in range(len(mask)):
            if mask[t]:
                last = t
            else:
                rec["rot"][t] = out[name]["rot"][last]
                rec["pos"][t] = out[name]["pos"][last]
    return out


# ---------------------------------------------------------------- metrics

def _root_aligned(world, joint):
    ref = world.get("pelvis")
    if ref is None:
        return world[joint]
    return world[joint] - ref


def body_errors(gt, pr):
    joints = [j for j in BODY if j in gt["world"] and j in pr["world"]]
    ang = np.stack([q_angle_deg(gt["local_q"][j], pr["local_q"][j])
                    for j in joints], axis=1)
    pos = np.stack([np.linalg.norm(_root_aligned(gt["world"], j)
                                   - _root_aligned(pr["world"], j), axis=1)
                    for j in joints], axis=1)
    return joints, ang, pos


def relative_gap_error(gt, pr, mask):
    """Gap completion error relative to the last observed anchor.

    Absolute gap error is dominated by the predictor's own bias, which the
    filler cannot repair.  The relative rotation/displacement from ``s-1``
    isolates what the filler is responsible for: continuing the motion.
    Hold-the-last-frame has an identity relative motion, so this is the
    metric on which "better than hold" is meaningful.
    """
    joints = [j for j in BODY if j in gt["local_q"] and j in pr["local_q"]]
    gaps = _gaps(mask)
    ang_vals, pos_vals = [], []
    for s, e in gaps:
        a = s - 1
        if a < 0:
            continue
        for t in range(s, e + 1):
            for j in joints:
                qg = _q_mul(_q_conj(gt["local_q"][j][a]), gt["local_q"][j][t])
                qp = _q_mul(_q_conj(pr["local_q"][j][a]), pr["local_q"][j][t])
                ang_vals.append(float(q_angle_deg(qg, qp)))
            for j in joints:
                dg = gt["world"][j][t] - gt["world"][j][a]
                dp = pr["world"][j][t] - pr["world"][j][a]
                pos_vals.append(float(np.linalg.norm(dg - dp)))
    if not ang_vals:
        return {"frames": 0}
    return {
        "frames": int(len(ang_vals) // max(1, len(joints))),
        "angle_mean_deg": round(float(np.mean(ang_vals)), 4),
        "angle_p95_deg": round(float(np.percentile(ang_vals, 95)), 4),
        "pos_mean_units": round(float(np.mean(pos_vals)), 5),
        "pos_p95_units": round(float(np.percentile(pos_vals, 95)), 5),
    }


def gap_error(ang, pos, mask):
    idx = np.where(~mask)[0]
    if len(idx) == 0:
        return {"frames": 0}
    a = ang[idx].mean(axis=1)
    p = pos[idx].mean(axis=1)
    return {
        "frames": int(len(idx)),
        "angle_mean_deg": round(float(a.mean()), 4),
        "angle_p95_deg": round(float(np.percentile(a, 95)), 4),
        "angle_max_deg": round(float(a.max()), 4),
        "pos_mean_pct_height": None,  # filled by caller (needs height)
        "pos_mean_units": round(float(p.mean()), 5),
        "pos_p95_units": round(float(np.percentile(p, 95)), 5),
        "pos_max_units": round(float(p.max()), 5),
    }


def motion_series(tr, joints):
    n = len(tr["world"][joints[0]])
    step = np.zeros(n, dtype=np.float64)
    for j in joints:
        q = tr["local_q"][j]
        d = q_angle_deg(q[1:], q[:-1])
        step[1:] = np.maximum(step[1:], d)
    ref = tr["world"].get("pelvis", tr["world"].get("root"))
    if ref is not None:
        root = np.zeros(n, dtype=np.float64)
        root[1:] = np.linalg.norm(np.diff(ref[:, [0, 2]], axis=0), axis=1)
    else:
        root = np.zeros(n, dtype=np.float64)
    return step, root


def _context_indices(s, e, n, mask, radius=15):
    a = np.arange(max(0, s - radius), max(0, s))
    b = np.arange(min(n, e + 1), min(n, e + 1 + radius))
    idx = np.concatenate([a, b])
    idx = idx[(idx >= 0) & (idx < n)]
    return idx[mask[idx]]


def freeze_metrics(step, root, mask, gaps):
    if not gaps:
        return {"gaps": 0}
    rows = []
    for s, e in gaps:
        ctx = _context_indices(s, e, len(mask), mask)
        gm = float(step[s:e + 1].mean())
        cm = float(step[ctx].mean()) if len(ctx) else 0.0
        gr = float(root[s:e + 1].mean())
        cr = float(root[ctx].mean()) if len(ctx) else 0.0
        rows.append({
            "start": int(s), "end": int(e), "len": int(e - s + 1),
            "motion": round(gm, 4),
            "context_motion": round(cm, 4),
            "ratio": round(gm / cm, 4) if cm > 1e-9 else 0.0,
            "root_motion": round(gr, 5),
            "root_context": round(cr, 5),
            "root_ratio": round(gr / cr, 4) if cr > 1e-9 else 0.0,
            "frozen_frames": int((step[s:e + 1] < 0.02).sum()),
        })
    ratios = np.array([r["ratio"] for r in rows], dtype=np.float64)
    root_ratios = np.array([r["root_ratio"] for r in rows], dtype=np.float64)
    gap_frames = int(sum(r["len"] for r in rows))
    frozen = int(sum(r["frozen_frames"] for r in rows))
    return {
        "gaps": len(rows),
        "ratio_median": round(float(np.median(ratios)), 4),
        "ratio_p10": round(float(np.percentile(ratios, 10)), 4),
        "ratio_min": round(float(ratios.min()), 4),
        "root_ratio_median": round(float(np.median(root_ratios)), 4),
        "frozen_frames": frozen,
        "gap_frames": gap_frames,
        "frozen_frac": round(frozen / max(1, gap_frames), 4),
        "rows": rows,
    }


def jump_metrics(step, root, mask, gaps, base_step=None, base_root=None):
    n = len(mask)
    if not gaps:
        return {"gaps": 0}
    gap_ids = np.zeros(n, dtype=bool)
    for s, e in gaps:
        gap_ids[max(0, s - 2):min(n, e + 3)] = True
    ref = base_step if base_step is not None else step
    ctx = ref[mask & ~gap_ids]
    p99 = float(np.percentile(ctx, 99)) if len(ctx) else 0.0
    rows = []
    jumps = 0
    for s, e in gaps:
        idx = [s - 1, s, s + 1, e - 1, e, e + 1, e + 2]
        idx = [t for t in idx if 0 <= t < n]
        bmax = float(max(step[t] for t in idx)) if idx else 0.0
        rmax = float(max(root[t] for t in idx)) if idx else 0.0
        base_bmax = float(max(ref[t] for t in idx)) if idx else 0.0
        # A step that the input already contains at the same boundary is not a
        # jump introduced by the filler.  Cap the relative 3*p99 term at 45 deg
        # so a very spiky source clip (T5 p99 ~31 deg) cannot hide a 70 deg
        # freeze-release jump behind a 92 deg threshold.
        thr = max(10.0, base_bmax + 5.0, min(3.0 * p99, 45.0))
        bad = bmax > thr
        jumps += int(bad)
        rows.append({"start": int(s), "end": int(e),
                     "boundary_step_deg": round(bmax, 3),
                     "base_boundary_step_deg": round(base_bmax, 3),
                     "boundary_root_step": round(rmax, 5),
                     "ctx_p99_deg": round(p99, 3),
                     "threshold_deg": round(thr, 3),
                     "jump": bool(bad)})
    return {
        "gaps": len(rows),
        "ctx_p99_deg": round(p99, 4),
        "jump_gaps": int(jumps),
        "jump_rate": round(jumps / len(rows), 4),
        "max_boundary_step_deg": round(max(r["boundary_step_deg"] for r in rows), 4),
        "max_excess_step_deg": round(max(
            max(0.0, r["boundary_step_deg"] - r["base_boundary_step_deg"])
            for r in rows), 4),
        "rows": rows,
    }


def recovery_metrics(ang_err, steps, mask, gaps, pre=15):
    if not gaps:
        return {"gaps": 0}
    n = len(mask)
    out = []
    for s, e in gaps:
        ctx = np.arange(max(0, s - pre), max(0, s))
        pre_err = float(ang_err[ctx].mean()) if len(ctx) else float(ang_err[s])
        thr = pre_err + 5.0
        rec = n - e
        for t in range(e + 1, n - 2):
            if all(ang_err[u] <= thr for u in (t, t + 1, t + 2)):
                rec = t - e
                break
        out.append(int(rec))
    return {"gaps": len(out), "median_frames": float(np.median(out)),
            "max_frames": int(max(out)), "rows": out}


def contact_gap(gt, pr, mask, margin, height, ground_y=0.0):
    gap = ~mask
    tp = fp = fn = 0
    slide_vals = []
    gt_slide_vals = []
    pen = 0.0
    hmae = []
    per_side = {}
    for side in ("l", "r"):
        gt_toe = gt["world"].get(f"toe_{side}")
        pr_toe = pr["world"].get(f"toe_{side}")
        if gt_toe is None or pr_toe is None:
            continue
        yg = gt_toe[:, 1]
        yp = pr_toe[:, 1]
        thr = max(float(ground_y), float(yg.min())) + margin * height
        gc = yg < thr
        pc = yp < thr
        tp += int((gc & pc & gap).sum())
        fp += int((~gc & pc & gap).sum())
        fn += int((gc & ~pc & gap).sum())
        pen = max(pen, float(np.maximum(0.0, ground_y - yp[gap]).max())
                  if gap.any() else 0.0)
        if (gc & gap).any():
            hmae.append(float(np.abs(yp[gc & gap] - yg[gc & gap]).mean()))
        # slide only inside gaps where the GT foot is planted at both ends.
        # The GT's own net movement is reported too: a foot that the GT calls
        # "in the contact band" at both ends can still legitimately travel
        # (a drag), in which case hold's zero slide is not a physically
        # meaningful target.
        for a, b in zip(np.where(np.diff(np.r_[False, gap, False].astype(int)) == 1)[0],
                        np.where(np.diff(np.r_[False, gap, False].astype(int)) == -1)[0] - 1):
            if gc[a] and gc[b]:
                slide_vals.append(float(np.linalg.norm(pr_toe[b, [0, 2]] - pr_toe[a, [0, 2]])))
                gt_slide_vals.append(float(np.linalg.norm(gt_toe[b, [0, 2]] - gt_toe[a, [0, 2]])))
        per_side[side] = {"thr": round(thr, 4),
                          "gt_gap_frames": int((gc & gap).sum()),
                          "pred_gap_frames": int((pc & gap).sum())}
    recall = tp / (tp + fn) if (tp + fn) else None
    precision = tp / (tp + fp) if (tp + fp) else None
    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "recall": None if recall is None else round(recall, 4),
        "precision": None if precision is None else round(precision, 4),
        "slide_mean_units": round(float(np.mean(slide_vals)), 5) if slide_vals else 0.0,
        "gt_slide_mean_units": round(float(np.mean(gt_slide_vals)), 5)
        if gt_slide_vals else 0.0,
        "penetration_units": round(pen, 5),
        "height_mae_units": round(float(np.mean(hmae)), 5) if hmae else 0.0,
        "sides": per_side,
    }


def jerk_boundary(world, gaps, radius=5):
    n = len(next(iter(world.values())))
    if n < 4:
        return {"full_rms": 0.0, "boundary_mean": 0.0}
    frames = []
    for p in world.values():
        acc = np.diff(np.asarray(p, dtype=np.float64), n=3, axis=0)
        frames.append(np.linalg.norm(acc, axis=1))
    if not frames:
        return {"full_rms": 0.0, "boundary_mean": 0.0}
    mag = np.stack(frames, axis=1).mean(axis=1)  # aligned to t=3..n-1
    full = float(np.sqrt(np.mean(mag ** 2)))
    idx = []
    for s, e in gaps:
        idx.extend(range(max(3, s - radius), min(n, e + radius + 1)))
    bm = float(np.mean(mag[np.asarray(idx) - 3])) if idx else 0.0
    return {"full_rms": round(full, 5), "boundary_mean": round(bm, 5)}


def score_run(gt_tr, pr_tr, mask, profile, margin, cfg=None,
              base_step=None, base_root=None):
    height = float(gt_tr["height"] or profile.get("height") or 1.0)
    joints, ang, pos = body_errors(gt_tr, pr_tr)
    ge = gap_error(ang, pos, mask)
    if ge.get("frames"):
        ge["pos_mean_pct_height"] = round(100.0 * ge["pos_mean_units"] / height, 4)
        ge["pos_p95_pct_height"] = round(100.0 * ge["pos_p95_units"] / height, 4)
    rg = relative_gap_error(gt_tr, pr_tr, mask)
    if rg.get("frames"):
        rg["pos_mean_pct_height"] = round(100.0 * rg["pos_mean_units"] / height, 4)
        rg["pos_p95_pct_height"] = round(100.0 * rg["pos_p95_units"] / height, 4)
    step, root = motion_series(pr_tr, joints)
    all_gaps = _gaps(mask)
    # Visible gaps: a one-frame hole is not a visible freeze, and holding it
    # is the lowest-error fill.  Gate the completion metrics on >= 3 frames.
    long_gaps = [(s, e) for s, e in all_gaps if e - s + 1 >= 3]
    long_mask = mask.copy()
    for s, e in all_gaps:
        if e - s + 1 < 3:
            long_mask[s:e + 1] = True
    ge_long = gap_error(ang, pos, long_mask)
    if ge_long.get("frames"):
        ge_long["pos_mean_pct_height"] = round(
            100.0 * ge_long["pos_mean_units"] / height, 4)
        ge_long["pos_p95_pct_height"] = round(
            100.0 * ge_long["pos_p95_units"] / height, 4)
    rg_long = relative_gap_error(gt_tr, pr_tr, long_mask)
    if rg_long.get("frames"):
        rg_long["pos_mean_pct_height"] = round(
            100.0 * rg_long["pos_mean_units"] / height, 4)
    ang_frame = ang.mean(axis=1)
    met = compute_metrics(gt_tr, pr_tr, profile, contact_margin=margin)
    return {
        "frames": int(len(mask)),
        "observed_frames": int(mask.sum()),
        "gap_frames": int((~mask).sum()),
        "long_gap_frames": int((~long_mask).sum()),
        "gaps": len(all_gaps),
        "long_gaps": len(long_gaps),
        "vector": met["vector"],
        "gap_error": ge,
        "relative_gap": rg,
        "gap_error_long": ge_long,
        "relative_gap_long": rg_long,
        "freeze": freeze_metrics(step, root, mask, long_gaps),
        "jump": jump_metrics(step, root, mask, long_gaps, base_step, base_root),
        "recovery": recovery_metrics(ang_frame, step, mask, long_gaps),
        "contact_gap": contact_gap(gt_tr, pr_tr, mask, margin, height),
        "jerk": jerk_boundary(pr_tr["world"], all_gaps),
        "groups": met["groups"],
    }


def observed_diff(base_tr, pr_tr, mask):
    if not mask.any():
        return {"angle_max_deg": 0.0, "pos_max_units": 0.0}
    joints = [j for j in BODY if j in base_tr["local_q"] and j in pr_tr["local_q"]]
    a = max(float(q_angle_deg(base_tr["local_q"][j][mask],
                              pr_tr["local_q"][j][mask]).max())
            for j in joints)
    p = max(float(np.abs(base_tr["world"][j][mask]
                         - pr_tr["world"][j][mask]).max()) for j in joints)
    return {"angle_max_deg": round(a, 9), "pos_max_units": round(p, 9)}


# ---------------------------------------------------------------- aggregate

def _mean(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return round(float(np.mean(vals)), 4) if vals else None


def aggregate(runs, pipeline):
    rows = [r[pipeline] for r in runs if pipeline in r]
    if not rows:
        return {}
    out = {
        "vector": ({k: _mean([r.get("vector", {}).get(k) for r in rows])
                    for k in sorted(set().union(*[set(r.get("vector", {}))
                                                  for r in rows]))}
                   if any(r.get("vector") for r in rows) else {}),
        "gap_angle_mean_deg": _mean([r.get("gap_error", {}).get("angle_mean_deg") for r in rows]),
        "gap_angle_p95_deg": _mean([r.get("gap_error", {}).get("angle_p95_deg") for r in rows]),
        "gap_pos_pct_height": _mean([r.get("gap_error", {}).get("pos_mean_pct_height") for r in rows]),
        "gap_pos_p95_pct_height": _mean([r.get("gap_error", {}).get("pos_p95_pct_height") for r in rows]),
        "rel_angle_mean_deg": _mean([r.get("relative_gap", {}).get("angle_mean_deg") for r in rows]),
        "rel_angle_p95_deg": _mean([r.get("relative_gap", {}).get("angle_p95_deg") for r in rows]),
        "rel_pos_pct_height": _mean([r.get("relative_gap", {}).get("pos_mean_pct_height") for r in rows]),
        "long_gap_frames": _mean([r.get("long_gap_frames") for r in rows]),
        "long_gap_angle_mean_deg": _mean([r.get("gap_error_long", {}).get("angle_mean_deg") for r in rows]),
        "long_gap_pos_pct_height": _mean([r.get("gap_error_long", {}).get("pos_mean_pct_height") for r in rows]),
        "long_rel_angle_mean_deg": _mean([r.get("relative_gap_long", {}).get("angle_mean_deg") for r in rows]),
        "long_rel_pos_pct_height": _mean([r.get("relative_gap_long", {}).get("pos_mean_pct_height") for r in rows]),
        "freeze_ratio_median": _mean([r.get("freeze", {}).get("ratio_median") for r in rows]),
        "freeze_ratio_p10": _mean([r.get("freeze", {}).get("ratio_p10") for r in rows]),
        "frozen_frac": _mean([r.get("freeze", {}).get("frozen_frac") for r in rows]),
        "jump_gaps": _mean([r.get("jump", {}).get("jump_gaps") for r in rows]),
        "jump_rate": _mean([r.get("jump", {}).get("jump_rate") for r in rows]),
        "max_boundary_step_deg": _mean([r.get("jump", {}).get("max_boundary_step_deg") for r in rows]),
        "recovery_median_frames": _mean([r.get("recovery", {}).get("median_frames") for r in rows]),
        "contact_recall": _mean([r.get("contact_gap", {}).get("recall") for r in rows]),
        "contact_precision": _mean([r.get("contact_gap", {}).get("precision") for r in rows]),
        "contact_slide_mean_units": _mean([r.get("contact_gap", {}).get("slide_mean_units") for r in rows]),
        "contact_gt_slide_mean_units": _mean([r.get("contact_gap", {}).get("gt_slide_mean_units") for r in rows]),
        "contact_penetration_units": _mean([r.get("contact_gap", {}).get("penetration_units") for r in rows]),
        "contact_height_mae_units": _mean([r.get("contact_gap", {}).get("height_mae_units") for r in rows]),
        "jerk_full_rms": _mean([r.get("jerk", {}).get("full_rms") for r in rows]),
        "jerk_boundary_mean": _mean([r.get("jerk", {}).get("boundary_mean") for r in rows]),
        "observed_angle_max_deg": _mean([r.get("observed_diff", {}).get("angle_max_deg") for r in rows]),
    }
    return out


def gate_for(agg, hold):
    def g(rec, key):
        return rec.get(key) if rec else None

    checks = {}
    checks["observed_unchanged"] = (g(agg, "observed_angle_max_deg") is not None
                                    and g(agg, "observed_angle_max_deg") < 1e-3)
    lf = g(agg, "long_gap_frames")
    has_long = lf is not None and lf > 0
    checks["has_long_gap"] = bool(has_long)
    if has_long:
        # No visible freeze: the gap must carry real motion and must not be
        # mostly frozen frames.  0.02 deg/frame is below a slow dance phrase's
        # natural motion, so up to 20% near-still frames is allowed as long as
        # the gap still carries >=15% of its observed context motion.
        fr = g(agg, "freeze_ratio_median")
        ff = g(agg, "frozen_frac")
        checks["no_freeze"] = (fr is not None and fr >= 0.15
                               and ff is not None and ff <= 0.20)
        jr = g(agg, "jump_rate")
        checks["no_jump"] = jr is not None and jr <= 0.05
        ga, hga = g(agg, "long_gap_angle_mean_deg"), g(hold, "long_gap_angle_mean_deg")
        gp, hgp = g(agg, "long_gap_pos_pct_height"), g(hold, "long_gap_pos_pct_height")
        ra, hra = g(agg, "long_rel_angle_mean_deg"), g(hold, "long_rel_angle_mean_deg")
        rp, hrp = g(agg, "long_rel_pos_pct_height"), g(hold, "long_rel_pos_pct_height")
        # Primary gate: absolute gap-frame error vs GT.  A 0.3 deg / 0.2%
        # height tie tolerance is below the meter's meaningful resolution
        # (the baseline itself is ~40 deg / 5-6% height); the strict delta is
        # reported too.
        checks["gap_error_better_than_hold"] = (
            (ga is None or hga is None or ga <= hga + 0.3)
            and (gp is None or hgp is None or gp <= hgp + 0.2))
        checks["gap_error_strict_better"] = (
            ga is not None and hga is not None and ga < hga)
        # Relative motion is the completion-only diagnostic: hold has zero
        # relative motion, so a noisy predictor can make hold look better
        # even when the absolute pose is closer.  Reported, not gated.
        checks["relative_gap_better_than_hold"] = (
            ra is not None and hra is not None and ra < hra
            and (rp is None or hrp is None or rp <= hrp + 1e-6))
    else:
        # Only one-frame holes: there is no visible freeze to fix and hold is
        # the lowest-error fill, so the completion gate is vacuous.
        checks["no_freeze"] = True
        checks["no_jump"] = True
        checks["gap_error_better_than_hold"] = True
        checks["gap_error_strict_better"] = True
        checks["relative_gap_better_than_hold"] = True
    rec, hrec = g(agg, "contact_recall"), g(hold, "contact_recall")
    pre, hpre = g(agg, "contact_precision"), g(hold, "contact_precision")
    sl, hsl = g(agg, "contact_slide_mean_units"), g(hold, "contact_slide_mean_units")
    pen, hpen = g(agg, "contact_penetration_units"), g(hold, "contact_penetration_units")
    checks["contact_state_not_worse"] = (
        (rec is None or hrec is None or rec >= hrec - 0.05)
        and (pre is None or hpre is None or pre >= hpre - 0.02)
        and (pen is None or hpen is None or pen <= hpen + 0.02))
    checks["contact_slide_not_worse"] = (
        sl is None or hsl is None or sl <= hsl + 0.05)
    checks["contact_not_worse"] = (checks["contact_state_not_worse"]
                                   and checks["contact_slide_not_worse"])
    checks["pass"] = all(bool(v) for k, v in checks.items()
                         if k not in ("has_long_gap", "gap_error_strict_better",
                                      "relative_gap_better_than_hold"))
    return checks


# ---------------------------------------------------------------- main

def load_view_spec(spec):
    """'front=path' -> ('front', path); 'path' -> ('main', path)."""
    if "=" in spec:
        name, path = spec.split("=", 1)
        return name.strip(), path.strip()
    return "main", spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "models.json"))
    ap.add_argument("--model", default="si")
    ap.add_argument("--gt", default=None, help="GT VMD; omit for a GT-free run")
    ap.add_argument("--vmd", nargs="+", required=True,
                    help="view=path ... (or a single path)")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "m25",
                                                  "dropout_bench.json"))
    ap.add_argument("--scenarios", default="random10,random20,random30,"
                                           "occlusion10,occlusion20,occlusion30")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--lookahead", type=float, default=0.4)
    ap.add_argument("--max-rollout", type=float, default=1.5)
    ap.add_argument("--contact-hold", type=float, default=1.0)
    ap.add_argument("--contact-margin", type=float, default=0.02)
    ap.add_argument("--no-contact", action="store_true")
    ap.add_argument("--no-smooth", action="store_true")
    ap.add_argument("--no-offline", action="store_true",
                    help="skip the offline pipeline (faster)")
    ap.add_argument("--views", default=None, help="comma separated view names")
    ap.add_argument("--anchor-stance", action="store_true",
                    help="M2.5 v0.1: anchor observed stance feet inside long holes")
    ap.add_argument("--stance-release", type=float, default=0.4)
    ap.add_argument("--stance-release-mode", choices=("budget", "full"),
                    default="budget")
    ap.add_argument("--stance-slide-budget", type=float, default=0.045)
    ap.add_argument("--stance-orientation", choices=("v0", "anchor"),
                    default="v0")
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8"))["models"][args.model]
    profile = json.load(open(os.path.join(ROOT, "profiles", f"{args.model}.json"),
                             encoding="utf-8"))
    model = parse_pmx(cfg["pmx"])
    scenarios = []
    for s in args.scenarios.split(","):
        s = s.strip()
        if not s:
            continue
        kind = "random" if s.startswith("random") else (
            "occlusion" if s.startswith("occlusion") else (
                "mixed" if s.startswith("mixed") else None))
        if kind is None:
            raise SystemExit(f"unknown scenario: {s}")
        p = float(s[len(kind):]) / 100.0
        scenarios.append({"name": s, "kind": kind, "p": p})
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    keep_views = set(args.views.split(",")) if args.views else None

    gt_vmd = parse_vmd(args.gt) if args.gt else None
    spec = dict(load_view_spec(x) for x in args.vmd)
    if keep_views:
        spec = {k: v for k, v in spec.items() if k in keep_views}

    scfg = StreamingConfig(
        lookahead_s=args.lookahead, max_rollout_s=args.max_rollout,
        contact_hold_s=args.contact_hold, smooth_gaps=not args.no_smooth,
        anchor_stance=args.anchor_stance)
    stance = None
    if args.anchor_stance:
        from vtv.stance_ik import StanceConfig
        stance = StanceConfig(
            high=scfg.high, low=scfg.low, fps=scfg.fps,
            release_s=args.stance_release,
            release_mode=args.stance_release_mode,
            slide_budget=args.stance_slide_budget,
            anchor_orientation=args.stance_orientation)
    report = {
        "model": args.model,
        "gt": os.path.abspath(args.gt) if args.gt else None,
        "contact_margin": args.contact_margin,
        "config": scfg.to_dict(),
        "stance_config": None if stance is None else stance.to_dict(),
        "scenarios": scenarios,
        "seeds": seeds,
        "views": {},
    }
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    for view, path in spec.items():
        print(f"\n=== {view}: {path} ===")
        inp = parse_vmd(path)
        frames, tracks = dense_tracks(inp)
        n = len(frames)
        gt_tr = None
        base_tr = None
        height = float(profile.get("height") or 1.0)
        if gt_vmd is not None:
            gt_tr = pose_track(model, gt_vmd, profile, frames=range(n))
            gt_tr["height"] = height
        base_tr = pose_track(model, inp, profile, frames=range(n))
        base_step, base_root = motion_series(base_tr, BODY)
        base_jerk = _jerk(base_tr["world"])
        view_rec = {
            "input": os.path.abspath(path),
            "frames": int(n),
            "baseline_vector": None,
            "baseline_jerk": {"rms": round(float(base_jerk["rms"]), 5),
                              "spikes": int(base_jerk["spike_count"])},
            "scenarios": {},
        }
        if gt_tr is not None:
            met = compute_metrics(gt_tr, base_tr, profile,
                                  contact_margin=args.contact_margin)
            view_rec["baseline_vector"] = met["vector"]
            print("baseline:", {k: met["vector"][k] for k in
                                ("angle_mean_deg", "position_pct_height",
                                 "contact_recall", "contact_precision",
                                 "contact_slide_delta", "jerk_rms_ratio")})

        for sc in scenarios:
            runs = []
            for seed in seeds:
                mask = make_mask(n, sc["kind"], sc["p"], seed)
                if mask.all():
                    continue
                hold = hold_tracks(tracks, mask)
                hold_vmd = tracks_to_vmd(inp, frames, hold)
                t0 = time.time()
                stream = fill_vmd(
                    inp, mask.astype(np.float64), model=model, profile=profile,
                    cfg=scfg, mode="stream",
                    anchor_stance=args.anchor_stance, stance=stance)
                t_stream = time.time() - t0
                t0 = time.time()
                offline = None if args.no_offline else fill_vmd(
                    inp, mask.astype(np.float64), model=model, profile=profile,
                    cfg=scfg, mode="offline",
                    anchor_stance=args.anchor_stance, stance=stance)
                t_off = time.time() - t0
                run = {"seed": int(seed),
                       "observed_fraction": round(float(mask.mean()), 4),
                       "gap_fraction": round(float((~mask).mean()), 4),
                       "fill_seconds_stream": round(t_stream, 3),
                       "fill_seconds_offline": round(t_off, 3)}
                # hold
                ht = pose_track(model, hold_vmd, profile, frames=range(n))
                if gt_tr is not None:
                    run["hold"] = score_run(gt_tr, ht, mask, profile,
                                            args.contact_margin,
                                            base_step=base_step, base_root=base_root)
                    run["hold"]["observed_diff"] = observed_diff(base_tr, ht, mask)
                else:
                    step, root = motion_series(ht, BODY)
                    gaps = _gaps(mask)
                    long_gaps = [g for g in gaps if g[1] - g[0] + 1 >= 3]
                    run["hold"] = {"freeze": freeze_metrics(step, root, mask, long_gaps),
                                    "jump": jump_metrics(step, root, mask, long_gaps, base_step, base_root),
                                    "jerk": jerk_boundary(ht["world"], gaps)}
                # stream
                st = pose_track(model, stream["vmd"], profile, frames=range(n))
                if gt_tr is not None:
                    run["stream"] = score_run(gt_tr, st, mask, profile,
                                              args.contact_margin,
                                              base_step=base_step, base_root=base_root)
                    run["stream"]["observed_diff"] = observed_diff(base_tr, st, mask)
                else:
                    step, root = motion_series(st, BODY)
                    gaps = _gaps(mask)
                    long_gaps = [g for g in gaps if g[1] - g[0] + 1 >= 3]
                    run["stream"] = {"freeze": freeze_metrics(step, root, mask, long_gaps),
                                      "jump": jump_metrics(step, root, mask, long_gaps, base_step, base_root),
                                      "jerk": jerk_boundary(st["world"], gaps),
                                      "observed_diff": observed_diff(base_tr, st, mask)}
                # offline
                if offline is not None:
                    ot = pose_track(model, offline["vmd"], profile, frames=range(n))
                    if gt_tr is not None:
                        run["offline"] = score_run(gt_tr, ot, mask, profile,
                                                   args.contact_margin,
                                                   base_step=base_step, base_root=base_root)
                        run["offline"]["observed_diff"] = observed_diff(base_tr, ot, mask)
                    else:
                        step, root = motion_series(ot, BODY)
                        gaps = _gaps(mask)
                        long_gaps = [g for g in gaps if g[1] - g[0] + 1 >= 3]
                        run["offline"] = {"freeze": freeze_metrics(step, root, mask, long_gaps),
                                           "jump": jump_metrics(step, root, mask, long_gaps, base_step, base_root),
                                           "jerk": jerk_boundary(ot["world"], gaps),
                                           "observed_diff": observed_diff(base_tr, ot, mask)}
                runs.append(run)
                run["stream"]["stance"] = _stance_summary(stream)
                if offline is not None:
                    run["offline"]["stance"] = _stance_summary(offline)
                print(f"  {sc['name']} seed={seed} gap={run['gap_fraction']:.2f} "
                      f"hold gapAng={_fmt_gap(run['hold'], 'angle_mean_deg')} "
                      f"stream={_fmt_gap(run['stream'], 'angle_mean_deg')} "
                      f"offline={_fmt_gap(run.get('offline'), 'angle_mean_deg')} "
                      f"freeze={_fmt(run['stream'].get('freeze'), 'ratio_median')} "
                      f"jump={_fmt(run['stream'].get('jump'), 'jump_rate')} "
                      f"slide={_fmt(run.get('offline', {}).get('contact_gap'), 'slide_mean_units')}"
                      + (f" stance={_fmt_stance(_stance_summary(offline))}"
                         if args.anchor_stance else ""))

            if not runs:
                continue
            agg = {}
            gates = {}
            for pipeline in ("hold", "stream", "offline"):
                agg[pipeline] = aggregate(runs, pipeline)
                if pipeline != "hold" and gt_tr is not None:
                    gates[pipeline] = gate_for(agg[pipeline], agg["hold"])
            sc_rec = {"mean_gap_fraction": round(
                float(np.mean([r["gap_fraction"] for r in runs])), 4),
                "n_runs": len(runs), "runs": runs,
                "aggregate": agg, "gate": gates}
            view_rec["scenarios"][sc["name"]] = sc_rec
            print(f"  {sc['name']} agg: hold={agg['hold'].get('gap_angle_mean_deg')} "
                  f"stream={agg['stream'].get('gap_angle_mean_deg')} "
                  f"offline={agg['offline'].get('gap_angle_mean_deg')} "
                  f"| rel hold={agg['hold'].get('rel_angle_mean_deg')} "
                  f"stream={agg['stream'].get('rel_angle_mean_deg')} "
                  f"offline={agg['offline'].get('rel_angle_mean_deg')} "
                  f"| stream_pass={gates.get('stream', {}).get('pass')} "
                  f"offline_pass={gates.get('offline', {}).get('pass')}")
        report["views"][view] = view_rec
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("\nWROTE", os.path.abspath(args.out))


def _gaps(mask):
    """Closed intervals [start, end] of *missing* frames (mask == False)."""
    miss = (~np.asarray(mask, dtype=bool)).astype(int)
    diff = np.diff(np.r_[0, miss, 0])
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _fmt(rec, key):
    if not rec:
        return "n/a"
    v = rec.get(key)
    return "n/a" if v is None else f"{v:.3f}"


def _stance_summary(res):
    st = ((res or {}).get("report") or {}).get("stance_ik")
    if not st:
        return None
    return {k: st.get(k) for k in (
        "enabled", "gaps_anchored", "sides_anchored", "frames_edited",
        "toe_xz_drift_max", "toe_xz_end_slide_max", "boundary_step_max_deg",
        "reach_limited_frames", "knee_max_deg", "thigh_max_deg",
        "weight_mean", "weight_min", "observed_changed_max_deg")}


def _fmt_stance(st):
    if not st:
        return "n/a"
    return (f"{st.get('gaps_anchored', 0)}g/"
            f"{st.get('frames_edited', 0)}f/"
            f"drift={st.get('toe_xz_drift_max')}/"
            f"step={st.get('boundary_step_max_deg')}")


def _fmt_gap(rec, key):
    if not rec:
        return "n/a"
    v = (rec.get("gap_error") or {}).get(key)
    return "n/a" if v is None else f"{v:.3f}"


if __name__ == "__main__":
    main()
