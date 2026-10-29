# -*- coding: utf-8 -*-
"""M2.4 target-space arm constraints and anatomical arm audit.

The stage is deliberately **not** a 2D fit.  It works on the target PMX and
uses the model's own geometry:

* the PMX **rigid bodies** (shape 0 sphere / shape 2 capsule) are the model's
  collision proxies; arm-arm and arm-torso penetration is measured on them;
* the **palm frame** from the finger rest positions gives the anatomical elbow
  hinge line (palm normal).  The elbow rotation's axis, its stability over the
  clip, and its deviation from that line are reported;
* the **upper-arm / forearm / hand directions** give the swing part of the arm
  pose; a two-bone (upper arm, forearm) plus hand decomposition separates swing
  from the residual roll about the bone axis.

Two corrections are available, both low-dimensional and both optional:

``resolve_self_intersections``
    rotate only the shoulder (3 DOF per frame) to separate a capsule pair that
    is deeper than ``pen_tol``; the elbow hinge and wrist are untouched, so the
    arm plane is preserved.  2-3 rounds of evaluate -> correct -> re-evaluate
    are the "micro-feedback" loop.  A trust region caps the rotation and the
    displacement; non-arm tracks, IK records and contact are byte-identical.

``project_hinge``
    rotate the *elbow* local rotation axis toward the anatomical hinge line
    while keeping the flexion angle.  This is the M2.4 hinge prior as an
    experiment.  It is **not** the default: on R2 it lowers the geodesic arm
    world angle but makes the actual forearm direction worse (25 -> 32 deg),
    i.e. it changes roll, not the pose.  The audit keeps both numbers visible.

Why the M2.4 goal is not simply "add a constraint until the arm world angle
drops 10 deg": the front-view upper-arm *direction* is 19-25 deg away from GT,
while the 2D-visible direction residual is only 3-6 deg.  The missing component
is depth, which a target-space capsule/hinge prior cannot observe.  The forearm
world angle is 53 deg with a 25 deg direction error and a 43 deg roll residual;
roll about a bone axis does not move any joint, so no joint-limit or capsule
constraint can determine it.  The report therefore separates:

    swing (bone direction, physically observable in 3D positions)
    roll  (rotation about the bone axis, only observable from the mesh/skin)

and never claims a geodesic world-angle win that came from roll alone.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import canonical as CANON
from .adapter import (QID, q_between, q_conj, q_from_axis_angle, qmul, qrot)
from .cleanup import _moving_average, dense_tracks, eval_tracks, tracks_to_vmd
from .mmd.mathutil import euler_xyz_to_q, qinv, qnorm

DEFAULT_PEN_TOL = 0.25      # units; allowed overlap (GT itself touches at ~0.2)
DEFAULT_MAX_DEG = 12.0      # per-frame shoulder correction cap
DEFAULT_MAX_DISP = 0.6      # units moved at the collision point per round
DEFAULT_ROUNDS = 3
DEFAULT_SMOOTH = 3
DEFAULT_HINGE_CAP = 15.0    # deg change of the elbow local rotation per frame

ARM_JOINTS = ("upperarm_l", "forearm_l", "hand_l",
              "upperarm_r", "forearm_r", "hand_r")
TORSO_JOINTS = ("pelvis", "spine", "spine1", "chest", "neck", "head")
ARM_SEGMENT = {"upperarm": "forearm", "forearm": "hand"}


# ---------------------------------------------------------------- small math

def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.zeros(3)


def _angle_between(a, b) -> float:
    a, b = _unit(a), _unit(b)
    return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0))))


def _q_axis_angle(q) -> Tuple[np.ndarray, float]:
    """Quaternion -> (unit axis, angle in degrees); identity -> (0, 0)."""
    q = np.asarray(q, dtype=np.float64)
    q = q / max(1e-12, float(np.linalg.norm(q)))
    if q[3] < 0:
        q = -q
    ang = 2.0 * math.acos(float(np.clip(q[3], -1.0, 1.0)))
    s = math.sin(ang * 0.5)
    if s < 1e-9:
        return np.zeros(3), 0.0
    return q[:3] / s, math.degrees(ang)


def _q_log(q) -> np.ndarray:
    axis, deg = _q_axis_angle(q)
    return axis * math.radians(deg)


def _q_dist(a, b) -> float:
    """Geodesic angle between two quaternions (degrees)."""
    d = abs(float(np.dot(np.asarray(a, dtype=np.float64),
                         np.asarray(b, dtype=np.float64))))
    return math.degrees(2.0 * math.acos(min(1.0, d)))


def _nlerp(q0, q1, t: float):
    q = (1.0 - t) * np.asarray(q0, dtype=np.float64) + t * np.asarray(q1, dtype=np.float64)
    return q / max(1e-12, float(np.linalg.norm(q)))


def _rot_about_axis(axis, deg: float):
    return q_from_axis_angle(*(_unit(axis) * math.radians(deg)))


# ------------------------------------------------------------- capsule math

def _seg_seg_closest(p1, q1, p2, q2) -> Tuple[np.ndarray, np.ndarray]:
    """Closest points of two segments (Ericson, real-time collision ded.)."""
    p1, q1, p2, q2 = (np.asarray(v, dtype=np.float64) for v in (p1, q1, p2, q2))
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    if a < 1e-12 and e < 1e-12:
        return p1, p2
    if a < 1e-12:
        s, t = 0.0, float(np.clip(f / e, 0.0, 1.0))
    else:
        c = float(d1 @ r)
        if e < 1e-12:
            t, s = 0.0, float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = float(d1 @ d2)
            den = a * e - b * b
            s = float(np.clip((b * f - c * e) / den, 0.0, 1.0)) if abs(den) > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t, s = 1.0, float(np.clip((b - c) / a, 0.0, 1.0))
    return p1 + d1 * s, p2 + d2 * t


def _rigid_frame(model: Dict, rb: Dict, world_q, world_pos) -> Tuple[np.ndarray, tuple]:
    """World pose of a PMX rigid body.

    PMX stores the rigid body position in *model* space, not bone-local space
    (the earlier diagnostic that added ``world_pos`` twice produced 18-unit
    clearances); the rest bone position must be subtracted first.
    """
    bone = model["bones"][rb["bone"]]
    rest = np.asarray(bone["pos"], dtype=np.float64)
    off = np.asarray(rb["pos"], dtype=np.float64) - rest
    pos = np.asarray(world_pos, dtype=np.float64) + np.asarray(
        qrot(tuple(world_q), tuple(off)), dtype=np.float64)
    rot = qmul(tuple(world_q), euler_xyz_to_q(rb["rot"]))
    return pos, rot


def _capsule(model: Dict, rb: Dict, world_q, world_pos) -> Optional[Dict]:
    if int(rb.get("shape", -1)) not in (0, 2):
        return None
    pos, rot = _rigid_frame(model, rb, world_q, world_pos)
    radius = float(rb["size"][0])
    half = float(rb["size"][1]) * 0.5 if int(rb["shape"]) == 2 else 0.0
    axis = np.asarray(qrot(rot, (0.0, 1.0, 0.0)), dtype=np.float64)
    return {"p0": pos - axis * half, "p1": pos + axis * half,
            "radius": radius, "center": pos}


def _primary_rb(model: Dict, profile: Dict, joint: str) -> Optional[Dict]:
    bone = profile["standard_map"].get(joint)
    if bone is None:
        return None
    idx = next((i for i, b in enumerate(model["bones"]) if b["name"] == bone), None)
    if idx is None:
        return None
    # the first capsule/sphere is the body proxy; later ones are accessories
    for rb in model.get("rigid_bodies") or []:
        if rb.get("bone") == idx and int(rb.get("shape", -1)) in (0, 2):
            return rb
    return None


def capsule_penetrations(track: Dict, model: Dict, profile: Dict,
                         arm_arm: bool = True) -> Dict:
    """Per-frame arm-arm / arm-torso capsule penetration.

    Returns ``{"frames": [{pair, depth, point, normal}, ...], "max": ...}``.
    Only the primary body proxy per joint is used; hair/skirt/weapon rigid
    bodies would otherwise dominate and are not body-arm contact.
    """
    arms = [j for j in ARM_JOINTS if j in track["world"]]
    torso = [j for j in TORSO_JOINTS if j in track["world"]]
    arm_side = {j: j.rsplit("_", 1)[1] for j in arms}
    torso_bones = {j: _primary_rb(model, profile, j) for j in torso}
    arm_bones = {j: _primary_rb(model, profile, j) for j in arms}
    torso_bones = {j: r for j, r in torso_bones.items() if r is not None}
    arm_bones = {j: r for j, r in arm_bones.items() if r is not None}

    pairs: List[Tuple[str, str, str]] = []
    for a in arm_bones:
        for t in torso_bones:
            pairs.append((a, t, "arm_torso"))
    if arm_arm:
        la = [a for a in arm_bones if a.endswith("_l")]
        ra = [a for a in arm_bones if a.endswith("_r")]
        for a in la:
            for b in ra:
                pairs.append((a, b, "arm_arm"))

    frames = len(track["frames"])
    out_frames = []
    for f in range(frames):
        recs = []
        for a, b, kind in pairs:
            ca = _capsule(model, arm_bones[a], track["world_q"][a][f],
                          track["world"][a][f])
            cb = _capsule(model, torso_bones[b] if kind == "arm_torso" else arm_bones[b],
                          track["world_q"][b][f], track["world"][b][f])
            if ca is None or cb is None:
                continue
            pa, pb = _seg_seg_closest(ca["p0"], ca["p1"], cb["p0"], cb["p1"])
            dist = float(np.linalg.norm(pa - pb))
            depth = ca["radius"] + cb["radius"] - dist
            if depth <= 0.0:
                continue
            n = pa - pb
            nn = float(np.linalg.norm(n))
            if nn < 1e-9:
                # capsules are crossing: use the centre line, then a fallback
                n = np.asarray(ca["center"]) - np.asarray(cb["center"])
                nn = float(np.linalg.norm(n))
            normal = n / nn if nn > 1e-9 else np.array([0.0, 0.0, -1.0])
            recs.append({
                "frame": f, "a": a, "b": b, "kind": kind,
                "side": arm_side[a],
                "depth": float(depth), "point_a": pa, "point_b": pb,
                "normal": normal,
            })
        if recs:
            out_frames.append(recs)

    all_recs = [r for recs in out_frames for r in recs]
    by_kind: Dict[str, List[Dict]] = {"arm_torso": [], "arm_arm": []}
    for r in all_recs:
        by_kind[r["kind"]].append(r)
    summary = {}
    for kind, recs in by_kind.items():
        if not recs:
            summary[kind] = {"frames": 0, "pairs": 0, "max": 0.0,
                             "median": 0.0, "pairs_detail": {}}
            continue
        depths = np.asarray([r["depth"] for r in recs], dtype=np.float64)
        detail: Dict[str, Dict] = {}
        for r in recs:
            key = f'{r["a"]} x {r["b"]}'
            d = detail.setdefault(key, {"frames": 0, "max": 0.0})
            d["frames"] += 1
            d["max"] = max(d["max"], r["depth"])
        summary[kind] = {
            "frames": len({r["frame"] for r in recs}),
            "pairs": len(recs),
            "max": round(float(depths.max()), 4),
            "median": round(float(np.median(depths)), 4),
            "pairs_detail": {k: {"frames": v["frames"], "max": round(v["max"], 4)}
                             for k, v in sorted(detail.items())},
        }
    return {"frames": out_frames, "summary": summary,
            "max_arm_torso": float(summary["arm_torso"].get("max") or 0.0),
            "max_arm_arm": float(summary["arm_arm"].get("max") or 0.0)}


# --------------------------------------------------------------- palm frame

def palm_frame(model: Dict, profile: Dict, side: str) -> Dict:
    """Anatomical hand frame from the target model's own rest skeleton.

    The elbow is a hinge; its axis is the palm's medio-lateral axis.  In this
    rig the hand rest is palm-down, so the palm normal is that hinge line
    (sign is irrelevant for a hinge axis).  Finger positions are used instead
    of a hard-coded axis, so the prior is model-specific and GT-free.
    """
    names = {b["name"]: i for i, b in enumerate(model["bones"])}
    smap = profile["standard_map"]

    def pos(bone_name: str) -> np.ndarray:
        if bone_name not in names:
            raise KeyError(f"rest bone not found: {bone_name}")
        return np.asarray(model["bones"][names[bone_name]]["pos"], dtype=np.float64)

    def finger(f: str, seg: str) -> Optional[np.ndarray]:
        for cand in CANON.finger_candidates(side, f, seg):
            if cand in names:
                return pos(cand)
        return None

    wrist = pos(smap[f"hand_{side}"])
    mid = finger("middle", "1")
    index = finger("index", "1")
    pinky = finger("pinky", "1")
    if mid is None or index is None or pinky is None:
        raise KeyError(f"finger rest bones missing for side {side}")
    hand_dir = _unit(mid - wrist)
    spread = _unit(pinky - index)
    palm = _unit(np.cross(hand_dir, spread))
    up = _unit(pos(smap[f"forearm_{side}"]) - pos(smap[f"upperarm_{side}"]))
    fore = _unit(pos(smap[f"hand_{side}"]) - pos(smap[f"forearm_{side}"]))
    return {
        "side": side,
        "wrist": wrist, "hand_dir": hand_dir, "spread": spread,
        "palm_normal": palm, "hinge_line": palm,
        "upperarm_rest_dir": up, "forearm_rest_dir": fore,
        "hinge_perp_upperarm_deg": round(90.0 - _angle_between(palm, up), 3),
        "hand_forearm_deg": round(_angle_between(hand_dir, fore), 3),
    }


def _hinge_stats(track: Dict, side: str) -> Dict:
    """Elbow local rotation axis (in the rest frame), stability and angle."""
    up = f"upperarm_{side}"
    fore = f"forearm_{side}"
    qg_up = track["world_q"][up]
    qg_fo = track["world_q"][fore]
    axes, angles, flex = [], [], []
    for f in range(len(qg_up)):
        # L = inv(G_up) * G_fore is the forearm rotation relative to the
        # upper arm in the world-aligned rest frame.  A pure hinge has a
        # constant axis, independent of the shoulder pose.
        L = qmul(q_conj(tuple(qg_up[f])), tuple(qg_fo[f]))
        axis, deg = _q_axis_angle(L)
        axes.append(axis)
        angles.append(deg)
        du = track["world"][fore][f] - track["world"][up][f]
        df = track["world"][f"hand_{side}"][f] - track["world"][fore][f]
        flex.append(_angle_between(du, df))
    A = np.asarray(axes)
    mean = _unit(A.mean(0)) if len(A) else np.zeros(3)
    scatter = np.asarray([math.degrees(math.acos(
        float(np.clip(abs(np.dot(a, mean)), -1.0, 1.0)))) for a in A]) if len(A) else np.zeros(0)
    return {
        "axis_mean": [round(float(v), 4) for v in mean],
        "angle_mean_deg": round(float(np.mean(angles)) if angles else 0.0, 3),
        "axis_scatter_deg_mean": round(float(scatter.mean()) if len(scatter) else 0.0, 3),
        "axis_scatter_deg_p95": round(float(np.percentile(scatter, 95))
                                      if len(scatter) else 0.0, 3),
        "swing_flex_deg_mean": round(float(np.mean(flex)) if flex else 0.0, 3),
    }


def arm_structure(track: Dict, model: Dict, profile: Dict) -> Dict:
    """GT-free structural audit: hinge, wrist, shoulder direction, swing."""
    out = {"sides": {}}
    for side in ("l", "r"):
        up, fore, hand = (f"{p}_{side}" for p in ("upperarm", "forearm", "hand"))
        if not all(j in track["world"] for j in (up, fore, hand)):
            continue
        du = track["world"][fore] - track["world"][up]
        df = track["world"][hand] - track["world"][fore]
        du_n = np.asarray([_unit(v) for v in du])
        df_n = np.asarray([_unit(v) for v in df])
        chest = track["world_q"].get("chest")
        shoulder_dir = []
        if chest is not None:
            for f in range(len(du)):
                shoulder_dir.append(qrot(q_conj(tuple(chest[f])), tuple(du[f])))
        # wrist local rotation (relative to forearm, rest frame)
        wrist = []
        for f in range(len(du)):
            L = qmul(q_conj(tuple(track["world_q"][fore][f])),
                     tuple(track["world_q"][hand][f]))
            _, deg = _q_axis_angle(L)
            wrist.append(deg)
        pf = palm_frame(model, profile, side)
        hinge = _hinge_stats(track, side)
        dev = math.degrees(math.acos(float(np.clip(
            abs(np.dot(np.asarray(hinge["axis_mean"]), pf["hinge_line"])), -1.0, 1.0))))
        out["sides"][side] = {
            "upperarm_direction_chest_mean": [round(float(v), 3)
                                              for v in np.mean(shoulder_dir, axis=0)]
            if shoulder_dir else None,
            "elbow_hinge": hinge,
            "hinge_vs_palm_deg": round(float(dev), 3),
            "wrist_local_deg_mean": round(float(np.mean(wrist)) if wrist else 0.0, 3),
            "palm_frame": {k: [round(float(c), 4) for c in pf[k]]
                           for k in ("palm_normal", "hinge_line", "upperarm_rest_dir",
                                     "forearm_rest_dir")},
            "hinge_perp_upperarm_deg": pf["hinge_perp_upperarm_deg"],
        }
    return out


# ------------------------------------------------------- GT comparison (eval)

def compare_arm(gt_track: Dict, pred_track: Dict, model: Dict,
                profile: Dict) -> Dict:
    """Swing/roll decomposition of the arm world-angle error (GT read-only)."""
    out = {"sides": {}}
    for side in ("l", "r"):
        rows = {}
        for joint, child in (("upperarm", "forearm"), ("forearm", "hand")):
            j = f"{joint}_{side}"
            c = f"{child}_{side}"
            if j not in gt_track["world_q"] or j not in pred_track["world_q"]:
                continue
            qg = gt_track["world_q"][j]
            qp = pred_track["world_q"][j]
            dg = gt_track["world"][c] - gt_track["world"][j]
            dp = pred_track["world"][c] - pred_track["world"][j]
            total, swing, roll = [], [], []
            for f in range(len(qg)):
                ug, up = _unit(dg[f]), _unit(dp[f])
                qs = q_between(tuple(up), tuple(ug))
                qe = qmul(tuple(qg[f]), q_conj(tuple(qp[f])))
                qt = qmul(q_conj(qs), qe)
                total.append(_q_dist(qg[f], qp[f]))
                swing.append(_q_dist(qs, QID))
                roll.append(_q_dist(qt, QID))
            rows[j] = {
                "world_total_deg": round(float(np.mean(total)), 3),
                "swing_deg": round(float(np.mean(swing)), 3),
                "roll_deg": round(float(np.mean(roll)), 3),
                "pos_child_units": round(float(np.mean(np.linalg.norm(
                    (gt_track["world"][c] - gt_track["world"]["pelvis"]) -
                    (pred_track["world"][c] - pred_track["world"]["pelvis"]), axis=1))), 4),
            }
        # hand: use the PMX hand rest direction as the leaf direction
        j = f"hand_{side}"
        if j in gt_track["world_q"] and j in pred_track["world_q"]:
            pf = palm_frame(model, profile, side)
            # The hand is a leaf: use the model's own rest hand direction as
            # the swing reference, so the decomposition is defined for both
            # tracks on the same rig.
            hd = pf["hand_dir"]
            total, swing, roll = [], [], []
            for f in range(len(gt_track["world_q"][j])):
                dg = np.asarray(qrot(tuple(gt_track["world_q"][j][f]), tuple(hd)))
                dp = np.asarray(qrot(tuple(pred_track["world_q"][j][f]), tuple(hd)))
                qs = q_between(tuple(dp), tuple(dg))
                qe = qmul(tuple(gt_track["world_q"][j][f]),
                          q_conj(tuple(pred_track["world_q"][j][f])))
                qt = qmul(q_conj(qs), qe)
                total.append(_q_dist(gt_track["world_q"][j][f],
                                     pred_track["world_q"][j][f]))
                swing.append(_q_dist(qs, QID))
                roll.append(_q_dist(qt, QID))
            rows[j] = {
                "world_total_deg": round(float(np.mean(total)), 3),
                "swing_deg": round(float(np.mean(swing)), 3),
                "roll_deg": round(float(np.mean(roll)), 3),
                "pos_child_units": None,
            }
        out["sides"][side] = rows
    vals = [r["world_total_deg"] for s in out["sides"].values() for r in s.values()]
    swings = [r["swing_deg"] for s in out["sides"].values() for r in s.values()]
    rolls = [r["roll_deg"] for s in out["sides"].values() for r in s.values()]
    out["arm_world_mean_deg"] = round(float(np.mean(vals)), 3) if vals else None
    out["arm_swing_mean_deg"] = round(float(np.mean(swings)), 3) if swings else None
    out["arm_roll_mean_deg"] = round(float(np.mean(rolls)), 3) if rolls else None
    return out


# --------------------------------------------------------- hinge projection

def project_hinge(vmd: Dict, model: Dict, profile: Dict,
                  alpha: float = 1.0, cap_deg: float = DEFAULT_HINGE_CAP,
                  keep_flexion: bool = True) -> Dict:
    """M2.4 experiment: rotate the elbow local axis toward the palm hinge line.

    ``keep_flexion`` preserves the local elbow rotation angle; otherwise only
    the hinge component ``r·n`` is kept (which also changes the flexion).  The
    per-frame geodesic change is capped at ``cap_deg`` by an nlerp.  This
    function is deliberately separate from ``resolve_self_intersections``:
    it is a *prior*, not a constraint, and on R2 it trades forearm swing for
    roll (see module docstring).
    """
    frames, tracks = dense_tracks(vmd)
    report = {"mode": "hinge", "alpha": float(alpha), "cap_deg": float(cap_deg),
              "keep_flexion": bool(keep_flexion), "sides": {}}
    for side in ("l", "r"):
        n = palm_frame(model, profile, side)["hinge_line"]
        bn = profile["standard_map"][f"forearm_{side}"]
        if bn not in tracks:
            continue
        changed, caps = 0, 0
        angles = []
        for f in range(len(frames)):
            q = tracks[bn]["rot"][f]
            r = _q_log(q)
            th = float(np.linalg.norm(r))
            if th < 1e-8:
                continue
            sign = 1.0 if float(r @ n) >= 0.0 else -1.0
            target_angle = sign * th if keep_flexion else float(r @ n)
            if abs(target_angle) < 1e-9:
                continue
            q_t = q_from_axis_angle(*(n * target_angle))
            q_t = _nlerp(q, q_t, float(alpha))
            delta = _q_dist(q, q_t)
            if delta > cap_deg:
                q_t = _nlerp(q, q_t, cap_deg / max(delta, 1e-9))
                caps += 1
            tracks[bn]["rot"][f] = np.asarray(qnorm(tuple(q_t)))
            angles.append(delta)
            changed += 1
        report["sides"][side] = {
            "bone": bn, "changed_frames": changed, "capped_frames": caps,
            "delta_deg_mean": round(float(np.mean(angles)) if angles else 0.0, 3),
            "delta_deg_max": round(float(np.max(angles)) if angles else 0.0, 3),
            "hinge_line": [round(float(c), 4) for c in n],
        }
    out = tracks_to_vmd(vmd, frames, tracks)
    return {"vmd": out, "report": report}


# -------------------------------------------- self-intersection resolution

def _group_deepest(pen: Dict, tol: float, arm_arm: bool) -> Dict[int, List[Dict]]:
    """Deepest violation per frame (and per side for arm-arm split)."""
    out: Dict[int, List[Dict]] = {}
    for recs in pen["frames"]:
        # only pairs above the tolerance; the deepest one per frame drives
        deep = [r for r in recs
                if r["depth"] > tol and (arm_arm or r["kind"] == "arm_torso")]
        if not deep:
            continue
        deep.sort(key=lambda r: -r["depth"])
        out[int(deep[0]["frame"])] = deep
    return out


def _shoulder_delta(track: Dict, frame: int, side: str, point: np.ndarray,
                    push: np.ndarray, disp: float) -> Tuple[np.ndarray, float]:
    """Minimal shoulder rotation moving ``point`` by ``disp`` along ``push``."""
    up = f"upperarm_{side}"
    shoulder = np.asarray(track["world"][up][frame], dtype=np.float64)
    r = np.asarray(point, dtype=np.float64) - shoulder
    rn = float(r @ r)
    if rn < 1e-9:
        return np.zeros(3), 0.0
    omega = np.cross(r, push * disp) / rn
    return omega, math.degrees(float(np.linalg.norm(omega)))


def _apply_shoulder(tracks: Dict, track: Dict, model: Dict, profile: Dict,
                    frame: int, side: str, omega: np.ndarray) -> None:
    if float(np.linalg.norm(omega)) < 1e-12:
        return
    up = f"upperarm_{side}"
    bn = profile["standard_map"][up]
    if bn not in tracks:
        return
    L = np.asarray(tracks[bn]["rot"][frame], dtype=np.float64)
    Wu = np.asarray(track["world_q"][up][frame], dtype=np.float64)
    Wp = qmul(tuple(Wu), qinv(tuple(L)))       # actual parent world (肩C + inherit)
    dq = q_from_axis_angle(*omega)
    newL = qmul(qmul(qinv(Wp), qmul(dq, Wp)), tuple(L))
    tracks[bn]["rot"][frame] = np.asarray(qnorm(newL))


def resolve_self_intersections(vmd: Dict, model: Dict, profile: Dict,
                               pen_tol: float = DEFAULT_PEN_TOL,
                               max_deg: float = DEFAULT_MAX_DEG,
                               max_disp: float = DEFAULT_MAX_DISP,
                               rounds: int = DEFAULT_ROUNDS,
                               smooth: int = DEFAULT_SMOOTH,
                               arm_arm: bool = False) -> Dict:
    """2-3 rounds of capsule feedback, shoulder-only correction.

    ``arm_arm=False`` (default) resolves only arm-torso penetration: the GT
    itself keeps the two arms close (1.3-2.5 units centre clearance) but the
    front prediction overlaps them; the separation direction is ambiguous and
    resolving it was measured to worsen the 3D pose, so it stays opt-in.
    """
    frames, tracks = dense_tracks(vmd)
    history = []
    # cumulative world rotation vector per (side, frame); the trust region is
    # on the *total* shoulder correction, not on each round (3 x 12 deg would
    # otherwise become a 36 deg change).
    cum = {s: np.zeros((len(frames), 3), dtype=np.float64) for s in ("l", "r")}
    for rnd in range(int(rounds)):
        tr = eval_tracks(model, profile, tracks)
        pen = capsule_penetrations(tr, model, profile, arm_arm=arm_arm)
        deep = _group_deepest(pen, pen_tol, arm_arm)
        if not deep:
            history.append({"round": rnd, "corrections": 0,
                            "max_depth": round(pen.get("max_arm_torso", 0.0), 4)})
            break
        # (frame, side) -> omega ; split arm-arm half/half
        deltas: Dict[Tuple[int, str], np.ndarray] = {}
        for f, recs in deep.items():
            for rec in recs[:1]:
                depth = rec["depth"]
                disp = min(float(max_disp), max(0.0, depth - pen_tol))
                if disp <= 1e-9:
                    continue
                if rec["kind"] == "arm_torso":
                    omega, _ = _shoulder_delta(tr, f, rec["side"], rec["point_a"],
                                               rec["normal"], disp)
                    deltas[(f, rec["side"])] = omega
                else:
                    # push each arm apart by half the excess
                    other = "r" if rec["side"] == "l" else "l"
                    omega, _ = _shoulder_delta(tr, f, rec["side"], rec["point_a"],
                                               rec["normal"], 0.5 * disp)
                    deltas[(f, rec["side"])] = omega
                    omega2, _ = _shoulder_delta(
                        tr, f, other, rec["point_b"], -rec["normal"], 0.5 * disp)
                    deltas[(f, other)] = omega2
        if not deltas:
            break
        # temporal smooth of the world rotation-vector per side
        for side in ("l", "r"):
            arr = np.zeros((len(frames), 3), dtype=np.float64)
            for (f, s), om in deltas.items():
                if s == side:
                    arr[f] += om
            if smooth and smooth > 1:
                arr = _moving_average(arr, int(smooth))
            for f in range(len(frames)):
                cand = cum[side][f] + arr[f]
                ang = math.degrees(float(np.linalg.norm(cand)))
                if ang > max_deg:
                    cand *= max_deg / ang
                inc = cand - cum[side][f]
                cum[side][f] = cand
                _apply_shoulder(tracks, tr, model, profile, f, side, inc)
        history.append({"round": rnd, "corrections": len(deltas),
                        "max_depth": round(pen.get("max_arm_torso", 0.0), 4)})
    out = tracks_to_vmd(vmd, frames, tracks)
    return {"vmd": out, "report": {"mode": "constraints",
                                   "pen_tol": float(pen_tol),
                                   "max_deg": float(max_deg),
                                   "max_disp": float(max_disp),
                                   "rounds": int(rounds),
                                   "arm_arm": bool(arm_arm),
                                   "history": history}}


# ------------------------------------------------------------------ verify

def _q_close(a, b, tol=1e-9) -> bool:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return min(float(np.max(np.abs(a - b))), float(np.max(np.abs(a + b)))) <= tol


def verify(input_vmd: Dict, output_vmd: Dict, model: Dict, profile: Dict,
           report: Dict) -> Dict:
    """GT-free self-checks; any failure means the caller must drop the edit."""
    checks: Dict[str, Dict] = {}
    fin, tin = dense_tracks(input_vmd)
    fout, tout = dense_tracks(output_vmd)
    arm_bones = {profile["standard_map"][j] for j in ARM_JOINTS
                 if j in profile["standard_map"]}
    checks["frames"] = {"ok": fin == fout, "in": len(fin), "out": len(fout)}
    checks["same_bones"] = {"ok": set(tin) == set(tout),
                            "in": len(tin), "out": len(tout)}
    # non-arm tracks must be identical at every sampled frame
    bad = []
    for bn in sorted(set(tin) & set(tout)):
        if bn in arm_bones:
            continue
        if not _q_close(tin[bn]["rot"], tout[bn]["rot"]) or not _q_close(
                tin[bn]["pos"], tout[bn]["pos"]):
            bad.append(bn)
    checks["non_arm_unchanged"] = {"ok": not bad, "changed": bad[:10]}
    # the elbow/wrist locals are never touched by the constraint solver
    changed_arm = [bn for bn in sorted(arm_bones & set(tin) & set(tout))
                   if not _q_close(tin[bn]["rot"], tout[bn]["rot"])]
    allowed = {profile["standard_map"][j] for j in ("upperarm_l", "upperarm_r")
               if j in profile["standard_map"]}
    if str(report.get("mode", "")).startswith("hinge"):
        allowed |= {profile["standard_map"][j] for j in ("forearm_l", "forearm_r")
                    if j in profile["standard_map"]}
    unexpected = [bn for bn in changed_arm if bn not in allowed]
    checks["arm_locals_changed"] = {"ok": not unexpected,
                                    "changed": changed_arm,
                                    "unexpected": unexpected}
    max_change = 0.0
    for bn in changed_arm:
        for a, b in zip(tin[bn]["rot"], tout[bn]["rot"]):
            max_change = max(max_change, _q_dist(a, b))
    # vector addition of world rotation vectors is a small-angle approximation;
    # allow 0.25 deg of composition error, not an unbounded overshoot.
    checks["correction_cap"] = {
        "ok": max_change <= float(report.get("max_deg", 90.0)) + 0.25,
        "max_change_deg": round(max_change, 3),
        "cap_deg": float(report.get("max_deg", 90.0))}
    checks["ik_frames"] = {
        "ok": input_vmd.get("ik_frames", []) == output_vmd.get("ik_frames", []),
        "count": len(output_vmd.get("ik_frames", []))}
    # bone lengths (arm chain) must not change
    tr_in = eval_tracks(model, profile, tin)
    tr_out = eval_tracks(model, profile, tout)
    lbad = []
    pairs = [("upperarm", "forearm"), ("forearm", "hand")]
    for side in ("l", "r"):
        for a, b in pairs:
            ja, jb = f"{a}_{side}", f"{b}_{side}"
            if ja in tr_in["world"] and jb in tr_in["world"]:
                li = np.linalg.norm(tr_in["world"][ja] - tr_in["world"][jb], axis=1)
                lo = np.linalg.norm(tr_out["world"][ja] - tr_out["world"][jb], axis=1)
                if float(np.max(np.abs(li - lo))) > 1e-6:
                    lbad.append(ja)
    checks["arm_lengths"] = {"ok": not lbad, "changed": lbad}
    # penetration must not increase (target-space validity); arm-arm is always
    # audited even when the solver is not allowed to separate the arms, because
    # an arm-torso push can create a new left/right overlap.
    pin = capsule_penetrations(tr_in, model, profile, arm_arm=True)
    pout = capsule_penetrations(tr_out, model, profile, arm_arm=True)
    pen_tol = float(report.get("pen_tol", DEFAULT_PEN_TOL))
    checks["arm_torso_penetration"] = {
        "ok": pout["max_arm_torso"] <= pin["max_arm_torso"] + 1e-9,
        "before": round(pin["max_arm_torso"], 4),
        "after": round(pout["max_arm_torso"], 4)}
    mode = str(report.get("mode", "constraints"))
    if mode in ("constraints", "hinge+constraints"):
        need = (max(0.05, 0.25 * (pin["max_arm_torso"] - pen_tol))
                if pin["max_arm_torso"] > pen_tol else 0.0)
        checks["arm_torso_improved"] = {
            "ok": (pin["max_arm_torso"] <= pen_tol + 1e-9
                   or pout["max_arm_torso"] <= pin["max_arm_torso"] - need + 1e-9),
            "before": round(pin["max_arm_torso"], 4),
            "after": round(pout["max_arm_torso"], 4),
            "required_reduce": round(need, 4)}
    else:
        checks["arm_torso_improved"] = {"ok": True, "skipped": True,
                                        "reason": f"mode={mode} does not correct"}
    if report.get("arm_arm"):
        limit = pin["max_arm_arm"] + 0.02
    else:
        limit = max(pin["max_arm_arm"] + 0.02, pen_tol + 0.02)
    checks["arm_arm_not_worse"] = {
        "ok": pout["max_arm_arm"] <= limit + 1e-9,
        "before": round(pin["max_arm_arm"], 4),
        "after": round(pout["max_arm_arm"], 4),
        "limit": round(limit, 4)}
    return {"ok": all(v["ok"] for v in checks.values()), "checks": checks}
