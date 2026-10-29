# -*- coding: utf-8 -*-
"""M2.4b multi-view 3D arm direction prior (experiment, not a pipeline stage).

The M2.4 audit showed that the front/3-4 arm error is dominated by **depth**:
the 2D-visible arm direction residual is 3-11 deg while the 3D direction error
is 19-45 deg, and target-space constraints (capsules, joint limits, hinge
planes) cannot observe depth.  M2.3 showed that fusing the three views'
``body_pose`` with a **fixed GVHMR root** is self-contradictory.  This module
does what M2.3 could not: it treats the three calibrated cameras as a
metrology device and triangulates the **3D arm direction** itself.

What it computes
----------------
For every frame and every arm segment (upper arm, forearm) ViTPose gives a 2D
direction in each view.  A 2D direction back-projects to a plane through the
camera centre; the 3D segment direction is the common null vector of the three
planes.  The direction does not depend on root translation, body scale, or
which skeleton the body model has, so it is not contaminated by the per-view
GVHMR root error the way M2.3's ``body_pose`` fusion was.

Two important limits, both reported rather than hidden:

1. **This is a multi-view upper bound, not a single-view prior.**  It needs a
   calibrated multi-camera rig.  A single-view 3D arm prior has to come from a
   torso-relative pose prior learned on AMASS/SMPL-X (the repo has no such
   weights; see the M2.4b report).
2. **A world-space arm direction is not a torso-relative arm pose.**  The 3/4
   view's 63 deg world arm error contains a 51 deg chest/root orientation
   error.  Rotating the arm to the world target removes that root error from
   the arm, but it can make the arm *worse* relative to the torso.  The
   ``body_relative_errors`` function is therefore part of the acceptance gate;
   a candidate that wins only in world space is rejected (the same rule M2.4
   applied to roll-only wins).

Application
-----------
``apply_direction_prior`` changes only the upper-arm and forearm local
rotations, by the world-space swing that takes the current direction to the
target.  The parent world is recovered from the evaluated track
(``Wp = W * inv(L)``), so shoulder-C helpers / inheritance are accounted for
without writing them.  The wrist local rotation is untouched, so the hand
follows the forearm and no new unobservable wrist DOF is invented.  Bone
lengths, non-arm tracks and IK records are byte-identical.
"""
from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .adapter import q_between, q_conj, q_from_axis_angle, qmul, qrot
from .arm_prior import capsule_penetrations
from .cleanup import dense_tracks, eval_tracks, tracks_to_vmd
from .mmd.mathutil import qinv

# PMX world axes differ from the Blender/scene metric by a reflection:
# scene (x, y, z) -> PMX (x, z, y).  Directions transform with the same map.
SCENE_TO_PMX = np.array([[1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0],
                         [0.0, 1.0, 0.0]], dtype=np.float64)

DEFAULT_CAP_DEG = 60.0
DEFAULT_SMOOTH = 3
DEFAULT_CONF = 0.30

# COCO-17 keypoint indices.
COCO_L_SHOULDER, COCO_R_SHOULDER = 5, 6
COCO_L_ELBOW, COCO_R_ELBOW = 7, 8
COCO_L_WRIST, COCO_R_WRIST = 9, 10

PAIRS = {
    "upper_l": (COCO_L_SHOULDER, COCO_L_ELBOW),
    "fore_l": (COCO_L_ELBOW, COCO_L_WRIST),
    "upper_r": (COCO_R_SHOULDER, COCO_R_ELBOW),
    "fore_r": (COCO_R_ELBOW, COCO_R_WRIST),
}

ARM_LOCAL_BONES = ("upperarm_l", "forearm_l", "upperarm_r", "forearm_r")


# --------------------------------------------------------------- camera / IO

def load_blender_camera(meta_path: str) -> Dict:
    """Read the synthetic-scene Blender camera as a metric camera.

    Returns ``R`` (camera -> Blender/scene world, proper rotation), ``C``
    (camera centre in the scene), and intrinsics ``fx,fy,cx,cy``.  The scene
    is a right-handed Blender world; ``scene_to_pmx_*`` maps results to the
    left-handed PMX world used by the VMD metrics.
    """
    from scipy.spatial.transform import Rotation as Rot

    meta = json.load(open(meta_path, encoding="utf-8"))
    cam = meta["camera"]
    W, H = meta.get("resolution", [1080, 1920])
    # Blender AUTO sensor fit uses sensor_width on the larger image dimension.
    fx = float(cam["lens"]) / float(cam["sensor_width"]) * float(max(W, H))
    fy = fx
    q = cam["rotation_quaternion"]  # Blender w, x, y, z
    R_bcw = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    # Blender camera looks -Z / +Y up; OpenCV (GVHMR) looks +Z / +Y down.
    R_cv2scene = R_bcw @ np.diag([1.0, -1.0, -1.0])
    return {
        "R": np.asarray(R_cv2scene, dtype=np.float64),
        "C": np.asarray(cam["location"], dtype=np.float64),
        "fx": fx, "fy": fy, "cx": float(W) / 2.0, "cy": float(H) / 2.0,
        "width": int(W), "height": int(H), "meta": meta_path,
    }


def load_keypoints(path: str) -> np.ndarray:
    """Load a ViTPose ``(F, 17, 3)`` tensor saved by GVHMR preprocess."""
    import torch

    t = torch.load(path, map_location="cpu")
    a = t.numpy() if hasattr(t, "numpy") else np.asarray(t)
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 3 or a.shape[1] < 17:
        raise ValueError(f"expected (F,17,3) keypoints, got {a.shape}")
    return a


def scene_to_pmx(v: np.ndarray) -> np.ndarray:
    """Map scene/Blender vectors (…,3) to the PMX world (a reflection)."""
    return np.einsum("ij,...j->...i", SCENE_TO_PMX, np.asarray(v, dtype=np.float64))


def _unit(v: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, 1e-12)


# --------------------------------------------------------------- triangulation

def _pixel_rays(kp: np.ndarray, cam: Dict, idx: int) -> np.ndarray:
    """World-space ray through keypoint ``idx`` for every frame (F,3)."""
    uv = np.asarray(kp[:, idx, :2], dtype=np.float64)
    x = (uv[:, 0] - cam["cx"]) / cam["fx"]
    y = (uv[:, 1] - cam["cy"]) / cam["fy"]
    d_cam = np.stack([x, y, np.ones_like(x)], axis=1)
    return d_cam @ cam["R"].T


def _smooth_dirs(d: np.ndarray, smooth: int, max_step_deg: float = 0.0,
                 median: int = 0) -> np.ndarray:
    """Robust temporal filter on unit directions.

    2D keypoint noise makes the plane-intersection direction jump by tens of
    degrees on single frames even when the view geometry is well conditioned
    (the smallest singular value is small but not the only small one).  A
    plain box filter cannot remove those impulses, so the chain is:

    1. a component-wise **median** over ``median`` frames (removes impulses);
    2. an optional box average;
    3. a per-frame angular step cap (``max_step_deg``), which is the temporal
       trust region: the prior may lag a genuinely fast move, but it cannot
       inject a one-frame 60 deg jump into the VMD.

    The caller has already chosen a single global hemisphere for the segment;
    this filter never flips a frame on its own.
    """
    d = np.array(d, dtype=np.float64, copy=True)
    if median and median > 1 and len(d) >= median:
        pad = int(median) // 2
        dp = np.concatenate([np.repeat(d[:1], pad, 0), d,
                             np.repeat(d[-1:], pad, 0)], axis=0)
        med = np.zeros_like(d)
        for f in range(len(d)):
            w = dp[f:f + int(median)].copy()
            for k in range(len(w)):
                if float(np.dot(w[k], d[f])) < 0.0:
                    w[k] *= -1.0
            med[f] = np.median(w, axis=0)
        d = _unit(med)
    if smooth and smooth > 1 and len(d) >= smooth:
        k = np.ones(int(smooth), dtype=np.float64)
        k /= k.sum()
        pad = int(smooth) // 2
        dp = np.concatenate([np.repeat(d[:1], pad, 0), d,
                             np.repeat(d[-1:], pad, 0)], axis=0)
        out = np.stack([np.convolve(dp[:, c], k, mode="valid")
                        for c in range(3)], axis=1)
        d = _unit(out)
    if max_step_deg and max_step_deg > 0.0:
        for f in range(1, len(d)):
            cosang = float(np.clip(np.dot(d[f], d[f - 1]), -1.0, 1.0))
            ang = math.degrees(math.acos(cosang))
            if ang > max_step_deg:
                # Slerp from the previous direction toward the measured one.
                t = max_step_deg / ang
                q = q_between(tuple(d[f - 1]), tuple(d[f]))
                axis = np.asarray(q[:3])
                n = float(np.linalg.norm(axis))
                if n > 1e-12:
                    q2 = q_from_axis_angle(*(axis / n * math.radians(max_step_deg)))
                    d[f] = np.asarray(qrot(q2, tuple(d[f - 1])))
                    d[f] = _unit(d[f:f + 1])[0]
    return d


def triangulate_directions(kps: Dict[str, np.ndarray],
                           cams: Dict[str, Dict],
                           pairs: Optional[Dict[str, Tuple[int, int]]] = None,
                           conf_min: float = DEFAULT_CONF,
                           smooth: int = DEFAULT_SMOOTH,
                           prior: Optional[Dict[str, np.ndarray]] = None,
                           lam: float = 0.0,
                           max_step_deg: float = 0.0,
                           median: int = 0) -> Dict[str, Dict]:
    """Triangulate 3D segment directions from multi-view 2D keypoints.

    Parameters
    ----------
    kps : view -> (F,17,3) keypoints with confidence in the last channel.
    cams : view -> camera dict from :func:`load_blender_camera`.
    pairs : name -> (keypoint_a, keypoint_b); defaults to the four arm
        segments (``upper_l/fore_l/upper_r/fore_r``).
    prior : optional name -> (F,3) PMX-world direction used for sign
        disambiguation and (with ``lam>0``) a linear blend.  Pass the current
        prediction; never pass GT to the solver.
    lam : prior blend weight; 0 = pure triangulation.

    Returns ``name -> {"dir": (F,3) PMX, "cond": (F,), "weight": (F,)}``.
    ``cond`` is the smallest/largest singular-value ratio of the weighted
    plane normals; near 0 means the three planes agree on a direction, near 1
    means the view geometry is degenerate (the target should not be trusted).
    """
    pairs = pairs or PAIRS
    views = [v for v in kps if v in cams]
    if len(views) < 2:
        raise ValueError("need at least two calibrated views")
    F = min(len(kps[v]) for v in views)
    out: Dict[str, Dict] = {}
    for name, (i0, i1) in pairs.items():
        norms: List[np.ndarray] = []
        weights: List[np.ndarray] = []
        for v in views:
            kp = kps[v][:F]
            r0 = _pixel_rays(kp, cams[v], i0)
            r1 = _pixel_rays(kp, cams[v], i1)
            n = np.cross(r0, r1)
            n = _unit(n)
            w = np.clip(kp[:, i0, 2] * kp[:, i1, 2], 0.0, 1.0)
            w = np.where(w >= conf_min, w, 0.0)
            norms.append(n)
            weights.append(w)
        N = np.stack(norms, axis=1)       # (F,V,3)
        Wt = np.stack(weights, axis=1)    # (F,V)
        d_scene = np.zeros((F, 3), dtype=np.float64)
        cond = np.zeros(F, dtype=np.float64)
        wsum = Wt.sum(axis=1)
        for f in range(F):
            A = N[f] * Wt[f, :, None]
            if float(wsum[f]) < 1e-9:
                # No usable 2D evidence: fall back to the prior if present.
                if prior is not None and name in prior:
                    d_scene[f] = np.asarray(SCENE_TO_PMX.T @ prior[name][f],
                                            dtype=np.float64)
                    cond[f] = 1.0
                continue
            _, s, vt = np.linalg.svd(A)
            d_scene[f] = vt[-1]
            cond[f] = float(s[-1] / max(s[0], 1e-12))
        d_pmx = scene_to_pmx(d_scene)
        # Per-frame hemisphere anchor on the prediction.  The plane
        # intersection is orientation-free; anchoring on the current arm keeps
        # the correction small.  This is simple and works on average, but it
        # flips when the line is near perpendicular to the prediction -- that
        # is the jerk the report quantifies; a learned torso-relative prior is
        # what removes it, not a longer filter.
        if prior is not None and name in prior:
            p0 = np.asarray(prior[name], dtype=np.float64)
            flip = np.sum(d_pmx * p0, axis=1) < 0.0
            d_pmx[flip] *= -1.0
        if lam > 0.0 and prior is not None and name in prior:
            p = _unit(np.asarray(prior[name], dtype=np.float64))
            d_pmx = _unit(d_pmx + float(lam) * p)
        d_pmx = _smooth_dirs(d_pmx, smooth, max_step_deg=max_step_deg,
                             median=median)
        out[name] = {"dir": d_pmx, "cond": cond, "weight": wsum}
    return out


# --------------------------------------------------------------- VMD apply

def _cap_swing(q: Sequence[float], cap_deg: float) -> Tuple[Tuple[float, ...], float]:
    """Clamp a world-swing quaternion to ``cap_deg``; returns (q, angle_before)."""
    q = np.asarray(q, dtype=np.float64)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    if q[3] < 0.0:
        q = -q
    ang = math.degrees(2.0 * math.acos(float(np.clip(q[3], -1.0, 1.0))))
    if ang <= cap_deg or ang < 1e-9:
        return tuple(q), ang
    axis = q[:3] / max(float(np.linalg.norm(q[:3])), 1e-12)
    capped = q_from_axis_angle(*(axis * math.radians(cap_deg)))
    return tuple(capped), ang


def _apply_world_swing(local_q, world_q, dq):
    """Change a bone's local rotation so its world rotation gains ``dq``.

    ``Wp`` is the *actual* parent world recovered from the evaluated track, so
    PMX shoulder-C helpers and inheritance need not be written.
    """
    Wp = qmul(tuple(world_q), qinv(tuple(local_q)))
    return qmul(qmul(qinv(Wp), qmul(tuple(dq), Wp)), tuple(local_q))


def _frame_depths(model: Dict, profile: Dict,
                  tracks: Dict[str, Dict[str, np.ndarray]], f: int) -> Dict:
    """Capsule penetration depth per (arm, other) pair for one frame."""
    sub = {bn: {"rot": tracks[bn]["rot"][f:f + 1],
                "pos": tracks[bn]["pos"][f:f + 1]} for bn in tracks}
    tr = eval_tracks(model, profile, sub)
    pen = capsule_penetrations(tr, model, profile, arm_arm=True)
    out: Dict = {}
    for recs in pen["frames"]:
        for r in recs:
            key = (r["a"], r["b"])
            out[key] = max(out.get(key, 0.0), float(r["depth"]))
    return out


def apply_direction_prior(vmd: Dict, model: Dict, profile: Dict,
                          targets: Dict[str, np.ndarray],
                          cap_deg: float = DEFAULT_CAP_DEG,
                          valid: Optional[Dict[str, np.ndarray]] = None,
                          sides: Sequence[str] = ("l", "r"),
                          pen_guard: bool = True,
                          pen_tol: float = 0.01,
                          signed: bool = False) -> Dict:
    """Rotate the upper arms / forearms toward the target world directions.

    ``targets`` is ``upper_l/fore_l/upper_r/fore_r`` -> (F,3) PMX-world unit
    directions from :func:`triangulate_directions` (or any other 3D prior).
    The wrist local rotation is not touched; the hand follows its forearm.

    ``signed`` says the target is a true direction, not an orientation-free
    line.  Triangulated directions are lines (the plane intersection has a
    180 deg sign ambiguity), so they are flipped per frame toward the
    prediction.  A learned torso-relative prior outputs a signed direction;
    flipping that would inject a 180 deg jump whenever the target is near
    perpendicular to the current arm.  The M2.4c oracle audit hit exactly this
    (122-179 deg output steps while the target itself moved <=23 deg/frame).

    With ``pen_guard`` (default) every frame's candidate is evaluated against
    the model's own capsules and **rejected** if any arm-torso / arm-arm pair
    gets deeper than it was on the current pose.  This is the same rule as
    M2.4's constraint gate: a pose win that creates a new self-intersection is
    not accepted.  The guard is deliberately frame-local, so it cannot say
    "the arm should be somewhere else"; it only prevents the prior from
    pushing the arm through the body.
    """
    frames, tracks = dense_tracks(vmd)
    tr = eval_tracks(model, profile, tracks)
    F = len(frames)
    report: Dict = {
        "mode": "arm3d_direction_prior", "cap_deg": float(cap_deg),
        "sides": {}, "frames": int(F), "pen_guard": bool(pen_guard),
        "pen_tol": float(pen_tol),
    }
    cur_depths: List[Dict] = []
    applied: Dict[str, np.ndarray] = {
        f"{seg}_{side}": np.zeros(F, dtype=bool)
        for side in ("l", "r") for seg in ("upper", "fore")}
    if pen_guard:
        base = capsule_penetrations(tr, model, profile, arm_arm=True)
        cur_depths = [{} for _ in range(F)]
        for recs in base["frames"]:
            for r in recs:
                key = (r["a"], r["b"])
                d = cur_depths[r["frame"]]
                d[key] = max(d.get(key, 0.0), float(r["depth"]))
    for side in sides:
        up, fo, ha = (f"{p}_{side}" for p in ("upperarm", "forearm", "hand"))
        bn_up = profile["standard_map"].get(up)
        bn_fo = profile["standard_map"].get(fo)
        bn_ha = profile["standard_map"].get(ha)
        if not (bn_up in tracks and bn_fo in tracks and bn_ha in tracks):
            continue
        tu = np.asarray(targets.get(f"upper_{side}", np.zeros((F, 3))),
                        dtype=np.float64)
        tf = np.asarray(targets.get(f"fore_{side}", np.zeros((F, 3))),
                        dtype=np.float64)
        vu = valid.get(f"upper_{side}") if valid else None
        vf = valid.get(f"fore_{side}") if valid else None
        changed_u, changed_f, caps_u, caps_f, rejected = 0, 0, 0, 0, 0
        max_du, max_df = 0.0, 0.0
        for f in range(F):
            if (vu is not None and not bool(vu[f])) or float(np.linalg.norm(tu[f])) < 1e-9:
                continue
            S = np.asarray(tr["world"][up][f], dtype=np.float64)
            E = np.asarray(tr["world"][fo][f], dtype=np.float64)
            W = np.asarray(tr["world"][ha][f], dtype=np.float64)
            L1 = float(np.linalg.norm(E - S))
            L2 = float(np.linalg.norm(W - E))
            if L1 < 1e-9 or L2 < 1e-9:
                continue
            u0 = (E - S) / L1
            f0 = (W - E) / L2
            du = np.asarray(tu[f], dtype=np.float64)
            du = du / max(float(np.linalg.norm(du)), 1e-12)
            # The target is a *line* (the triangulation is orientation-free
            # up to 180 deg); pick the endpoint closest to the current arm so
            # the correction is the small swing, not a spin.  A signed
            # prior target must not be flipped.
            if not signed and float(np.dot(du, u0)) < 0.0:
                du = -du
            dq_u = q_between(tuple(u0), tuple(du))
            dq_u, ang_u = _cap_swing(dq_u, cap_deg)
            if ang_u > cap_deg:
                caps_u += 1
            f1 = np.asarray(qrot(tuple(dq_u), tuple(f0)), dtype=np.float64)
            df = np.asarray(tf[f], dtype=np.float64)
            if float(np.linalg.norm(df)) < 1e-9:
                df = f1
            df = df / max(float(np.linalg.norm(df)), 1e-12)
            if not signed and float(np.dot(df, f1)) < 0.0:
                df = -df
            dq_f = q_between(tuple(f1), tuple(df))
            dq_f, ang_f = _cap_swing(dq_f, cap_deg)
            if ang_f > cap_deg:
                caps_f += 1

            Lu = np.array(tracks[bn_up]["rot"][f], dtype=np.float64)
            Wu = np.asarray(tr["world_q"][up][f], dtype=np.float64)
            new_u = _apply_world_swing(Lu, Wu, dq_u)
            Lf = np.array(tracks[bn_fo]["rot"][f], dtype=np.float64)
            Wf = np.asarray(tr["world_q"][fo][f], dtype=np.float64)
            Wf_after_parent = qmul(tuple(dq_u), tuple(Wf))
            new_f = _apply_world_swing(Lf, Wf_after_parent, dq_f)

            old_u = tracks[bn_up]["rot"][f].copy()
            old_f = tracks[bn_fo]["rot"][f].copy()
            tracks[bn_up]["rot"][f] = np.asarray(new_u, dtype=np.float64)
            tracks[bn_fo]["rot"][f] = np.asarray(new_f, dtype=np.float64)
            if pen_guard:
                before = cur_depths[f]
                after = _frame_depths(model, profile, tracks, f)
                bad = False
                for key, depth in after.items():
                    if depth > before.get(key, 0.0) + pen_tol:
                        bad = True
                        break
                if bad:
                    tracks[bn_up]["rot"][f] = old_u
                    tracks[bn_fo]["rot"][f] = old_f
                    rejected += 1
                    continue
                cur_depths[f] = after
            changed_u += 1
            changed_f += 1
            applied[f"upper_{side}"][f] = True
            applied[f"fore_{side}"][f] = True
            max_du = max(max_du, min(ang_u, cap_deg))
            max_df = max(max_df, min(ang_f, cap_deg))
        report["sides"][side] = {
            "frames_upper": changed_u, "frames_fore": changed_f,
            "cap_frames_upper": caps_u, "cap_frames_fore": caps_f,
            "pen_guard_rejected": rejected,
            "max_upper_correction_deg": round(max_du, 3),
            "max_fore_correction_deg": round(max_df, 3),
        }
    out = tracks_to_vmd(vmd, frames, tracks)
    return {"vmd": out, "report": report, "tracks": tracks,
            "applied": applied}


# --------------------------------------------------------------- evaluation

def current_directions(vmd: Dict, model: Dict, profile: Dict) -> Dict[str, np.ndarray]:
    """Current upper-arm / forearm world directions of a VMD.

    Used as the hemisphere anchor and as the no-op target in tests.  The
    vectors are unnormalised; callers normalise.
    """
    frames, tracks = dense_tracks(vmd)
    tr = eval_tracks(model, profile, tracks)
    out = {}
    for side in ("l", "r"):
        up, fo, ha = (f"{p}_{side}" for p in ("upperarm", "forearm", "hand"))
        out[f"upper_{side}"] = np.asarray(tr["world"][fo]) - np.asarray(tr["world"][up])
        out[f"fore_{side}"] = np.asarray(tr["world"][ha]) - np.asarray(tr["world"][fo])
    return out


def body_relative_errors(gt_track: Dict, pred_track: Dict,
                         model: Dict, profile: Dict) -> Dict:
    """Torso-relative arm direction errors (the meaningful arm-pose gate).

    A world-space arm correction can win by cancelling a root/chest
    orientation error; this metric rotates every arm direction into its own
    track's chest frame first, so a root rotation applied to the whole body
    does not change it.  It is the check that separates "the arm moved" from
    "the whole character was turned".
    """
    from .arm_prior import palm_frame

    out: Dict = {"sides": {}, "arm_body_mean_deg": None,
                 "chest_world_mean_deg": None}
    qcg = gt_track["world_q"]["chest"]
    qcp = pred_track["world_q"]["chest"]
    chest = []
    vals = []
    for side in ("l", "r"):
        rows = {}
        for bone, child in (("upperarm", "forearm"), ("forearm", "hand")):
            j = f"{bone}_{side}"
            c = f"{child}_{side}"
            if j not in gt_track["world"] or j not in pred_track["world"]:
                continue
            errs = []
            for f in range(len(gt_track["frames"])):
                dg = np.asarray(gt_track["world"][c][f]) - gt_track["world"][j][f]
                dp = np.asarray(pred_track["world"][c][f]) - pred_track["world"][j][f]
                dg = dg / max(float(np.linalg.norm(dg)), 1e-12)
                dp = dp / max(float(np.linalg.norm(dp)), 1e-12)
                dg_c = np.asarray(qrot(tuple(q_conj(tuple(qcg[f]))), tuple(dg)))
                dp_c = np.asarray(qrot(tuple(q_conj(tuple(qcp[f]))), tuple(dp)))
                errs.append(math.degrees(math.acos(float(np.clip(
                    np.dot(dg_c, dp_c), -1.0, 1.0)))))
            rows[bone] = round(float(np.mean(errs)), 3) if errs else None
            if errs:
                vals.append(float(np.mean(errs)))
        # Hand is a leaf: use the model's own rest hand direction, as
        # compare_arm does, and bring both into their own chest frames.
        j = f"hand_{side}"
        if j in gt_track["world_q"] and j in pred_track["world_q"]:
            hd = np.asarray(palm_frame(model, profile, side)["hand_dir"])
            errs = []
            for f in range(len(gt_track["frames"])):
                dg = np.asarray(qrot(tuple(gt_track["world_q"][j][f]), tuple(hd)))
                dp = np.asarray(qrot(tuple(pred_track["world_q"][j][f]), tuple(hd)))
                dg_c = np.asarray(qrot(tuple(q_conj(tuple(qcg[f]))), tuple(dg)))
                dp_c = np.asarray(qrot(tuple(q_conj(tuple(qcp[f]))), tuple(dp)))
                errs.append(math.degrees(math.acos(float(np.clip(
                    np.dot(dg_c, dp_c), -1.0, 1.0)))))
            rows["hand"] = round(float(np.mean(errs)), 3) if errs else None
            if errs:
                vals.append(float(np.mean(errs)))
        out["sides"][side] = rows
    for f in range(len(gt_track["frames"])):
        chest.append(math.degrees(2.0 * math.acos(min(1.0, abs(float(
            np.dot(qcg[f], qcp[f])))))))
    out["chest_world_mean_deg"] = round(float(np.mean(chest)), 3) if chest else None
    out["arm_body_mean_deg"] = round(float(np.mean(vals)), 3) if vals else None
    return out


def penetration(vmd: Dict, model: Dict, profile: Dict) -> Dict:
    frames, tracks = dense_tracks(vmd)
    tr = eval_tracks(model, profile, tracks)
    return capsule_penetrations(tr, model, profile, arm_arm=True)["summary"]


# --------------------------------------------------------------- verify

def _q_close(a, b, tol: float = 1e-9) -> bool:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return min(float(np.max(np.abs(a - b))), float(np.max(np.abs(a + b)))) <= tol


def verify(input_vmd: Dict, output_vmd: Dict, model: Dict, profile: Dict,
           report: Dict, targets: Optional[Dict[str, np.ndarray]] = None,
           applied: Optional[Dict[str, np.ndarray]] = None,
           max_target_err_deg: float = 0.5) -> Dict:
    """GT-free self-checks; a failure means the caller must drop the edit."""
    fin, tin = dense_tracks(input_vmd)
    fout, tout = dense_tracks(output_vmd)
    checks: Dict[str, Dict] = {}
    checks["frames"] = {"ok": fin == fout, "in": len(fin), "out": len(fout)}
    checks["same_bones"] = {"ok": set(tin) == set(tout),
                            "in": len(tin), "out": len(tout)}
    arm_locals = {profile["standard_map"][j] for j in ARM_LOCAL_BONES
                  if j in profile["standard_map"]}
    hand_locals = {profile["standard_map"][j] for j in ("hand_l", "hand_r")
                   if j in profile["standard_map"]}
    bad = []
    for bn in sorted(set(tin) & set(tout)):
        if bn in arm_locals or bn in hand_locals:
            continue
        if not _q_close(tin[bn]["rot"], tout[bn]["rot"]) or not _q_close(
                tin[bn]["pos"], tout[bn]["pos"]):
            bad.append(bn)
    checks["non_arm_unchanged"] = {"ok": not bad, "changed": bad[:10]}
    changed_hand = [bn for bn in sorted(hand_locals & set(tin) & set(tout))
                    if not _q_close(tin[bn]["rot"], tout[bn]["rot"])]
    checks["hand_local_unchanged"] = {"ok": not changed_hand,
                                      "changed": changed_hand}
    changed = [bn for bn in sorted(arm_locals & set(tin) & set(tout))
               if not _q_close(tin[bn]["rot"], tout[bn]["rot"])]
    checks["arm_locals_changed"] = {"ok": True, "changed": changed}
    max_change = 0.0
    for bn in changed:
        for a, b in zip(tin[bn]["rot"], tout[bn]["rot"]):
            d = abs(float(np.dot(np.asarray(a), np.asarray(b))))
            max_change = max(max_change, math.degrees(2.0 * math.acos(min(1.0, d))))
    checks["correction_cap"] = {
        "ok": max_change <= float(report.get("cap_deg", 90.0)) + 0.25,
        "max_change_deg": round(max_change, 3),
        "cap_deg": float(report.get("cap_deg", 90.0))}
    checks["ik_frames"] = {
        "ok": input_vmd.get("ik_frames", []) == output_vmd.get("ik_frames", []),
        "count": len(output_vmd.get("ik_frames", []))}
    checks["track_count"] = {"ok": len(tout) == 23, "count": len(tout)}
    tr_in = eval_tracks(model, profile, tin)
    tr_out = eval_tracks(model, profile, tout)
    lbad = []
    for side in ("l", "r"):
        for a, b in (("upperarm", "forearm"), ("forearm", "hand")):
            ja, jb = f"{a}_{side}", f"{b}_{side}"
            li = np.linalg.norm(tr_in["world"][ja] - tr_in["world"][jb], axis=1)
            lo = np.linalg.norm(tr_out["world"][ja] - tr_out["world"][jb], axis=1)
            if float(np.max(np.abs(li - lo))) > 1e-6:
                lbad.append(ja)
    checks["arm_lengths"] = {"ok": not lbad, "changed": lbad}

    def _steps(tr):
        vals = []
        for side in ("l", "r"):
            for seg, child in (("upperarm", "forearm"), ("forearm", "hand")):
                j, c = f"{seg}_{side}", f"{child}_{side}"
                d = np.asarray(tr["world"][c]) - tr["world"][j]
                d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
                if len(d) > 1:
                    vals.append(np.degrees(np.arccos(np.clip(
                        (d[1:] * d[:-1]).sum(1), -1, 1.0))))
        return np.concatenate(vals) if vals else np.zeros(1)

    s_in, s_out = _steps(tr_in), _steps(tr_out)
    limit = max(15.0, float(np.max(s_in))) + 5.0
    checks["temporal_smoothness"] = {
        "ok": float(np.max(s_out)) <= limit,
        "max_step_in_deg": round(float(np.max(s_in)), 3),
        "max_step_out_deg": round(float(np.max(s_out)), 3),
        "p99_step_out_deg": round(float(np.percentile(s_out, 99)), 3),
        "limit_deg": round(limit, 3),
        "note": ("a per-frame 3D target with a hemisphere/outlier jump injects "
                 "a visible jerk spike; the output may not be less smooth than "
                 "the input by more than 5 deg/frame")}
    pin = capsule_penetrations(tr_in, model, profile, arm_arm=True)["summary"]
    pout = capsule_penetrations(tr_out, model, profile, arm_arm=True)["summary"]
    checks["arm_torso_penetration"] = {
        "ok": pout["arm_torso"]["max"] <= pin["arm_torso"]["max"] + 1e-9,
        "before": round(pin["arm_torso"]["max"], 4),
        "after": round(pout["arm_torso"]["max"], 4)}
    checks["arm_arm_penetration"] = {
        "ok": pout["arm_arm"]["max"] <= pin["arm_arm"]["max"] + 0.02,
        "before": round(pin["arm_arm"]["max"], 4),
        "after": round(pout["arm_arm"]["max"], 4),
        "limit": round(pin["arm_arm"]["max"] + 0.02, 4)}
    # The global max can hide a new local contact (e.g. the right forearm
    # entering the chest while the left forearm still dominates the max).
    pair_bad = []
    for group in ("arm_torso", "arm_arm"):
        before = (pin.get(group) or {}).get("pairs_detail", {}) or {}
        after = (pout.get(group) or {}).get("pairs_detail", {}) or {}
        for name, rec in after.items():
            b = float((before.get(name) or {}).get("max", 0.0))
            a = float(rec.get("max", 0.0))
            if a > b + 0.02:
                pair_bad.append({"pair": name, "group": group,
                                 "before": round(b, 4), "after": round(a, 4)})
    checks["penetration_pairs"] = {"ok": not pair_bad,
                                   "increased": pair_bad[:12],
                                   "tolerance": 0.02}
    if targets:
        terr = []
        for side in ("l", "r"):
            for seg in ("upper", "fore"):
                name = f"{seg}_{side}"
                if name not in targets:
                    continue
                tgt = np.asarray(targets[name], dtype=np.float64)
                if seg == "upper":
                    parent, child = f"upperarm_{side}", f"forearm_{side}"
                else:
                    parent, child = f"forearm_{side}", f"hand_{side}"
                for f in range(min(len(fout), len(tgt))):
                    if float(np.linalg.norm(tgt[f])) < 1e-9:
                        continue
                    if applied is not None and name in applied and not bool(applied[name][f]):
                        continue
                    d = np.asarray(tr_out["world"][child][f]) - tr_out["world"][parent][f]
                    d = d / max(float(np.linalg.norm(d)), 1e-12)
                    t = tgt[f] / max(float(np.linalg.norm(tgt[f])), 1e-12)
                    terr.append(math.degrees(math.acos(float(np.clip(
                        abs(float(np.dot(d, t))), -1.0, 1.0)))))
        checks["target_reached"] = {
            "ok": (not terr) or max(terr) <= max_target_err_deg + 1e-6,
            "max_err_deg": round(float(max(terr)), 3) if terr else 0.0,
            "max_allowed_deg": max_target_err_deg}
    return {"ok": all(bool(v.get("ok", True)) for v in checks.values()),
            "checks": checks,
            "penetration_summary": {"before": pin, "after": pout}}


# --------------------------------------------------------------- CLI helpers

def targets_from_result(tri: Dict, conf_floor: float = 0.02) -> Tuple[Dict, Dict]:
    """Split a triangulation result into direction and validity maps.

    Frames with almost no usable 2D evidence (or degenerate view geometry,
    ``cond`` near 1) are marked invalid so the solver keeps the prediction
    there instead of inventing a direction.
    """
    dirs, valid = {}, {}
    for name, rec in tri.items():
        d = np.asarray(rec["dir"], dtype=np.float64)
        ok = np.isfinite(d).all(axis=1)
        w = np.asarray(rec.get("weight", np.ones(len(d))), dtype=np.float64)
        ok &= w > conf_floor
        c = np.asarray(rec.get("cond", np.zeros(len(d))), dtype=np.float64)
        ok &= c < 0.98
        dirs[name] = d
        valid[name] = ok
    return dirs, valid
