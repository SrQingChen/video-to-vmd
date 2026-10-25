# -*- coding: utf-8 -*-
"""M1 target adapter: canonical body motion -> PMX local bone animation.

Pipeline (v2 §4.3):

    G_tgt(j)   = D_j · Q · G_src(j) · Q⁻¹ · D_j⁻¹        (world rotation)
    R_local    = inv( W(parent) · inherit ) · G_tgt(j)   (via the *real* PMX chain)
    root_t     = P_tgt(pelvis) − C_rest − R_root·(P_rest − C_rest)

* ``G_src`` is the canonical world rotation of a joint (SMPL-X or PMX source);
* ``D_j`` maps the source rest bone direction to the target rest bone direction
  (A-pose vs T-pose compensation); leaves inherit their parent's ``D``; the
  root/pelvis use the identity because their frame *is* the body frame;
* ``Q`` is the source-world -> target-world frame change (identity for GVHMR:
  its world is already X-right / Y-up / Z-back, verified on T5/T1);
* inheritance (肩C −1.0, twist helpers) is evaluated exactly as MMD does, so a
  bone whose parent is an *unwritten* helper still gets the right local value;
* twist is split off the upper arm/forearm and written to the model's main
  twist bones (腕捩/手捩): ``L = Swing·Twist``, 腕 gets the swing, 腕捩 the
  twist.  The skeleton result is identical (the twist is about the bone axis,
  so positions are unchanged) and the arm mesh twists where the rig expects.

Root movement: only ``センター`` is translated (safe_mode rule) and only as a
*delta* from the target rest pelvis, scaled by the leg-length ratio and
anchored to the source rest stance.  Absolute ``transl`` is never written.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import canonical_body as CB
from .mmd.animation import Skeleton
from .mmd.evaluate import VmdEvaluator
from .mmd.mathutil import (QID, qaxis, qinv, qmul, qnorm, qrot, qslerp,
                           vcross, vdot, vlen, vnorm)

FPS = 30.0

# Joints whose rest *direction* is not the same anatomical segment in the two
# rigs, so a direction-based swing compensation would be wrong:
#   SMPL-X  ankle->foot joint = the metatarsal (toe base), nearly horizontal;
#   PMX     足首->つま先      = ankle to toe tip, ~50 deg down-forward.
# Compensating that difference rotates the whole foot ~100 deg (it sank T5 by
# 4 units and pointed the toes up).  The foot is a frame (sole orientation),
# not a limb segment, so the ankle keeps the identity alignment.
NO_SWING = ("foot_l", "foot_r", "toe_l", "toe_r")


# ---------------------------------------------------------------- quaternions

def q_angle_deg(q: Sequence[float]) -> float:
    w = max(-1.0, min(1.0, abs(q[3])))
    return math.degrees(2.0 * math.acos(w))


def q_conj(q: Sequence[float]) -> Tuple[float, float, float, float]:
    return (-q[0], -q[1], -q[2], q[3])


def q_between(u: Sequence[float], v: Sequence[float]) -> Tuple[float, float, float, float]:
    """Minimal (swing) rotation taking unit vector ``u`` to unit vector ``v``."""
    u, v = vnorm(tuple(u)), vnorm(tuple(v))
    d = max(-1.0, min(1.0, vdot(u, v)))
    if d > 1.0 - 1e-12:
        return QID
    if d < -1.0 + 1e-9:
        axis = vcross(u, (1.0, 0.0, 0.0))
        if vlen(axis) < 1e-6:
            axis = vcross(u, (0.0, 1.0, 0.0))
        return qaxis(vnorm(axis), math.pi)
    axis = vcross(u, v)
    s = math.sqrt((1.0 + d) * 2.0)
    return (axis[0] / s, axis[1] / s, axis[2] / s, s * 0.5)


def q_twist_swing(q: Sequence[float], axis: Sequence[float]):
    """Split ``q = swing · twist`` with ``twist`` about ``axis`` (unit)."""
    a = vnorm(tuple(axis))
    proj = vdot(q[:3], a)
    if abs(proj) < 1e-12:
        return qnorm(tuple(q)), QID
    twist = qnorm((a[0] * proj, a[1] * proj, a[2] * proj, q[3]))
    swing = qmul(q, qinv(twist))
    return qnorm(swing), twist


def q_from_axis_angle(rx, ry, rz) -> Tuple[float, float, float, float]:
    """Axis-angle vector -> quaternion (stable for small angles)."""
    ang = math.sqrt(rx * rx + ry * ry + rz * rz)
    if ang < 1e-12:
        return QID
    s = math.sin(ang * 0.5) / ang
    return (rx * s, ry * s, rz * s, math.cos(ang * 0.5))


def yaw_quat(deg: float):
    return qaxis((0.0, 1.0, 0.0), math.radians(deg))


def qrot_batch(q: np.ndarray, v: Sequence[float]) -> np.ndarray:
    """Rotate one vector by every quaternion in ``q`` (F,4) -> (F,3)."""
    v = np.asarray(v, dtype=np.float64)
    xyz = q[:, :3]
    w = q[:, 3:4]
    t = 2.0 * np.cross(xyz, v)
    return v[None, :] + w * t + np.cross(xyz, t)


def fk_positions(world_rot: Dict[str, np.ndarray],
                 rest_joints: Dict[str, Sequence[float]],
                 root_pos: np.ndarray) -> Dict[str, np.ndarray]:
    """Canonical forward kinematics (SMPL-X / MMD convention).

    ``J_j = J_parent + R_parent · (rest_j - rest_parent)``; a joint's own
    rotation only moves its children.  ``root_pos`` is the pelvis position.
    """
    rest = {j: np.asarray(v, dtype=np.float64) for j, v in rest_joints.items()}
    out = {"pelvis": np.asarray(root_pos, dtype=np.float64)}
    for j in ["thigh_l", "shin_l", "foot_l", "toe_l",
              "thigh_r", "shin_r", "foot_r", "toe_r",
              "spine", "spine1", "chest", "neck", "head",
              "clavicle_l", "upperarm_l", "forearm_l", "hand_l",
              "clavicle_r", "upperarm_r", "forearm_r", "hand_r"]:
        p = CB.CANONICAL_PARENT[j]
        if p not in out or j not in rest or p not in rest:
            continue
        out[j] = out[p] + qrot_batch(world_rot[p], rest[j] - rest[p])
    return out


# ---------------------------------------------------------------- canonical

def _segment_frame(model: Dict, name2i: Dict[str, int], smap: Dict[str, str],
                   j: str) -> int:
    """Bone whose world rotation is the *segment* frame of joint ``j``.

    MMD may put helper bones between a joint and its semantic child (腕捩
    between 腕 and ひじ, 肩C between 肩 and 腕).  The rest offset that actually
    moves the child is transformed by the frame of its own parent, so walk up
    from the child and take the first non-degenerate offset's parent.
    """
    child = CB.CANONICAL_CHILD.get(j)
    jb = smap.get(j)
    if child is None or jb not in name2i:
        return name2i.get(jb, -1)
    cb = smap.get(child)
    if cb not in name2i:
        return name2i[jb]
    cur = name2i[cb]
    for _ in range(8):
        if cur == name2i[jb] or cur < 0:
            break
        p = model["bones"][cur]["parent"]
        if p < 0:
            break
        a = model["bones"][cur]["pos"]
        b = model["bones"][p]["pos"]
        if sum((a[c] - b[c]) ** 2 for c in range(3)) > 1e-12:
            return p
        cur = p
    return name2i[jb]


def canonical_from_pmx(model: Dict, profile: Dict, vmd: Dict,
                       frames: Optional[Iterable[int]] = None,
                       solve_ik: bool = False,
                       unit_to_m: Optional[float] = None) -> Dict:
    """Read a VMD on a PMX skeleton into a canonical (world rotation) motion.

    Joint rotations are read from each joint's *segment frame* (the bone that
    moves the rest offset to the semantic child) and then corrected so that
    ``G·rest_dir`` equals the observed segment direction.  That makes the
    conversion loss-free even when the rig puts a twist/helper bone in the
    chain (R2's 腕捩/手捩 carry ~4.5 deg of non-axial rotation) and keeps a
    same-model round-trip exact to float precision.
    """
    unit = float(unit_to_m or profile.get("unit_to_m") or 0.08)
    if frames is None:
        frames = list(range(vmd["max_frame"] + 1))
    frames = [int(f) for f in frames]
    ev = VmdEvaluator(model, vmd, solve_ik=solve_ik, force_ik_off=not solve_ik,
                      apply_all_translations=False)
    name2i = {b["name"]: i for i, b in enumerate(model["bones"])}
    smap = profile["standard_map"]

    missing = [j for j in CB.ADAPTER_JOINTS if smap.get(j) not in name2i]
    idx = {j: name2i[smap[j]] for j in CB.ADAPTER_JOINTS if smap.get(j) in name2i}
    frame_bone = {j: _segment_frame(model, name2i, smap, j) for j in idx}

    world_rot, pos = {}, {}
    for j, i in idx.items():
        fidx = frame_bone[j] if frame_bone[j] >= 0 else i
        world_rot[j] = np.asarray([ev.frame(f)["world_q"][fidx] for f in frames],
                                  dtype=np.float64)
        pos[j] = np.asarray([ev.frame(f)["world_pos"][i] for f in frames],
                            dtype=np.float64)

    rest_joints = {j: [c * unit for c in model["bones"][i]["pos"]]
                   for j, i in idx.items()}
    rest_dir = CB.rest_dirs(rest_joints)
    # direction correction: G·rest_dir must equal the observed child direction
    for j in idx:
        child = CB.CANONICAL_CHILD.get(j)
        if not child or child not in idx or j not in rest_dir:
            continue
        u = tuple(rest_dir[j])
        d = pos[child] - pos[j]
        n = np.linalg.norm(d, axis=1, keepdims=True)
        good = n[:, 0] > 1e-9
        d = np.divide(d, n, out=np.zeros_like(d), where=n > 0)
        for k in range(len(frames)):
            if not good[k]:
                continue
            got = qrot(tuple(world_rot[j][k]), u)
            swing = q_between(tuple(got), tuple(d[k]))
            world_rot[j][k] = np.asarray(qmul(swing, tuple(world_rot[j][k])))

    reach = [f"toe_{s}" for s in ("l", "r") if f"toe_{s}" in idx]
    root_pos = pos["pelvis"] * unit
    toe_y = np.asarray([min(pos[j][k][1] for j in reach)
                        for k in range(len(frames))]) * unit
    ground_toe = min(model["bones"][idx[j]]["pos"][1] for j in reach) * unit

    return {
        "frames": frames,
        "world_rot": world_rot,
        "root_pos": root_pos,
        "rest_joints": rest_joints,
        "rest_dir": rest_dir,
        "rest_pelvis": np.asarray(rest_joints["pelvis"], dtype=np.float64),
        "span": CB.joint_span(rest_joints),
        "toe_y": toe_y,
        "ground_toe": float(ground_toe),
        "segment_frame": {j: model["bones"][frame_bone[j]]["name"]
                          for j in idx if frame_bone[j] >= 0},
        "height": float(profile.get("height") or 0.0) * unit,
        "unit_to_m": unit,
        "missing": missing,
        "meta": {"source": "pmx", "vmd_frames": [frames[0], frames[-1]],
                 "solve_ik": bool(solve_ik), "model": profile.get("model_key")},
    }


def canonical_from_smplx(params: Dict, rest_joints: Dict[str, Sequence[float]],
                         slow_factor: float = 1.0) -> Dict:
    """Build a canonical motion from SMPL-X parameters.

    ``params``: body_pose (F,63), global_orient (F,3), transl (F,3) in metres
    and axis-angle radians, in the GVHMR *world* frame.
    ``rest_joints``: canonical -> rest joint position (metres) from the SMPL-X
    model at zero pose with the clip's betas — the exact rest the body_pose
    rotations are relative to.
    ``slow_factor``: 1.0 normal speed, 0.25 for T1 (0.25x slow motion), 0.5 for
    T2.  Output frame k samples source frame k / slow_factor, so the exported
    VMD runs at true dance speed: 968 -> 242, 484 -> 242, 242 -> 242.
    """
    body_pose = np.asarray(params["body_pose"], dtype=np.float64)
    global_orient = np.asarray(params["global_orient"], dtype=np.float64)
    transl = np.asarray(params["transl"], dtype=np.float64)
    n_src = body_pose.shape[0]
    if transl.shape[0] != n_src or global_orient.shape[0] != n_src:
        raise ValueError("SMPL-X parameter frame mismatch")
    if slow_factor <= 0:
        raise ValueError("slow_factor must be > 0")

    n_out = int(round(n_src * slow_factor))
    src_idx = np.clip(np.round(np.arange(n_out) / slow_factor).astype(int),
                      0, n_src - 1)
    frames = list(range(n_out))

    root_q = np.asarray([q_from_axis_angle(*global_orient[k]) for k in src_idx])
    rot = {"root": root_q, "pelvis": root_q.copy()}
    local = {j: np.asarray([q_from_axis_angle(*body_pose[k, bi * 3:bi * 3 + 3])
                            for k in src_idx])
             for j, bi in CB.BODY_POSE_INDEX.items()}
    for j in CB.BODY_JOINTS:
        if j == "pelvis":
            continue
        p = CB.CANONICAL_PARENT[j]
        rot[j] = np.asarray([qnorm(qmul(rot[p][k], local[j][k]))
                             for k in range(n_out)])

    # ---- SMPL-X model frame -> canonical (MMD-like) frame ------------------
    # SMPL-X's rest frame is +X = character's left, +Y = up, +Z = FORWARD, while
    # PMX/MMD is +X = left, +Y = up, +Z = BACKWARD (verified: SMPL-X Lhip +X and
    # ankle->toe +Z; PMX 左足 +X and 足首->つま先 -Z).  The two are related by the
    # Z reflection S = diag(1,1,-1); a rotation conjugates as S R S, i.e.
    # (x,y,z,w) -> (-x,-y,z,w).  Skipping this makes every rest direction that is
    # not exactly "straight down" get a mirrored swing: on T5 it turned the arms
    # forward-in-front into backwards (fixed 2026-10-25).
    rot = {j: np.asarray([(-q[0], -q[1], q[2], q[3]) for q in v])
           for j, v in rot.items()}
    rest_joints = {j: [v[0], v[1], -v[2]] for j, v in rest_joints.items()}
    rest = {j: list(v) for j, v in rest_joints.items()}
    j0 = np.asarray(rest["pelvis"], dtype=np.float64)
    foot_y = [rest[f"toe_{s}"][1] for s in ("l", "r") if f"toe_{s}" in rest]
    if not foot_y:
        raise ValueError("rest_joints needs toe_l/toe_r")
    # standing pelvis height above the ground; the SMPL-X model frame does not
    # necessarily put the pelvis at y=0 (BodyModelSMPLX adds a constant), so use
    # the pelvis-to-lowest-foot distance, not the absolute foot height.
    h_stand = float(j0[1] - min(foot_y))
    root_pos = transl[src_idx] + j0
    root_pos = root_pos * np.array([1.0, 1.0, -1.0])
    rest_joints = rest
    rest_pelvis = np.array([j0[0], h_stand, j0[2]])
    height = float(max(rest["head"][1] - min(foot_y), 1e-6))
    # posed toe height: SMPL-X rest stance defines its own ground
    J = fk_positions(rot, rest, root_pos)
    toe_y = np.minimum(J["toe_l"][:, 1], J["toe_r"][:, 1])

    return {
        "frames": frames,
        "world_rot": {j: np.asarray(v) for j, v in rot.items()},
        "root_pos": root_pos,
        "rest_joints": rest,
        "rest_dir": CB.rest_dirs(rest),
        "rest_pelvis": rest_pelvis,
        "span": CB.joint_span(rest),
        "toe_y": toe_y,
        "ground_toe": 0.0,
        "height": height,
        "unit_to_m": 1.0,
        "missing": [],
        "meta": {"source": "smplx", "slow_factor": float(slow_factor),
                 "src_frames": int(n_src), "out_frames": int(n_out)},
    }


# ---------------------------------------------------------------- target pose

class TargetPose:
    """Rotation-only MMD pose: locals per bone, worlds in topological order.

    Mirrors ``mmd.animation.Skeleton`` (inherit rotation with ratio, parent *
    local) without IK/physics.
    """

    def __init__(self, model: Dict):
        bones = model["bones"]
        self.bones = bones
        self.n = len(bones)
        self.parent = [b["parent"] for b in bones]
        self.name2i = {b["name"]: i for i, b in enumerate(bones)}
        self.inherit = []
        for b in bones:
            if (b["flags"] & 0x0100) and "inherit_parent" in b:
                self.inherit.append((b["inherit_parent"],
                                     float(b.get("inherit_ratio", 1.0))))
            else:
                self.inherit.append(None)
        self.order = self._topo()
        self.local: List[Tuple] = []
        self.world: List[Tuple] = []
        self.reset()

    def _topo(self) -> List[int]:
        out, seen = [], [False] * self.n

        def visit(i):
            if seen[i]:
                return
            p = self.parent[i]
            if 0 <= p < self.n:
                visit(p)
            seen[i] = True
            out.append(i)

        for i in range(self.n):
            visit(i)
        return out

    def reset(self):
        self.local = [QID] * self.n
        self.world = [QID] * self.n

    def build(self):
        """Recompute world rotations from locals in deform order."""
        for i in self.order:
            p = self.parent[i]
            pw = self.world[p] if 0 <= p < self.n else QID
            ln = self.local[i]
            inh = self.inherit[i]
            if inh is not None:
                src, ratio = inh
                if 0 <= src < self.n:
                    ln = qmul(qslerp(QID, self.local[src], ratio), ln)
            self.world[i] = qmul(pw, ln)


# ---------------------------------------------------------------- adapter

def adapt(motion: Dict, model: Dict, profile: Dict,
          twist_split: bool = False,
          ground_calibrate: bool = True,
          ground_percentile: float = 5.0,
          stage_yaw_deg: float = 0.0,
          max_frames: Optional[int] = None) -> Dict:
    """Map a canonical motion onto ``model``; returns VMD tracks + diagnostics."""
    unit = float(profile.get("unit_to_m") or 0.08)
    smap = profile["standard_map"]
    pose = TargetPose(model)
    name2i = pose.name2i

    tgt_bone: Dict[str, str] = {}
    missing: List[str] = []
    for j in CB.ADAPTER_JOINTS:
        bn = smap.get(j)
        if bn is None or bn not in name2i:
            missing.append(j)
        else:
            tgt_bone[j] = bn
    if missing:
        raise ValueError(f"target is missing canonical bones: {missing}")

    tgt_idx = {j: name2i[bn] for j, bn in tgt_bone.items()}
    joint_at = {i: j for j, i in tgt_idx.items()}
    rest_m = {j: [c * unit for c in model["bones"][i]["pos"]]
              for j, i in tgt_idx.items()}
    tgt_dir = CB.rest_dirs(rest_m)
    tgt_span = CB.joint_span(rest_m)
    src_span = motion.get("span")
    scale = (tgt_span / src_span) if (tgt_span and src_span) else 1.0

    # ---- rest compensation D (+ optional whole-character yaw) -------------
    # The yaw is a *left* multiplication of every world rotation and of the root
    # delta: that is a rigid turn of the whole character (locals unchanged).
    # Conjugating instead would re-twist every bone (M1 bug, fixed 2026-10-25).
    yaw = yaw_quat(stage_yaw_deg) if abs(stage_yaw_deg) > 1e-9 else QID
    D: Dict[str, Tuple] = {}
    rest_report: Dict[str, Dict] = {}
    for j in CB.ADAPTER_JOINTS:
        if j in ("root", "pelvis"):
            D[j] = QID
            rest_report[j] = {"src_dir": None, "tgt_dir": None, "deg": 0.0,
                              "note": "body frame, identity"}
            continue
        u = motion["rest_dir"].get(j)
        v = tgt_dir.get(j)
        if j in NO_SWING:
            d = QID
        else:
            d = q_between(u, v) if (u is not None and v is not None) else QID
        D[j] = d
        rest_report[j] = {
            "src_dir": [round(float(c), 5) for c in u] if u else None,
            "tgt_dir": [round(float(c), 5) for c in v] if v else None,
            "deg": round(q_angle_deg(d), 3),
        }

    # ---- writable set (safe_mode) + main twist bones ----------------------
    writable_vals = set(profile["writable_bones"].values())
    written = [tgt_bone[j] for j in CB.ADAPTER_JOINTS if tgt_bone[j] in writable_vals]
    twist_bones: Dict[str, str] = {}
    if twist_split:
        for seg in ("forearm_l", "forearm_r", "wrist_l", "wrist_r"):
            main = ((profile.get("twist") or {}).get(seg) or {}).get("main")
            if main and main in name2i:
                twist_bones[seg] = main
    written = [n for n in written if n not in twist_bones.values()]
    written += sorted(twist_bones.values())
    written = sorted(set(written), key=lambda n: name2i[n])
    written_idx = {name2i[n]: n for n in written}

    frames = list(motion["frames"][:max_frames]) if max_frames else list(motion["frames"])
    nf = len(frames)

    c_rest = np.asarray(rest_m["root"], dtype=np.float64)
    p_rest = np.asarray(rest_m["pelvis"], dtype=np.float64)
    tgt_rest_pelvis = p_rest.copy()
    src_rest_pelvis = np.asarray(motion["rest_pelvis"], dtype=np.float64)
    root_i = name2i[tgt_bone["root"]]

    root_t: List[Tuple[float, float, float]] = []
    local_tracks: Dict[int, List[Tuple]] = {i: [] for i in written_idx}
    toe_rest = {s: rest_m[f"toe_{s}"][1] for s in ("l", "r") if f"toe_{s}" in rest_m}
    toe_idx = [name2i[tgt_bone[f"toe_{s}"]] for s in ("l", "r")
               if f"toe_{s}" in tgt_bone]

    for fi in range(nf):
        pose.reset()
        local = pose.local
        # 1) desired world rotations
        G = {}
        for j in CB.ADAPTER_JOINTS:
            q = motion["world_rot"][j][fi]
            d = D[j]
            G[j] = qmul(yaw, qmul(qmul(d, q), q_conj(d)))
        # 2) walk the real PMX chain
        for i in pose.order:
            p = pose.parent[i]
            pw = pose.world[p] if 0 <= p < pose.n else QID
            inh = pose.inherit[i]
            if inh is not None and 0 <= inh[0] < pose.n:
                pre = qmul(pw, qslerp(QID, local[inh[0]], inh[1]))
            else:
                pre = pw
            j = joint_at.get(i)
            if j is not None:
                loc = qnorm(qmul(qinv(pre), G[j]))
                if twist_split:
                    child = twist_target = None
                    if j in ("upperarm_l", "upperarm_r"):
                        child = "forearm_l" if j.endswith("_l") else "forearm_r"
                        twist_target = twist_bones.get(child)
                    elif j in ("forearm_l", "forearm_r"):
                        child = "hand_l" if j.endswith("_l") else "hand_r"
                        twist_target = twist_bones.get(
                            "wrist_l" if j.endswith("_l") else "wrist_r")
                    if child and twist_target:
                        axis = vnorm(tuple(np.asarray(rest_m[child])
                                           - np.asarray(rest_m[j])))
                        swing, tw = q_twist_swing(loc, axis)
                        loc = swing
                        local[name2i[twist_target]] = tw
                local[i] = loc
            pose.world[i] = qmul(pre, local[i])
        # 3) root translation: delta from source rest stance, scaled, no absolutes
        rq = G["root"]
        anchor = c_rest + np.asarray(qrot(rq, tuple(p_rest - c_rest)))
        # metres from the source -> target units; the anchor is already units
        delta = motion["root_pos"][fi] - src_rest_pelvis
        if yaw != QID:
            delta = np.asarray(qrot(yaw, tuple(delta)))
        target_pelvis = tgt_rest_pelvis + scale * delta / unit
        t = target_pelvis - anchor
        root_t.append((float(t[0]), float(t[1]), float(t[2])))
        for i, lst in local_tracks.items():
            lst.append(local[i])

    # ---- ground calibration: one global Y shift, pose preserving ----------
    # Preserve how far the source's lowest toe sits above *its* ground:
    #   target_ref = target_rest_toe + scale * (p5(src_toe) - src_ground_toe)
    # so a same-model round-trip shifts by exactly 0.
    ground_shift = 0.0
    toe_min = None
    if toe_idx:
        sk = Skeleton(model)
        ys = []
        for fi in range(nf):
            sampled = {}
            for i in written_idx:
                sampled[i] = (root_t[fi] if i == root_i else (0.0, 0.0, 0.0),
                              local_tracks[i][fi])
            sk.pose(sampled, ik_enabled=set())
            ys.append(min(sk.world(i)[1][1] for i in toe_idx))
        ys = np.asarray(ys)
        toe_min = float(ys.min())
        src_toe = motion.get("toe_y")
        if ground_calibrate and toe_rest and src_toe is not None:
            ref = min(toe_rest.values()) / unit
            gap_src = max(0.0, float(np.percentile(np.asarray(src_toe),
                                                   ground_percentile))
                          - float(motion.get("ground_toe") or 0.0))
            target_ref = ref + scale * gap_src / unit
            ground_shift = target_ref - float(np.percentile(ys, ground_percentile))
            root_t = [(x, y + ground_shift, z) for x, y, z in root_t]

    # ---- assemble tracks --------------------------------------------------
    root_name = tgt_bone["root"]
    bone_frames: Dict[str, List[Dict]] = {}
    for n in written:
        i = name2i[n]
        rows = []
        for fi in range(nf):
            pos = root_t[fi] if n == root_name else (0.0, 0.0, 0.0)
            rows.append({"frame": int(frames[fi]), "pos": pos,
                         "rot": tuple(float(c) for c in local_tracks[i][fi])})
        bone_frames[n] = rows

    return {
        "bone_frames": bone_frames,
        "ik_frames": _ik_off(model),
        "frames": [int(f) for f in frames],
        "bones": written,
        "twist_bones": twist_bones,
        "scale": float(scale),
        "ground_shift_units": float(ground_shift),
        "toe_min_units": toe_min,
        "target_rest_toe_units": (min(toe_rest.values()) / unit if toe_rest else None),
        "rest_compensation_deg": rest_report,
        "meta": {
            "unit_to_m": unit,
            "frames": nf,
            "frame_range": [int(frames[0]), int(frames[-1])] if nf else None,
            "written_bones": len(written),
            "twist_split": bool(twist_split),
            "stage_yaw_deg": float(stage_yaw_deg),
            "source": motion.get("meta"),
            "src_span_m": src_span,
            "tgt_span_m": tgt_span,
        },
    }


def _ik_off(model: Dict) -> List[Dict]:
    names = [b["name"] for b in model["bones"] if b["flags"] & 0x0020 and "ik" in b]
    return [{"frame": 0, "show": True, "ik": {n: False for n in names}}] if names else []
