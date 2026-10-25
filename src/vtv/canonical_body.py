# -*- coding: utf-8 -*-
"""Canonical 23-joint body: SMPL-X semantics, PMX target names.

The canonical layer is the identity-independent middle language of v2 §2.3.
It carries **world rotations** (not PMX local rotations) plus the root position,
so an adapter can re-express one body on another skeleton without caring how
either model distributes twist/helpers.

SMPL-X facts used here (verified 2026-10-25 against ``SMPLX_NEUTRAL.npz``):

* 22 joints, parents ``[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]``;
* ``body_pose`` is 21 axis-angle triples for joints 1..21 (joint 0 = global_orient);
* the local rotations are expressed in a **world aligned** frame (pose 0 = rest),
  the same convention MMD uses, so no per-bone frame conversion is needed;
* rest frame: **+X = character's left, +Y = up, +Z = forward**;
  forward is +Z (ankle->foot joint has +Z), left is +X (left hip has +X).

MMD/PMX rest frame: +X = character's right, +Y = up, +Z = backward (-Z forward).
The source world frame used by GVHMR is already MMD-like: world +X = right,
+Y = up, +Z = backward (measured: the subject faces -Z with the left wrist at
-X).  ``Q`` (source frame -> target frame) is therefore the identity; only a
presentation yaw would change it.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

# ---- SMPL-X ---------------------------------------------------------------
SMPLX_JOINTS = [
    "pelvis", "Lhip", "Rhip", "spine1", "Lknee", "Rknee", "spine2", "Lankle",
    "Rankle", "spine3", "Lfoot", "Rfoot", "neck", "Lcollar", "Rcollar", "head",
    "Lshoulder", "Rshoulder", "Lelbow", "Relbow", "Lwrist", "Rwrist",
]
SMPLX_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
                 16, 17, 18, 19]

# canonical joint -> SMPL-X joint index
SMPLX_INDEX = {
    "pelvis": 0,
    "thigh_l": 1, "thigh_r": 2,
    "spine": 3,
    "shin_l": 4, "shin_r": 5,
    "spine1": 6,
    "foot_l": 7, "foot_r": 8,
    "chest": 9,
    "toe_l": 10, "toe_r": 11,
    "neck": 12,
    "clavicle_l": 13, "clavicle_r": 14,
    "head": 15,
    "upperarm_l": 16, "upperarm_r": 17,
    "forearm_l": 18, "forearm_r": 19,
    "hand_l": 20, "hand_r": 21,
}

# body_pose has 21 triples, one per joint 1..21
BODY_POSE_INDEX = {j: i - 1 for j, i in SMPLX_INDEX.items() if i > 0}

SMPLX_PARENT = {SMPLX_JOINTS[i]: (SMPLX_JOINTS[p] if p >= 0 else None)
                for i, p in enumerate(SMPLX_PARENTS)}

# ---- canonical tree -------------------------------------------------------
# 23 joints: root (VMD root translation/rotation) + 22 body joints.
BODY_JOINTS = [
    "pelvis", "spine", "spine1", "chest", "neck", "head",
    "clavicle_l", "upperarm_l", "forearm_l", "hand_l",
    "clavicle_r", "upperarm_r", "forearm_r", "hand_r",
    "thigh_l", "shin_l", "foot_l", "toe_l",
    "thigh_r", "shin_r", "foot_r", "toe_r",
]
ADAPTER_JOINTS = ["root"] + BODY_JOINTS

CANONICAL_PARENT = {
    "root": None, "pelvis": "root",
    "spine": "pelvis", "spine1": "spine", "chest": "spine1",
    "neck": "chest", "head": "neck",
    "clavicle_l": "chest", "upperarm_l": "clavicle_l",
    "forearm_l": "upperarm_l", "hand_l": "forearm_l",
    "clavicle_r": "chest", "upperarm_r": "clavicle_r",
    "forearm_r": "upperarm_r", "hand_r": "forearm_r",
    "thigh_l": "pelvis", "shin_l": "thigh_l", "foot_l": "shin_l", "toe_l": "foot_l",
    "thigh_r": "pelvis", "shin_r": "thigh_r", "foot_r": "shin_r", "toe_r": "foot_r",
}

# The child that defines a joint's *bone direction* (rest-pose compensation).
# None = no meaningful segment: pelvis/root keep the identity frame, and true
# leaves (head/hand/toe) inherit their parent's direction.
CANONICAL_CHILD = {
    "root": None, "pelvis": None,
    "spine": "spine1", "spine1": "chest", "chest": "neck", "neck": "head",
    "head": None,
    "clavicle_l": "upperarm_l", "upperarm_l": "forearm_l",
    "forearm_l": "hand_l", "hand_l": None,
    "clavicle_r": "upperarm_r", "upperarm_r": "forearm_r",
    "forearm_r": "hand_r", "hand_r": None,
    "thigh_l": "shin_l", "shin_l": "foot_l", "foot_l": "toe_l", "toe_l": None,
    "thigh_r": "shin_r", "shin_r": "foot_r", "foot_r": "toe_r", "toe_r": None,
}


def rest_dirs(rest_joints: Dict[str, Sequence[float]]) -> Dict[str, List[float]]:
    """Unit bone direction per joint from rest joint positions (any unit).

    ``rest_joints`` is keyed by canonical name.  Leaves inherit the parent's
    direction; ``root``/``pelvis`` get no entry (callers use the identity).
    """
    out: Dict[str, List[float]] = {}

    def unit(a, b):
        v = [b[c] - a[c] for c in range(3)]
        n = sum(c * c for c in v) ** 0.5
        return [c / n for c in v] if n > 1e-9 else None

    def resolve(j: str) -> Optional[List[float]]:
        if j in out:
            return out[j]
        child = CANONICAL_CHILD.get(j)
        d = None
        if child and child in rest_joints and j in rest_joints:
            d = unit(rest_joints[j], rest_joints[child])
        if d is None:
            parent = CANONICAL_PARENT.get(j)
            d = resolve(parent) if parent else None
        if d is not None:
            out[j] = d
        return d

    for j in ADAPTER_JOINTS:
        if j not in rest_joints:
            continue
        resolve(j)
    return out


def joint_span(rest_joints: Dict[str, Sequence[float]]) -> Optional[float]:
    """Pelvis->ankle distance (mean of both sides), the scale reference.

    Leg length is the quantity that decides how far the pelvis sits above the
    ground for a given pose, so it is the right scale for root translation.
    """
    if "pelvis" not in rest_joints:
        return None
    ds = []
    for side in ("l", "r"):
        k = f"foot_{side}"
        if k in rest_joints:
            p, q = rest_joints["pelvis"], rest_joints[k]
            ds.append(sum((q[c] - p[c]) ** 2 for c in range(3)) ** 0.5)
    return sum(ds) / len(ds) if ds else None


def check_mapping() -> List[str]:
    """Self-check of the canonical tables (import-time cheap, used by tests)."""
    errs = []
    for j in BODY_JOINTS:
        if j not in SMPLX_INDEX:
            errs.append(f"{j}: not in SMPLX_INDEX")
        if j not in CANONICAL_PARENT:
            errs.append(f"{j}: no canonical parent")
        if j not in CANONICAL_CHILD:
            errs.append(f"{j}: no CANONICAL_CHILD entry")
    if len(SMPLX_JOINTS) != 22 or len(SMPLX_PARENTS) != 22:
        errs.append("SMPL-X table size")
    for i, p in enumerate(SMPLX_PARENTS):
        if p >= i:
            errs.append(f"SMPL-X parent order at {i}")
    return errs
