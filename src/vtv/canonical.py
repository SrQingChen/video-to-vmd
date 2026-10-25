# -*- coding: utf-8 -*-
"""Canonical body definition shared by profile / adapter / metrics.

The canonical layer is identity-independent: it is *not* SMPL-X and it is
*not* the PMX hierarchy.  It is a small semantic skeleton whose bone lengths
are instantiated from the target model (see ``profile.py``).

Naming: ``<joint>_<side>`` with side in ``l``/``r``.  The PMX/MMD names are
resolved per model; ``STANDARD_JOINTS`` lists the canonical keys the M0
audit/profile/adapters understand.
"""
from __future__ import annotations

# MMD root / torso chain (all three test models share this chain).
ROOT_CHAIN = [
    "全ての親", "操作中心", "センター", "センター2",
    "グルーブ", "グルーブ2", "腰",
]

# Canonical body joints (no fingers).  Order is stable for reports.
STANDARD_JOINTS = [
    "root", "all_parent", "groove", "waist",
    "pelvis", "spine", "spine1", "chest", "neck", "head",
    "clavicle_p_l", "clavicle_l", "clavicle_c_l",
    "upperarm_l", "forearm_l", "hand_l",
    "clavicle_p_r", "clavicle_r", "clavicle_c_r",
    "upperarm_r", "forearm_r", "hand_r",
    "thigh_l", "shin_l", "foot_l", "toe_l",
    "thigh_r", "shin_r", "foot_r", "toe_r",
    "forearm_twist_l", "forearm_twist_helper_l",
    "wrist_twist_l", "wrist_twist_helper_l",
    "forearm_twist_r", "forearm_twist_helper_r",
    "wrist_twist_r", "wrist_twist_helper_r",
    "thigh_d_l", "shin_d_l", "foot_d_l",
    "thigh_d_r", "shin_d_r", "foot_d_r",
]

# Canonical fingers: thumb has 0/1/2 in MMD, other fingers 1/2/3.
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
FINGER_SEGMENTS = {
    "thumb": ("0", "1", "2"),
    "index": ("1", "2", "3"),
    "middle": ("1", "2", "3"),
    "ring": ("1", "2", "3"),
    "pinky": ("1", "2", "3"),
}
FINGER_JP = {
    "thumb": "親指",
    "index": "人指",
    "middle": "中指",
    "ring": "薬指",
    "pinky": "小指",
}
SIDE_JP = {"l": "左", "r": "右"}

# Direct canonical -> PMX name candidates (first hit wins).  Candidates cover
# half/full-width digits and the common "IK"/"ＩＫ" spellings.
_PMX_CANDIDATES = {
    "root": ["センター"],
    "all_parent": ["全ての親"],
    "groove": ["グルーブ"],
    "waist": ["腰"],
    "pelvis": ["下半身"],
    "spine": ["上半身"],
    "spine1": ["上半身1"],
    "chest": ["上半身2", "上半身"],
    "neck": ["首"],
    "head": ["頭"],
}
for _side in ("l", "r"):
    _s = SIDE_JP[_side]
    _PMX_CANDIDATES.update({
        f"clavicle_p_{_side}": [f"{_s}肩P"],
        f"clavicle_{_side}": [f"{_s}肩"],
        f"clavicle_c_{_side}": [f"{_s}肩C"],
        f"upperarm_{_side}": [f"{_s}腕"],
        f"forearm_{_side}": [f"{_s}ひじ"],
        f"hand_{_side}": [f"{_s}手首"],
        f"thigh_{_side}": [f"{_s}足"],
        f"shin_{_side}": [f"{_s}ひざ"],
        f"foot_{_side}": [f"{_s}足首"],
        f"toe_{_side}": [f"{_s}つま先"],
        f"forearm_twist_{_side}": [f"{_s}腕捩"],
        f"forearm_twist_helper_{_side}": [f"{_s}腕捩1"],
        f"wrist_twist_{_side}": [f"{_s}手捩"],
        f"wrist_twist_helper_{_side}": [f"{_s}手捩1"],
        f"thigh_d_{_side}": [f"{_s}足D"],
        f"shin_d_{_side}": [f"{_s}ひざD"],
        f"foot_d_{_side}": [f"{_s}足首D"],
    })

# IK bones (first entry is the leg IK, second the toe IK).
IK_CANDIDATES = {
    "l": {"leg": ["左足ＩＫ", "左足IK"], "toe": ["左つま先ＩＫ", "左つま先IK"]},
    "r": {"leg": ["右足ＩＫ", "右足IK"], "toe": ["右つま先ＩＫ", "右つま先IK"]},
}


def finger_candidates(side: str, finger: str, segment: str):
    s = SIDE_JP[side]
    jp = FINGER_JP[finger]
    fw = {"0": "０", "1": "１", "2": "２", "3": "３"}
    return [f"{s}{jp}{fw[segment]}", f"{s}{jp}{segment}"]


def pmx_candidates(joint: str):
    """Return PMX name candidates for a canonical joint."""
    if joint in _PMX_CANDIDATES:
        return list(_PMX_CANDIDATES[joint])
    if "_" in joint:
        base, side = joint.rsplit("_", 1)
        if side in SIDE_JP:
            for finger in FINGERS:
                if base.startswith(finger):
                    seg = base[len(finger):]
                    if seg in FINGER_SEGMENTS[finger]:
                        return finger_candidates(side, finger, seg)
    return []


def iter_finger_joints():
    for side in ("l", "r"):
        for finger in FINGERS:
            for seg in FINGER_SEGMENTS[finger]:
                yield f"{finger}{seg}_{side}"
