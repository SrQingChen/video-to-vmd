# -*- coding: utf-8 -*-
"""PMX -> target profile: the deterministic description the adapter and the
compatibility audit consume.

A profile is intentionally a plain JSON-able dict: bone table, canonical
standard map, helper chains (P/C, D-chain, twist), IK, morphs, scale.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import unicodedata
from typing import Dict, Iterable, List, Optional

from .pmx import parse_pmx
from .. import canonical as C

FLAG_BITS = {
    0x0001: "tail_is_bone",
    0x0002: "rotatable",
    0x0004: "movable",
    0x0008: "visible",
    0x0010: "enabled",
    0x0020: "ik",
    0x0080: "local_inherit",
    0x0100: "inherit_rotation",
    0x0200: "inherit_translation",
    0x0400: "fixed_axis",
    0x0800: "local_axes",
    0x1000: "after_physics",
    0x2000: "external_parent",
}
MORPH_TYPES = {
    0: "group", 1: "vertex", 2: "bone", 3: "uv", 4: "add_uv1",
    5: "add_uv2", 6: "add_uv3", 7: "add_uv4", 8: "material",
    9: "flip", 10: "impulse",
}


def norm_name(name: str) -> str:
    return unicodedata.normalize("NFKC", name).replace(" ", "").strip()


def decode_flags(flags: int) -> List[str]:
    return [label for bit, label in FLAG_BITS.items() if flags & bit]


def _dist(a, b) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _bone_dict(b: Dict, by_index: Dict[int, Dict]) -> Dict:
    out = {
        "name": b["name"],
        "name_en": b.get("name_en", ""),
        "index": None,  # filled by caller
        "parent": b["parent"],
        "layer": b["layer"],
        "flags": b["flags"],
        "flags_decoded": decode_flags(b["flags"]),
        "pos": [round(float(v), 6) for v in b["pos"]],
        "movable": bool(b["flags"] & 0x0004),
        "rotatable": bool(b["flags"] & 0x0002),
        "after_physics": bool(b["flags"] & 0x1000),
        "tail_bone": b.get("tail_bone"),
        "tail_offset": b.get("tail_offset"),
    }
    if "inherit_parent" in b:
        out["inherit"] = {
            "parent": b["inherit_parent"],
            "parent_name": by_index.get(b["inherit_parent"], {}).get("name"),
            "ratio": b["inherit_ratio"],
        }
    if "fixed_axis" in b:
        out["fixed_axis"] = [round(float(v), 6) for v in b["fixed_axis"]]
    if "local_x" in b:
        out["local_x"] = [round(float(v), 6) for v in b["local_x"]]
        out["local_z"] = [round(float(v), 6) for v in b["local_z"]]
    if "ik" in b:
        ik = b["ik"]
        out["ik"] = {
            "target": ik["target"],
            "target_name": by_index.get(ik["target"], {}).get("name"),
            "loop": ik["loop"],
            "limit_angle": ik["limit_angle"],
            "links": [{
                "bone": link["bone"],
                "bone_name": by_index.get(link["bone"], {}).get("name"),
                "min": link.get("min"),
                "max": link.get("max"),
            } for link in ik["links"]],
        }
    return out


def _bbox(vertices) -> Dict[str, List[float]]:
    if not vertices:
        return {"min": [0, 0, 0], "max": [0, 0, 0], "size": [0, 0, 0]}
    lo = [min(v["pos"][i] for v in vertices) for i in range(3)]
    hi = [max(v["pos"][i] for v in vertices) for i in range(3)]
    return {
        "min": [round(v, 6) for v in lo],
        "max": [round(v, 6) for v in hi],
        "size": [round(hi[i] - lo[i], 6) for i in range(3)],
    }


def build_profile(pmx_path: str, model_key: Optional[str] = None) -> Dict:
    model = parse_pmx(pmx_path)
    bones_raw = model["bones"]
    by_index = {i: b for i, b in enumerate(bones_raw)}
    bones = []
    for i, b in enumerate(bones_raw):
        d = _bone_dict(b, by_index)
        d["index"] = i
        d["parent_name"] = by_index.get(b["parent"], {}).get("name")
        bones.append(d)

    index_by_name = {b["name"]: b["index"] for b in bones}
    index_by_norm: Dict[str, int] = {}
    for b in bones:
        index_by_norm.setdefault(norm_name(b["name"]), b["index"])

    def find(candidates: Iterable[str]) -> Optional[int]:
        for cand in candidates:
            if cand in index_by_name:
                return index_by_name[cand]
            n = norm_name(cand)
            if n in index_by_norm:
                return index_by_norm[n]
        return None

    # ---- canonical map ---------------------------------------------------
    standard_map: Dict[str, str] = {}
    standard_index: Dict[str, int] = {}
    missing: List[str] = []
    for joint in C.STANDARD_JOINTS:
        i = find(C.pmx_candidates(joint))
        if i is None:
            missing.append(joint)
        else:
            standard_index[joint] = i
            standard_map[joint] = bones[i]["name"]

    finger_map: Dict[str, str] = {}
    finger_missing: List[str] = []
    for joint in C.iter_finger_joints():
        i = find(C.pmx_candidates(joint))
        if i is None:
            finger_missing.append(joint)
        else:
            finger_map[joint] = bones[i]["name"]

    # ---- IK --------------------------------------------------------------
    ik_info = {}
    for side, kinds in C.IK_CANDIDATES.items():
        ik_info[side] = {}
        for kind, cands in kinds.items():
            i = find(cands)
            if i is None:
                continue
            b = bones[i]
            ik_info[side][kind] = {"bone": b["name"], **(b.get("ik") or {})}

    # ---- helper chains ---------------------------------------------------
    def bone_of(joint):
        i = standard_index.get(joint)
        return bones[i] if i is not None else None

    twists = {}
    for side in ("l", "r"):
        main = bone_of(f"forearm_twist_{side}")
        helper = bone_of(f"forearm_twist_helper_{side}")
        if main or helper:
            twists[f"forearm_{side}"] = {
                "main": main["name"] if main else None,
                "main_fixed_axis": main.get("fixed_axis") if main else None,
                "helper": helper["name"] if helper else None,
                "helper_inherit": helper.get("inherit") if helper else None,
            }
        main = bone_of(f"wrist_twist_{side}")
        helper = bone_of(f"wrist_twist_helper_{side}")
        if main or helper:
            twists[f"wrist_{side}"] = {
                "main": main["name"] if main else None,
                "main_fixed_axis": main.get("fixed_axis") if main else None,
                "helper": helper["name"] if helper else None,
                "helper_inherit": helper.get("inherit") if helper else None,
            }

    pc = {}
    for side in ("l", "r"):
        pc[side] = {
            "p": (bone_of(f"clavicle_p_{side}") or {}).get("name"),
            "shoulder": (bone_of(f"clavicle_{side}") or {}).get("name"),
            "c": (bone_of(f"clavicle_c_{side}") or {}).get("name"),
            "c_inherit": (bone_of(f"clavicle_c_{side}") or {}).get("inherit"),
            "upperarm_parent": (
                bones[standard_index[f"upperarm_{side}"]]["parent_name"]
                if f"upperarm_{side}" in standard_index else None),
        }

    d_chain = {}
    for side in ("l", "r"):
        chain = [bone_of(f"thigh_d_{side}"), bone_of(f"shin_d_{side}"),
                 bone_of(f"foot_d_{side}")]
        d_chain[side] = [{
            "name": b["name"],
            "inherit": b.get("inherit"),
            "after_physics": b.get("after_physics"),
        } for b in chain if b]

    # ---- root chain ------------------------------------------------------
    root_chain = []
    for name in C.ROOT_CHAIN:
        i = find([name])
        if i is not None:
            root_chain.append(bones[i]["name"])

    # ---- geometry / scale ------------------------------------------------
    bbox = _bbox(model["vertices"])
    height = bbox["size"][1]
    lengths = {}
    for joint in ("upperarm_l", "forearm_l", "hand_l", "thigh_l", "shin_l",
                  "foot_l", "upperarm_r", "forearm_r", "hand_r", "thigh_r",
                  "shin_r", "foot_r"):
        if joint not in standard_index:
            continue
        child = {
            "upperarm": "forearm", "forearm": "hand", "hand": None,
            "thigh": "shin", "shin": "foot", "foot": "toe",
        }[joint.rsplit("_", 1)[0]]
        if child is None:
            continue
        cjoint = f"{child}_{joint.rsplit('_', 1)[1]}"
        if cjoint in standard_index:
            lengths[joint] = round(_dist(bones[standard_index[joint]]["pos"],
                                        bones[standard_index[cjoint]]["pos"]), 6)
    if "upperarm_l" in standard_index and "upperarm_r" in standard_index:
        shoulder_width = _dist(bones[standard_index["upperarm_l"]]["pos"],
                               bones[standard_index["upperarm_r"]]["pos"])
    else:
        shoulder_width = None
    if "thigh_l" in standard_index and "thigh_r" in standard_index:
        hip_width = _dist(bones[standard_index["thigh_l"]]["pos"],
                          bones[standard_index["thigh_r"]]["pos"])
    else:
        hip_width = None

    # ---- writable whitelist (safe mode seed) -----------------------------
    writable = {joint: standard_map[joint] for joint in standard_map
                if joint not in (
                    "all_parent", "groove", "waist", "clavicle_p_l", "clavicle_p_r",
                    "clavicle_c_l", "clavicle_c_r",
                    "forearm_twist_l", "forearm_twist_helper_l",
                    "forearm_twist_r", "forearm_twist_helper_r",
                    "wrist_twist_l", "wrist_twist_helper_l",
                    "wrist_twist_r", "wrist_twist_helper_r",
                    "thigh_d_l", "shin_d_l", "foot_d_l",
                    "thigh_d_r", "shin_d_r", "foot_d_r",
                )}
    writable.update(finger_map)

    physics_bones = [b["name"] for b in bones if b["flags"] & 0x1000]
    morphs = [{
        "name": mo["name"],
        "name_en": mo.get("name_en", ""),
        "panel": mo["panel"],
        "type": mo["type"],
        "type_name": MORPH_TYPES.get(mo["type"], str(mo["type"])),
        "offset_count": len(mo["offsets"]),
    } for mo in model["morphs"]]

    with open(pmx_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()

    return {
        "source": os.path.abspath(pmx_path),
        "model_key": model_key or os.path.splitext(os.path.basename(pmx_path))[0],
        "pmx_sha256": digest,
        "pmx": {
            "name": model["name"],
            "name_en": model["name_en"],
            "version": model["version"],
            "encoding": model["encoding"],
            "vertex_count": len(model["vertices"]),
            "material_count": len(model["materials"]),
            "bone_count": len(bones),
            "morph_count": len(morphs),
            "rigid_body_count": len(model["rigid_bodies"]),
            "joint_count": len(model["joints"]),
        },
        "bbox": bbox,
        "height": round(height, 6),
        "unit_to_m": 0.08,
        "root_chain": root_chain,
        "standard_map": standard_map,
        "standard_missing": missing,
        "finger_map": finger_map,
        "finger_missing": finger_missing,
        "twist": twists,
        "pc": pc,
        "d_chain": d_chain,
        "ik": ik_info,
        "lengths": lengths,
        "shoulder_width": None if shoulder_width is None else round(shoulder_width, 6),
        "hip_width": None if hip_width is None else round(hip_width, 6),
        "writable_bones": writable,
        "physics_bone_count": len(physics_bones),
        "physics_bones_sample": physics_bones[:40],
        "morphs": morphs,
        "bones": bones,
    }


def save_profile(profile: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=1)


def load_profile(path: str) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
