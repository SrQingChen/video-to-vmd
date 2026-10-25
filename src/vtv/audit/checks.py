# -*- coding: utf-8 -*-
"""Compatibility audit checks (§2.8.3 / §4.3.0).

Each check returns plain JSON data; no rendering is required.  The report
distinguishes "playback environment is wrong" from "VMD/retarget is wrong":
the PMX<->Blender rest/topology comparison and the same-model R2/祀 FK check
are the two anchors.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Set

from .. import canonical as C
from ..mmd.evaluate import VmdEvaluator, build_name_map, ik_bones
from ..mmd.writer import identity_vmd, write_vmd
from ..mmd.animation import Skeleton

STD_BODY = [
    "root", "pelvis", "spine", "spine1", "chest", "neck", "head",
    "clavicle_l", "upperarm_l", "forearm_l", "hand_l",
    "clavicle_r", "upperarm_r", "forearm_r", "hand_r",
    "thigh_l", "shin_l", "foot_l", "toe_l",
    "thigh_r", "shin_r", "foot_r", "toe_r",
]


def _qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _qconj(q):
    return (-q[0], -q[1], -q[2], q[3])


def q_angle(a, b) -> float:
    d = abs(sum(a[i] * b[i] for i in range(4)))
    d = max(-1.0, min(1.0, d))
    return 2.0 * math.acos(d)


def axis_quat(axis: str, deg: float):
    h = math.radians(deg) * 0.5
    s = math.sin(h)
    return {"x": (s, 0.0, 0.0, math.cos(h)),
            "y": (0.0, s, 0.0, math.cos(h)),
            "z": (0.0, 0.0, s, math.cos(h))}[axis]


# ---------------------------------------------------------------------------
# PMX <-> Blender armature
# ---------------------------------------------------------------------------
def compare_pmx_blender(profile: Dict, armature: Dict, scale_to_mmd: float = 12.5) -> Dict:
    pmx = {b["name"]: b for b in profile["bones"]}
    bl = {b["name"]: b for b in armature["bones"]}
    common = sorted(set(pmx) & set(bl))
    pmx_only = sorted(set(pmx) - set(bl))
    bl_only = sorted(set(bl) - set(pmx))

    parent_mismatch = []
    rest_err = []
    dir_dot = []
    for name in common:
        p, b = pmx[name], bl[name]
        if (p.get("parent_name") or None) != (b.get("parent") or None):
            parent_mismatch.append({"bone": name, "pmx": p.get("parent_name"),
                                    "blender": b.get("parent")})
        # MMD Tools axis swap: (x, y, z)_mmd = (x, z, y)_blender * 1/0.08
        h = b["head"]
        conv = [h[0] * scale_to_mmd, h[2] * scale_to_mmd, h[1] * scale_to_mmd]
        rest_err.append((math.dist(p["pos"], conv), name))
        # bone direction, same swap
        if p.get("tail_bone") is not None and 0 <= p["tail_bone"] < len(profile["bones"]):
            tail = profile["bones"][p["tail_bone"]]["pos"]
            tv = [tail[i] - p["pos"][i] for i in range(3)]
        elif p.get("tail_offset"):
            tv = p["tail_offset"]
        else:
            continue
        bv = [b["tail"][i] - b["head"][i] for i in range(3)]
        bv = [bv[0], bv[2], bv[1]]
        nl = math.sqrt(sum(x * x for x in tv))
        nv = math.sqrt(sum(x * x for x in bv))
        if nl > 1e-9 and nv > 1e-9:
            dir_dot.append((sum(tv[i] * bv[i] for i in range(3)) / (nl * nv), name))
    rest_err.sort()
    dir_dot.sort()

    return {
        "pmx_bones": len(pmx),
        "blender_bones": len(bl),
        "common_bones": len(common),
        "pmx_only": pmx_only,
        "blender_only": bl_only,
        "blender_only_is_mmd_helper": all(
            n.startswith("_dummy_") or n.startswith("_shadow_") for n in bl_only),
        "parent_mismatch": parent_mismatch,
        "rest_median_err": rest_err[len(rest_err) // 2][0] if rest_err else None,
        "rest_max_err": rest_err[-1][0] if rest_err else None,
        "rest_max_bone": rest_err[-1][1] if rest_err else None,
        "direction_min_dot": dir_dot[0][0] if dir_dot else None,
        "direction_min_bone": dir_dot[0][1] if dir_dot else None,
        "axis_convention": "(x,y,z)_mmd = (x,z,y)_blender * 12.5",
        "verdict": ("faithful_import" if not parent_mismatch
                    and (not rest_err or rest_err[-1][0] < 1e-3)
                    and (not dir_dot or dir_dot[0][0] > 0.999) else "mismatch"),
    }


# ---------------------------------------------------------------------------
# VMD coverage / semantics
# ---------------------------------------------------------------------------
def vmd_coverage(model: Dict, vmd: Dict, profile: Dict) -> Dict:
    name_map = build_name_map(model)
    keys = {n.encode("shift-jis", errors="replace")[:15] for n in vmd["bones"]}
    matched = keys & set(name_map)
    unmatched = [n for n in vmd["bones"]
                 if n.encode("shift-jis", errors="replace")[:15] not in name_map]
    standard_missing = [
        j for j, bn in profile["standard_map"].items()
        if bn.encode("shift-jis", errors="replace")[:15] not in keys
    ]
    finger_missing = [
        j for j, bn in profile["finger_map"].items()
        if bn.encode("shift-jis", errors="replace")[:15] not in keys
    ]
    morph_names = {m["name"] for m in profile["morphs"]}
    vmd_morphs = set(vmd.get("morphs", {}))
    return {
        "vmd_model_name": vmd.get("model_name", ""),
        "vmd_max_frame": vmd.get("max_frame", 0),
        "vmd_bone_tracks": len(vmd["bones"]),
        "matched_tracks": len(matched),
        "matched_ratio": round(len(matched) / max(1, len(vmd["bones"])), 4),
        "unmatched_tracks": len(unmatched),
        "unmatched_sample": unmatched[:20],
        "standard_missing": standard_missing,
        "finger_missing_count": len(finger_missing),
        "vmd_morphs": len(vmd_morphs),
        "profile_morphs": len(morph_names),
        "morph_match": len(vmd_morphs & morph_names),
        "morph_unmatched_sample": sorted(vmd_morphs - morph_names)[:15],
        "ik_frames": len(vmd.get("ik_frames", [])),
    }


def semantic_audit(model: Dict, vmd: Dict, profile: Dict) -> Dict:
    """Flag VMD semantics that change on MMD vs Blender playback."""
    by_name = {b["name"]: b for b in model["bones"]}
    non_movable = []
    for name, keys in vmd["bones"].items():
        b = by_name.get(name)
        if b is None:
            continue
        if not (b["flags"] & 0x0004):
            moved = sum(1 for k in keys
                        if abs(k["pos"][0]) + abs(k["pos"][1]) + abs(k["pos"][2]) > 1e-6)
            if moved:
                non_movable.append({"bone": name, "frames_with_translation": moved})
    non_movable.sort(key=lambda x: -x["frames_with_translation"])

    ik_names = [b["name"] for b in model["bones"] if b["flags"] & 0x0020 and "ik" in b]
    ik_tracked = [n for n in ik_names if n in vmd["bones"]]
    ik_state = {}
    for fr in vmd.get("ik_frames", []):
        if int(fr["frame"]) == 0:
            ik_state.update(fr.get("ik", {}))
    # MMD default: IK is on unless a frame says off.
    on_at_zero = [n for n in ik_names if ik_state.get(n, True)]
    return {
        "non_movable_translation_bones": non_movable,
        "non_movable_translation_count": len(non_movable),
        "ik_bones": ik_names,
        "ik_tracks_present": ik_tracked,
        "ik_off_frames_present": bool(vmd.get("ik_frames")),
        "ik_on_at_frame0": on_at_zero,
        "ik_default_on": len(on_at_zero) > 0,
        "warning": (
            "MMD ignores VMD translation on non-movable bones; Blender/MMD Tools applies it. "
            "Put root motion on センター/グルーブ only."
            if non_movable else ""
        ),
    }


def ik_toggle_effect(model: Dict, vmd: Dict, joints: Sequence[str],
                     profile: Dict) -> Dict:
    """World-space effect of IK on/off (should be ~0 for a safe VMD)."""
    name2i = {b["name"]: i for i, b in enumerate(model["bones"])}
    ev_on = VmdEvaluator(model, vmd, solve_ik=True,
                         apply_all_translations=True)
    ev_off = VmdEvaluator(model, vmd, solve_ik=False,
                          apply_all_translations=True)
    out = {}
    for j in joints:
        bn = profile["standard_map"].get(j)
        if not bn or bn not in name2i:
            continue
        i = name2i[bn]
        vals = []
        for f in range(vmd["max_frame"] + 1):
            p_on = ev_on.frame(f)["world_pos"][i]
            p_off = ev_off.frame(f)["world_pos"][i]
            vals.append(math.dist(p_on, p_off))
        out[j] = {
            "bone": bn,
            "mean": round(sum(vals) / len(vals), 4),
            "max": round(max(vals), 4),
        }
    return out


# ---------------------------------------------------------------------------
# Identity / single-bone checks (FK only, IK forced off)
# ---------------------------------------------------------------------------
def identity_check(model: Dict, profile: Dict) -> Dict:
    names = list(profile["writable_bones"].values())
    ik_names = [b["name"] for b in model["bones"] if b["flags"] & 0x0020 and "ik" in b]
    data = identity_vmd(names, ik_names=ik_names)
    ev = VmdEvaluator(model, {"bones": data["bone_frames"],
                              "morphs": {}, "ik_frames": data["ik_frames"],
                              "max_frame": 0},
                      force_ik_off=True, apply_all_translations=True)
    rec = ev.frame(0)
    name2i = {b["name"]: i for i, b in enumerate(model["bones"])}
    errs = []
    for name in names:
        i = name2i[name]
        errs.append((math.dist(model["bones"][i]["pos"], rec["world_pos"][i]), name))
    errs.sort()
    return {
        "bones": len(names),
        "max_err": round(errs[-1][0], 8),
        "worst_bone": errs[-1][1],
        "verdict": "bind_pose_preserved" if errs[-1][0] < 1e-5 else "moved",
    }


def single_bone_check(model: Dict, profile: Dict, angle: float = 30.0) -> Dict:
    """Rotate one standard bone at a time, report child displacement.

    This validates hierarchy/axes, not aesthetics: a working bone must move
    its child joint and must not move its parent.
    """
    name2i = {b["name"]: i for i, b in enumerate(model["bones"])}
    parent_of = {b["name"]: b.get("parent_name") for b in model["bones"]}
    results = []
    for side in ("l", "r"):
        for base in ("clavicle", "upperarm", "forearm", "hand",
                     "thigh", "shin", "foot"):
            joint = f"{base}_{side}"
            child = {"clavicle": "upperarm", "upperarm": "forearm",
                     "forearm": "hand", "hand": None, "thigh": "shin",
                     "shin": "foot", "foot": "toe"}[base]
            if child is None:
                continue
            cjoint = f"{child}_{side}"
            if joint not in profile["standard_map"] or cjoint not in profile["standard_map"]:
                continue
            bn = profile["standard_map"][joint]
            cb = profile["standard_map"][cjoint]
            pb = profile["standard_map"].get(f"pelvis")
            frames = {bn: [{"frame": 0, "pos": (0, 0, 0), "rot": axis_quat("x", angle)}]}
            ev = VmdEvaluator(model, {"bones": frames, "morphs": {},
                                      "ik_frames": [], "max_frame": 0},
                              force_ik_off=True, apply_all_translations=True)
            rec = ev.frame(0)
            di = name2i[bn]
            ci = name2i[cb]
            rest_c = model["bones"][ci]["pos"]
            moved = math.dist(rest_c, rec["world_pos"][ci])
            parent_name = parent_of[bn]
            parent_moved = None
            if parent_name in name2i:
                pi = name2i[parent_name]
                parent_moved = math.dist(model["bones"][pi]["pos"],
                                         rec["world_pos"][pi])
            results.append({
                "joint": joint, "bone": bn, "angle_deg": angle, "axis": "x",
                "child": cb, "child_displacement": round(moved, 5),
                "parent": parent_name,
                "parent_displacement": None if parent_moved is None else round(parent_moved, 8),
                "ok": moved > 1e-4 and (parent_moved is None or parent_moved < 1e-5),
            })
    return {"results": results,
            "failed": [r for r in results if not r["ok"]]}


# ---------------------------------------------------------------------------
# Same-model known-good check
# ---------------------------------------------------------------------------
def known_good_check(model: Dict, vmd: Dict, profile: Dict, blender_pose: Dict,
                     joints: Optional[Sequence[str]] = None,
                     scale_to_mmd: float = 12.5) -> Dict:
    """Compare VMD FK against a Blender action on the same model.

    Evaluates four semantics combinations; the 'blender' mode applies VMD
    translation on non-movable bones (MMD Tools behaviour) and uses IK on.
    """
    joints = list(joints or STD_BODY)
    name2i = {b["name"]: i for i, b in enumerate(model["bones"])}
    modes = {
        "mmd_ik_on": dict(solve_ik=True, apply_all_translations=False),
        "mmd_ik_off": dict(solve_ik=False, apply_all_translations=False),
        "blender_ik_on": dict(solve_ik=True, apply_all_translations=True),
        "blender_ik_off": dict(solve_ik=False, apply_all_translations=True),
    }
    frame_count = vmd["max_frame"] + 1
    out = {"modes": {}, "per_joint": {}}
    best = None
    for mode, kw in modes.items():
        ev = VmdEvaluator(model, vmd, **kw)
        errs = []
        per_joint = {j: [] for j in joints}
        for f in range(frame_count):
            rec = ev.frame(f)
            for j in joints:
                bn = profile["standard_map"].get(j)
                if not bn or bn not in blender_pose["pose"]:
                    continue
                if f >= len(blender_pose["pose"][bn]["head"]):
                    continue
                i = name2i[bn]
                bh = blender_pose["pose"][bn]["head"][f]
                bp = [bh[0] * scale_to_mmd, bh[2] * scale_to_mmd, bh[1] * scale_to_mmd]
                d = math.dist(rec["world_pos"][i], bp)
                errs.append(d)
                per_joint[j].append(d)
        rmse = math.sqrt(sum(x * x for x in errs) / len(errs)) if errs else None
        mx = max(errs) if errs else None
        out["modes"][mode] = {"rmse": None if rmse is None else round(rmse, 5),
                              "max": None if mx is None else round(mx, 5)}
        if rmse is not None and (best is None or rmse < best[0]):
            best = (rmse, mode, per_joint)
    if best:
        out["best_mode"] = best[1]
        out["per_joint"] = {
            j: {"rmse": round(math.sqrt(sum(x * x for x in v) / len(v)), 5),
                "max": round(max(v), 5)}
            for j, v in best[2].items() if v
        }
    out["verdict"] = ("same_model_match" if out.get("best_mode") == "blender_ik_on"
                      and out["modes"]["blender_ik_on"]["rmse"] < 0.1
                      else "check_needed")
    return out


# ---------------------------------------------------------------------------
# Safe mode
# ---------------------------------------------------------------------------
def build_safe_mode(profile: Dict) -> Dict:
    ik_names = []
    for side in profile.get("ik", {}).values():
        for rec in side.values():
            if isinstance(rec, dict) and rec.get("bone"):
                ik_names.append(rec["bone"])
    helpers = []
    for side in ("l", "r"):
        helpers += [b["name"] for b in profile.get("d_chain", {}).get(side, [])]
    helpers += [v["helper"] for v in profile.get("twist", {}).values() if v.get("helper")]
    helpers += [profile["pc"][s]["c"] for s in ("l", "r")
                if profile["pc"][s].get("c")]
    helpers += [profile["pc"][s]["p"] for s in ("l", "r")
                if profile["pc"][s].get("p")]
    return {
        "model_key": profile.get("model_key"),
        "pmx": profile.get("source"),
        "height_units": profile.get("height"),
        "unit_scale_m": 0.08,
        "writable_bones": profile["writable_bones"],
        "write_bone_count": len(profile["writable_bones"]),
        "never_write_prefixes": ["_dummy_", "_shadow_", "_face", "face", "Bone_"],
        "never_write_helpers": sorted(set(helpers)),
        "force_ik_off": ik_names,
        "root_translation_bone": profile["standard_map"].get("root"),
        "translation_bones": [n for n in (
            profile["standard_map"].get("root"),
            profile["standard_map"].get("groove")) if n],
        "physics_freeze": True,
        "morph_whitelist": [m["name"] for m in profile["morphs"]],
        "quirks": [
            "MMD ignores translation on non-movable bones; Blender import applies it",
            "VMD without IK frames leaves leg IK on (default); always write IK off",
            "D-chain/twist helpers/肩C are driven by inheritance; R2 animates 肩 and leaves P/C at zero",
            "Blender armature has _dummy_/_shadow_ helper bones; never export them",
        ],
    }
