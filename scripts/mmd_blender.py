# -*- coding: utf-8 -*-
"""Blender-side MMD motion helpers: VMD import + inherit-bone bake.

Why this exists
---------------
MMD evaluates 付与 (inherit) at playback time: ``左足D/左ひざD/左足首D`` copy
``左足/左ひざ/左足首`` (ratio 1.0), ``肩C`` copies ``肩P`` (-1.0) and the
``捩1`` helpers copy ``捩`` (0.5).  A VMD only needs the 23 FK body tracks;
MMD reconstructs the rest.  Blender has no inherit concept, so those bones stay
at rest and leg meshes (whose weights live on the D chain) never follow.

The bake writes the *local* transform of every inherit bone as explicit pose
keys, computed with the same ``VmdEvaluator`` used by the audit / closed loop.
The VMD itself is never modified: writing D bones into the MMD asset would
double-apply the inheritance inside MMD.

Coordinate convention (verified against MMD Tools imports, error < 1e-5):
  * position:  Blender = 0.08 * (x, z, y)          [MMD unit = 12.5/m]
  * rotation:  R_blender = S @ R_mmd @ S^T,  S = swap(Y, Z)
  * VMD frame f lands on Blender frame f + 1 (MMD Tools stores key 0 at 1).
"""
from __future__ import annotations

import math
import os
import sys

import bpy
from mathutils import Matrix, Quaternion, Vector

UNIT = 0.08
AXES = Matrix(((1, 0, 0), (0, 0, 1), (0, 1, 0)))
VMD_FRAME_OFFSET = 1  # MMD Tools: VMD key 0 -> Blender frame 1


# ---------------------------------------------------------------------------
# math
# ---------------------------------------------------------------------------
def quat_to_matrix3(q):
    """vtv quaternion (x, y, z, w) -> 3x3 rotation matrix."""
    x, y, z, w = q
    return Matrix((
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    ))


def deformation_matrix(world_q, world_pos, rest_pos):
    """MMD bone deformation in Blender's armature space.

    MMD applies ``p' = R (p - rest_head) + world_pos``; the equivalent Blender
    matrix is ``S R S^T`` with translation ``0.08 * S (world_pos - R rest_head)``.
    """
    R = quat_to_matrix3(world_q)
    p = Vector(world_pos)
    h = Vector(rest_pos)
    T = (AXES @ R @ AXES.transposed()).to_4x4()
    T.translation = AXES @ (p - R @ h) * UNIT
    return T


def rest_head(rest_pos):
    return AXES @ Vector(rest_pos) * UNIT


# ---------------------------------------------------------------------------
# model / motion
# ---------------------------------------------------------------------------
def repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def ensure_vtv_importable():
    src = os.path.join(repo_root(), "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def load_model_and_vmd(cfg, vmd_path):
    ensure_vtv_importable()
    from vtv.mmd.pmx import parse_pmx
    from vtv.mmd.vmd import parse_vmd
    return parse_pmx(cfg["pmx"]), parse_vmd(vmd_path)


def inheriting_bones(model):
    """Bones whose pose MMD derives from 付与親 (and Blender therefore misses)."""
    bones = model["bones"]
    out = []
    for b in bones:
        if not (b["flags"] & (0x0100 | 0x0200)):
            continue
        src = b.get("inherit_parent", -1)
        if 0 <= src < len(bones):
            out.append(b["name"])
    return out


# ---------------------------------------------------------------------------
# fcurve helpers (legacy + slotted actions)
# ---------------------------------------------------------------------------
def _fcurve_containers(action):
    """Yield (container, fcurve) for every fcurve of an action.

    Blender >=4.4 stores curves in layers/strips/channelbags; ``action.fcurves``
    is a legacy view that may be empty or partial, so both are visited and
    de-duplicated by identity.
    """
    seen = set()
    try:
        for fc in action.fcurves:
            key = fc.as_pointer()
            if key not in seen:
                seen.add(key)
                yield action.fcurves, fc
    except Exception:
        pass
    for layer in getattr(action, "layers", []):
        for strip in getattr(layer, "strips", []):
            for bag in getattr(strip, "channelbags", []):
                for fc in bag.fcurves:
                    key = fc.as_pointer()
                    if key not in seen:
                        seen.add(key)
                        yield bag.fcurves, fc


def remove_bone_fcurves(action, names):
    prefixes = tuple('pose.bones["%s"].' % n for n in names)
    removed = 0
    for container, fc in list(_fcurve_containers(action)):
        if fc.data_path.startswith(prefixes):
            container.remove(fc)
            removed += 1
    return removed


def set_bone_curves_linear(action, names):
    prefixes = tuple('pose.bones["%s"].' % n for n in names)
    for _container, fc in _fcurve_containers(action):
        if not fc.data_path.startswith(prefixes):
            continue
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"


# ---------------------------------------------------------------------------
# bake
# ---------------------------------------------------------------------------
def _bone_parent_name(model, i):
    p = model["bones"][i]["parent"]
    if 0 <= p < len(model["bones"]):
        return model["bones"][p]["name"]
    return None


def bake_inherit(arm, model, vmd, frames, vmd_offset=VMD_FRAME_OFFSET,
                 names=None, check_scale=True):
    """Bake inherit bones onto ``arm`` for every Blender frame in ``frames``.

    Returns a report dict.  ``arm`` must already carry the FK action (i.e. the
    VMD was imported by MMD Tools); frames outside the action keep the last
    key, mirroring MMD's interpolation clamp.
    """
    ensure_vtv_importable()
    from vtv.mmd.evaluate import VmdEvaluator

    bones = model["bones"]
    name2i = {b["name"]: i for i, b in enumerate(bones)}
    bake_names = list(names or inheriting_bones(model))
    missing = [n for n in bake_names if n not in arm.pose.bones]
    bake_names = [n for n in bake_names if n in arm.pose.bones]

    rest = {n: arm.data.bones[n].matrix_local.copy() for n in bake_names}
    parent_name = {n: _bone_parent_name(model, name2i[n]) for n in bake_names}
    constrained = {n: [c.type for c in arm.pose.bones[n].constraints]
                   for n in bake_names if arm.pose.bones[n].constraints}
    # A different Blender parent would invalidate the local-space formula.
    parent_mismatch = {}
    for n in bake_names:
        bp = arm.data.bones[n].parent
        got = bp.name if bp else None
        want = parent_name[n]
        if got != want:
            parent_mismatch[n] = {"blender": got, "pmx": want}
    bake_names = [n for n in bake_names if n not in parent_mismatch]
    parent_name = {n: parent_name[n] for n in bake_names}

    ev = VmdEvaluator(model, vmd, force_ik_off=True)
    frames = list(frames)
    action = arm.animation_data.action if arm.animation_data else None
    removed = remove_bone_fcurves(action, bake_names) if action else 0

    def target(name, rec):
        i = name2i[name]
        return deformation_matrix(rec["world_q"][i], rec["world_pos"][i],
                                  bones[i]["pos"])

    for n in bake_names:
        arm.pose.bones[n].rotation_mode = "QUATERNION"

    prev_q = {}
    scale_warns = []
    for f in frames:
        rec = ev.frame(f - vmd_offset)
        for n in bake_names:
            pb = arm.pose.bones[n]
            Mi = rest[n]
            Tp_name = parent_name[n]
            Tp = target(Tp_name, rec) if Tp_name else Matrix.Identity(4)
            B = Mi.inverted() @ Tp.inverted() @ target(n, rec) @ Mi
            pb.matrix_basis = B
            q = pb.rotation_quaternion.copy()
            if prev_q.get(n) is not None and prev_q[n].dot(q) < 0.0:
                q.negate()
                pb.rotation_quaternion = q
            prev_q[n] = q.copy()
            if check_scale:
                s = pb.scale
                if max(abs(s[0] - 1), abs(s[1] - 1), abs(s[2] - 1)) > 1e-4:
                    scale_warns.append((n, f))
            pb.keyframe_insert("rotation_quaternion", frame=f, group=n)
            pb.keyframe_insert("location", frame=f, group=n)

    if action:
        set_bone_curves_linear(action, bake_names)

    return {
        "baked_bones": bake_names,
        "missing_bones": missing,
        "constrained_bones": constrained,
        "parent_mismatch": parent_mismatch,
        "vmd_frame_offset": vmd_offset,
        "frames": [frames[0], frames[-1]] if frames else None,
        "removed_fcurves": removed,
        "scale_warnings": scale_warns[:20],
        "scale_warning_count": len(scale_warns),
        "action": action.name if action else None,
    }


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------
def verify_deformations(arm, model, vmd, blender_frames, vmd_offset=VMD_FRAME_OFFSET,
                        names=None, tol=1e-3):
    """Compare Blender's evaluated deformation with the PMX evaluator.

    Returns max head-position error (Blender units) and max rotation error
    (degrees) over every mapped PMX bone, plus the worst offenders.
    """
    ensure_vtv_importable()
    from vtv.mmd.evaluate import VmdEvaluator

    bones = model["bones"]
    name2i = {b["name"]: i for i, b in enumerate(bones)}
    mapped = [b["name"] for b in bones if b["name"] in arm.pose.bones]
    ev = VmdEvaluator(model, vmd, force_ik_off=True)
    scene = bpy.context.scene
    worst_pos = (-1.0, None, None)
    worst_rot = (-1.0, None, None)
    rows = []
    for bf in blender_frames:
        scene.frame_set(bf)
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        ea = arm.evaluated_get(dg)
        rec = ev.frame(bf - vmd_offset)
        f_pos = (0.0, None)
        f_rot = (0.0, None)
        for nm in mapped:
            pb = ea.pose.bones.get(nm)
            if pb is None:
                continue
            i = name2i[nm]
            hb = rest_head(bones[i]["pos"])
            head = pb.matrix.translation
            target_head = rest_head(rec["world_pos"][i])
            dp = (head - target_head).length
            D = pb.matrix @ pb.bone.matrix_local.inverted()
            T = deformation_matrix(rec["world_q"][i], rec["world_pos"][i],
                                   bones[i]["pos"])
            dq = math.degrees(D.to_quaternion().rotation_difference(
                T.to_quaternion()).angle)
            if dp > f_pos[0]:
                f_pos = (dp, nm)
            if dq > f_rot[0]:
                f_rot = (dq, nm)
        rows.append({"frame": bf, "vmd_frame": bf - vmd_offset,
                     "max_pos_err": f_pos[0], "worst_pos_bone": f_pos[1],
                     "max_rot_err_deg": f_rot[0], "worst_rot_bone": f_rot[1]})
        if f_pos[0] > worst_pos[0]:
            worst_pos = (f_pos[0], bf, f_pos[1])
        if f_rot[0] > worst_rot[0]:
            worst_rot = (f_rot[0], bf, f_rot[1])
    return {
        "max_head_error": worst_pos[0],
        "max_head_error_frame": worst_pos[1],
        "max_head_error_bone": worst_pos[2],
        "max_rot_error_deg": worst_rot[0],
        "max_rot_error_bone": worst_rot[2],
        "tol": tol,
        "ok": worst_pos[0] < tol,
        "samples": rows,
        "mapped_bones": len(mapped),
    }


def d_chain_vs_fk(model, vmd, frames, vmd_offset=VMD_FRAME_OFFSET):
    """Sanity: with ratio 1.0 and coincident rest heads, D bones == FK bones.

    This is an MMD-semantics check that does not involve Blender at all; if it
    fails, the inheritance interpretation is wrong and the bake cannot be
    trusted.
    """
    ensure_vtv_importable()
    from vtv.mmd.evaluate import VmdEvaluator

    bones = model["bones"]
    name2i = {b["name"]: i for i, b in enumerate(bones)}
    pairs = []
    for b in bones:
        if not (b["flags"] & 0x0100):
            continue
        src = b.get("inherit_parent", -1)
        if not (0 <= src < len(bones)) or abs(b.get("inherit_ratio", 0.0) - 1.0) > 1e-9:
            continue
        nm = b["name"]
        if not nm.endswith("D"):
            continue
        pairs.append((nm, bones[src]["name"]))
    ev = VmdEvaluator(model, vmd, force_ik_off=True)
    worst_pos = (-1.0, None, None)
    worst_rot = (-1.0, None, None)
    for f in frames:
        rec = ev.frame(f - vmd_offset)
        for dn, fn in pairs:
            if dn not in name2i or fn not in name2i:
                continue
            di, fi = name2i[dn], name2i[fn]
            Td = deformation_matrix(rec["world_q"][di], rec["world_pos"][di], bones[di]["pos"])
            Tf = deformation_matrix(rec["world_q"][fi], rec["world_pos"][fi], bones[fi]["pos"])
            dp = (Td.translation - Tf.translation).length
            dq = math.degrees(Td.to_quaternion().rotation_difference(
                Tf.to_quaternion()).angle)
            if dp > worst_pos[0]:
                worst_pos = (dp, f, dn)
            if dq > worst_rot[0]:
                worst_rot = (dq, f, dn)
    return {"pairs": pairs,
            "max_pos_err": worst_pos[0], "pos_frame": worst_pos[1],
            "max_rot_err_deg": worst_rot[0], "rot_frame": worst_rot[1],
            "ok": worst_pos[0] < 1e-5 and worst_rot[0] < 1e-3}


# ---------------------------------------------------------------------------
# import helper used by render_synth
# ---------------------------------------------------------------------------
def import_vmd(arm, vmd_path, import_morphs=True):
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    # MMD Tools imports bone animation for a selected armature and morph
    # animation for a selected mesh/root.  Select the model's meshes too so a
    # VMD with finger/morph tracks imports both halves.
    if import_morphs:
        for coll in arm.users_collection:
            for ob in coll.all_objects:
                if ob.type == "MESH" and ob.data.shape_keys:
                    # Drop the .blend's idle face action: the VMD owns the
                    # face for this render, otherwise old keys keep blinking
                    # on top of the new morph track.
                    ad = ob.data.shape_keys.animation_data
                    if ad is not None:
                        ad.action = None
                    ob.select_set(True)
    if arm.animation_data:
        arm.animation_data.action = None
    for pb in arm.pose.bones:
        pb.matrix_basis.identity()
    bpy.context.view_layer.update()
    bpy.ops.mmd_tools.import_vmd(filepath=os.path.abspath(vmd_path),
                                 scale=UNIT, bone_mapper="BLENDER", margin=0)
    bpy.context.view_layer.update()
    act = arm.animation_data.action if arm.animation_data else None
    return act


# ---------------------------------------------------------------------------
# measurements straight from the Blender rig (no PMX evaluator involved)
# ---------------------------------------------------------------------------
KNEE_CHAINS = {
    "l": {"fk": ("左足", "左ひざ", "左足首"), "d": ("左足D", "左ひざD", "左足首D")},
    "r": {"fk": ("右足", "右ひざ", "右足首"), "d": ("右足D", "右ひざD", "右足首D")},
}


def _bend_deg(armature, chain):
    """Return (flex, interior): flex 0 = straight; interior 180 = straight.

    ``interior`` is the convention the M1 report used ("right knee 51.2°" at
    T5 frame 26 = 128.8° of actual flexion).
    """
    pts = []
    for name in chain:
        pb = armature.pose.bones.get(name)
        if pb is None:
            return None, None
        pts.append(pb.matrix.translation)
    u = pts[1] - pts[0]
    v = pts[2] - pts[1]
    if u.length < 1e-9 or v.length < 1e-9:
        return None, None
    c = max(-1.0, min(1.0, u.normalized().dot(v.normalized())))
    flex = math.degrees(math.acos(c))
    return flex, 180.0 - flex


def measure_knees(arm, frames):
    """Knee flexion and ankle height from the posed rig.

    Measured on both the FK and the D chain: the mesh is weighted to the D
    chain, so D/FK agreement plus a non-zero flex is the visual acceptance
    signal that the bake worked.
    """
    scene = bpy.context.scene
    rows = []
    for bf in frames:
        scene.frame_set(bf)
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        ea = arm.evaluated_get(dg)
        row = {"frame": bf}
        for side, chains in KNEE_CHAINS.items():
            for kind, chain in chains.items():
                flex, interior = _bend_deg(ea, chain)
                row[f"{side}_{kind}_flex"] = flex
                row[f"{side}_{kind}_interior"] = interior
            ankle = ea.pose.bones.get(chains["d"][2])
            row[f"{side}_ankle_z"] = float(ankle.matrix.translation.z) if ankle else None
        rows.append(row)
    return rows
