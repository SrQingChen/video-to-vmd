"""Export Blender armature rest/topology metadata to JSON.

Run headless:
  blender --background <scene.blend> --python scripts/blender_export_armature.py -- <out.json> [arm1,arm2]

The output is the Blender side of the compatibility audit (§4.3.0): bone
names/parents/rest matrices, action key coverage, constraints, extra bones
(MMD Tools `_dummy_` / `_shadow_` helpers).
"""
import json
import os
import sys

import bpy
from mathutils import Matrix


def vec(v):
    return [round(float(x), 6) for x in v]


def mat(m):
    n = len(m)
    return [round(float(m[r][c]), 6) for r in range(n) for c in range(len(m[0]))]


def main():
    argv = sys.argv
    out = None
    only = None
    if "--" in argv:
        pos = argv.index("--")
        rest = argv[pos + 1:]
        if rest:
            out = rest[0]
        if len(rest) > 1:
            only = set(rest[1].split(","))

    scene = bpy.context.scene
    data = {
        "blend": bpy.data.filepath,
        "blender": bpy.app.version_string,
        "fps": scene.render.fps / scene.render.fps_base,
        "frame_range": [scene.frame_start, scene.frame_end],
        "armatures": {},
    }

    for ob in bpy.data.objects:
        if ob.type != "ARMATURE":
            continue
        if only and ob.name not in only:
            continue
        arm = ob.data
        action = ob.animation_data.action if ob.animation_data else None
        fcurve_prefix = {}
        if action:
            for fc in action.fcurves:
                if fc.data_path.startswith('pose.bones["'):
                    name = fc.data_path.split('"')[1]
                    fcurve_prefix[name] = fcurve_prefix.get(name, 0) + 1

        bones = []
        for b in arm.bones:
            pb = ob.pose.bones.get(b.name)
            bones.append({
                "name": b.name,
                "parent": b.parent.name if b.parent else None,
                "head": vec(b.head_local),
                "tail": vec(b.tail_local),
                "length": round(float(b.length), 6),
                "matrix_local": mat(b.matrix_local),
                "matrix_local_3x3": mat(b.matrix_local.to_3x3()),
                "use_connect": bool(b.use_connect),
                "use_deform": bool(b.use_deform),
                "use_inherit_rotation": bool(b.use_inherit_rotation),
                "inherit_scale": b.inherit_scale,
                "fcurves": fcurve_prefix.get(b.name, 0),
                "constraints": ([c.type for c in pb.constraints] if pb else []),
                "rotation_mode": pb.rotation_mode if pb else None,
            })

        data["armatures"][ob.name] = {
            "armature_data": arm.name,
            "object_matrix_world": mat(ob.matrix_world),
            "scale": vec(ob.scale),
            "bone_count": len(bones),
            "bones": bones,
            "extra_bones_dummy": [b["name"] for b in bones if b["name"].startswith("_dummy_")],
            "extra_bones_shadow": [b["name"] for b in bones if b["name"].startswith("_shadow_")],
            "action": action.name if action else None,
            "action_fcurves": len(action.fcurves) if action else 0,
            "action_frame_range": vec(action.frame_range) if action else None,
            "keyed_bones": len([b for b in bones if b["fcurves"] > 0]),
            "unkeyed_bones_sample": [b["name"] for b in bones if b["fcurves"] == 0][:30],
        }

    data["actions"] = [{
        "name": a.name,
        "frame_range": vec(a.frame_range),
        "fcurves": len(a.fcurves),
        "users": a.users,
    } for a in bpy.data.actions]

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print("WROTE", out)


main()
