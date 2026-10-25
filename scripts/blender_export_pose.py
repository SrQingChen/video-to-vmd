"""Export posed bone matrices/heads from a Blender armature action.

Run headless:
  blender --background <scene.blend> --python scripts/blender_export_pose.py -- <out.json> <armature> <start:end:step> [bones.json]

bones.json may be a profile (uses standard_map/finger_map) or a plain list.
Coordinates are armature space (no object transform), matching the PMX
skeleton space after the MMD Tools axis swap (x, y, z)_mmd = (x, z, y)_blender.
"""
import json
import os
import sys

import bpy


def vec(v):
    return [round(float(x), 6) for x in v]


def mat(m):
    n = len(m)
    return [round(float(m[r][c]), 6) for r in range(n) for c in range(len(m[0]))]


def bone_list_from(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return set(data)
    names = set()
    for key in ("standard_map", "finger_map", "writable_bones"):
        vals = data.get(key) or {}
        names.update(vals.values() if isinstance(vals, dict) else vals)
    for side in (data.get("ik") or {}).values():
        for rec in side.values():
            if isinstance(rec, dict) and rec.get("bone"):
                names.add(rec["bone"])
    names.update(data.get("root_chain") or [])
    return names


def main():
    argv = sys.argv
    if "--" in argv:
        rest = argv[argv.index("--") + 1:]
    else:
        rest = []
    out, arm_name, frames_spec = rest[0], rest[1], rest[2]
    bones_filter = bone_list_from(rest[3]) if len(rest) > 3 else None
    start, end, step = [int(x) for x in frames_spec.split(":")]

    ob = bpy.data.objects.get(arm_name)
    if ob is None or ob.type != "ARMATURE":
        raise SystemExit(f"armature not found: {arm_name}")
    scene = bpy.context.scene
    arm = ob.data
    names = [b.name for b in arm.bones
             if bones_filter is None or b.name in bones_filter]

    frames = list(range(start, end + 1, step))
    data = {
        "blend": bpy.data.filepath,
        "armature": arm_name,
        "armature_matrix_world": mat(ob.matrix_world),
        "fps": scene.render.fps / scene.render.fps_base,
        "action": ob.animation_data.action.name if ob.animation_data and ob.animation_data.action else None,
        "bones": names,
        "frames": frames,
        "pose": {name: {"head": [], "matrix": [], "loc": [], "quat": []}
                 for name in names},
    }
    for f in frames:
        scene.frame_set(f)
        bpy.context.view_layer.update()
        for name in names:
            pb = ob.pose.bones[name]
            rec = data["pose"][name]
            rec["head"].append(vec(pb.head))
            rec["matrix"].append(mat(pb.matrix))
            rec["loc"].append(vec(pb.location))
            rec["quat"].append(vec(pb.rotation_quaternion))

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print("WROTE", out, "frames", len(frames), "bones", len(names))


main()
