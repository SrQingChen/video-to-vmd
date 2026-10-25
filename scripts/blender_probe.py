"""Inventory a .blend scene: objects, armatures, actions, cameras, MMD setup.

Run headless:
  blender --background <scene.blend> --python scripts/blender_probe.py -- <out.json>
"""
import json
import os
import sys

import bpy


def vec(v):
    return [round(float(x), 6) for x in v]


def main():
    argv = sys.argv
    out = None
    if "--" in argv:
        pos = argv.index("--")
        if len(argv) > pos + 1:
            out = argv[pos + 1]

    scene = bpy.context.scene
    data = {
        "blend": bpy.data.filepath,
        "blender": bpy.app.version_string,
        "fps": scene.render.fps / scene.render.fps_base,
        "frame_start": scene.frame_start,
        "frame_end": scene.frame_end,
        "resolution": [scene.render.resolution_x, scene.render.resolution_y,
                       scene.render.resolution_percentage],
        "engine": scene.render.engine,
        "camera": scene.camera.name if scene.camera else None,
        "objects": [],
        "armatures": {},
        "actions": [],
        "mmd_roots": [],
        "collections": [c.name for c in bpy.data.collections],
    }

    for ob in bpy.data.objects:
        item = {
            "name": ob.name,
            "type": ob.type,
            "parent": ob.parent.name if ob.parent else None,
            "parent_type": ob.parent_type,
            "hide_render": ob.hide_render,
            "hide_viewport": ob.hide_viewport,
            "collections": [c.name for c in ob.users_collection],
            "location": vec(ob.location),
            "scale": vec(ob.scale),
            "animation_data": bool(ob.animation_data and ob.animation_data.action),
            "action": (ob.animation_data.action.name
                       if ob.animation_data and ob.animation_data.action else None),
        }
        if ob.type == "ARMATURE":
            item["bones"] = len(ob.data.bones)
            item["modifiers"] = [m.type for m in ob.modifiers]
            item["constraints"] = [c.type for c in ob.constraints]
        if ob.type == "CAMERA":
            item["camera_type"] = ob.data.type
            item["lens"] = float(ob.data.lens)
            item["animation_data_all"] = bool(ob.animation_data)
            if ob.animation_data:
                item["nla_tracks"] = len(ob.animation_data.nla_tracks)
                item["drivers"] = len(ob.animation_data.drivers)
        data["objects"].append(item)

    for arm in bpy.data.armatures:
        obj = None
        for ob in bpy.data.objects:
            if ob.type == "ARMATURE" and ob.data == arm:
                obj = ob
                break
        data["armatures"][arm.name] = {
            "object": obj.name if obj else None,
            "bones": len(arm.bones),
            "bone_names_sample": [b.name for b in arm.bones[:60]],
            "bone_names_tail": [b.name for b in arm.bones[-20:]],
            "rest_pose": arm.pose_position,
            "collections": [c.name for c in arm.collections],
            "constraints": ([c.type for c in obj.pose.bones[0].constraints]
                            if obj and len(obj.pose.bones) else []),
        }

    for act in bpy.data.actions:
        data["actions"].append({
            "name": act.name,
            "frame_range": vec(act.frame_range),
            "fcurves": len(act.fcurves),
            "users": act.users,
        })

    for ob in bpy.data.objects:
        if "mmd" in ob.name.lower() or ob.type == "EMPTY":
            keys = [k for k in ob.keys() if "mmd" in k.lower()]
            if "mmd_type" in ob.keys() or keys:
                data["mmd_roots"].append({
                    "name": ob.name,
                    "mmd_type": ob.get("mmd_type"),
                    "keys": keys,
                })

    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        print("WROTE", out)
    else:
        print(json.dumps(data, ensure_ascii=False, indent=1))


main()
