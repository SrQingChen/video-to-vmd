"""Render a single-character synthetic-GT clip with a fixed full-body camera.

Run headless:
  blender --background <three_person.blend> --python scripts/render_synth.py -- \
    --model si --vmd outputs/gt/si_R2_safe.vmd --view front \
    --out outputs/synth/si_R2/front --frames 1:60 --res 1080x1920

Rules enforced here (v2 §2.9.1):
  * only the target character collection is rendered;
  * the camera is fixed and its animation is cleared (no camera VMD);
  * camera parameters + GT VMD + frame range are saved to meta.json.
"""
import argparse
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    argv = sys.argv
    rest = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "models.json"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--vmd", default=None)
    ap.add_argument("--view", default="front",
                    choices=["front", "three_quarter", "side", "back", "existing"])
    ap.add_argument("--camera", default=None, help="existing camera name for --view existing")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", default="1:60")
    ap.add_argument("--res", default="1080x1920")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--lens", type=float, default=50.0)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--margin", type=float, default=1.15)
    ap.add_argument("--no-import", action="store_true",
                    help="use the existing action instead of importing --vmd")
    ap.add_argument("--bbox-frames", type=int, default=8,
                    help="sample this many frames for the camera framing "
                         "(0 = first frame only); avoids walking out of frame")
    return ap.parse_args(rest)


def char_bbox(collection):
    dg = bpy.context.evaluated_depsgraph_get()
    lo = Vector((1e9, 1e9, 1e9))
    hi = Vector((-1e9, -1e9, -1e9))
    found = False
    for ob in collection.all_objects:
        if ob.type != "MESH":
            continue
        ev = ob.evaluated_get(dg)
        for corner in ev.bound_box:
            p = ev.matrix_world @ Vector(corner)
            lo = Vector((min(lo[i], p[i]) for i in range(3)))
            hi = Vector((max(hi[i], p[i]) for i in range(3)))
            found = True
    if not found:
        lo, hi = Vector((-0.5, -0.5, 0)), Vector((0.5, 0.5, 1.8))
    return lo, hi


def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out)
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)["models"][args.model]
    target = cfg["collection"]
    arm = bpy.data.objects.get(cfg["armature"])
    if arm is None:
        raise SystemExit(f"armature not found: {cfg['armature']}")

    scene = bpy.context.scene
    start, end = [int(x) for x in args.frames.split(":")]
    scene.render.fps = args.fps
    scene.render.fps_base = 1.0
    rx, ry = [int(x) for x in args.res.lower().split("x")]
    scene.render.resolution_x, scene.render.resolution_y = rx, ry
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.use_file_extension = True
    try:
        scene.eevee.taa_render_samples = args.samples
    except Exception:
        pass

    # ---- single character ------------------------------------------------
    hidden = []
    for coll in bpy.data.collections:
        if coll.name.startswith("CHR_"):
            hide = coll.name != target
            coll.hide_render = hide
            coll.hide_viewport = hide
            if hide:
                hidden.append(coll.name)

    # ---- motion ---------------------------------------------------------
    if not args.no_import and args.vmd:
        bpy.ops.object.select_all(action="DESELECT")
        arm.select_set(True)
        bpy.context.view_layer.objects.active = arm
        if arm.animation_data:
            arm.animation_data.action = None
        for pb in arm.pose.bones:
            pb.matrix_basis.identity()
        bpy.ops.mmd_tools.import_vmd(filepath=os.path.abspath(args.vmd),
                                     scale=0.08, bone_mapper="BLENDER", margin=0)
    action = arm.animation_data.action.name if arm.animation_data and arm.animation_data.action else None

    # MMD Tools import changes the scene frame range; set it after import.
    scene.frame_start, scene.frame_end = start, end
    scene.frame_set(start)
    bpy.context.view_layer.update()
    coll = bpy.data.collections.get(target)
    lo, hi = char_bbox(coll)
    if args.bbox_frames and args.bbox_frames > 1:
        # union the character bbox over the clip so the framing holds for the
        # whole motion (a 2 m root path otherwise leaves the auto frame)
        pad = max(0.02, (hi.z - lo.z) * 0.02)
        for f in np.linspace(start, end, args.bbox_frames).astype(int):
            scene.frame_set(int(f))
            bpy.context.view_layer.update()
            l2, h2 = char_bbox(coll)
            lo = Vector((min(lo[i], l2[i]) for i in range(3)))
            hi = Vector((max(hi[i], h2[i]) for i in range(3)))
        lo = lo - Vector((pad, pad, pad))
        hi = hi + Vector((pad, pad, pad))
    scene.frame_set(start)

    # ---- fixed camera ----------------------------------------------------
    if args.view == "existing":
        cam = bpy.data.objects.get(args.camera) or scene.camera
        if cam is None:
            raise SystemExit("no existing camera")
        cam.animation_data_clear()
        cam.data.animation_data_clear() if cam.data.animation_data else None
        scene.camera = cam
        cam_info = {"name": cam.name, "location": list(cam.location),
                    "rotation_quaternion": list(cam.rotation_quaternion),
                    "lens": float(cam.data.lens), "reused": True}
    else:
        center = (lo + hi) * 0.5
        height = max(1e-3, hi.z - lo.z)
        width = max(1e-3, hi.x - lo.x)
        cam_data = bpy.data.cameras.new("SYNTH CAM")
        cam = bpy.data.objects.new("SYNTH CAM", cam_data)
        scene.collection.objects.link(cam)
        cam_data.lens = args.lens
        cam_data.sensor_fit = "AUTO"
        cam_data.clip_start = 0.05
        cam_data.clip_end = 100.0
        # portrait: AUTO uses the 36mm sensor on the larger (vertical) side
        vfov = 2.0 * math.atan(cam_data.sensor_width / (2.0 * args.lens))
        dist = (height * args.margin * 0.5) / math.tan(vfov * 0.5)
        dist = max(dist, width * args.margin * 1.2)
        az = {"front": 0.0, "three_quarter": 35.0, "side": 90.0, "back": 180.0}[args.view]
        az = math.radians(az)
        # MMD model faces -Y in Blender (PMX -Z after the xyz<->xzy swap)
        direction = Vector((math.sin(az), -math.cos(az), 0.0))
        cam.location = center + direction * dist
        look = center - cam.location
        cam.rotation_mode = "QUATERNION"
        cam.rotation_quaternion = look.to_track_quat("-Z", "Y")
        scene.camera = cam
        cam_info = {
            "name": cam.name, "location": list(cam.location),
            "rotation_quaternion": list(cam.rotation_quaternion),
            "lens": args.lens, "sensor_width": cam_data.sensor_width,
            "sensor_fit": cam_data.sensor_fit, "distance": dist,
            "view": args.view, "reused": False,
        }

    # camera animation must not survive (no camera VMD)
    if cam.animation_data:
        cam.animation_data_clear()

    os.makedirs(os.path.join(out_dir, "frames"), exist_ok=True)
    scene.render.filepath = os.path.join(out_dir, "frames", "frame_")

    meta = {
        "model": args.model,
        "collection": target,
        "armature": cfg["armature"],
        "vmd": os.path.abspath(args.vmd) if args.vmd else None,
        "action": action,
        "camera": cam_info,
        "bbox": {"min": list(lo), "max": list(hi),
                 "height": hi.z - lo.z, "width": hi.x - lo.x},
        "frames": [start, end],
        "fps": args.fps,
        "resolution": [rx, ry],
        "engine": scene.render.engine,
        "samples": args.samples,
        "hidden_collections": hidden,
        "blend": bpy.data.filepath,
        "blender": bpy.app.version_string,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    print("SYNTH_META", json.dumps(meta, ensure_ascii=False))
    bpy.ops.render.render(animation=True)
    print("RENDER_DONE", out_dir)


main()
