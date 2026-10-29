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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import mmd_blender  # noqa: E402
from vtv import framing  # noqa: E402


def parse_args():
    argv = sys.argv
    rest = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "models.json"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--vmd", default=None)
    ap.add_argument("--view", default="front",
                    choices=["front", "front_left", "front_right",
                             "three_quarter", "side", "back", "existing"])
    ap.add_argument("--shot", default="full",
                    choices=["full", "medium", "face"],
                    help="full body, the fixed waist-up shot, or a face close-up "
                         "used for M6 face evidence")
    ap.add_argument("--camera", default=None, help="existing camera name for --view existing")
    ap.add_argument("--camera-from", default=None,
                    help="meta.json whose camera (location/quaternion/lens) is "
                         "reused verbatim; for before/after comparisons")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", default="1:60")
    ap.add_argument("--stills", default=None,
                    help="comma separated Blender frames to render as stills "
                         "instead of an animation range (bake still covers 1:242)")
    ap.add_argument("--res", default="1080x1920")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--lens", type=float, default=50.0)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--margin", type=float, default=1.15)
    ap.add_argument("--no-import", action="store_true",
                    help="use the existing action instead of importing --vmd")
    ap.add_argument("--bbox-frames", type=int, default=0,
                    help="0 = every frame in the bbox scope (M5 default; an "
                         "8-frame sample missed wide arm frames); >1 = that many "
                         "uniform samples; 1 = first frame only")
    ap.add_argument("--bbox-scope", default="clip", choices=["clip", "range"],
                    help="clip = union over the whole VMD (short clips: no frame "
                         "can leave the frame); range = union over the rendered "
                         "range (long videos: keeps the character large)")
    ap.add_argument("--no-framing-gate", action="store_true",
                    help="do not fail the render when a sampled frame leaves the "
                         "image (framing is still recorded in meta.json)")
    ap.add_argument("--no-bake-inherit", action="store_true",
                    help="do NOT bake MMD inherit bones (D chain/twist/肩C) "
                         "onto the Blender rig; renders will show straight legs")
    ap.add_argument("--verify-frames", type=int, default=5,
                    help="sample this many frames to check baked pose == PMX "
                         "evaluator (0 = skip)")
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
    # Stills outside --frames must still be inside the inherited-bake range,
    # otherwise the D chain/twist bones are left at rest on those frames
    # (straight legs).  Expand the bake range to cover every requested still.
    still_frames = ([int(x) for x in args.stills.split(",") if x.strip()]
                    if args.stills else [])
    if still_frames:
        start = min([start] + still_frames)
        end = max([end] + still_frames)
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
        action_obj = mmd_blender.import_vmd(arm, args.vmd)
        action = action_obj.name if action_obj else None
    else:
        action = (arm.animation_data.action.name
                  if arm.animation_data and arm.animation_data.action else None)

    # MMD Tools import changes the scene frame range; set it after import.
    scene.frame_start, scene.frame_end = start, end

    # The auto-frame needs the *baked* pose (D chain/twist/肩C move the mesh).
    # M5: the union is computed over every frame of the chosen scope; an 8-frame
    # sample missed wide-arm frames (tiflos) and the old horizontal fit was wrong.
    vmd_info = None
    bake_start, bake_end = start, end
    bake_report = None
    if not args.no_import and args.vmd and not args.no_bake_inherit:
        model, vmd = mmd_blender.load_model_and_vmd(cfg, os.path.abspath(args.vmd))
        if args.bbox_scope == "clip":
            bake_start, bake_end = 1, int(vmd["max_frame"]) + 1
        bake_report = mmd_blender.bake_inherit(arm, model, vmd,
                                               range(bake_start, bake_end + 1))
        sample = sorted({int(round(x)) for x in
                         np.linspace(start, end, min(8, end - start + 1))})
        bake_report["d_chain_vs_fk"] = mmd_blender.d_chain_vs_fk(
            model, vmd, sample)
        if args.verify_frames > 0:
            vs = sorted({int(round(x)) for x in
                         np.linspace(start, end, args.verify_frames)})
            bake_report["verify"] = mmd_blender.verify_deformations(
                arm, model, vmd, vs)
        bake_report["knees"] = mmd_blender.measure_knees(arm, sample)
        print("INHERIT_BAKE", json.dumps(bake_report, ensure_ascii=False))
        # Hard gate: a render whose rig disagrees with the PMX evaluator is
        # not evidence.  Fail loudly instead of producing a misleading clip.
        if not bake_report["d_chain_vs_fk"]["ok"]:
            raise SystemExit("INHERIT BAKE FAILED: D chain != FK (MMD semantics)")
        if not bake_report.get("verify", {}).get("ok", True):
            raise SystemExit("INHERIT BAKE FAILED: Blender != PMX evaluator")

    coll = bpy.data.collections.get(target)

    def scan_frames(a, b):
        if args.bbox_frames <= 0:
            return list(range(int(a), int(b) + 1))
        if args.bbox_frames == 1:
            return [int(a)]
        return sorted({int(round(x)) for x in
                       np.linspace(a, b, int(args.bbox_frames))})

    scan = scan_frames(bake_start, bake_end)
    lo = Vector((1e9, 1e9, 1e9))
    hi = Vector((-1e9, -1e9, -1e9))
    frame_bboxes = []
    for f in scan:
        scene.frame_set(int(f))
        bpy.context.view_layer.update()
        l2, h2 = char_bbox(coll)
        lo = Vector((min(lo[i], l2[i]) for i in range(3)))
        hi = Vector((max(hi[i], h2[i]) for i in range(3)))
        if start <= int(f) <= end:
            frame_bboxes.append({"frame": int(f),
                                 "min": [float(c) for c in l2],
                                 "max": [float(c) for c in h2]})
    if lo.x > hi.x:  # no frame scanned (defensive)
        lo, hi = Vector((-0.5, -0.5, 0.0)), Vector((0.5, 0.5, 1.8))
        frame_bboxes = [{"frame": start, "min": list(lo), "max": list(hi)}]
    scene.frame_set(start)

    # ---- fixed camera ----------------------------------------------------
    if args.camera_from:
        with open(args.camera_from, encoding="utf-8") as f:
            src = json.load(f)["camera"]
        cam_data = bpy.data.cameras.new("SYNTH CAM")
        cam = bpy.data.objects.new("SYNTH CAM", cam_data)
        scene.collection.objects.link(cam)
        cam_data.lens = float(src["lens"])
        cam_data.sensor_width = float(src.get("sensor_width", 36.0))
        cam_data.sensor_fit = src.get("sensor_fit", "AUTO")
        cam_data.clip_start = 0.05
        cam_data.clip_end = 100.0
        cam.location = Vector(src["location"])
        cam.rotation_mode = "QUATERNION"
        cam.rotation_quaternion = src["rotation_quaternion"]
        scene.camera = cam
        if cam.animation_data:
            cam.animation_data_clear()
        cam_info = dict(src, reused=True, source_meta=os.path.abspath(args.camera_from))
    elif args.view == "existing":
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
        az = {"front": 0.0, "front_left": -15.0, "front_right": 15.0,
              "three_quarter": 35.0, "side": 90.0, "back": 180.0}[args.view]
        az = math.radians(az)
        # MMD model faces -Y in Blender (PMX -Z after the xyz<->xzy swap)
        direction = Vector((math.sin(az), -math.cos(az), 0.0))
        cam_data = bpy.data.cameras.new("SYNTH CAM")
        cam = bpy.data.objects.new("SYNTH CAM", cam_data)
        scene.collection.objects.link(cam)
        cam_data.lens = args.lens
        cam_data.sensor_fit = "AUTO"
        cam_data.clip_start = 0.05
        cam_data.clip_end = 100.0
        if args.shot in ("medium", "face"):
            # Fixed chest/waist-up or face framing for hand/face evidence.
            # ``medium`` is the M3.5 shot; ``face`` is a tight head shot for
            # M6 face-evidence training (the detector needs a much larger face
            # than the full-body portrait provides).
            full_h = height
            if args.shot == "face":
                center = Vector((center.x, center.y, lo.z + 0.86 * full_h))
                height = 0.32 * full_h
                width = 0.32 * full_h
            else:
                center = Vector((center.x, center.y, lo.z + 0.72 * full_h))
                height = 0.45 * full_h
                width = max(0.70 * full_h, width * 0.75)
            vfov = 2.0 * math.atan(cam_data.sensor_width / (2.0 * args.lens))
            dist = (height * args.margin * 0.5) / math.tan(vfov * 0.5)
            dist = max(dist, width * args.margin * 1.2)
            fit = {"distance": dist, "mode": f"{args.shot}-legacy"}
        else:
            # M5 true two-axis fit: portrait 2K has a much narrower horizontal
            # FOV, so a vertical-only distance crops wide arms/sleeves.
            corners = framing.bbox_corners(lo, hi)
            fit = framing.fit_distance(
                corners, center, direction, (rx, ry), args.lens,
                cam_data.sensor_width, cam_data.sensor_fit, args.margin)
            dist = fit["distance"]
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
            "view": args.view, "reused": False, "fit": fit,
        }

    # camera animation must not survive (no camera VMD)
    if cam.animation_data:
        cam.animation_data_clear()

    # ---- framing verification (M5): every sampled frame must stay in frame --
    R = cam.rotation_quaternion.to_matrix()
    right = R @ Vector((1.0, 0.0, 0.0))
    up = R @ Vector((0.0, 1.0, 0.0))
    forward = -(R @ Vector((0.0, 0.0, 1.0)))
    hfov, vfov = framing.sensor_fovs((rx, ry), cam.data.lens,
                                     cam.data.sensor_width, cam.data.sensor_fit)
    framing_report = framing.framing_report_basis(
        frame_bboxes, [float(c) for c in cam.location],
        [float(c) for c in right], [float(c) for c in up],
        [float(c) for c in forward], hfov, vfov, args.margin)
    framing_report["shot"] = args.shot
    framing_report["bbox_scope"] = args.bbox_scope
    framing_report["frame_range"] = [start, end]
    print("FRAMING", json.dumps(framing_report, ensure_ascii=False))
    if (args.shot == "full" and not framing_report["ok"]
            and not args.no_framing_gate):
        ndc = framing_report.get("max_abs_ndc")
        raise SystemExit(
            "FRAMING FAILED: a sampled frame leaves the image "
            f"(max |ndc| {ndc if ndc is not None else 'n/a'} at frame "
            f"{framing_report.get('worst_frame')}); raise --margin or use "
            "--bbox-scope clip")

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
                 "height": hi.z - lo.z, "width": hi.x - lo.x,
                 "scope": args.bbox_scope,
                 "frames_sampled": len(scan)},
        "framing": framing_report,
        "frames": [start, end],
        "bake_frames": [bake_start, bake_end],
        "fps": args.fps,
        "resolution": [rx, ry],
        "engine": scene.render.engine,
        "samples": args.samples,
        "hidden_collections": hidden,
        "inherit_bake": bake_report,
        "blend": bpy.data.filepath,
        "blender": bpy.app.version_string,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    print("SYNTH_META", json.dumps(meta, ensure_ascii=False))
    if still_frames:
        for f in still_frames:
            scene.frame_set(f)
            bpy.context.view_layer.update()
            scene.render.filepath = os.path.join(out_dir, "frames",
                                                 "frame_%04d.png" % f)
            bpy.ops.render.render(write_still=True)
    else:
        bpy.ops.render.render(animation=True)
    print("RENDER_DONE", out_dir)


main()
