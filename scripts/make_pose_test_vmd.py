"""Build a single-joint test VMD (default: 左ひざ +60 deg) for the D-chain check.

  python scripts/make_pose_test_vmd.py --bone 左ひざ --degrees 60 \
      --out outputs/tests/knee60_l.vmd

All 23 body bones are keyed at identity so the pose is unambiguous, then the
target bone is rotated about a world axis.  Leg IK is written as explicitly
OFF (the pipeline's safe convention).
"""
import argparse
import json
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from vtv.mmd.evaluate import ik_bones  # noqa: E402
from vtv.mmd.pmx import parse_pmx  # noqa: E402
from vtv.mmd.writer import write_vmd  # noqa: E402

BODY23 = ["センター", "グルーブ", "下半身", "上半身", "上半身1", "上半身2",
          "首", "頭", "右肩", "右腕", "右ひじ", "右手首",
          "左肩", "左腕", "左ひじ", "左手首",
          "右足", "右ひざ", "右足首", "右つま先",
          "左足", "左ひざ", "左足首", "左つま先"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "models.json"))
    ap.add_argument("--model", default="si")
    ap.add_argument("--bone", default="左ひざ")
    ap.add_argument("--axis", default="x", choices=["x", "y", "z"])
    ap.add_argument("--degrees", type=float, default=60.0)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8"))["models"][args.model]
    model = parse_pmx(cfg["pmx"])
    names = {b["name"] for b in model["bones"]}
    ik = [model["bones"][i]["name"] for i in ik_bones(model)]

    half = math.radians(args.degrees) * 0.5
    s = math.sin(half)
    q = {"x": (s, 0.0, 0.0, math.cos(half)),
         "y": (0.0, s, 0.0, math.cos(half)),
         "z": (0.0, 0.0, s, math.cos(half))}[args.axis]

    tracks = {}
    for name in BODY23:
        if name not in names:
            print("missing body bone:", name)
            continue
        rot = q if name == args.bone else (0.0, 0.0, 0.0, 1.0)
        tracks[name] = [
            {"frame": 0, "pos": (0.0, 0.0, 0.0), "rot": rot},
            {"frame": args.frames, "pos": (0.0, 0.0, 0.0), "rot": rot},
        ]
    if args.bone not in tracks:
        raise SystemExit("unknown bone: " + args.bone)

    ik_off = [{"frame": 0, "show": True, "ik": {n: False for n in ik}}]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_vmd(args.out, cfg.get("display", args.model), tracks, {}, ik_off)
    print("WROTE", os.path.abspath(args.out), "bone", args.bone,
          "axis", args.axis, "deg", args.degrees, "frames", args.frames)


main()
