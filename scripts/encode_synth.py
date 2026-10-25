"""Encode rendered synthetic-GT PNG sequences and write manifest.json.

Usage:
  python scripts/encode_synth.py --base outputs/synth/si_R2 --gt-vmd outputs/gt/si_R2_safe.vmd
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FFMPEG = os.environ.get("FFMPEG", r"D:\ffmpeg\bin\ffmpeg.exe")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--gt-vmd", default=None)
    ap.add_argument("--views", default="front,three_quarter,side")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    base = os.path.abspath(args.base)
    manifest = {
        "base": base,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "fps": args.fps,
        "views": {},
    }
    if args.gt_vmd:
        manifest["gt_vmd"] = os.path.abspath(args.gt_vmd)
        manifest["gt_vmd_sha256"] = sha256(args.gt_vmd)

    for view in [v for v in args.views.split(",") if v]:
        vdir = os.path.join(base, view)
        frames = os.path.join(vdir, "frames")
        meta_path = os.path.join(vdir, "meta.json")
        if not os.path.isdir(frames):
            print("skip", view, "(no frames)")
            continue
        pngs = sorted(f for f in os.listdir(frames) if f.endswith(".png"))
        if not pngs:
            print("skip", view, "(empty)")
            continue
        video = os.path.join(vdir, f"gt_{view}.mp4")
        cmd = [FFMPEG, "-y", "-loglevel", "error", "-framerate", str(args.fps),
               "-i", os.path.join(frames, "frame_%04d.png"),
               "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", video]
        subprocess.run(cmd, check=True)
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        manifest["views"][view] = {
            "frames": len(pngs),
            "video": video,
            "video_bytes": os.path.getsize(video),
            "meta": meta_path,
            "camera": meta.get("camera"),
            "hidden_collections": meta.get("hidden_collections"),
            "action": meta.get("action"),
        }
        print("encoded", video, len(pngs), "frames")

    out = os.path.join(base, "manifest.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print("WROTE", out)


if __name__ == "__main__":
    main()
