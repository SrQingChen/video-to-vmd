"""Render the M3.5 clip grid: 3 motions x 3 models x 3 fixed cameras.

One render per clip from the M3.5 GT VMD (body + fingers + face morphs), at a
fixed camera with the MMD inheritance bake certificate checked before the clip
is accepted.  The script is resumable: a clip with the full frame count and an
mp4 is skipped.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLENDER = "D:/blender/blender.exe"
BLEND = "E:/zcodeAI项目/AIblender/三人_王牌竞速_20261024.blend"
FFMPEG = "D:/ffmpeg/bin/ffmpeg.exe"
MOTIONS = ("R1", "R2", "R3")
MODELS = ("si", "luoxi", "tiflos")
VIEWS = ("front", "three_quarter", "side")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/m35/render"))
    ap.add_argument("--res", default="1080x1920")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--motions", default=",".join(MOTIONS))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--views", default=",".join(VIEWS))
    ap.add_argument("--shot", default="full", choices=["full", "medium", "face"],
                    help="full body, the fixed waist-up shot, or a face close-up")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--retries", type=int, default=2,
                    help="retry a crashed Blender render this many times")
    args = ap.parse_args()

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {"clips": {}, "res": args.res, "samples": args.samples}
    jobs = [(m, mo, v) for m in args.motions.split(",")
            for mo in args.models.split(",") for v in args.views.split(",")]
    for i, (motion, model, view) in enumerate(jobs, 1):
        gt = ROOT / f"outputs/m35/gt/{motion}/{model}.vmd"
        out = out_root / f"{motion}_{model}_{view}"
        n = None
        if not gt.exists():
            print(f"[{i}/{len(jobs)}] SKIP missing GT {gt}", flush=True)
            continue
        # frame count from the GT VMD header (fast parse of max frame)
        import struct
        d = gt.read_bytes()
        o = 30 + 20
        (nb,) = struct.unpack_from("<I", d, o)
        o += 4
        mx = 0
        for _ in range(nb):
            (f,) = struct.unpack_from("<I", d, o + 15)
            mx = max(mx, f)
            o += 111
        n = mx + 1
        frames = out / "frames"
        mp4 = out / f"{motion}_{model}_{view}.mp4"
        have = len(list(frames.glob("frame_*.png"))) if frames.exists() else 0
        if not args.force and have == n and mp4.exists():
            manifest["clips"][f"{motion}_{model}_{view}"] = {
                "frames": n, "dir": str(out.relative_to(ROOT)),
                "mp4": str(mp4.relative_to(ROOT)), "cached": True}
            print(f"[{i}/{len(jobs)}] cached {out.name} ({n})", flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        cmd = [BLENDER, "--background", BLEND, "--python",
               str(ROOT / "scripts/render_synth.py"), "--",
               "--model", model, "--vmd", str(gt), "--view", view,
               "--shot", args.shot, "--out", str(out),
               "--frames", f"1:{n}", "--res", args.res,
               "--samples", str(args.samples)]
        accepted = False
        for attempt in range(args.retries + 1):
            if attempt:
                shutil.rmtree(frames, ignore_errors=True)
                print(f"[{i}/{len(jobs)}] RETRY {out.name} "
                      f"({attempt}/{args.retries})", flush=True)
                time.sleep(3)
            print(f"[{i}/{len(jobs)}] RENDER {out.name} frames=1:{n}", flush=True)
            log = out / "render.log"
            with open(log, "w", encoding="utf-8") as f:
                r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
            meta_p = out / "meta.json"
            ok = r.returncode == 0 and meta_p.exists()
            meta = (json.loads(meta_p.read_text(encoding="utf-8"))
                    if meta_p.exists() else {})
            bake = meta.get("inherit_bake") or {}
            d_ok = bool((bake.get("d_chain_vs_fk") or {}).get("ok"))
            v_ok = bool((bake.get("verify") or {}).get("ok"))
            count = len(list(frames.glob("frame_*.png")))
            accepted = ok and d_ok and v_ok and count == n
            if accepted:
                break
        if accepted:
            subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", "30",
                            "-start_number", "1", "-i", str(frames / "frame_%04d.png"),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                            str(mp4)], check=True)
        manifest["clips"][f"{motion}_{model}_{view}"] = {
            "frames": n, "rendered": count, "ok": accepted,
            "d_chain_ok": d_ok, "verify_ok": v_ok,
            "verify_max": (bake.get("verify") or {}).get("max_head_error"),
            "dir": str(out.relative_to(ROOT)),
            "mp4": str(mp4.relative_to(ROOT)),
            "camera": meta.get("camera"), "resolution": meta.get("resolution"),
            "vmd": str(gt.relative_to(ROOT)),
        }
        (out_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{i}/{len(jobs)}] {'OK' if accepted else 'FAIL'} {out.name} "
              f"frames={count}/{n} dchain={d_ok} verify={v_ok}", flush=True)
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    n_ok = sum(1 for v in manifest["clips"].values() if v.get("ok") or v.get("cached"))
    print(f"RENDER_ALL_DONE {n_ok}/{len(manifest['clips'])}", flush=True)
    failed = [k for k, v in manifest["clips"].items()
              if not (v.get("ok") or v.get("cached"))]
    if failed:
        print("FAILED", failed, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
