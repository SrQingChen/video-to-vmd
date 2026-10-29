"""Build a long single-camera test video by concatenating short clips.

Used by M5 to exercise the chunked pipeline without shipping any restricted
media: the inputs are the project's own T1/T2/T5 test clips (not in the public
repo), or any user-provided clips with the same resolution/fps.

    python scripts/make_long_video.py \
        --inputs testdata/T5_.../001.mp4,testdata/T2_.../0.5.mp4,... \
        --out outputs/m5/long/long_80s.mp4

The output is 30 fps (already the pipeline's native rate, so no second
resample), 1080x1920 by default, and a manifest with the source hash, per-clip
frames and the total duration is written next to it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vtv.longvideo import video_fps, video_frames  # noqa: E402


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description="concatenate clips into one long video")
    ap.add_argument("--inputs", required=True,
                    help="comma separated input videos, played in order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--res", default=None,
                    help="WxH (default: first input; other inputs are scaled)")
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--ffmpeg", default="ffmpeg")
    args = ap.parse_args(argv)

    inputs = [Path(p.strip()) for p in args.inputs.split(",") if p.strip()]
    for p in inputs:
        if not p.exists():
            raise SystemExit(f"input not found: {p}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.res:
        w, h = [int(x) for x in args.res.lower().split("x")]
    else:
        import cv2
        cap = cv2.VideoCapture(str(inputs[0]))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

    parts = []
    chains = []
    for i, p in enumerate(inputs):
        parts += ["-i", str(p)]
        chains.append(
            f"[{i}:v]fps={args.fps},scale={w}:{h}:force_original_aspect_ratio"
            f"=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}]")
    concat = "".join(f"[v{i}]" for i in range(len(inputs)))
    filt = ";".join(chains) + f";{concat}concat=n={len(inputs)}:v=1:a=0[v]"
    cmd = [args.ffmpeg, "-y", "-loglevel", "error", *parts,
           "-filter_complex", filt, "-map", "[v]", "-an",
           "-c:v", "libx264", "-crf", str(args.crf),
           "-pix_fmt", "yuv420p", "-r", str(args.fps), str(out)]
    print("CMD", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"ffmpeg failed ({r.returncode})")

    n = video_frames(out)
    manifest = {
        "out": str(out),
        "out_frames": n,
        "out_fps": video_fps(out),
        "duration_seconds": round(n / float(args.fps), 3),
        "resolution": [w, h],
        "sha256": sha256_file(out),
        "inputs": [{"path": str(p), "frames": video_frames(p),
                    "fps": video_fps(p), "sha256": sha256_file(p)}
                   for p in inputs],
        "note": ("30 fps 单机位长片，用于 M5 分段/合并验收；素材不入库，"
                 "可用任意同分辨率片段替换"),
    }
    (out.with_suffix(".json")).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("out", "out_frames",
                                               "duration_seconds")},
                     ensure_ascii=False, indent=1))
    print("manifest:", out.with_suffix(".json"))


if __name__ == "__main__":
    main()
