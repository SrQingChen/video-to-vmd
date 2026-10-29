"""Time-aligned side-by-side comparison: source video vs VMD render.

M1's ``cmp_121`` compared source frame 121 with VMD frame 121, but the VMDs
are 30 fps while every source clip is 60 fps *and* the T1/T2 clips were
retimed by ``slow_factor``.  The correct mapping is

    source_frame = round(vmd_frame * source_fps / (30 * slow_factor))

    T5: x2   (VMD 121 <-> src 242)
    T2: x4   (VMD 121 <-> src 484)
    T1: x8   (VMD 121 <-> src 968)

Usage:
  python scripts/compare_aligned.py \
      --video testdata/T5_高速舞蹈_001/001.mp4 \
      --render-dir outputs/t5_m1/front_dchain \
      --slow-factor 1.0 --vmd-frames 0,60,120,180,240 \
      --out outputs/t5_m1/aligned

The output images are labelled ``VMD f<i> / src f<j>`` plus the wall-clock
time, and a JSON records the exact mapping that was used.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VMD_FPS = 30.0
FONT_CANDIDATES = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
]


def load_font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit("cannot open video: " + path)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": fps, "frames": n, "width": w, "height": h,
            "duration": n / fps if fps else 0.0}


def read_video_frame(path, index):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit("cannot open video: " + path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def read_render_frame(render_dir, vmd_frame):
    path = os.path.join(render_dir, "frames", "frame_%04d.png" % (vmd_frame + 1))
    if not os.path.exists(path):
        return None, path
    return Image.open(path).convert("RGB"), path


def compose(source_img, render_img, title, left_label, right_label,
            header_h):
    h = render_img.height
    scale = h / source_img.height
    sw = int(round(source_img.width * scale))
    src = source_img.resize((sw, h), Image.LANCZOS)
    gap = 8
    sheet = Image.new("RGB", (sw + gap + render_img.width, h + header_h),
                      (18, 18, 20))
    sheet.paste(src, (0, header_h))
    sheet.paste(render_img, (sw + gap, header_h))
    d = ImageDraw.Draw(sheet)
    font = load_font(max(14, header_h // 2))
    small = load_font(max(12, header_h // 3))
    d.text((10, 6), title, fill=(240, 240, 240), font=font)
    d.text((10, header_h + 6), left_label, fill=(255, 220, 120), font=small)
    d.text((sw + gap + 10, header_h + 6), right_label,
           fill=(120, 220, 255), font=small)
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--render-dir", required=True)
    ap.add_argument("--slow-factor", type=float, default=1.0,
                    help="must match the adapter --slow-factor used for the VMD")
    ap.add_argument("--time-scale", type=float, default=None,
                    help="override source_frames per VMD frame")
    ap.add_argument("--vmd-frames", default="0,60,120,180,240")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sheet", default=None,
                    help="optional contact-sheet path (default <out>/sheet.png)")
    ap.add_argument("--header", type=int, default=64)
    args = ap.parse_args()

    info = video_info(args.video)
    scale = args.time_scale
    if scale is None:
        scale = info["fps"] / (VMD_FPS * args.slow_factor)
    vmd_frames = [int(x) for x in args.vmd_frames.split(",") if x.strip()]
    os.makedirs(args.out, exist_ok=True)

    rows = []
    tiles = []
    for vf in vmd_frames:
        sf = int(round(vf * scale))
        clamped = min(max(sf, 0), info["frames"] - 1)
        src_rgb = read_video_frame(args.video, clamped)
        render, rpath = read_render_frame(args.render_dir, vf)
        if src_rgb is None or render is None:
            rows.append({"vmd_frame": vf, "source_frame": clamped,
                         "missing": "source" if src_rgb is None else rpath})
            print("MISSING", vf, "src_frame", clamped,
                  "render" if src_rgb is not None else "source")
            continue
        t = vf / VMD_FPS
        title = ("VMD f%d  <->  SRC f%d   (x%.3g, %.0f fps, t=%.3fs)"
                 % (vf, clamped, scale, info["fps"], t))
        left = "SOURCE  f%d   t=%.3fs" % (clamped, clamped / info["fps"])
        right = "VMD RENDER  f%d  (blender %d)" % (vf, vf + 1)
        img = compose(Image.fromarray(src_rgb), render, title, left, right,
                      args.header)
        path = os.path.join(args.out, "aligned_vmd%04d_src%04d.png" % (vf, clamped))
        img.save(path)
        tiles.append(img)
        rows.append({"vmd_frame": vf, "source_frame": clamped,
                     "source_time": clamped / info["fps"], "vmd_time": t,
                     "image": os.path.abspath(path),
                     "source_frame_requested": sf,
                     "clamped": clamped != sf})
        print("WROTE", path)

    sheet_path = args.sheet or os.path.join(args.out, "sheet.png")
    if tiles:
        w = max(t.width for t in tiles)
        total = sum(t.height for t in tiles) + 6 * (len(tiles) - 1)
        sheet = Image.new("RGB", (w, total), (18, 18, 20))
        y = 0
        for t in tiles:
            sheet.paste(t, (0, y))
            y += t.height + 6
        sheet.save(sheet_path)
        print("WROTE", sheet_path)

    report = {
        "video": os.path.abspath(args.video),
        "video_info": info,
        "render_dir": os.path.abspath(args.render_dir),
        "vmd_fps": VMD_FPS,
        "slow_factor": args.slow_factor,
        "time_scale": scale,
        "formula": "source_frame = round(vmd_frame * source_fps / (30 * slow_factor))",
        "warning": ("M1's cmp_121 compared source 121 with VMD 121; that is "
                    "off by the time_scale factor and must not be used."),
        "pairs": rows,
    }
    with open(os.path.join(args.out, "mapping.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("ALIGNED", json.dumps({k: report[k] for k in
                                 ("time_scale", "slow_factor")}),
          "pairs", len(rows))


if __name__ == "__main__":
    main()
