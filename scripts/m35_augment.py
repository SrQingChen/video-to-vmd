"""Observation-side augmentation for M3.5 (never touches the GT motion).

Two deterministic variants per clip:

* ``occ``  - 1-2 solid occluder rectangles over a body region for 0.5-1.5 s,
             plus mild sensor noise.
* ``mix``  - 1-2 frame-freeze runs (0.3-1.5 s, same frame count), a 0.65x
             downscale/upscale and strong compression noise.

Frame count and 30 fps are preserved, so GVHMR output stays frame-aligned with
the GT.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def occluders(rng, w, h):
    out = []
    for _ in range(rng.integers(1, 3)):
        bw = int(rng.uniform(0.22, 0.48) * w)
        bh = int(rng.uniform(0.18, 0.42) * h)
        cx = int(rng.uniform(0.25, 0.75) * w)
        cy = int(rng.uniform(0.25, 0.75) * h)
        out.append((cx - bw // 2, cy - bh // 2, bw, bh))
    return out


def intervals(rng, n, count, lo, hi):
    out = []
    for _ in range(count):
        length = int(rng.integers(lo, hi + 1))
        start = int(rng.integers(0, max(1, n - length)))
        out.append((start, min(n - 1, start + length)))
    return out


def augment(src, dst, variant, seed):
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {src}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))
    occ = intervals(rng, n, int(rng.integers(1, 3)), 15, 45) if variant == "occ" else []
    freeze = intervals(rng, n, int(rng.integers(1, 3)), 10, 45) if variant == "mix" else []
    boxes = occluders(rng, w, h)
    frozen = {}
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for a, b in freeze:
            if a <= i <= b:
                if a not in frozen:
                    frozen[a] = frame.copy()
                frame = frozen[a].copy()
                break
        if variant == "mix":
            small = cv2.resize(frame, (max(1, int(w * 0.65)), max(1, int(h * 0.65))),
                               interpolation=cv2.INTER_AREA)
            frame = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
            okj, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 38])
            if okj:
                frame = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            noise = rng.normal(0, 5.0, frame.shape)
            frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        elif variant == "lowres":
            # M6: harder resolution/compression stress than ``mix``; the
            # frame count and timing are unchanged so the GT stays aligned.
            small = cv2.resize(frame, (max(1, int(w * 0.45)), max(1, int(h * 0.45))),
                               interpolation=cv2.INTER_AREA)
            frame = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
            okj, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 28])
            if okj:
                frame = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            noise = rng.normal(0, 7.0, frame.shape)
            frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        elif variant == "blur":
            # M6: optical blur + compression (occlusion/truncation stress is
            # already covered by ``occ``/``mix``).
            frame = cv2.GaussianBlur(frame, (5, 5), 1.2)
            okj, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 35])
            if okj:
                frame = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            noise = rng.normal(0, 4.0, frame.shape)
            frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        elif variant == "occ":
            for a, b in occ:
                if a <= i <= b:
                    for (x, y, bw, bh) in boxes:
                        cv2.rectangle(frame, (x, y), (x + bw, y + bh),
                                      (38, 36, 34), -1)
                    break
            noise = rng.normal(0, 3.0, frame.shape)
            frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)
        i += 1
    cap.release()
    writer.release()
    return {"src": str(src), "dst": str(dst), "variant": variant, "frames": i,
            "fps": float(fps), "size": [w, h],
            "occlusions": occ, "freezes": freeze, "seed": int(seed)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", default=str(ROOT / "outputs/m35/render"))
    ap.add_argument("--out", default=str(ROOT / "outputs/m35/aug"))
    ap.add_argument("--clips", default=None,
                    help="comma separated clip ids (default: train clips only)")
    ap.add_argument("--variants", default="occ,mix")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from vtv.feedback_data import TRAIN_MOTIONS, TRAIN_MODELS, TRAIN_VIEWS, assign_split

    if args.clips:
        clips = args.clips.split(",")
    else:
        clips = [f"{m}_{mo}_{v}" for m in TRAIN_MOTIONS for mo in TRAIN_MODELS
                 for v in TRAIN_VIEWS if assign_split(m, mo, v) == "train"]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {"variants": {}, "clips": {}}
    prev_p = out_root / "manifest.json"
    if prev_p.exists():
        try:
            prev = json.loads(prev_p.read_text(encoding="utf-8"))
            manifest["clips"].update(prev.get("clips", {}))
        except Exception:
            pass
    for cid in clips:
        src = Path(args.render) / cid / f"{cid}.mp4"
        if not src.exists():
            print("MISSING", src)
            continue
        for k, variant in enumerate(args.variants.split(",")):
            name = f"{cid}__{variant}"
            dst = out_root / f"{name}.mp4"
            seed = abs(hash(cid)) % (2 ** 31) + 7919 * (k + 1)
            if dst.exists():
                cap = cv2.VideoCapture(str(dst))
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                manifest["clips"][name] = {
                    "cached": True, "dst": str(dst), "variant": variant,
                    "frames": n, "size": [w, h]}
                print("cached", name, n)
                continue
            rec = augment(src, dst, variant, seed)
            manifest["clips"][name] = rec
            print("AUG", name, rec["frames"], variant)
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print("AUG_DONE", len(manifest["clips"]))


if __name__ == "__main__":
    main()
