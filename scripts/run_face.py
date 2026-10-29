"""Run MediaPipe Face Landmarker over a 30 fps clip and save semantic weights.

Outputs:
  outputs/m3/<name>/face.npz   weights (N,12) semantic, level (N,),
                               detected (N,), bbox (N,4), geom (N,3) EAR/MAR,
                               patch_ok (N,), bbox_ok (N,),
                               blendshapes (N,52), lm (N,478,2)
  outputs/m3/<name>/face.json  detection/level summary

The semantic channels are defined in ``src/vtv/face.py``; the per-model morph
alias mapping happens in ``assemble_handface_vmd.py`` so the same face evidence
can drive si / luoxi / tiflos with their own morph names.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mediapipe as mp  # noqa: E402
from mediapipe.tasks.python import vision  # noqa: E402

from vtv.face import (  # noqa: E402
    SEMANTIC, build_face_track, face_geometry,
)

FACE_SCALES = (1.4, 2.0, 2.8, 3.8, 5.0)
BS_NAMES = None  # filled from the first detection


def _mp_path(p: str) -> str:
    try:
        return os.path.relpath(os.path.abspath(p), os.getcwd())
    except ValueError:
        return p


def _smooth_xy(xy: np.ndarray, k: int = 7) -> np.ndarray:
    out = np.array(xy, float)
    for i in range(len(xy)):
        lo, hi = max(0, i - k // 2), min(len(xy), i + k // 2 + 1)
        out[i] = np.median(xy[lo:hi], axis=0)
    return out


def _crop(img, cx, cy, half):
    h, w = img.shape[:2]
    x0, y0 = max(0, int(cx - half)), max(0, int(cy - half))
    x1, y1 = min(w, int(cx + half)), min(h, int(cy + half))
    if x1 - x0 < 32 or y1 - y0 < 32:
        return None, (0, 0, 0, 0)
    return img[y0:y1, x0:x1], (x0, y0, x1, y1)


def _patch_stats(gray, x, y, radius):
    h, w = gray.shape
    x0, x1 = max(0, int(x - radius)), min(w, int(x + radius) + 1)
    y0, y1 = max(0, int(y - radius)), min(h, int(y + radius) + 1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    p = gray[y0:y1, x0:x1]
    return float(p.mean()), float(p.std())


def _patch_ok(gray, lm, eye_ids, cheek_id):
    """Eyes are dark/high-contrast on this character; a mesh on hair is not."""
    a, b = eye_ids
    cx = 0.5 * (lm[a, 0] + lm[b, 0])
    cy = 0.5 * (lm[a, 1] + lm[b, 1])
    r = max(3.0, 0.28 * float(np.linalg.norm(lm[a] - lm[b])))
    eye = _patch_stats(gray, cx, cy, r)
    cheek = _patch_stats(gray, lm[cheek_id, 0], lm[cheek_id, 1], r)
    if eye is None or cheek is None:
        return False, {}
    darker = eye[0] < cheek[0] + 8.0
    contrast = eye[1] > 18.0
    return bool(darker or contrast), {"eye_mean": eye[0], "eye_std": eye[1],
                                      "cheek_mean": cheek[0], "cheek_std": cheek[1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--kp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--face-model", default=str(ROOT / "models/mediapipe/face_landmarker.task"))
    ap.add_argument("--frames", default=None)
    ap.add_argument("--input-size", type=int, default=512)
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--overlays", type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    kp = torch.load(args.kp, map_location="cpu").numpy()
    cap = cv2.VideoCapture(args.video)
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    n_frames = min(len(kp), n_video) if n_video > 0 else len(kp)
    if args.max_frames > 0:
        n_frames = min(n_frames, args.max_frames)
    start, end = 0, n_frames
    if args.frames:
        start, end = [int(x) for x in args.frames.split(":")]
        end = min(end, n_frames)
    span = end - start
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    opts = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=_mp_path(args.face_model)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=args.conf,
        min_face_presence_confidence=args.conf,
        output_face_blendshapes=True,
    )
    det = vision.FaceLandmarker.create_from_options(opts)

    face_pts = _smooth_xy(kp[:, :5, :2])
    records = [None] * span
    lm_arr = np.full((span, 478, 2), np.nan, np.float32)
    bs_arr = np.zeros((span, 52), np.float32)
    geom_arr = np.zeros((span, 3), np.float32)
    bbox_arr = np.zeros((span, 4), np.float32)
    patch_arr = np.zeros(span, bool)
    bbox_ok_arr = np.zeros(span, bool)
    bs_names = None
    t_total = 0.0
    n_calls = 0
    n_det = 0

    for fi in range(start, end):
        ok, img = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        span_px = float(max(np.ptp(face_pts[fi, :, 0]), np.ptp(face_pts[fi, :, 1])))
        cx, cy = face_pts[fi].mean(0)
        best = None
        best_key = None
        for fac in FACE_SCALES:
            half = max(60.0, span_px * fac * 0.5)
            crop, box = _crop(img, cx, cy, half)
            if crop is None:
                continue
            up = cv2.resize(crop, (args.input_size, args.input_size),
                            interpolation=cv2.INTER_CUBIC)
            t0 = time.perf_counter()
            res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                      data=cv2.cvtColor(up, cv2.COLOR_BGR2RGB)))
            dt = time.perf_counter() - t0
            t_total += dt
            n_calls += 1
            if not res.face_landmarks:
                continue
            x0, y0, x1, y1 = box
            lm = np.array([[p.x * (x1 - x0) + x0, p.y * (y1 - y0) + y0]
                           for p in res.face_landmarks[0]], float)
            bs = {b.category_name: float(b.score) for b in res.face_blendshapes[0]}
            bw = float(np.ptp(lm[:, 0]))
            bh = float(np.ptp(lm[:, 1]))
            # plausible face box inside the crop and not a hair blob
            ratio = bw / max(bh, 1e-6)
            fill = max(bw, bh) / max(x1 - x0, y1 - y0)
            bbox_ok = (0.55 <= ratio <= 1.5 and 0.30 <= fill <= 1.0
                       and bw >= 40 and bh >= 40)
            ok_l, stats_l = _patch_ok(gray, lm, (33, 133), 234)
            ok_r, stats_r = _patch_ok(gray, lm, (362, 263), 454)
            patch_ok = bool(ok_l and ok_r)
            geom = face_geometry(lm)
            cand = {
                "lm": lm, "blendshapes": bs, "geom": geom,
                "bbox": np.array([x0, y0, x1, y1], float),
                "bbox_ok": bool(bbox_ok), "patch_ok": patch_ok,
                "patch": {"l": stats_l, "r": stats_r},
                "bbox_wh": (bw, bh), "fac": fac, "dt": dt,
            }
            key = (1 if patch_ok else 0, 1 if bbox_ok else 0, bw * bh)
            if best_key is None or key > best_key:
                best, best_key = cand, key
        if best is not None:
            li = fi - start
            n_det += 1
            records[li] = best
            lm_arr[li] = best["lm"]
            if bs_names is None:
                bs_names = sorted(best["blendshapes"])
            for j, name in enumerate(bs_names):
                bs_arr[li, j] = best["blendshapes"].get(name, 0.0)
            geom_arr[li] = [best["geom"].get("ear_l", 0.0),
                            best["geom"].get("ear_r", 0.0),
                            best["geom"].get("mar", 0.0)]
            bbox_arr[li] = best["bbox"]
            patch_arr[li] = best["patch_ok"]
            bbox_ok_arr[li] = best["bbox_ok"]
        if (fi - start) % 40 == 0:
            print(f"  frame {fi}/{end}", flush=True)

    cap.release()
    print(f"face inference: {n_calls} crops, {1000 * t_total / max(n_calls, 1):.1f} ms/crop")

    weights, levels, diag = build_face_track(records, span)
    # L2 = head/wrist visible but no trustworthy face; L3 = not visible at all.
    face_vis = kp[start:end, :5, 2].mean(axis=1) >= 0.40
    for i in range(span):
        if levels[i] == 3 and face_vis[i]:
            levels[i] = 2
    diag["level_counts"] = {int(k): int((levels == k).sum()) for k in (0, 1, 2, 3)}
    diag["face_visible_rate"] = float(face_vis.mean())
    diag["semantic"] = list(SEMANTIC)
    diag["blendshape_names"] = bs_names
    diag["detect_count"] = n_det
    diag["detect_rate"] = n_det / max(span, 1)
    diag["patch_ok_rate_detected"] = float(patch_arr.sum() / max(n_det, 1))
    diag["runtime"] = {"crops": n_calls, "ms_per_crop": 1000 * t_total / max(n_calls, 1)}

    np.savez_compressed(
        out_dir / "face.npz",
        weights=weights.astype(np.float32),
        level=levels.astype(np.int8),
        detected=np.array([r is not None for r in records], bool),
        patch_ok=patch_arr,
        bbox=bbox_arr.astype(np.float32),
        bbox_ok=bbox_ok_arr,
        geom=geom_arr.astype(np.float32),
        blendshapes=bs_arr,
        blendshape_names=np.array(bs_names or [], dtype=object),
        lm=lm_arr.astype(np.float32),
        frames=np.array([start, end]),
    )
    with open(out_dir / "face.json", "w", encoding="utf-8") as f:
        json.dump(diag, f, ensure_ascii=False, indent=2, default=lambda o: float(o))

    # overlays of accepted / rejected detections
    saved = 0
    cap = cv2.VideoCapture(args.video)
    for fi in range(start, end):
        rec = records[fi - start]
        if not rec:
            continue
        if rec["patch_ok"] is False and saved >= 3:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, img = cap.read()
        if not ok:
            continue
        vis = img.copy()
        for (x, y) in rec["lm"][::4]:
            cv2.circle(vis, (int(x), int(y)), 1, (0, 0, 255), -1)
        x0, y0, x1, y1 = [int(v) for v in rec["bbox"]]
        col = (0, 255, 0) if rec["patch_ok"] else (0, 165, 255)
        cv2.rectangle(vis, (x0, y0), (x1, y1), col, 2)
        cv2.putText(vis, f"f{fi} {'ok' if rec['patch_ok'] else 'weak'} fac={rec['fac']}",
                    (max(4, x0), max(18, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        cv2.imwrite(os.path.relpath(out_dir / f"face_f{fi:03d}.png", os.getcwd()), vis)
        saved += 1
        if saved >= args.overlays:
            break
    cap.release()

    print(json.dumps({k: diag[k] for k in ("level_counts", "detect_count", "detect_rate",
                                           "patch_ok_rate_detected", "unreliable",
                                           "procedural_blink_frames", "face_visible_rate")},
                     indent=2, ensure_ascii=False))
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
