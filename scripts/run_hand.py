"""Run MediaPipe hands over a 30 fps clip and save anatomical angles (M3).

The heavy lifting (level assignment, smoothing, limits) lives in
``src/vtv/hands.py``; this script only produces per-frame evidence:

  outputs/m3/<name>/hand.npz   angles (N,2,19), level (N,2), conf (N,2),
                               detected (N,2), lm2d (N,2,21,2),
                               world3d (N,2,21,3), score (N,2), bbox (N,2,4),
                               wrist_err (N,2), fails (N,2)
  outputs/m3/<name>/hand.json  summary + per-segment diagnostics

Side convention: 'l'/'r' are the *character's* sides, taken from COCO wrist
keypoints 9/10.  MediaPipe's handedness label is recorded but never used for
assignment (it is unreliable on anime hands and mirrored crops).
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

from vtv.hands import (  # noqa: E402
    JOINT_KEYS, LEVEL_CONF, LEVEL_L0, LEVEL_L1, LEVEL_L2, LEVEL_L3,
    HandTrackConfig, assign_levels, build_angle_track, fit_l1_gain,
    hand_plausibility, mp_joint_angles,
)

COCO_WRIST = {"l": 9, "r": 10}
COCO_ELBOW = {"l": 7, "r": 8}
CROP_SPECS = ((0.35, 1.10), (0.60, 1.50), (0.20, 2.05))


def _mp_path(p: str) -> str:
    """MediaPipe's C layer cannot open Windows paths with non-ASCII chars."""
    try:
        return os.path.relpath(os.path.abspath(p), os.getcwd())
    except ValueError:
        return p


def _smooth_xy(xy: np.ndarray, k: int = 5) -> np.ndarray:
    out = np.array(xy, float)
    for i in range(len(xy)):
        lo, hi = max(0, i - k // 2), min(len(xy), i + k // 2 + 1)
        out[i] = np.median(xy[lo:hi], axis=0)
    return out


def _crop(img, cx, cy, half):
    h, w = img.shape[:2]
    x0, y0 = max(0, int(cx - half)), max(0, int(cy - half))
    x1, y1 = min(w, int(cx + half)), min(h, int(cy + half))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None, (0, 0, 0, 0)
    return img[y0:y1, x0:x1], (x0, y0, x1, y1)


def _draw_overlay(img, rec, side, frame, path):
    vis = img.copy()
    x0, y0, x1, y1 = [int(v) for v in rec["bbox"]]
    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
    if rec["lm2d"] is not None:
        for (x, y) in rec["lm2d"]:
            cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)
    cv2.putText(vis, f"f{frame}{side} L{rec['level']} s={rec['score']:.2f}",
                (max(4, x0), max(18, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2)
    cv2.imwrite(path, vis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="VMD-aligned 30 fps mp4")
    ap.add_argument("--kp", required=True, help="ViTPose .pt (N,17,3) in source pixels")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--hand-model", default=str(ROOT / "models/mediapipe/hand_landmarker.task"))
    ap.add_argument("--frames", default=None, help="a:b frame range (VMD frames)")
    ap.add_argument("--input-size", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.12)
    ap.add_argument("--overlays", type=int, default=12)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    kp = torch.load(args.kp, map_location="cpu").numpy()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    n_frames = min(len(kp), n_video) if n_video > 0 else len(kp)
    if args.max_frames > 0:
        n_frames = min(n_frames, args.max_frames)
    start, end = 0, n_frames
    if args.frames:
        start, end = [int(x) for x in args.frames.split(":")]
        end = min(end, n_frames)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    opts = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=_mp_path(args.hand_model)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=args.conf,
        min_hand_presence_confidence=args.conf,
        min_tracking_confidence=args.conf,
    )
    det = vision.HandLandmarker.create_from_options(opts)

    wrist_xy = {s: _smooth_xy(kp[:, COCO_WRIST[s], :2]) for s in ("l", "r")}
    elbow_xy = {s: _smooth_xy(kp[:, COCO_ELBOW[s], :2]) for s in ("l", "r")}
    wrist_conf = {s: kp[:, COCO_WRIST[s], 2] for s in ("l", "r")}

    span = end - start
    records = {s: [None] * span for s in ("l", "r")}
    raw_arrays = {s: {"lm2d": np.full((span, 21, 2), np.nan, np.float32),
                     "world3d": np.full((span, 21, 3), np.nan, np.float32),
                     "score": np.zeros(span, np.float32),
                     "valid": np.zeros(span, bool),
                     "bbox": np.zeros((span, 4), np.float32),
                     "wrist_err": np.zeros(span, np.float32),
                     "fails": np.zeros(span, np.int16)} for s in ("l", "r")}
    labels = {s: [] for s in ("l", "r")}
    t_total = 0.0
    n_calls = 0

    for fi in range(start, end):
        ok, img = cap.read()
        if not ok:
            break
        for side in ("l", "r"):
            wx, wy = wrist_xy[side][fi]
            ex, ey = elbow_xy[side][fi]
            fore = float(np.hypot(wx - ex, wy - ey))
            palm = max(16.0, 0.34 * fore)
            if fore > 1e-6:
                ux, uy = (wx - ex) / fore, (wy - ey) / fore
            else:
                ux, uy = 0.0, -1.0
            best = None
            best_key = None
            for off, hf in CROP_SPECS:
                cx, cy = wx + ux * palm * off, wy + uy * palm * off
                crop, box = _crop(img, cx, cy, palm * hf)
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
                if not res.hand_landmarks:
                    continue
                x0, y0, x1, y1 = box
                lm = np.array([[p.x * (x1 - x0) + x0, p.y * (y1 - y0) + y0]
                               for p in res.hand_landmarks[0]], float)
                world = np.array([[p.x, p.y, p.z] for p in res.hand_world_landmarks[0]], float)
                score = float(res.handedness[0][0].score)
                label = res.handedness[0][0].category_name
                valid, fails, metrics = hand_plausibility(lm, world, (wx, wy), palm)
                # Body-wrist disambiguation: if the detected wrist is clearly
                # closer to the *other* body wrist, this crop caught the wrong
                # hand (hands crossing / close together).
                other = "r" if side == "l" else "l"
                ox, oy = wrist_xy[other][fi]
                d_self = float(np.hypot(lm[0, 0] - wx, lm[0, 1] - wy))
                d_other = float(np.hypot(lm[0, 0] - ox, lm[0, 1] - oy))
                side_ok = d_other > d_self - 0.5 * palm
                metrics["side_ok"] = bool(side_ok)
                metrics["d_self_px"] = d_self
                metrics["d_other_px"] = d_other
                if not side_ok:
                    valid = False
                    fails += 1
                cand = {
                    "lm2d": lm, "world3d": world, "score": score, "label": label,
                    "palm_px": float(np.linalg.norm(lm[9] - lm[0])),
                    "bbox": np.array(box, float), "valid": bool(valid),
                    "fails": int(fails), "metrics": metrics,
                    "angles2d": mp_joint_angles(lm),
                    "angles3d": mp_joint_angles(world) if metrics.get("world_ok") else None,
                    "world_ok": bool(metrics.get("world_ok")),
                    "side_ok": bool(side_ok),
                    "wrist_conf": float(wrist_conf[side][fi]),
                    "crop": (off, hf),
                }
                key = (1 if valid else 0, -fails, score)
                if best_key is None or key > best_key:
                    best, best_key = cand, key
            if best is not None:
                li = fi - start
                records[side][li] = best
                arr = raw_arrays[side]
                arr["lm2d"][li] = best["lm2d"]
                arr["world3d"][li] = best["world3d"]
                arr["score"][li] = best["score"]
                arr["valid"][li] = best["valid"]
                arr["bbox"][li] = best["bbox"]
                arr["wrist_err"][li] = best["metrics"].get("wrist_err_px", np.nan)
                arr["fails"][li] = best["fails"]
                labels[side].append(best["label"])
        if (fi - start) % 40 == 0:
            print(f"  frame {fi}/{end}", flush=True)

    cap.release()
    print(f"hand inference: {n_calls} crops, {1000 * t_total / max(n_calls, 1):.1f} ms/crop")

    # ---- tracks ---------------------------------------------------------
    cfg = HandTrackConfig()
    angle_tracks = {}
    level_tracks = {}
    diagnostics = {}
    gains = {}
    for side in ("l", "r"):
        recs = records[side]
        gain, gain_info = fit_l1_gain(recs, cfg.l1_gain)
        gains[side] = gain_info
        cfg_side = HandTrackConfig(**{**cfg.__dict__, "l1_gain": gain})
        levels, segments, diag = assign_levels(recs, wrist_conf[side][start:end], cfg_side)
        angles, target = build_angle_track(recs, levels, cfg_side)
        angle_tracks[side] = angles
        level_tracks[side] = levels
        diag["fit_l1_gain"] = gain_info
        diag["level_counts"] = {int(k): int((levels == k).sum()) for k in (0, 1, 2, 3)}
        diag["detect_count"] = int(sum(1 for r in recs if r))
        diag["valid_count"] = int(sum(1 for r in recs if r and r["valid"]))
        diag["detect_rate"] = diag["detect_count"] / max(len(recs), 1)
        diag["valid_rate"] = diag["valid_count"] / max(len(recs), 1)
        diag["mp_label_counts"] = {k: labels[side].count(k) for k in set(labels[side])}
        diag["wrong_side_rejected"] = int(sum(
            1 for r in recs if r and not r.get("side_ok", True)))
        diagnostics[side] = diag

    # ---- save ------------------------------------------------------------
    n = span
    angles = np.stack([angle_tracks["l"], angle_tracks["r"]], axis=1)  # (N,2,19)
    levels = np.stack([level_tracks["l"], level_tracks["r"]], axis=1)  # (N,2)
    conf = np.zeros((n, 2), np.float32)
    for si, s in enumerate(("l", "r")):
        for k in range(n):
            conf[k, si] = LEVEL_CONF[int(levels[k, si])]
    np.savez_compressed(
        out_dir / "hand.npz",
        angles=angles.astype(np.float32),
        level=levels.astype(np.int8),
        conf=conf.astype(np.float32),
        detected=np.stack([raw_arrays["l"]["valid"], raw_arrays["r"]["valid"]], axis=1),
        lm2d=np.stack([raw_arrays["l"]["lm2d"], raw_arrays["r"]["lm2d"]], axis=1),
        world3d=np.stack([raw_arrays["l"]["world3d"], raw_arrays["r"]["world3d"]], axis=1),
        score=np.stack([raw_arrays["l"]["score"], raw_arrays["r"]["score"]], axis=1),
        bbox=np.stack([raw_arrays["l"]["bbox"], raw_arrays["r"]["bbox"]], axis=1),
        wrist_err=np.stack([raw_arrays["l"]["wrist_err"], raw_arrays["r"]["wrist_err"]], axis=1),
        fails=np.stack([raw_arrays["l"]["fails"], raw_arrays["r"]["fails"]], axis=1),
    )
    summary = {
        "video": args.video,
        "frames": [start, end],
        "n_frames": n,
        "video_size": [width, height],
        "input_size": args.input_size,
        "conf": args.conf,
        "joint_keys": list(JOINT_KEYS),
        "sides": diagnostics,
        "runtime": {"crops": n_calls,
                    "ms_per_crop": 1000 * t_total / max(n_calls, 1)},
    }
    with open(out_dir / "hand.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---- overlays --------------------------------------------------------
    saved = 0
    cap = cv2.VideoCapture(args.video)
    for side in ("l", "r"):
        for fi in range(start, end):
            rec = records[side][fi - start]
            if not rec or level_tracks[side][fi - start] not in (LEVEL_L0, LEVEL_L1):
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, img = cap.read()
            if not ok:
                continue
            rec2 = dict(rec)
            rec2["level"] = int(level_tracks[side][fi - start])
            _draw_overlay(img, rec2, side, fi,
                          os.path.relpath(out_dir / f"hand_f{fi:03d}{side}.png", os.getcwd()))
            saved += 1
            if saved >= args.overlays:
                break
        if saved >= args.overlays:
            break
    cap.release()

    print(json.dumps({s: {"L0": diagnostics[s]["level_counts"][0],
                          "L1": diagnostics[s]["level_counts"][1],
                          "L2": diagnostics[s]["level_counts"][2],
                          "L3": diagnostics[s]["level_counts"][3],
                          "detect": diagnostics[s]["detect_rate"],
                          "valid": diagnostics[s]["valid_rate"],
                          "gain": gains[s]["gain"]} for s in ("l", "r")},
                     indent=2, ensure_ascii=False))
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
