"""M3.0 audit: MediaPipe Hands / Face Landmarker feasibility on T5.

Runs both landmarkers over the whole 30 fps source clip (242 frames) with
body-guided crops, verifies the detected geometry, and saves a JSON report
plus a handful of overlays for visual inspection.

The goal is a measured decision, not a pretty demo:
  * hand detection rate (raw / after geometric verification)
  * temporal stability of finger angles on consecutive verified frames
  * handedness agreement with the body wrist side
  * per-crop runtime
  * face landmarker detection rate / blendshape stability

Usage:
  python scripts/m30_handface_audit.py \
      --video outputs/gvhmr/T5/0_input_video.mp4 \
      --kp outputs/gvhmr/T5/preprocess/vitpose.pt \
      --out outputs/m3/audit
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

# MediaPipe hand landmark topology (per hand, 21 points).
FINGERS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
WRIST_LM = 0
MIDDLE_MCP = 9
INDEX_MCP = 5
PINKY_MCP = 17

# COCO-17 body keypoints used to guide the crops.
COCO = {"nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4,
        "l_shoulder": 5, "r_shoulder": 6, "l_elbow": 7, "r_elbow": 8,
        "l_wrist": 9, "r_wrist": 10}


def _angle(a, b):
    """Angle between two vectors in degrees, 0 when degenerate."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    c = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def hand_metrics(landmarks_xy, world_xyz, expected_wrist_xy, expected_hand_px):
    """Geometric plausibility of a detected hand.

    Returns (ok, score, metrics).  Ratios are relative to the MP hand scale so
    they are independent of image resolution; the wrist/pixel-size checks are
    relative to the body keypoints.
    """
    p = np.asarray(landmarks_xy, float)
    w = np.asarray(world_xyz, float)
    wrist = p[WRIST_LM]
    palm = np.linalg.norm(p[MIDDLE_MCP] - wrist)
    width = np.linalg.norm(p[INDEX_MCP] - p[PINKY_MCP])
    m = {"palm_px": float(palm), "width_px": float(width)}
    checks = {}
    if palm < 6:
        return False, 0.0, m
    m["palm_ratio"] = float(width / palm)
    checks["palm_ratio"] = 0.35 <= m["palm_ratio"] <= 2.0

    finger_len = {}
    for name, (a, b, c, d) in FINGERS.items():
        # chain length MCP->PIP->DIP->TIP, relative to wrist->middle-MCP
        if name == "thumb":
            chain = sum(np.linalg.norm(p[y] - p[x]) for x, y in ((a, b), (b, c), (c, d)))
            rest = np.linalg.norm(p[a] - wrist)
            finger_len[name] = float(chain / palm)
            checks["thumb_ratio"] = 0.4 <= finger_len[name] <= 3.0
            checks["thumb_base_ratio"] = 0.15 <= rest / palm <= 2.0
        else:
            chain = sum(np.linalg.norm(p[y] - p[x]) for x, y in ((a, b), (b, c), (c, d)))
            finger_len[name] = float(chain / palm)
    m["finger_ratio"] = finger_len
    ref = [finger_len[k] for k in ("index", "middle", "ring", "pinky")]
    checks["finger_ratio"] = all(0.45 <= v <= 2.0 for v in ref)
    checks["finger_spread"] = (max(ref) - min(ref)) <= 0.9

    # 3D hand must not be degenerate: metric hand size in a human-ish range and
    # enough depth variation to be a real 3D prediction.
    wscale = float(np.linalg.norm(w[MIDDLE_MCP] - w[WRIST_LM]))
    m["world_scale_m"] = wscale
    m["world_depth_m"] = float(np.ptp(w[:, 2]))
    checks["world_scale"] = 0.05 <= wscale <= 0.20
    checks["world_depth"] = 0.01 <= m["world_depth_m"] <= 0.25

    # Wrist position must agree with the body wrist, and the hand size must be
    # within a factor of the expected body-proportioned size.
    m["wrist_err_px"] = float(np.linalg.norm(wrist - np.asarray(expected_wrist_xy, float)))
    m["wrist_err_ratio"] = float(m["wrist_err_px"] / max(expected_hand_px, 1e-6))
    checks["wrist_pos"] = m["wrist_err_ratio"] <= 2.0
    m["size_ratio"] = float(palm / max(expected_hand_px, 1e-6))
    checks["size"] = 0.55 <= m["size_ratio"] <= 2.0

    m["checks"] = checks
    m["n_fail"] = sum(1 for v in checks.values() if not v)
    return m["n_fail"] == 0, float(m["n_fail"]), m


def joint_angles(pts):
    """Per-joint flexion (deg, unsigned) and MCP spread (signed, index side +).

    Works for both the 3D world landmarks and 2D image landmarks.
    """
    pts = np.asarray(pts, float)
    wrist = pts[WRIST_LM]
    u = pts[MIDDLE_MCP] - wrist
    l = pts[INDEX_MCP] - pts[PINKY_MCP]
    lu = float(np.dot(l, u) / max(np.dot(u, u), 1e-9))
    l = l - lu * u
    out = {}
    for name, (a, b, c, d) in FINGERS.items():
        if name == "thumb":
            d0 = pts[b] - pts[a]
            d1 = pts[c] - pts[b]
            d2 = pts[d] - pts[c]
            out[name] = {
                "cmc_spread": _angle(d0, l),
                "mcp": _angle(d0, d1),
                "ip": _angle(d1, d2),
            }
        else:
            d0 = pts[a] - wrist
            d1 = pts[b] - pts[a]
            d2 = pts[c] - pts[b]
            d3 = pts[d] - pts[c]
            # spread: in-plane angle vs middle-finger axis, signed toward index
            denom = max(np.linalg.norm(d1), 1e-8)
            dl = float(np.dot(d1, l) / denom)
            du = float(np.dot(d1, u) / max(np.linalg.norm(u), 1e-8))
            out[name] = {
                "spread": float(np.degrees(np.arctan2(dl, max(du, 1e-6)))),
                "mcp": _angle(d0, d1),
                "pip": _angle(d1, d2),
                "dip": _angle(d2, d3),
            }
    return out


def flat_angles(angles, prefix=""):
    out = {}
    for name, joints in angles.items():
        for k, v in joints.items():
            out[prefix + name + "." + k] = float(v)
    return out


def crop(img, cx, cy, half):
    h, w = img.shape[:2]
    x0, y0 = max(0, int(cx - half)), max(0, int(cy - half))
    x1, y1 = min(w, int(cx + half)), min(h, int(cy + half))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None, (0, 0, 0, 0)
    return img[y0:y1, x0:x1], (x0, y0, x1, y1)


def candidate_boxes(wrist, elbow, scale):
    """Three crop candidates around the hand, along the forearm direction."""
    wx, wy = wrist
    ex, ey = elbow
    d = float(np.hypot(wx - ex, wy - ey))
    if d < 1e-6:
        ux, uy = 0.0, -1.0
    else:
        ux, uy = (wx - ex) / d, (wy - ey) / d
    hand = max(22.0, 0.34 * d)  # expected PALM length (wrist->middle MCP)
    specs = [
        (0.40, 1.05),   # tight: centre just past the wrist, one hand length
        (0.60, 1.45),   # medium
        (0.25, 1.95),   # wide (includes the sleeve/forearm)
    ]
    out = []
    for off, hf in specs:
        cx, cy = wx + ux * hand * off, wy + uy * hand * off
        half = hand * hf
        out.append((cx, cy, half, hand))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(ROOT / "outputs/gvhmr/T5/0_input_video.mp4"))
    ap.add_argument("--kp", default=str(ROOT / "outputs/gvhmr/T5/preprocess/vitpose.pt"))
    ap.add_argument("--out", default=str(ROOT / "outputs/m3/audit"))
    ap.add_argument("--hand-model", default=str(ROOT / "models/mediapipe/hand_landmarker.task"))
    ap.add_argument("--face-model", default=str(ROOT / "models/mediapipe/face_landmarker.task"))
    ap.add_argument("--overlays", type=int, default=14)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # MediaPipe's C layer cannot open Windows paths containing non-ASCII
    # characters (the repo lives under E:\zcodeAI项目\...).  Feed it a path
    # relative to the current directory instead of the absolute one.
    def _mp_path(p):
        try:
            return os.path.relpath(os.path.abspath(p), os.getcwd())
        except ValueError:
            return p

    args.hand_model = _mp_path(args.hand_model)
    args.face_model = _mp_path(args.face_model)
    kp = torch.load(args.kp, map_location="cpu").numpy()
    n_frames = len(kp)

    hand_opts = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=args.hand_model),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.15,
        min_hand_presence_confidence=0.15,
        min_tracking_confidence=0.15,
    )
    hand_det = vision.HandLandmarker.create_from_options(hand_opts)
    face_opts = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=args.face_model),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.2,
        min_face_presence_confidence=0.2,
        output_face_blendshapes=True,
    )
    face_det = vision.FaceLandmarker.create_from_options(face_opts)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    records = {"hand": {}, "face": {}}
    overlay_pick = []
    t_hand_total = 0.0
    t_face_total = 0.0
    n_hand_calls = 0
    n_face_calls = 0

    for fi in range(n_frames):
        ok, img = cap.read()
        if not ok:
            break
        rec = {"l": [], "r": []}
        for side, wi, ei, si in (("l", COCO["l_wrist"], COCO["l_elbow"], COCO["l_shoulder"]),
                                 ("r", COCO["r_wrist"], COCO["r_elbow"], COCO["r_shoulder"])):
            wrist = kp[fi, wi, :2]
            elbow = kp[fi, ei, :2]
            wconf = float(kp[fi, wi, 2])
            d = float(np.linalg.norm(wrist - elbow))
            for cx, cy, half, hand_px in candidate_boxes(wrist, elbow, None):
                im, box = crop(img, cx, cy, half)
                if im is None:
                    continue
                up = cv2.resize(im, (320, 320), interpolation=cv2.INTER_CUBIC)
                t0 = time.perf_counter()
                res = hand_det.detect(mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(up, cv2.COLOR_BGR2RGB)))
                dt = time.perf_counter() - t0
                t_hand_total += dt
                n_hand_calls += 1
                if not res.hand_landmarks:
                    rec[side].append({"ok": False, "dt": dt, "box": box})
                    continue
                lm = np.array([[p.x * im.shape[1] + box[0], p.y * im.shape[0] + box[1]]
                               for p in res.hand_landmarks[0]], float)
                world = np.array([[p.x, p.y, p.z] for p in res.hand_world_landmarks[0]], float)
                score = float(res.handedness[0][0].score)
                label = res.handedness[0][0].category_name
                valid, fails, metrics = hand_metrics(lm, world, wrist, hand_px)
                rec[side].append({
                    "ok": True, "valid": bool(valid), "fails": int(fails),
                    "score": score, "label": label, "dt": dt, "box": box,
                    "metrics": metrics,
                    "angles3d": flat_angles(joint_angles(world)),
                    "angles2d": flat_angles(joint_angles(lm)),
                    "lm2d": lm.tolist(), "world3d": world.tolist(),
                })
        # keep the best candidate per side
        for side in ("l", "r"):
            cands = rec[side]
            dets = [c for c in cands if c.get("ok")]
            if not dets:
                rec[side] = None
                continue
            dets.sort(key=lambda c: (c["valid"], -c["fails"], c["score"]), reverse=True)
            rec[side] = dets[0]
        records["hand"][str(fi)] = rec

        # ---- face: head crop from the five face keypoints ------------------
        pts = kp[fi, :5, :2]
        cx, cy = pts.mean(0)
        span = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))
        face_best = None
        for fac in (2.2, 3.0, 4.0):
            half = max(56.0, span * fac * 0.5)
            im, box = crop(img, cx, cy, half)
            if im is None:
                continue
            up = cv2.resize(im, (512, 512), interpolation=cv2.INTER_CUBIC)
            t0 = time.perf_counter()
            fres = face_det.detect(mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(up, cv2.COLOR_BGR2RGB)))
            dt = time.perf_counter() - t0
            t_face_total += dt
            n_face_calls += 1
            if not fres.face_landmarks:
                continue
            lm = np.array([[p.x * im.shape[1] + box[0], p.y * im.shape[0] + box[1]]
                           for p in fres.face_landmarks[0]], float)
            bs = {b.category_name: float(b.score) for b in fres.face_blendshapes[0]}
            face_best = {
                "ok": True, "dt": dt, "box": box, "fac": fac,
                "bbox_wh": [float(np.ptp(lm[:, 0])), float(np.ptp(lm[:, 1]))],
                "blendshapes": bs,
            }
            break
        records["face"][str(fi)] = face_best
        if face_best and len(overlay_pick) < 4:
            overlay_pick.append(("face", fi))

        if fi % 40 == 0:
            print(f"  frame {fi}/{n_frames}", flush=True)

    cap.release()

    # ------------------------------------------------------------------
    # summaries
    # ------------------------------------------------------------------
    summary = {"n_frames": n_frames, "fps": fps, "size": [width, height],
               "hand": {}, "face": {}}

    def _joint_series(side, key):
        track = []
        for fi in range(n_frames):
            c = records["hand"].get(str(fi), {}).get(side)
            if c and c.get("valid"):
                track.append((fi, c[key]))
        return track

    for side in ("l", "r"):
        raw = sum(1 for fi in range(n_frames)
                  if records["hand"].get(str(fi), {}).get(side))
        valid = sum(1 for fi in range(n_frames)
                    if (records["hand"].get(str(fi), {}).get(side) or {}).get("valid"))
        fails = {}
        labels = {}
        scores = []
        metas = []
        for fi in range(n_frames):
            c = records["hand"].get(str(fi), {}).get(side)
            if not c:
                continue
            labels[c["label"]] = labels.get(c["label"], 0) + 1
            if c.get("valid"):
                scores.append(c["score"])
                for k, v in c["metrics"]["checks"].items():
                    if not v:
                        fails[k] = fails.get(k, 0) + 1
                metas.append(c["metrics"])
        expect_label = "Right" if side == "l" else "Left"  # character L = image Right in a front-facing mirror? see doc
        # stability on consecutive verified frames
        for key in ("angles3d", "angles2d"):
            track = _joint_series(side, key)
            diffs = []
            for (f0, a0), (f1, a1) in zip(track, track[1:]):
                if f1 - f0 != 1:
                    continue
                diffs.extend(abs(a1[k] - a0[k]) for k in a0 if k in a1)
            summary["hand"].setdefault(side, {})[key + "_step"] = {
                "n": len(diffs),
                "p50": float(np.percentile(diffs, 50)) if diffs else None,
                "p95": float(np.percentile(diffs, 95)) if diffs else None,
                "max": float(np.max(diffs)) if diffs else None,
            }
        summary["hand"][side] = {
            "raw_detect": raw, "raw_rate": raw / max(n_frames, 1),
            "valid_detect": valid, "valid_rate": valid / max(n_frames, 1),
            "fail_counts_valid": fails,
            "mp_label_counts_raw": labels,
            "score_median_valid": float(np.median(scores)) if scores else None,
            "size_ratio_median": float(np.median([m["size_ratio"] for m in metas])) if metas else None,
            "wrist_err_ratio_median": float(np.median([m["wrist_err_ratio"] for m in metas])) if metas else None,
            "world_scale_median": float(np.median([m["world_scale_m"] for m in metas])) if metas else None,
            **summary["hand"].get(side, {}),
        }

    face_ok = sum(1 for f in records["face"].values() if f)
    bs_steps = []
    bs_names = None
    prev = None
    for fi in range(n_frames):
        f = records["face"].get(str(fi))
        if not f:
            prev = None
            continue
        cur = f["blendshapes"]
        if bs_names is None:
            bs_names = sorted(cur)
        if prev is not None and prev[0] == fi - 1:
            bs_steps.extend(abs(cur[k] - prev[1][k]) for k in bs_names)
        prev = (fi, cur)
    summary["face"] = {
        "detect": face_ok, "rate": face_ok / max(n_frames, 1),
        "dt_ms_median": float(np.median([f["dt"] for f in records["face"].values() if f])) * 1000
        if face_ok else None,
        "bbox_w_median": float(np.median([f["bbox_wh"][0] for f in records["face"].values() if f]))
        if face_ok else None,
        "blendshape_step": {
            "n": len(bs_steps),
            "p50": float(np.percentile(bs_steps, 50)) if bs_steps else None,
            "p95": float(np.percentile(bs_steps, 95)) if bs_steps else None,
            "max": float(np.max(bs_steps)) if bs_steps else None,
        },
    }
    summary["runtime"] = {
        "hand_calls": n_hand_calls,
        "hand_ms_per_call": 1000.0 * t_hand_total / max(n_hand_calls, 1),
        "face_calls": n_face_calls,
        "face_ms_per_call": 1000.0 * t_face_total / max(n_face_calls, 1),
    }
    summary["stage_fraction"] = 100 * n_hand_calls / max(n_frames * 2 * 3, 1)

    def _jsonable(o):
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        raise TypeError(f"not JSON serializable: {type(o)}")

    with open(out_dir / "audit.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f,
                  ensure_ascii=False, default=_jsonable)

    # overlays: first few accepted hand detections + rejected ones
    cap = cv2.VideoCapture(args.video)
    saved = 0
    for side in ("l", "r"):
        for fi in range(n_frames):
            c = records["hand"].get(str(fi), {}).get(side)
            if not c or not c.get("valid"):
                continue
            ok, img = cap.read() if False else (None, None)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, img = cap.read()
            if not ok:
                continue
            x0, y0, x1, y1 = c["box"]
            vis = img.copy()
            for (x, y) in np.array(c["lm2d"]):
                if x0 - 2 <= x <= x1 + 2 and y0 - 2 <= y <= y1 + 2:
                    cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(vis, f"f{fi}{side} valid s={c['score']:.2f}", (max(4, x0), max(18, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imwrite(os.path.relpath(out_dir / f"hand_ok_f{fi:03d}{side}.png", os.getcwd()), vis)
            saved += 1
            if saved >= 6:
                break
        if saved >= 6:
            break
    cap.release()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
