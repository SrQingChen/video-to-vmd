"""M6 training-only synthetic hand evidence from the GT VMD.

M3.5's bottleneck is hand supervision coverage: MediaPipe finds L0/L1 hands on
only 462 of 7,572 train frames.  This script builds a *synthetic observation*
for every training frame from the GT hand landmarks (``feedback_data`` already
has the exact same 21-point convention), adds landmark noise and a small
dropout rate, and writes the same ``hand.npz`` schema as ``run_hand.py``.

Boundary: this is a training-time evidence augmentation, not an inference
shortcut.  Production still uses MediaPipe; the network only sees this stream
when the ``handgt`` training variant is selected.  L2/L3 frames remain masked
in the teacher, so the fallback pose is never supervised.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vtv import feedback_data as FD  # noqa: E402
from vtv.hands import mp_joint_angles  # noqa: E402
from vtv.mmd.pmx import parse_pmx  # noqa: E402
from vtv.mmd.vmd import parse_vmd  # noqa: E402


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def make_for_clip(base: str, out_root: Path, seed: int = 7,
                  sigma_m: float = 0.004, dropout: float = 0.05):
    motion, model_key = base.split("_")[:2]
    cfg = load_json(ROOT / "configs/models.json")["models"][model_key]
    profile = load_json(ROOT / f"profiles/{model_key}.json")
    model = parse_pmx(cfg["pmx"])
    gt = parse_vmd(str(ROOT / f"outputs/m35/gt/{motion}/{model_key}.vmd"))
    n = int(gt["max_frame"]) + 1
    lm = FD._synthetic_landmarks(model, profile, gt, range(n))
    rng = np.random.default_rng(seed + (abs(hash(base)) % 10000))
    angles = np.zeros((n, 2, len(FD.HAND_JOINT_KEYS)), np.float32)
    level = np.full((n, 2), 3, np.int8)
    score = np.zeros((n, 2), np.float32)
    detected = np.zeros((n, 2), np.int8)
    for si, side in enumerate(("l", "r")):
        pts = lm[side].copy()
        finite = np.all(np.isfinite(pts), axis=(1, 2))
        # 3D landmark jitter, then the same anatomical-angle extractor as
        # MediaPipe's output path (so the units/convention are identical).
        noisy = pts + rng.normal(0.0, sigma_m, pts.shape)
        for fi in range(n):
            if not finite[fi]:
                continue
            a = mp_joint_angles(noisy[fi])
            for j, key in enumerate(FD.HAND_JOINT_KEYS):
                finger, joint = key.split(".")
                angles[fi, si, j] = a.get(finger, {}).get(joint, 0.0)
            keep = rng.random() >= dropout
            level[fi, si] = 0 if keep else 2
            score[fi, si] = 0.95 if keep else 0.0
            detected[fi, si] = 1
    out = out_root / base
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "hand.npz",
        angles=angles, level=level, conf=(level <= 1).astype(np.float32),
        detected=detected, score=score,
        lm2d=np.zeros((n, 2, 21, 2), np.float32),
        world3d=np.zeros((n, 2, 21, 3), np.float32),
        bbox=np.zeros((n, 2, 4), np.float32),
    )
    report = {
        "base": base, "frames": n, "sigma_m": sigma_m, "dropout": dropout,
        "level_counts": {s: {int(k): int((level[:, si] == k).sum())
                             for k in range(4)} for si, s in enumerate(("l", "r"))},
        "note": "synthetic GT hand landmarks with noise; train-only evidence",
    }
    (out / "hand.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default=None,
                    help="base clip ids; default all M6 train base clips")
    ap.add_argument("--out", default=str(ROOT / "outputs/m6/handgt"))
    ap.add_argument("--sigma", type=float, default=0.004)
    ap.add_argument("--dropout", type=float, default=0.05)
    args = ap.parse_args()
    if args.clips:
        clips = [c for c in args.clips.split(",") if c]
    else:
        sys.path.insert(0, str(ROOT / "scripts"))
        from m35_data_engine import all_clips
        clips = [c for c in all_clips() if "__" not in c
                 and FD.assign_split(*c.split("_")[:1], *c.split("_")[1:2],
                                     "_".join(c.split("_")[2:])) == "train"]
    reports = []
    for cid in clips:
        r = make_for_clip(cid, Path(args.out), sigma_m=args.sigma,
                          dropout=args.dropout)
        reports.append(r)
        print(cid, "frames", r["frames"], "L0/L1",
              {s: r["level_counts"][s].get(0, 0) + r["level_counts"][s].get(1, 0)
               for s in ("l", "r")}, flush=True)
    (Path(args.out) / "manifest.json").write_text(
        json.dumps({"clips": reports}, indent=1), encoding="utf-8")
    print("WROTE", Path(args.out) / "manifest.json")


if __name__ == "__main__":
    main()
