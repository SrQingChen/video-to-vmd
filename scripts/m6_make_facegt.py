"""M6 training-only synthetic face evidence from the GT morph weights.

MediaPipe's face landmarker is unreliable on anime faces (M3: 33% detection,
mostly L1; a face close-up still produced L2 on this model).  The M6 guide
allows synthetic L0 blendshape supervision, so this script builds a noisy
version of the GT 12-channel semantic weights and writes the same
``face.npz`` schema as ``run_face.py``.  L2/L3 frames stay masked in the
teacher; production still uses the real detector.
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


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def make_for_clip(base: str, out_root: Path, seed: int = 11,
                  sigma: float = 0.05, dropout: float = 0.05):
    motion, model_key = base.split("_")[:2]
    gt = np.load(ROOT / f"outputs/m35/gt/{motion}/{model_key}.gt.npz")
    w = np.asarray(gt["face_weights"], np.float32)
    chan = np.asarray(gt["face_mask"], np.float32)
    n = len(w)
    rng = np.random.default_rng(seed + (abs(hash(base)) % 10000))
    noisy = np.clip(w + rng.normal(0.0, sigma, w.shape), 0.0, 1.0)
    level = np.where(rng.random(n) < dropout, 2, 0).astype(np.int8)
    detected = np.ones(n, np.int8)
    score = np.where(level <= 1, 0.95, 0.0).astype(np.float32)
    out = out_root / base
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "face.npz",
        weights=noisy.astype(np.float32),
        level=level,
        detected=detected,
        score=score,
        blendshapes=np.zeros((n, 52), np.float32),
        face_mask=chan.astype(np.float32),
    )
    report = {
        "base": base, "frames": n, "sigma": sigma, "dropout": dropout,
        "level_counts": {int(k): int((level == k).sum()) for k in range(4)},
        "channels": {s: int(chan[i] > 0) for i, s in enumerate(FD.SEMANTIC)},
        "note": "synthetic GT morph weights with noise; train-only evidence",
    }
    (out / "face.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default=None)
    ap.add_argument("--out", default=str(ROOT / "outputs/m6/facegt"))
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.05)
    args = ap.parse_args()
    if args.clips:
        clips = [c for c in args.clips.split(",") if c]
    else:
        clips = [f"{m}_{mo}_{v}" for m in FD.TRAIN_MOTIONS
                 for mo in FD.TRAIN_MODELS for v in FD.TRAIN_VIEWS]
    reports = []
    for cid in clips:
        r = make_for_clip(cid, Path(args.out), sigma=args.sigma,
                          dropout=args.dropout)
        reports.append(r)
        print(cid, r["frames"], "L0", r["level_counts"].get(0, 0), flush=True)
    (Path(args.out) / "manifest.json").write_text(
        json.dumps({"clips": reports}, indent=1), encoding="utf-8")
    print("WROTE", Path(args.out) / "manifest.json")


if __name__ == "__main__":
    main()
