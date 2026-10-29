# -*- coding: utf-8 -*-
"""Shared evaluation helpers for the M3.5 feedback network.

Kept in ``src`` so the held-out evaluator, the T5 no-GT check and the unit
tests use exactly the same gating and stability code.
"""
from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from . import feedback_data as FD
from .feedback_net import FeedbackNet, NetConfig
from .metrics.core import _intervals, _jerk, _slide_drift, pose_track


def load_net(ckpt_path):
    ck = torch.load(str(ckpt_path), map_location="cpu")
    cfg = NetConfig(**ck["cfg"])
    net = FeedbackNet(cfg, ck["limits"])
    net.load_state_dict(ck["model"])
    net.eval()
    return net, ck


def predict(net, ck, feats: np.ndarray, device: str = "cpu") -> np.ndarray:
    x = np.clip((feats - ck["feat_mean"]) / ck["feat_std"], -6.0, 6.0)
    with torch.no_grad():
        pred = net(torch.tensor(x[None], device=device))[0].cpu().numpy()
    return pred.astype(np.float32)


def gate_residual(res: np.ndarray, hand_level: np.ndarray,
                  face_level: np.ndarray,
                  target_names: Mapping[str, str],
                  hand_supported: Optional[np.ndarray] = None,
                  face_supported: Optional[np.ndarray] = None) -> np.ndarray:
    """Zero hand/face corrections outside M3 L0/L1, missing morphs and
    channels that had no training supervision (stay at baseline)."""
    out = np.array(res, np.float32, copy=True)
    for s in range(2):
        bad = np.asarray(hand_level[:, s]) > 1
        s0 = s * 19 + FD.HAND_SLICE.start
        out[bad, s0:s0 + 19] = 0.0
        if hand_supported is not None and not bool(hand_supported[s]):
            out[:, s0:s0 + 19] = 0.0
    chan = np.asarray([1.0 if s in target_names else 0.0 for s in FD.SEMANTIC])
    fmask = FD.face_channel_mask(face_level, chan)
    if face_supported is not None:
        fmask = fmask * np.asarray(face_supported, np.float32)[None, :]
    out[:, FD.FACE_SLICE] *= fmask
    return out


def smooth_residual(res: np.ndarray, hand_level: np.ndarray,
                    face_level: Optional[np.ndarray] = None,
                    win: int = 3) -> np.ndarray:
    """Temporal smoothing of the *predicted* correction (targets stay raw).

    Body/root are dense; hand and face residuals are averaged only inside
    L0/L1 runs so a correction cannot bleed into (or pop on) fallback frames.
    """
    out = np.array(res, np.float32, copy=True)
    if win <= 1:
        return out
    k = np.ones(win, np.float64) / win

    def conv_dense(x):
        pad = win // 2
        xp = np.concatenate([np.repeat(x[:1], pad, axis=0), x,
                             np.repeat(x[-1:], pad, axis=0)], axis=0)
        return np.stack([np.convolve(xp[:, c], k, mode="valid")
                         for c in range(x.shape[1])], axis=1)

    out[:, FD.ROOT_SLICE] = conv_dense(out[:, FD.ROOT_SLICE])
    out[:, FD.ROT_SLICE] = conv_dense(out[:, FD.ROT_SLICE])
    for s in range(2):
        valid = (np.asarray(hand_level[:, s]) <= 1).astype(np.float64)
        s0 = s * 19 + FD.HAND_SLICE.start
        block = out[:, s0:s0 + 19]
        num = np.stack([np.convolve((block[:, j] * valid), k, mode="same")
                        for j in range(19)], axis=1)
        den = np.convolve(valid, k, mode="same")
        sm = num / np.maximum(den[:, None], 1e-9)
        out[:, s0:s0 + 19] = np.where(valid[:, None] > 0, sm, 0.0)
    if face_level is not None:
        valid = (np.asarray(face_level) <= 1).astype(np.float64)
        block = out[:, FD.FACE_SLICE]
        num = np.stack([np.convolve(block[:, j] * valid, k, mode="same")
                        for j in range(block.shape[1])], axis=1)
        den = np.convolve(valid, k, mode="same")
        sm = num / np.maximum(den[:, None], 1e-9)
        out[:, FD.FACE_SLICE] = np.where(valid[:, None] > 0, sm, 0.0)
    return out.astype(np.float32)


def _finger_step(vmd: Mapping, profile: Mapping) -> Dict[str, float]:
    names = sorted({n for n in profile.get("finger_map", {}).values()
                    if n in vmd["bones"]})
    steps = []
    for name in names:
        keys = sorted(vmd["bones"][name], key=lambda k: k["frame"])
        for a, b in zip(keys, keys[1:]):
            if int(b["frame"]) - int(a["frame"]) != 1:
                continue
            dot = abs(sum(x * y for x, y in zip(a["rot"], b["rot"])))
            steps.append(2.0 * math.degrees(math.acos(min(1.0, dot))))
    if not steps:
        return {"p95": 0.0, "max": 0.0}
    return {"p95": float(np.percentile(steps, 95)), "max": float(np.max(steps))}


def _morph_step(vmd: Mapping, n: int) -> float:
    worst = 0.0
    for name, keys in vmd.get("morphs", {}).items():
        if len(keys) < 2:
            continue
        fr = np.asarray([k[0] for k in keys], float)
        w = np.asarray([k[1] for k in keys], float)
        d = np.interp(np.arange(n, dtype=float), fr, w,
                      left=w[0], right=w[-1])
        worst = max(worst, float(np.abs(np.diff(d)).max()) if len(d) > 1 else 0.0)
    return worst


def stability(model: Mapping, profile: Mapping, vmd: Mapping,
              n: Optional[int] = None) -> Dict[str, float]:
    """GT-free stability of one VMD (used for T5 and as a sanity check)."""
    n = int(vmd["max_frame"]) + 1 if n is None else int(n)
    frames = list(range(n))
    tr = pose_track(model, vmd, profile, frames=frames)
    height = float(profile.get("height") or 1.0)
    jerk = _jerk(tr["world"])
    toe = {}
    for side in ("l", "r"):
        if f"toe_{side}" in tr["world"]:
            toe[side] = np.asarray(tr["world"][f"toe_{side}"])
    slide, pen = 0.0, 0.0
    if toe:
        ymin = min(float(v.min()) for v in (t[:, 1] for t in toe.values()))
        thr = ymin + 0.02 * height
        for side, p in toe.items():
            mask = p[:, 1] < thr
            slide += _slide_drift(p, _intervals(mask))
            pen = max(pen, float(np.maximum(0.0, -p[:, 1]).max()))
    fs = _finger_step(vmd, profile)
    return {
        "frames": n,
        "jerk_rms": round(float(jerk["rms"]), 4),
        "jerk_spikes": int(jerk["spike_count"]),
        "foot_slide": round(float(slide), 5),
        "foot_penetration": round(float(pen), 5),
        "finger_step_p95": round(fs["p95"], 4),
        "finger_step_max": round(fs["max"], 4),
        "morph_step_max": round(_morph_step(vmd, n), 4),
        "bone_tracks": len(vmd["bones"]),
    }
