# -*- coding: utf-8 -*-
"""Face retargeting: MediaPipe blendshapes/landmarks -> PMX morph weights (M3).

Two evidence sources are combined:

* **L0** MediaPipe ``FaceLandmarker`` blendshapes (52 ARKit-style scores) when
  the detected mesh passes a patch-contrast check and is temporally stable.
* **L1** eye/mouth geometry from the 478 landmarks (EAR / MAR), used when the
  mesh is usable but the blendshapes are noisy.
* **L2/L3** procedural neutral face + natural blinking.  This is the correct
  result when the face cannot be trusted; it must never jitter.

Only morphs that exist in the target profile are emitted (the alias table is
built from ``profiles/*.json``).  Weight composition is capped so that several
additive morphs cannot blow up the face.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

LEVEL_L0, LEVEL_L1, LEVEL_L2, LEVEL_L3 = 0, 1, 2, 3
LEVEL_CONF = {LEVEL_L0: 1.0, LEVEL_L1: 0.7, LEVEL_L2: 0.3, LEVEL_L3: 0.1}

# Semantic channels produced by run_face and consumed by the assembler.
SEMANTIC = (
    "blink", "wink_l", "wink_r", "smile", "surprise", "angry", "sad",
    "mouth_a", "mouth_i", "mouth_u", "mouth_e", "mouth_o",
)
SEM_INDEX = {k: i for i, k in enumerate(SEMANTIC)}

# Candidate PMX morph names per semantic channel, in order of preference.
# Names observed in the three target profiles (si/luoxi: 41, tiflos: 64).
ALIASES = {
    "blink": ("まばたき",),
    "wink_l": ("ウィンク", "ウィンク２"),
    "wink_r": ("ウィンク右", "ｳｨﾝｸ２右", "ウィンク２右"),
    "smile": ("笑い", "にこり", "にっこり"),
    "surprise": ("びっくり",),
    "angry": ("怒り目", "怒り"),
    "sad": ("悲しい目", "困る"),
    "mouth_a": ("あ", "あ２"),
    "mouth_i": ("い",),
    "mouth_u": ("う",),
    "mouth_e": ("え",),
    "mouth_o": ("お",),
}

# MediaPipe face landmark indices (478-point mesh).
LM_LEFT_EYE = {"corner_a": 33, "corner_b": 133, "upper": 159, "lower": 145}
LM_RIGHT_EYE = {"corner_a": 362, "corner_b": 263, "upper": 386, "lower": 374}
LM_MOUTH = {"corner_a": 61, "corner_b": 291, "upper": 13, "lower": 14}
LM_CHEEK_LEFT = 234
LM_CHEEK_RIGHT = 454


@dataclass
class MorphMap:
    model_key: str
    available: Dict[str, str] = field(default_factory=dict)   # semantic -> morph name
    missing: List[str] = field(default_factory=list)

    @classmethod
    def from_profile(cls, profile: Mapping) -> "MorphMap":
        names = {m["name"] for m in profile.get("morphs", [])}
        out = {}
        missing = []
        for sem in SEMANTIC:
            for cand in ALIASES[sem]:
                if cand in names:
                    out[sem] = cand
                    break
            else:
                missing.append(sem)
        return cls(model_key=str(profile.get("model_key", "")),
                   available=out, missing=missing)

    def morph_names(self) -> List[str]:
        return sorted(set(self.available.values()))


@dataclass
class FaceTrackConfig:
    core_frames: int = 4
    core_lm_step_px: float = 9.0
    l1_frames: int = 3
    l1_lm_step_px: float = 22.0
    blink_median: int = 1          # median kills 3-frame blinks -> no median
    weight_ema: float = 0.34
    blink_ema: float = 0.55
    max_step: float = 0.22
    blink_max_step: float = 0.55
    max_active_morphs: int = 4
    total_cap: float = 1.15
    procedural_blink_min_s: float = 2.2
    procedural_blink_max_s: float = 4.6
    procedural_blink_frames: int = 4
    l1_mouth: bool = False       # anime mouth landmarks are not trustworthy
    l1_wink: bool = False        # one-sided EAR closure -> soft bilateral blink
    seed: int = 7
    # reliability
    max_blink_hold_frames: int = 12
    max_jaw_hold_frames: int = 45


def face_geometry(landmarks: Sequence[Sequence[float]]) -> Dict[str, float]:
    """EAR / MAR from 478 MediaPipe face landmarks (2D pixels)."""
    p = np.asarray(landmarks, float)
    if p.shape[0] < 478:
        return {}
    def dist(i, j):
        return float(np.linalg.norm(p[i] - p[j]))
    ear_l = dist(LM_LEFT_EYE["upper"], LM_LEFT_EYE["lower"]) / max(
        dist(LM_LEFT_EYE["corner_a"], LM_LEFT_EYE["corner_b"]), 1e-6)
    ear_r = dist(LM_RIGHT_EYE["upper"], LM_RIGHT_EYE["lower"]) / max(
        dist(LM_RIGHT_EYE["corner_a"], LM_RIGHT_EYE["corner_b"]), 1e-6)
    mar = dist(LM_MOUTH["upper"], LM_MOUTH["lower"]) / max(
        dist(LM_MOUTH["corner_a"], LM_MOUTH["corner_b"]), 1e-6)
    return {"ear_l": ear_l, "ear_r": ear_r, "mar": mar}


def blendshape_weights(bs: Mapping[str, float]) -> np.ndarray:
    """Map MediaPipe blendshapes to semantic channels (0..1 each)."""
    def g(k, d=0.0):
        return float(bs.get(k, d))
    bl = g("eyeBlinkLeft")
    br = g("eyeBlinkRight")
    smile = 0.5 * (g("mouthSmileLeft") + g("mouthSmileRight"))
    frown = 0.5 * (g("mouthFrownLeft") + g("mouthFrownRight"))
    brow_down = 0.5 * (g("browDownLeft") + g("browDownRight"))
    brow_up = g("browInnerUp")
    eye_wide = 0.5 * (g("eyeWideLeft") + g("eyeWideRight"))
    jaw = g("jawOpen")
    pucker = g("mouthPucker")
    funnel = g("mouthFunnel")
    stretch = 0.5 * (g("mouthStretchLeft") + g("mouthStretchRight"))
    shrug = g("mouthShrugUpper")

    out = np.zeros(len(SEMANTIC), float)
    # blink vs wink: only call a wink when the other eye is clearly open
    if min(bl, br) > 0.30 and abs(bl - br) < 0.35:
        out[SEM_INDEX["blink"]] = max(bl, br)
    elif bl > 0.45 and br < 0.35:
        out[SEM_INDEX["wink_l"]] = bl
    elif br > 0.45 and bl < 0.35:
        out[SEM_INDEX["wink_r"]] = br
    else:
        out[SEM_INDEX["blink"]] = min(bl, br) * 0.8

    out[SEM_INDEX["smile"]] = min(1.0, smile * 1.25)
    out[SEM_INDEX["surprise"]] = min(1.0, brow_up * 0.7 + eye_wide * 0.45)
    out[SEM_INDEX["angry"]] = min(1.0, brow_down * 0.9)
    out[SEM_INDEX["sad"]] = min(1.0, frown * 0.9 + brow_up * 0.25)

    # mouth shapes: jawOpen drives "a"; pucker/funnel drive "o"/"u";
    # stretch widens toward "i"/"e".
    out[SEM_INDEX["mouth_a"]] = min(1.0, jaw * 0.85)
    out[SEM_INDEX["mouth_o"]] = min(1.0, pucker * 0.85 + jaw * 0.30)
    out[SEM_INDEX["mouth_u"]] = min(1.0, funnel * 0.75 + pucker * 0.15)
    out[SEM_INDEX["mouth_i"]] = min(1.0, stretch * 0.55)
    out[SEM_INDEX["mouth_e"]] = min(1.0, stretch * 0.30 + shrug * 0.25)
    return np.clip(out, 0.0, 1.0)


def geometry_weights(geom: Mapping[str, float],
                     ear_baseline: Mapping[str, float],
                     mar_open: float = 0.18,
                     use_mouth: bool = True,
                     allow_wink: bool = False) -> np.ndarray:
    """L1 eye/mouth geometry -> semantic channels."""
    out = np.zeros(len(SEMANTIC), float)
    bl = ear_baseline.get("ear_l", 0.0)
    br = ear_baseline.get("ear_r", 0.0)
    el = float(geom.get("ear_l", bl))
    er = float(geom.get("ear_r", br))

    def close_amt(ear, base):
        if base <= 1e-6:
            return 0.0
        return float(np.clip((base - ear) / (0.5 * base), 0.0, 1.0))

    cl = close_amt(el, bl)
    cr = close_amt(er, br)
    if allow_wink:
        if cl > 0.25 and cr > 0.25:
            out[SEM_INDEX["blink"]] = max(cl, cr)
        elif cl > 0.3:
            out[SEM_INDEX["wink_l"]] = cl
        elif cr > 0.3:
            out[SEM_INDEX["wink_r"]] = cr
    else:
        # One-sided EAR closure on an anime face is usually mesh noise; use a
        # softer bilateral blink instead of a visible wink.
        c = max(cl, cr)
        if c > 0.25:
            out[SEM_INDEX["blink"]] = float(min(0.9, 0.65 * c + 0.15))
    if use_mouth:
        mar = float(geom.get("mar", 0.0))
        out[SEM_INDEX["mouth_a"]] = float(np.clip(mar / max(mar_open, 1e-6), 0.0, 1.0))
    return out


def _limit_total(weights: np.ndarray, cfg: FaceTrackConfig) -> np.ndarray:
    """Keep at most N active channels and cap the sum."""
    w = np.array(weights, float)
    if cfg.max_active_morphs > 0:
        idx = np.argsort(w)[::-1]
        for j in idx[cfg.max_active_morphs:]:
            w[j] = 0.0
    total = float(w.sum())
    if total > cfg.total_cap:
        w *= cfg.total_cap / total
    return np.clip(w, 0.0, 1.0)


def _segment_stats(records: Sequence[Optional[Mapping]]) -> List[List[int]]:
    segs = []
    i = 0
    n = len(records)
    while i < n:
        if not records[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and records[j + 1]:
            j += 1
        segs.append(list(range(i, j + 1)))
        i = j + 1
    return segs


def _lm_step_p90(records: Sequence[Optional[Mapping]], seg: Sequence[int]) -> Optional[float]:
    steps = []
    prev = None
    for k in seg:
        rec = records[k]
        lm = (rec.get("landmarks") or rec.get("lm")) if rec else None
        if lm is None:
            prev = None
            continue
        eye_pts = np.asarray(lm, float)[[33, 133, 362, 263]]
        if prev is not None:
            steps.extend(np.linalg.norm(eye_pts - prev, axis=1).tolist())
        prev = eye_pts
    return float(np.percentile(steps, 90)) if steps else None


def assign_face_levels(records: Sequence[Optional[Mapping]],
                       cfg: Optional[FaceTrackConfig] = None
                       ) -> Tuple[np.ndarray, List[List[int]], Dict]:
    cfg = cfg or FaceTrackConfig()
    n = len(records)
    levels = np.full(n, LEVEL_L3, np.int8)
    diag = {"segments": []}
    for seg in _segment_stats(records):
        if len(seg) < cfg.l1_frames:
            continue
        step = _lm_step_p90(records, seg)
        patch_ok = sum(1 for k in seg if records[k].get("patch_ok")) / len(seg)
        size_stable = sum(1 for k in seg if records[k].get("bbox_ok")) / len(seg)
        info = {"start": seg[0], "end": seg[-1], "len": len(seg),
                "lm_step_p90": None if step is None else round(step, 2),
                "patch_ok": round(patch_ok, 2), "bbox_ok": round(size_stable, 2)}
        core = (len(seg) >= cfg.core_frames and step is not None
                and step <= cfg.core_lm_step_px and patch_ok >= 0.6 and size_stable >= 0.6)
        weak = (step is not None and step <= cfg.l1_lm_step_px and patch_ok >= 0.35)
        if core:
            for k in seg:
                if records[k].get("patch_ok") and records[k].get("bbox_ok"):
                    levels[k] = LEVEL_L0
                else:
                    levels[k] = LEVEL_L1
            info["level"] = "core"
        elif weak:
            for k in seg:
                levels[k] = LEVEL_L1
            info["level"] = "weak"
        else:
            info["level"] = "rejected"
        diag["segments"].append(info)
    for k in range(n):
        if levels[k] == LEVEL_L2:
            levels[k] = LEVEL_L2
    return levels, _segment_stats(records), diag


def _blink_curve(n_frames: int, intervals: Sequence[Tuple[int, float]],
                 width: int) -> np.ndarray:
    out = np.zeros(n_frames, float)
    shape = np.array([0.15, 0.62, 1.0, 0.55, 0.12])[:max(3, width)]
    shape = shape / shape.max()
    for start, _ in intervals:
        for k, v in enumerate(shape):
            if start + k < n_frames:
                out[start + k] = max(out[start + k], float(v))
    return out


def procedural_blinks(n_frames: int, cfg: FaceTrackConfig,
                      occupied: Optional[np.ndarray] = None) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    intervals = []
    t = rng.uniform(cfg.procedural_blink_min_s, cfg.procedural_blink_max_s)
    while True:
        frame = int(round(t * 30.0))
        if frame >= n_frames:
            break
        # do not double-blink on top of a detected blink
        if occupied is not None and occupied[max(0, frame - 3):frame + cfg.procedural_blink_frames + 3].any():
            frame += cfg.procedural_blink_frames + 2
            if frame >= n_frames:
                break
        intervals.append((frame, t))
        t += rng.uniform(cfg.procedural_blink_min_s, cfg.procedural_blink_max_s)
    return _blink_curve(n_frames, intervals, cfg.procedural_blink_frames + 1)


def build_face_track(records: Sequence[Optional[Mapping]], n_frames: int,
                     cfg: Optional[FaceTrackConfig] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Return semantic weights (N, 12), per-frame levels, diagnostics."""
    cfg = cfg or FaceTrackConfig()
    levels, segments, diag = assign_face_levels(records, cfg)
    weights = np.zeros((n_frames, len(SEMANTIC)), float)
    geom = np.zeros((n_frames, 3), float)
    for i in range(n_frames):
        rec = records[i]
        g = None
        if rec:
            g = rec.get("geometry") or rec.get("geom")
        if g:
            geom[i] = [g.get("ear_l", 0.0), g.get("ear_r", 0.0), g.get("mar", 0.0)]

    # EAR baseline: a clip-level median over stable frames is far more
    # robust than a rolling window when accepted segments are sparse.
    idx = [k for k in range(n_frames) if levels[k] in (LEVEL_L0, LEVEL_L1)
           and geom[k, 0] > 0 and geom[k, 1] > 0]
    if len(idx) >= 5:
        base = np.median(geom[idx, :2], axis=0)
        base = np.maximum(base, 0.12)
    else:
        base = np.array([0.28, 0.28])
    ear_base = np.tile(base, (n_frames, 1))

    # raw weights per frame according to level
    for i in range(n_frames):
        lvl = int(levels[i])
        rec = records[i]
        if lvl == LEVEL_L0 and rec and rec.get("blendshapes"):
            weights[i] = _limit_total(blendshape_weights(rec["blendshapes"]), cfg)
        elif lvl == LEVEL_L1 and rec:
            g = rec.get("geometry") or rec.get("geom")
            if g:
                weights[i] = _limit_total(
                    geometry_weights(g, {"ear_l": ear_base[i, 0],
                                         "ear_r": ear_base[i, 1]},
                                     use_mouth=cfg.l1_mouth,
                                     allow_wink=cfg.l1_wink), cfg)
        else:
            weights[i] = 0.0

    # temporal smoothing: median3 (not for blink) then EMA with a slope limit
    blink_idx = SEM_INDEX["blink"]
    wink_idx = [SEM_INDEX["wink_l"], SEM_INDEX["wink_r"]]
    out = np.zeros_like(weights)
    for i in range(n_frames):
        if i >= 1 and i + 1 < n_frames:
            med = np.median(weights[i - 1:i + 2], axis=0)
            med[blink_idx] = weights[i, blink_idx]
            for j in wink_idx:
                med[j] = weights[i, j]
        else:
            med = weights[i]
        if i == 0:
            out[i] = med
            continue
        alpha = cfg.blink_ema if (med[blink_idx] > 0.05 or out[i - 1, blink_idx] > 0.05) else cfg.weight_ema
        sm = out[i - 1] * (1 - alpha) + med * alpha
        step = np.full(len(SEMANTIC), cfg.max_step)
        step[blink_idx] = cfg.blink_max_step
        for j in wink_idx:
            step[j] = cfg.blink_max_step
        out[i] = np.clip(out[i - 1] + np.clip(sm - out[i - 1], -step, step), 0.0, 1.0)

    # procedural blink where the real signal is missing
    real_blink = out[:, blink_idx].copy()
    occupied = real_blink > 0.2
    proc = procedural_blinks(n_frames, cfg, occupied)
    out[:, blink_idx] = np.maximum(out[:, blink_idx], proc)

    # reliability: a blink held for many frames or a jaw stuck open is not a
    # usable signal; drop the corresponding channels and fall back.
    def longest_run(mask):
        best = cur = 0
        for v in mask:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        return best
    unreliable = {}
    if longest_run(out[:, blink_idx] > 0.6) > cfg.max_blink_hold_frames:
        unreliable["blink_hold"] = int(longest_run(out[:, blink_idx] > 0.6))
        out[:, blink_idx] = proc
    if longest_run(out[:, SEM_INDEX["mouth_a"]] > 0.5) > cfg.max_jaw_hold_frames:
        unreliable["jaw_hold"] = int(longest_run(out[:, SEM_INDEX["mouth_a"]] > 0.5))
        out[:, SEM_INDEX["mouth_a"]] *= 0.2
        out[:, SEM_INDEX["mouth_o"]] *= 0.2

    diag["level_counts"] = {int(k): int((levels == k).sum()) for k in (0, 1, 2, 3)}
    diag["detect_count"] = int(sum(1 for r in records if r))
    diag["detect_rate"] = diag["detect_count"] / max(n_frames, 1)
    diag["unreliable"] = unreliable
    diag["procedural_blink_frames"] = int((proc > 0.2).sum())
    return out, levels, diag


def weights_to_morphs(weights: np.ndarray, morph_map: MorphMap,
                      cfg: Optional[FaceTrackConfig] = None) -> Dict[str, np.ndarray]:
    """Semantic weights -> {morph_name: per-frame weight} (existing morphs only)."""
    cfg = cfg or FaceTrackConfig()
    out: Dict[str, np.ndarray] = {}
    n = len(weights)
    for sem in SEMANTIC:
        name = morph_map.available.get(sem)
        if not name:
            continue
        w = np.clip(weights[:, SEM_INDEX[sem]], 0.0, 1.0)
        if name in out:
            out[name] = np.maximum(out[name], w)
        else:
            out[name] = w
    # cap the number of simultaneous morphs per frame
    names = list(out)
    for i in range(n):
        order = sorted(names, key=lambda nm: out[nm][i], reverse=True)
        for nm in order[cfg.max_active_morphs:]:
            out[nm][i] = 0.0
        total = sum(out[nm][i] for nm in names)
        if total > cfg.total_cap:
            for nm in names:
                out[nm][i] *= cfg.total_cap / total
    # drop all-zero tracks
    return {k: v for k, v in out.items() if float(np.max(v)) > 1e-4}


def compress_morph_keys(weights: np.ndarray, eps: float = 1e-4
                        ) -> List[Tuple[int, float]]:
    """Sparse keys for a VMD morph track: emit changes, close zero runs."""
    keys: List[Tuple[int, float]] = []
    prev = 0.0
    for i, w in enumerate(weights):
        w = float(max(0.0, min(1.0, w)))
        if abs(w - prev) <= eps and not (w == 0.0 and prev != 0.0):
            continue
        if w == 0.0 and prev != 0.0:
            keys.append((i, 0.0))
            prev = 0.0
        else:
            keys.append((i, w))
            prev = w
    if prev != 0.0:
        keys.append((len(weights) - 1, 0.0))
    return keys
