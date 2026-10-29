# -*- coding: utf-8 -*-
"""M5 long-video support: chunk planning, cutting, alignment, VMD merge, seams.

The M4 pipeline processed a video as one clip.  A 60-120 s video (1,800-3,600
frames at 30 fps) does not fit the 8 GB GVHMR budget as a single sequence, so
M5 runs the *complete* per-model pipeline on overlapping chunks and merges the
resulting VMDs.

Design decisions (measured on the T5 probe, docs/M5_...):

* **Chunks overlap by construction.**  ``chunk_frames=300`` (10 s) and
  ``overlap_frames=30`` (1 s); consecutive chunks start ``chunk-overlap``
  frames apart, so the union is exactly ``[0, n)`` with no gaps or duplicates.
* **Every chunk is aligned before blending.**  GVHMR's global trajectory origin
  differs per clip and its root heading differs by a few degrees.  The alignment
  is a yaw about world Y (from the root orientation) plus a 3-D translation
  (from the root position in the overlap).  It is applied to the root track
  only: a global yaw left-multiplies the root local rotation and rotates the
  root translation; every other local rotation is unchanged.
* **Overlaps are cross-faded, not cut.**  Quaternions are slerped and the root
  translation is lerped with a smoothstep weight, so a boundary cannot become a
  hard switch.  Normalised weights sum to 1 on every frame.
* **The merge is auditable.**  ``merge.json`` records the plan, per-chunk
  yaw/translation/residual, frame coverage and the source step at each seam;
  the QC layer compares the merged seam steps/slide with the interior of the
  same clip (``boundary_report``).

The pipeline merges both ``before.vmd`` (deterministic M3 output) and
``final.vmd`` with the *same* transforms, so the "feedback must not touch
root/legs/hands/face" audit remains valid on the merged clip.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .cleanup import dense_tracks
from .face import compress_morph_keys
from .mmd.mathutil import qinv, qmul, qnorm
from .mmd.vmd import parse_vmd
from .mmd.writer import DEFAULT_CURVE, write_vmd

FPS = 30.0
DEFAULT_CHUNK_FRAMES = 300       # 10 s at 30 fps
DEFAULT_OVERLAP_FRAMES = 30      # 1 s
DEFAULT_MIN_TAIL_FRAMES = 30
DEFAULT_CRF = 16

# boundary gate floors (see boundary_report)
JUMP_ABS_DEG = 2.0
FREEZE_STEP_DEG = 0.02
FREEZE_MAX_FRAMES = 1
ROOT_STEP_ABS = 0.05             # MMD units per frame
TOE_STEP_ABS = 0.05
SLIDE_RATE_ABS = 0.005           # units per frame inside a seam window


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------
def plan_chunks(n_frames: int, chunk_frames: int = DEFAULT_CHUNK_FRAMES,
                overlap_frames: int = DEFAULT_OVERLAP_FRAMES,
                min_tail_frames: int = DEFAULT_MIN_TAIL_FRAMES) -> Dict:
    """Split ``[0, n_frames)`` into overlapping chunks.

    Returns explicit starts/ends (source frames, end exclusive).  Invariants
    (asserted, and covered by tests):

    * ``starts[0] == 0``, ``ends[-1] == n_frames``;
    * ``ends[i] == starts[i] + min(chunk_frames, n - starts[i])``;
    * consecutive chunks overlap by exactly ``overlap_frames``;
    * a tail shorter than ``min_tail_frames`` is absorbed by the previous chunk
      instead of becoming a 5-frame fragment.
    """
    n = int(n_frames)
    c = int(chunk_frames)
    o = int(overlap_frames)
    tail = int(min_tail_frames)
    if n <= 0:
        raise ValueError("n_frames must be > 0")
    if c <= 0:
        raise ValueError("chunk_frames must be > 0")
    if o < 0 or o >= c:
        raise ValueError("0 <= overlap_frames < chunk_frames")
    if tail < 0:
        raise ValueError("min_tail_frames must be >= 0")
    if n <= c:
        return {"n_frames": n, "chunk_frames": c, "overlap_frames": o,
                "min_tail_frames": tail,
                "starts": [0], "ends": [n], "lengths": [n],
                "output_offsets": [0], "n_chunks": 1}
    starts = [0]
    while starts[-1] + c < n:
        starts.append(starts[-1] + (c - o))
    ends = [min(n, s + c) for s in starts]
    if len(starts) > 1 and ends[-1] - starts[-1] < tail:
        starts.pop()
        ends.pop()
        ends[-1] = n
    else:
        ends[-1] = n
    lengths = [e - s for s, e in zip(starts, ends)]
    # A tail absorption can only make the last chunk longer than C by < tail.
    for i, L in enumerate(lengths):
        if L > c + tail:
            raise AssertionError(f"chunk {i} too long: {L} > {c + tail}")
    plan = {
        "n_frames": n, "chunk_frames": c, "overlap_frames": o,
        "min_tail_frames": tail,
        "starts": starts, "ends": ends, "lengths": lengths,
        # slow_motion is not supported across chunks; slow_factor==1 => offset=start
        "output_offsets": list(starts), "n_chunks": len(starts),
    }
    check_plan(plan)
    return plan


def check_plan(plan: Mapping) -> None:
    n = int(plan["n_frames"])
    starts = [int(x) for x in plan["starts"]]
    ends = [int(x) for x in plan["ends"]]
    o = int(plan["overlap_frames"])
    if starts[0] != 0 or ends[-1] != n:
        raise AssertionError("chunk plan does not cover [0, n)")
    for i in range(len(starts)):
        if ends[i] <= starts[i]:
            raise AssertionError(f"empty chunk {i}")
        if i == 0:
            continue
        overlap = ends[i - 1] - starts[i]
        if overlap != o:
            raise AssertionError(
                f"chunk {i - 1}->{i} overlap is {overlap}, expected {o}")
        if starts[i] >= ends[i - 1]:
            raise AssertionError(f"gap between chunk {i - 1} and {i}")
    # exact tiling of the output timeline (slow_factor == 1 by contract)
    seen = np.zeros(n, dtype=np.int32)
    for s, e in zip(starts, ends):
        seen[s:e] += 1
    if seen.min() < 1 or seen.max() > 2:
        raise AssertionError("chunk coverage is not 1-2 per frame")
    overlaps = [i for i in range(1, n) if seen[i] == 2]
    if len(overlaps) != o * (len(starts) - 1):
        raise AssertionError("overlap frame count mismatch")


# ---------------------------------------------------------------------------
# video io (ffmpeg)
# ---------------------------------------------------------------------------
def _run(cmd: Sequence[str], log: Optional[Path] = None) -> None:
    if log:
        with open(log, "a", encoding="utf-8") as f:
            f.write("CMD " + " ".join(str(c) for c in cmd) + "\n")
            f.flush()
            r = subprocess.run([str(c) for c in cmd], stdout=f,
                               stderr=subprocess.STDOUT)
    else:
        r = subprocess.run([str(c) for c in cmd], capture_output=True)
    if r.returncode != 0:
        tail = ""
        if not log:
            tail = (r.stderr or b"").decode("utf-8", "replace")[-2000:]
        raise RuntimeError(f"ffmpeg failed (exit {r.returncode}) {tail}")


def video_frames(path) -> int:
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def video_fps(path) -> float:
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return fps


def make_30fps_video(src, dst, ffmpeg: str = "ffmpeg", fps: int = 30,
                     crf: int = DEFAULT_CRF, log: Optional[Path] = None) -> Dict:
    """Re-encode ``src`` at ``fps`` so every later cut is frame-exact.

    A source that already runs at ``fps`` is copied verbatim: re-encoding it
    would add a generation without changing a single frame.
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    same_fps = abs(video_fps(src) - float(fps)) < 1e-3
    if same_fps:
        if dst.exists():
            dst.unlink()
        shutil.copyfile(src, dst)
    else:
        if dst.exists():
            dst.unlink()
        _run([ffmpeg, "-y", "-loglevel", "error", "-i", str(src),
              "-vf", f"fps={int(fps)}", "-an", "-c:v", "libx264",
              "-crf", str(int(crf)), "-pix_fmt", "yuv420p", str(dst)], log)
    n = video_frames(dst)
    if n < 2:
        raise RuntimeError(f"30 fps video has {n} frames: {dst}")
    return {"frames": int(n), "fps": int(fps), "path": str(dst),
            "copied": bool(same_fps)}


def cut_frames(src, dst, start: int, count: int, ffmpeg: str = "ffmpeg",
               crf: int = DEFAULT_CRF, log: Optional[Path] = None) -> Dict:
    """Cut an exact frame range ``[start, start+count)`` (re-encoded).

    Frame numbers are the *decoded* ones, not timestamps; the ``select``
    filter is used instead of input seeking so a chunk cannot start on a
    neighbouring frame.
    """
    start, count = int(start), int(count)
    if count <= 0:
        raise ValueError("count must be > 0")
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    last = start + count - 1
    sel = f"select='between(n\\,{start}\\,{last})'"
    _run([ffmpeg, "-y", "-loglevel", "error", "-i", str(src),
          "-vf", sel, "-vsync", "0", "-an", "-c:v", "libx264",
          "-crf", str(int(crf)), "-pix_fmt", "yuv420p", str(dst)], log)
    n = video_frames(dst)
    if n != count:
        raise RuntimeError(
            f"cut produced {n} frames, expected {count} ({src} {start}:{last})")
    return {"frames": n, "start": start, "end": last, "path": str(dst)}


# ---------------------------------------------------------------------------
# quaternion helpers (vectorised where it matters)
# ---------------------------------------------------------------------------
def _q_inv(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate([-q[..., :3], q[..., 3:]], axis=-1)


def _q_mul(a, b) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], axis=-1)


def _q_angle_deg(a, b) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = np.abs(np.sum(a * b, axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(d, -1.0, 1.0)))


def _slerp_batch(qa, qb, w) -> np.ndarray:
    qa = np.array(qa, dtype=np.float64, copy=True)
    qb = np.array(qb, dtype=np.float64, copy=True)
    w = np.asarray(w, dtype=np.float64).reshape(-1, 1)
    dot = np.sum(qa * qb, axis=-1, keepdims=True)
    flip = dot < 0.0
    qb = np.where(flip, -qb, qb)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    theta = np.arccos(dot)
    sin_t = np.sin(theta)
    small = sin_t < 1e-8
    w1 = np.where(small, 1.0 - w, np.sin((1.0 - w) * theta) / np.where(small, 1.0, sin_t))
    w2 = np.where(small, w, np.sin(w * theta) / np.where(small, 1.0, sin_t))
    q = w1 * qa + w2 * qb
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)


def _rot_y(deg: float) -> np.ndarray:
    a = math.radians(float(deg))
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _yaw_quat(deg: float) -> np.ndarray:
    a = math.radians(float(deg)) * 0.5
    return np.array([0.0, math.sin(a), 0.0, math.cos(a)], dtype=np.float64)


def _twist_y_deg(q) -> np.ndarray:
    """Yaw component of a quaternion (swing-twist about world Y)."""
    q = np.asarray(q, dtype=np.float64)
    y = q[..., 1]
    w = q[..., 3]
    return np.degrees(2.0 * np.arctan2(y, w))


def _circular_mean_deg(angles) -> Tuple[float, float]:
    a = np.radians(np.asarray(angles, dtype=np.float64))
    if len(a) == 0:
        return 0.0, 0.0
    c = float(np.cos(a).mean())
    s = float(np.sin(a).mean())
    mean = math.degrees(math.atan2(s, c))
    dev = np.degrees(np.arcsin(np.clip(np.abs(np.sin(a - math.radians(mean))), 0, 1)))
    return float(mean), float(np.sqrt(np.mean(dev ** 2)))


# ---------------------------------------------------------------------------
# per-chunk alignment
# ---------------------------------------------------------------------------
def _root_tracks(vmd: Mapping, profile: Mapping, n: int) -> Tuple[np.ndarray, np.ndarray]:
    name = profile["standard_map"]["root"]
    if name not in (vmd.get("bones") or {}):
        raise ValueError(f"root bone {name!r} missing from VMD")
    _, tracks = dense_tracks(vmd)
    return np.asarray(tracks[name]["rot"], np.float64), np.asarray(
        tracks[name]["pos"], np.float64)


def alignment(prev_vmd: Mapping, cur_vmd: Mapping, profile: Mapping,
              prev_frames: Sequence[int], cur_frames: Sequence[int]) -> Dict:
    """Rigid transform mapping ``cur`` onto ``prev`` in their overlap.

    ``prev_frames``/``cur_frames`` are VMD frame numbers of the same content.
    """
    if len(prev_frames) != len(cur_frames) or len(prev_frames) < 2:
        raise ValueError("overlap needs the same >=2 frames on both sides")
    qp, pp = _root_tracks(prev_vmd, profile, int(prev_vmd["max_frame"]) + 1)
    qc, pc = _root_tracks(cur_vmd, profile, int(cur_vmd["max_frame"]) + 1)
    pf = np.asarray(prev_frames, np.int64)
    cf = np.asarray(cur_frames, np.int64)
    p_prev, p_cur = pp[pf], pc[cf]
    q_prev, q_cur = qp[pf], qc[cf]

    # heading difference: q_yaw * q_cur == q_prev  ->  q_yaw = q_prev * q_cur^-1
    rel = _q_mul(q_prev, _q_inv(q_cur))
    yaw_i = _twist_y_deg(rel)
    yaw_deg, yaw_spread = _circular_mean_deg(yaw_i)
    ry = _rot_y(yaw_deg)
    p_cur_rot = (ry @ p_cur.T).T
    t = p_prev.mean(axis=0) - p_cur_rot.mean(axis=0)
    res = np.linalg.norm(p_prev - (p_cur_rot + t), axis=1)
    # diagnostics: the same fit from the translation path (only meaningful if
    # the person actually travels; reported, never used)
    xz_span = float(np.linalg.norm(p_cur[:, [0, 2]] - p_cur[:, [0, 2]].mean(0),
                                   axis=1).max())
    return {
        "yaw_deg": round(float(yaw_deg), 6),
        "yaw_spread_deg": round(float(yaw_spread), 6),
        "translation": [round(float(x), 9) for x in t],
        "residual_root_mean_units": round(float(res.mean()), 6),
        "residual_root_max_units": round(float(res.max()), 6),
        "root_rotation_diff_before_deg": {
            "mean": round(float(_q_angle_deg(q_prev, q_cur).mean()), 4),
            "max": round(float(_q_angle_deg(q_prev, q_cur).max()), 4)},
        "root_rotation_residual_after_deg": {
            "mean": round(float(_q_angle_deg(
                q_prev, _q_mul(_yaw_quat(yaw_deg), q_cur)).mean()), 4),
            "max": round(float(_q_angle_deg(
                q_prev, _q_mul(_yaw_quat(yaw_deg), q_cur)).max()), 4)},
        "xz_path_span_units": round(xz_span, 4),
        "overlap_frames": [int(prev_frames[0]), int(prev_frames[-1])],
        "note": ("yaw from the root world orientation (twist about Y); a "
                 "translation-only Kabsch yaw is unreliable for in-place "
                 "dances and is not used"),
    }


def apply_alignment(vmd: Mapping, profile: Mapping, yaw_deg: float,
                    translation: Sequence[float]) -> Dict:
    """Apply a world yaw + translation to the root track (copy-on-write)."""
    root = profile["standard_map"]["root"]
    if root not in (vmd.get("bones") or {}):
        raise ValueError(f"root bone {root!r} missing from VMD")
    ry = _rot_y(yaw_deg)
    qy = _yaw_quat(yaw_deg)
    t = np.asarray(translation, dtype=np.float64)
    out = dict(vmd)
    out["bones"] = dict(vmd["bones"])
    keys = []
    for k in vmd["bones"][root]:
        p = ry @ np.asarray(k["pos"], np.float64) + t
        q = _q_mul(qy, np.asarray(k["rot"], np.float64))
        q = q / max(1e-12, float(np.linalg.norm(q)))
        keys.append({**k, "pos": tuple(float(x) for x in p),
                     "rot": tuple(float(x) for x in q)})
    out["bones"][root] = keys
    return out


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------
def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def merge_weights(offsets: Sequence[int], lengths: Sequence[int],
                  n_out: int, overlap: int) -> np.ndarray:
    """(n_out, n_parts) normalised cross-fade weights."""
    n_parts = len(offsets)
    if len(lengths) != n_parts:
        raise ValueError("offsets/lengths mismatch")
    W = np.zeros((int(n_out), n_parts), dtype=np.float64)
    for i, (off, L) in enumerate(zip(offsets, lengths)):
        off, L = int(off), int(L)
        w = np.ones(L, dtype=np.float64)
        if i > 0:
            o = int(offsets[i - 1]) + int(lengths[i - 1]) - off
            if o != overlap:
                raise ValueError(f"chunk {i}: overlap is {o}, expected {overlap}")
            k = min(o, L)
            w[:k] = _smoothstep((np.arange(k) + 0.5) / float(o))
        if i < n_parts - 1:
            nxt = int(offsets[i + 1])
            o = off + L - nxt
            if o != overlap:
                raise ValueError(f"chunk {i}: overlap is {o}, expected {overlap}")
            k = min(o, L)
            w[L - k:] = 1.0 - _smoothstep((np.arange(k) + 0.5) / float(o))
        W[off:off + L, i] = w
    sums = W.sum(axis=1)
    if np.any(sums <= 1e-9):
        bad = np.where(sums <= 1e-9)[0][:5].tolist()
        raise ValueError(f"frames without a chunk: {bad}")
    return W / sums[:, None]


def _morph_dense(keys, n: int) -> np.ndarray:
    if not keys:
        return np.zeros(int(n), dtype=np.float64)
    fr = np.asarray([k[0] for k in keys], dtype=np.float64)
    w = np.asarray([k[1] for k in keys], dtype=np.float64)
    return np.interp(np.arange(int(n), dtype=np.float64), fr, w,
                     left=w[0], right=w[-1])


def merge_vmds(parts: Sequence[Mapping], offsets: Sequence[int], n_out: int,
               overlap: int) -> Tuple[Dict, Dict]:
    """Cross-fade already-aligned chunk VMDs into one VMD.

    All parts must carry the same bone set (the 53-track contract); every
    frame of ``[0, n_out)`` must be covered by at least one part.
    """
    n_out = int(n_out)
    if not parts:
        raise ValueError("no parts to merge")
    lengths = [int(p["max_frame"]) + 1 for p in parts]
    W = merge_weights(offsets, lengths, n_out, overlap)
    bone_sets = [set(p.get("bones") or {}) for p in parts]
    if any(s != bone_sets[0] for s in bone_sets[1:]):
        diff = [sorted(s ^ bone_sets[0]) for s in bone_sets[1:]]
        raise ValueError(f"chunk bone sets differ: {diff[:3]}")
    bones = sorted(bone_sets[0])
    dense = [dense_tracks(p)[1] for p in parts]

    out_bones: Dict[str, List[Dict]] = {b: [] for b in bones}
    for b in bones:
        arrs = [np.asarray(d[b]["rot"], np.float64) for d in dense]
        poss = [np.asarray(d[b]["pos"], np.float64) for d in dense]
        rot = np.zeros((n_out, 4), dtype=np.float64)
        pos = np.zeros((n_out, 3), dtype=np.float64)
        for f in range(n_out):
            idx = np.where(W[f] > 0.0)[0]
            if len(idx) == 1:
                i = int(idx[0])
                rot[f] = arrs[i][f - int(offsets[i])]
                pos[f] = poss[i][f - int(offsets[i])]
            elif len(idx) == 2:
                i, j = int(idx[0]), int(idx[1])
                wi, wj = W[f, i], W[f, j]
                q = _slerp_batch(arrs[i][f - int(offsets[i])][None, :],
                                 arrs[j][f - int(offsets[j])][None, :],
                                 [wj / max(1e-12, wi + wj)])[0]
                rot[f] = q
                pos[f] = (wi * poss[i][f - int(offsets[i])]
                          + wj * poss[j][f - int(offsets[j])]) / (wi + wj)
            else:
                raise ValueError(f"frame {f} covered by {len(idx)} chunks")
        # VMD interpolates between neighbouring keys: q and -q are the same
        # rotation but interpolate the long way, so make the track continuous.
        for f in range(1, n_out):
            if float(np.dot(rot[f], rot[f - 1])) < 0.0:
                rot[f] = -rot[f]
        for f in range(n_out):
            out_bones[b].append({
                "frame": f,
                "pos": tuple(float(x) for x in pos[f]),
                "rot": tuple(float(x) for x in rot[f]),
                "curves": ((20, 20, 107, 107),) * 4,
            })
    # curves: copy the per-bone interpolation from the first part so the
    # merge does not silently switch every key to the default curve
    for b in bones:
        src = parts[0]["bones"][b][0].get("curves") if parts[0]["bones"][b] else None
        if src:
            for row in out_bones[b]:
                row["curves"] = src

    # morphs: evaluate, blend, compress
    names = sorted({m for p in parts for m in (p.get("morphs") or {})})
    out_morphs: Dict[str, List[Tuple[int, float]]] = {}
    for m in names:
        dense_m = [_morph_dense(p.get("morphs", {}).get(m) or [], L)
                   for p, L in zip(parts, lengths)]
        vals = np.zeros(n_out, dtype=np.float64)
        for f in range(n_out):
            idx = np.where(W[f] > 0.0)[0]
            acc = 0.0
            for i in idx:
                i = int(i)
                acc += W[f, i] * dense_m[i][f - int(offsets[i])]
            vals[f] = acc
        keys = compress_morph_keys(np.clip(vals, 0.0, 1.0))
        if keys:
            # MMD holds the first key's value before it, so make an explicit
            # zero at frame 0 when the track starts silent; dedupe same-frame
            # keys (compress can append a closing 0 on the last frame).
            if keys[0][0] > 0 and float(vals[0]) <= 1e-9:
                keys.insert(0, (0, 0.0))
            dedup = {}
            for k in keys:
                dedup[int(k[0])] = float(k[1])
            out_morphs[m] = sorted(dedup.items())

    # IK: one frame-0 state with every involved IK bone off (VMD state persists)
    ik_names = sorted({name for p in parts for fr in (p.get("ik_frames") or [])
                       for name in (fr.get("ik") or {})})
    ik_frames = ([{"frame": 0, "show": True,
                   "ik": {n: False for n in ik_names}}] if ik_names else [])

    merged = {"model_name": parts[0].get("model_name", ""),
              "bones": out_bones, "morphs": out_morphs,
              "ik_frames": ik_frames, "max_frame": n_out - 1}
    report = {
        "frames": n_out,
        "bones": len(bones),
        "morph_tracks": sorted(out_morphs),
        "ik_off": sorted(ik_names),
        "overlap_frames": int(overlap),
        "parts": [{"offset": int(o), "length": int(L)} for o, L in zip(offsets, lengths)],
        "weights": {"min_sum": float(W.sum(1).min()), "max_sum": float(W.sum(1).max())},
        "ok": bool(all(len(v) == n_out for v in out_bones.values())),
    }
    if not report["ok"]:
        raise AssertionError("merged tracks are not dense")
    return merged, report


# ---------------------------------------------------------------------------
# certificates / feedback / merge report
# ---------------------------------------------------------------------------
def merge_feedback(reports: Sequence[Mapping], offsets: Sequence[int],
                   lengths: Sequence[int], mode: str) -> Dict:
    """Aggregate per-chunk feedback.json (fallback windows shifted to global)."""
    if not reports:
        raise ValueError("no feedback reports")
    ckpts = {r.get("ckpt") for r in reports}
    shas = {r.get("ckpt_sha256") for r in reports}
    modes = {r.get("mode") for r in reports}
    ckpt_mismatch = len(ckpts) > 1 or len(shas) > 1
    mode_mismatch = len(modes) > 1
    total = max(1, int(sum(lengths)))
    groups = ("root", "body", "hands", "face")
    mean_abs = {}
    raw_mean_abs = {}
    for g in groups:
        vals = [float((r.get("residual_mean_abs") or {}).get(g, 0.0)) * int(L)
                for r, L in zip(reports, lengths)]
        mean_abs[g] = round(sum(vals) / total, 6)
        vals = [float((r.get("raw_residual_mean_abs") or {}).get(g, 0.0)) * int(L)
                for r, L in zip(reports, lengths)]
        raw_mean_abs[g] = round(sum(vals) / total, 6)
    fallback = []
    for i, (r, off) in enumerate(zip(reports, offsets)):
        for w in (r.get("segments_fallback") or []):
            fallback.append({
                "chunk": int(i),
                "start": int(off) + int(w.get("start", 0)),
                "end": int(off) + int(w.get("end", 0)),
                "reasons": list(w.get("reasons") or []),
                "before": w.get("before"), "after": w.get("after"),
            })
    return {
        "mode": mode,
        "config": reports[0].get("config"),
        "applied": any(bool(r.get("applied")) for r in reports),
        "ckpt": next(iter(ckpts)) if len(ckpts) == 1 else None,
        "ckpt_sha256": next(iter(shas)) if len(shas) == 1 else None,
        "ckpt_mismatch": bool(ckpt_mismatch),
        "mode_mismatch": bool(mode_mismatch),
        "segments_fallback": fallback,
        "whole_clip_fallback": any(bool(r.get("whole_clip_fallback")) for r in reports),
        "handface_fallback": any(bool(r.get("handface_fallback")) for r in reports),
        "residual_mean_abs": mean_abs,
        "raw_residual_mean_abs": raw_mean_abs,
        "per_chunk": [{"offset": int(o), "length": int(L),
                       "whole_clip_fallback": bool(r.get("whole_clip_fallback")),
                       "segments_fallback": len(r.get("segments_fallback") or [])}
                      for r, o, L in zip(reports, offsets, lengths)],
        "note": "per-chunk M3.5 feedback merged; fallback windows are global frames",
    }


# ---------------------------------------------------------------------------
# boundary continuity
# ---------------------------------------------------------------------------
def local_step_series(vmd: Mapping, n: int,
                      bones: Optional[Iterable[str]] = None,
                      tracks: Optional[Mapping] = None) -> np.ndarray:
    """Per-frame max local-quaternion step (degrees) over the selected bones."""
    if tracks is None:
        _, tracks = dense_tracks(vmd)
    names = [b for b in (bones or tracks.keys()) if b in tracks]
    out = np.zeros(int(n), dtype=np.float64)
    for name in names:
        q = np.asarray(tracks[name]["rot"], np.float64)
        if len(q) < 2:
            continue
        step = _q_angle_deg(q[1:], q[:-1])
        out[1:len(step) + 1] = np.maximum(out[1:len(step) + 1], step)
    return out


def root_step_series(vmd: Mapping, profile: Mapping, n: int,
                     tracks: Optional[Mapping] = None) -> np.ndarray:
    root = profile["standard_map"]["root"]
    if tracks is None:
        _, tracks = dense_tracks(vmd)
    p = np.asarray(tracks[root]["pos"], np.float64)
    out = np.zeros(int(n), dtype=np.float64)
    if len(p) > 1:
        out[1:len(p)] = np.linalg.norm(np.diff(p, axis=0), axis=1)
    return out


def _toe_step_series(track: Mapping, n: int) -> np.ndarray:
    out = np.zeros(int(n), dtype=np.float64)
    for side in ("l", "r"):
        k = f"toe_{side}"
        if k not in (track.get("world") or {}):
            continue
        p = np.asarray(track["world"][k], np.float64)
        if len(p) > 1:
            s = np.linalg.norm(np.diff(p, axis=0), axis=1)
            out[1:len(s) + 1] = np.maximum(out[1:len(s) + 1], s)
    return out


def _longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for x in mask:
        if x:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def boundary_report(before_track: Mapping, final_track: Mapping,
                    final_vmd: Mapping, profile: Mapping,
                    boundaries: Sequence[int], overlap: int, n: int,
                    source_max_step_deg: Optional[Sequence[float]] = None,
                    source_max_root_step: Optional[Sequence[float]] = None,
                    source_max_toe_step: Optional[Sequence[float]] = None) -> Dict:
    """Did the merge add a freeze / jump / foot-slide spike at any seam?

    Reference is the *interior of the same clip* (seam windows excluded), so a
    fast dance is not judged against an absolute step.  A genuine source
    discontinuity (a cut between concatenated test clips) is captured by the
    ``source_*`` arguments and is not blamed on the merge.
    """
    from . import qc

    n = int(n)
    height = float(profile.get("height") or 1.0)
    _, ftracks = dense_tracks(final_vmd)
    step = local_step_series(final_vmd, n, tracks=ftracks)
    root_step = root_step_series(final_vmd, profile, n, tracks=ftracks)
    toe_step = _toe_step_series(final_track, n)
    # before (pre-feedback) world-rotation steps: reporting only, so a seam
    # jump introduced by the feedback layer would be visible next to the final
    before_step = np.zeros(n, dtype=np.float64)
    for name, q in (before_track.get("world_q") or {}).items():
        q = np.asarray(q, np.float64)
        if len(q) < 2:
            continue
        s = _q_angle_deg(q[1:], q[:-1])
        before_step[1:len(s) + 1] = np.maximum(before_step[1:len(s) + 1], s)

    seam_windows = []
    for b in boundaries:
        b = int(b)
        seam_windows.append((max(1, b - 2), min(n - 1, b + int(overlap) + 2)))
    interior = np.ones(n, dtype=bool)
    interior[:2] = False
    interior[-2:] = False
    for a, z in seam_windows:
        interior[a:z + 1] = False
    if interior.sum() < 10:  # extremely short clip: fall back to everything
        interior[:] = True
    ref = {
        "frames": int(interior.sum()),
        "step_p95_deg": float(np.percentile(step[interior], 95)),
        "step_p99_deg": float(np.percentile(step[interior], 99)),
        "step_median_deg": float(np.percentile(step[interior], 50)),
        "root_step_p95": float(np.percentile(root_step[interior], 95)),
        "toe_step_p95": float(np.percentile(toe_step[interior], 95)),
        "freeze_run_max": int(_longest_run(step[interior] < FREEZE_STEP_DEG)),
    }
    # interior slide per-frame rate, same window length as a seam window
    win_len = max(8, int(overlap) + 4)
    rates = []
    centers = list(range(win_len // 2, max(win_len // 2 + 1, n - win_len // 2),
                         max(5, win_len // 2)))
    for c in centers:
        a, z = max(0, c - win_len // 2), min(n - 1, c + win_len // 2)
        if a >= z:
            continue
        if not interior[a:z + 1].all():
            continue
        w = qc.window_metrics(final_track, a, z, height)
        rates.append(w["foot_slide"] / max(1, z - a + 1))
    ref["slide_rate_p95"] = float(np.percentile(rates, 95)) if rates else 0.0
    ref["slide_windows"] = len(rates)

    seams = []
    for k, b in enumerate(boundaries):
        b = int(b)
        a = max(0, b)
        z = min(n - 1, b + int(overlap) - 1)
        w0, w1 = seam_windows[k]
        win_step = float(step[a:z + 1].max()) if z >= a else 0.0
        win_root = float(root_step[a:z + 1].max()) if z >= a else 0.0
        win_toe = float(toe_step[a:z + 1].max()) if z >= a else 0.0
        freeze = int(_longest_run(step[a:z + 1] < FREEZE_STEP_DEG))
        wm = qc.window_metrics(final_track, w0, w1, height)
        slide_rate = float(wm["foot_slide"]) / max(1, w1 - w0 + 1)
        src = None
        if source_max_step_deg is not None and k < len(source_max_step_deg):
            v = source_max_step_deg[k]
            src = None if v is None else float(v)
        src_root = None
        if source_max_root_step is not None and k < len(source_max_root_step):
            v = source_max_root_step[k]
            src_root = None if v is None else float(v)
        src_toe = None
        if source_max_toe_step is not None and k < len(source_max_toe_step):
            v = source_max_toe_step[k]
            src_toe = None if v is None else float(v)
        jump_limit = max(ref["step_p95_deg"] * 1.5, JUMP_ABS_DEG)
        if src is not None:
            jump_limit = max(jump_limit, float(src) * 1.15)
        root_limit = max(ref["root_step_p95"] * 1.5, ROOT_STEP_ABS,
                         (src_root or 0.0) * 1.15)
        toe_limit = max(ref["toe_step_p95"] * 1.5, TOE_STEP_ABS,
                        (src_toe or 0.0) * 1.15)
        checks = {
            "jump": win_step <= jump_limit,
            "root_step": win_root <= root_limit,
            "toe_step": win_toe <= toe_limit,
            "freeze": freeze <= max(ref["freeze_run_max"], FREEZE_MAX_FRAMES),
            "slide": slide_rate <= max(ref["slide_rate_p95"] * 1.5, SLIDE_RATE_ABS),
        }
        seams.append({
            "frame": b,
            "window": [a, z],
            "max_step_deg": round(win_step, 4),
            "before_world_step_max_deg": round(
                float(before_step[a:z + 1].max()) if z >= a else 0.0, 4),
            "root_step_max": round(win_root, 5),
            "toe_step_max": round(win_toe, 5),
            "freeze_run": freeze,
            "slide": round(float(wm["foot_slide"]), 6),
            "slide_rate": round(slide_rate, 6),
            "source_max_step_deg": None if src is None else round(float(src), 4),
            "source_max_root_step": None if src_root is None else round(float(src_root), 5),
            "source_max_toe_step": None if src_toe is None else round(float(src_toe), 5),
            "limits": {"jump": round(jump_limit, 4),
                       "root_step": round(root_limit, 5),
                       "toe_step": round(toe_limit, 5),
                       "slide_rate": round(max(ref["slide_rate_p95"] * 1.5, SLIDE_RATE_ABS), 6)},
            "checks": {k: bool(v) for k, v in checks.items()},
            "ok": bool(all(checks.values())),
        })
    out = {
        "enabled": True,
        "frames": n,
        "overlap_frames": int(overlap),
        "seams": seams,
        "reference": {k: (round(v, 6) if isinstance(v, float) else v)
                      for k, v in ref.items()},
        "ok": bool(all(s["ok"] for s in seams)),
        "note": ("seam windows are compared with the interior of the same clip "
                 "(p95 x1.5 + absolute floors); a source discontinuity from a "
                 "hard content cut is allowed via source_max_step_deg"),
    }
    return out


# ---------------------------------------------------------------------------
# small io helpers used by the pipeline
# ---------------------------------------------------------------------------
def load_json(path) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=_json_default)
    tmp.replace(path)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")
