# -*- coding: utf-8 -*-
"""M3.5 paired-data schema, low-dimensional residual targets and evidence features.

Boundary rules enforced here:

* **Inference features** (``build_features``) read only the current predicted
  pose, its velocities, 2D keypoints/confidence, contact state, uncertainty and
  the target profile.  They never read GT, GT morphs or source pixels.
* **Residual targets** (``extract_residuals``) read GT only as a teacher.
  They are low dimensional and bounded: root XZ, 9 upper-body bones (local
  rotation vector in the parent frame), per-hand 19 anatomical angles and 12
  face channels.  Legs/hips stay with the deterministic contact IK.
* Hands/face are supervised only on M3 L0/L1 frames; L2/L3 fallback frames are
  masked out so the network cannot learn "default slight flexion".
* Splits are whole segments (motion x model x camera).  The test motion (R3)
  never appears in train/val; ``tiflos`` and the ``side`` camera are held out of
  training.

The module also contains the application path (``apply_*``) so the same code
that makes training targets makes the corrected VMD used for evaluation.
"""
from __future__ import annotations

import copy
import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .mmd.mathutil import QID, qaxis, qinv, qmul, qnorm, qrot, vlen, vnorm

# ---------------------------------------------------------------------------
# factor layout
# ---------------------------------------------------------------------------
# (bone, reference bone used by the body-relative metric)
BODY_FACTORS: Tuple[Tuple[str, str], ...] = (
    # Learned corrections cover the upper body + root XZ only.
    #
    # Legs and hips are deliberately excluded: the root Y is never corrected
    # (guide rule), so a learned leg rotation computed against the GT local
    # pose fights M2.2 contact IK and lifts the feet off the floor.  The
    # deterministic contact IK stays authoritative for legs; the leg pose is
    # still an input feature (context) and a reported metric.
    ("spine", "pelvis"),
    ("spine1", "spine"),
    ("chest", "spine1"),
    # The clavicles must be corrected too: GVHMR's shoulder error is large and
    # an upper-arm local correction would otherwise rotate the arm inside a
    # wrong shoulder frame.
    ("clavicle_l", "chest"),
    ("upperarm_l", "clavicle_l"),
    ("forearm_l", "upperarm_l"),
    ("clavicle_r", "chest"),
    ("upperarm_r", "clavicle_r"),
    ("forearm_r", "upperarm_r"),
)
BODY_FACTOR_NAMES = tuple(b for b, _ in BODY_FACTORS)
BODY_FACTOR_REF = dict(BODY_FACTORS)
# Metric references are deliberately *not* the residual parents: the upper arm
# is scored relative to the chest, so a shoulder/clavicle error still counts.
METRIC_FACTORS: Tuple[Tuple[str, str], ...] = (
    ("spine", "pelvis"), ("spine1", "spine"), ("chest", "spine1"),
    ("upperarm_l", "chest"), ("forearm_l", "upperarm_l"),
    ("upperarm_r", "chest"), ("forearm_r", "upperarm_r"),
    ("thigh_l", "pelvis"), ("shin_l", "thigh_l"), ("foot_l", "shin_l"),
    ("thigh_r", "pelvis"), ("shin_r", "thigh_r"), ("foot_r", "shin_r"),
)
# child joint whose position defines the bone direction (for metrics)
BONE_CHILD = {
    "spine": "spine1", "spine1": "chest", "chest": "neck",
    "clavicle_l": "upperarm_l", "upperarm_l": "forearm_l", "forearm_l": "hand_l",
    "clavicle_r": "upperarm_r", "upperarm_r": "forearm_r", "forearm_r": "hand_r",
    "thigh_l": "shin_l", "shin_l": "foot_l", "foot_l": "toe_l",
    "thigh_r": "shin_r", "shin_r": "foot_r", "foot_r": "toe_r",
}
# Evidence directions: every bone with a well-defined direction, including the
# legs even though the network does not correct them (context for the upper
# body / root correction).  The pelvis has two leg children, so it is
# represented by its local quaternion instead.
FEATURE_DIR_BONES = (
    "spine", "spine1", "chest",
    "clavicle_l", "upperarm_l", "forearm_l",
    "clavicle_r", "upperarm_r", "forearm_r",
    "thigh_l", "shin_l", "foot_l",
    "thigh_r", "shin_r", "foot_r",
)

# Per-component bound on the residual rotation vector (degrees) in the parent
# frame.  Arms carry the largest systematic error on anime footage; legs are
# tighter because contact IK already solved the foot.
# Degrees, per rotvec component.  GVHMR/SMPL-X and the PMX rest pose split the
# torso bend between spine1/chest and the shoulder bend between clavicle/upper
# arm very differently: measured GT-vs-pred local deltas reach ~80-100 deg even
# when the *world* chest direction is only ~8-14 deg off.  A +-15 deg bound
# cannot represent that, so the bound covers the measured decomposition.  The
# output is still a bounded, low-dimensional local correction.
ROT_BOUND = {
    "spine": 30.0, "spine1": 90.0, "chest": 90.0,
    "clavicle_l": 60.0, "upperarm_l": 90.0, "forearm_l": 90.0,
    "clavicle_r": 60.0, "upperarm_r": 90.0, "forearm_r": 90.0,
}
# M6 teacher variant: the 60 deg clavicle bound is the measured bottleneck
# (M3.5 leaves ~24 deg world shoulder error).  The wider bound lets the
# teacher express a shoulder chain correction; it is opt-in so the M3.5
# checkpoint and its limits remain byte-compatible.
ROT_BOUND_V2 = dict(ROT_BOUND, clavicle_l=120.0, clavicle_r=120.0,
                    upperarm_l=150.0, upperarm_r=150.0,
                    forearm_l=150.0, forearm_r=150.0)
ROOT_BOUND = 0.8          # MMD units per component (XZ only; Y is never corrected)
HAND_BOUND = {
    # (per-joint bound in degrees)
    "spread": 12.0, "mcp": 20.0, "pip": 20.0, "dip": 16.0,
    "cmc": 15.0, "thumb_mcp": 20.0, "thumb_ip": 20.0,
}
FACE_BOUND = 0.30         # morph weight per channel

HAND_JOINT_KEYS = tuple(
    [f"{f}.{k}" for f in ("index", "middle", "ring", "pinky")
     for k in ("spread", "mcp", "pip", "dip")]
    + ["thumb.cmc", "thumb.mcp", "thumb.ip"]
)

# residual vector layout
ROOT_SLICE = slice(0, 2)
ROT_SLICE = slice(2, 2 + 3 * len(BODY_FACTOR_NAMES))
HAND_SLICE = slice(ROT_SLICE.stop, ROT_SLICE.stop + 2 * len(HAND_JOINT_KEYS))
FACE_SLICE = slice(HAND_SLICE.stop, HAND_SLICE.stop + 12)
RESIDUAL_DIM = FACE_SLICE.stop

ROT_BOUND_VEC = np.asarray(
    [ROT_BOUND[b] for b in BODY_FACTOR_NAMES for _ in range(3)], np.float32)
ROT_BOUND_VEC_V2 = np.asarray(
    [ROT_BOUND_V2[b] for b in BODY_FACTOR_NAMES for _ in range(3)], np.float32)


def _hand_joint_limit(key: str) -> float:
    finger, joint = key.split(".")
    if finger == "thumb":
        return HAND_BOUND[{"cmc": "cmc", "mcp": "thumb_mcp",
                           "ip": "thumb_ip"}[joint]]
    return HAND_BOUND[joint]


HAND_BOUND_VEC = np.asarray([_hand_joint_limit(k) for k in HAND_JOINT_KEYS] * 2,
                            np.float32)
FACE_BOUND_VEC = np.full(12, FACE_BOUND, np.float32)
ROOT_BOUND_VEC = np.full(2, ROOT_BOUND, np.float32)


def residual_limits(version: str = "v1") -> np.ndarray:
    """(RESIDUAL_DIM,) positive per-component bounds.

    ``v1`` is the M3.5 checkpoint contract; ``v2`` widens the shoulder chain
    only (see ``ROT_BOUND_V2``).
    """
    rot = ROT_BOUND_VEC_V2 if version == "v2" else ROT_BOUND_VEC
    return np.concatenate([ROOT_BOUND_VEC, rot, HAND_BOUND_VEC,
                           FACE_BOUND_VEC]).astype(np.float32)


def residual_groups() -> Dict[str, slice]:
    return {"root": ROOT_SLICE, "body": ROT_SLICE,
            "hands": HAND_SLICE, "face": FACE_SLICE}


# ---------------------------------------------------------------------------
# GT face: raw motion VMD morph names -> 12 semantic channels
# ---------------------------------------------------------------------------
RAW_MORPH_SEMANTICS: Dict[str, Tuple[Tuple[str, float], ...]] = {
    "blink": (("まばたき", 1.0),),
    "wink_l": (("ウィンク", 1.0), ("ウィンク２", 1.0)),
    "wink_r": (("ウィンク右", 1.0), ("ｳｨﾝｸ２右", 1.0), ("ウィンク２右", 1.0)),
    "smile": (("笑い", 1.0), ("にこり", 1.0), ("にっこり", 1.0),
              ("にやり２", 0.8), ("なごみ", 0.8)),
    "surprise": (("びっくり", 1.0),),
    "angry": (("怒り", 1.0), ("怒り目", 1.0)),
    "sad": (("困る", 1.0), ("悲しむ", 1.0), ("悲しい目", 1.0)),
    "mouth_a": (("あ", 1.0), ("あ２", 1.0), ("ワ", 1.0)),
    "mouth_i": (("い", 1.0),),
    "mouth_u": (("う", 1.0),),
    "mouth_e": (("え", 1.0),),
    "mouth_o": (("お", 1.0),),
}
SEMANTIC = tuple(RAW_MORPH_SEMANTICS.keys())


def dense_morph(vmd: Mapping, name: str, n_frames: int) -> Optional[np.ndarray]:
    """Linear morph track sampled per frame (MMD holds first/last key)."""
    keys = vmd.get("morphs", {}).get(name)
    if not keys:
        return None
    fr = np.asarray([k[0] for k in keys], float)
    w = np.asarray([k[1] for k in keys], float)
    x = np.arange(n_frames, dtype=float)
    out = np.interp(x, fr, w, left=w[0], right=w[-1])
    return np.clip(out, 0.0, 1.0)


def gt_semantic_weights(raw_vmd: Mapping, n_frames: int,
                        target_names: Optional[Mapping[str, str]] = None
                        ) -> Tuple[np.ndarray, Dict[str, str]]:
    """Raw VMD morphs -> (N,12) semantic GT weights + per-channel source.

    ``target_names`` maps semantic -> morph name on the target model; channels
    whose morph does not exist there are masked to zero and recorded as
    missing (they cannot be written, so they are not supervised).
    """
    n = len(SEMANTIC)
    out = np.zeros((n_frames, n), float)
    sources: Dict[str, str] = {}
    for i, sem in enumerate(SEMANTIC):
        if target_names is not None and sem not in target_names:
            continue
        best = np.zeros(n_frames, float)
        src = None
        for raw_name, scale in RAW_MORPH_SEMANTICS[sem]:
            d = dense_morph(raw_vmd, raw_name, n_frames)
            if d is None:
                continue
            if float(d.max()) * scale > float(best.max()):
                src = raw_name
            best = np.maximum(best, d * scale)
        if src is not None and float(best.max()) > 1e-4:
            out[:, i] = np.clip(best, 0.0, 1.0)
            sources[sem] = src
    return out, sources


# ---------------------------------------------------------------------------
# GT hands: evaluate the GT VMD and measure the same 19 anatomical angles
# ---------------------------------------------------------------------------
def _synthetic_landmarks(model: Mapping, profile: Mapping, vmd: Mapping,
                         frames: Sequence[int]) -> Dict[str, np.ndarray]:
    """(N,2,21,3) hand landmarks built from GT bone directions.

    MediaPipe measures angles from 21 points; feeding it the GT joint origins
    (plus the distal tip reconstructed from the bone's own rotation) gives a GT
    angle vector in exactly the same convention as the predicted one.
    """
    from .hands import HandRig
    from .mmd.evaluate import VmdEvaluator

    rigs = {s: HandRig.from_profile(profile, s) for s in ("l", "r")}
    name2i = {b["name"]: i for i, b in enumerate(model["bones"])}
    ev = VmdEvaluator(model, vmd, solve_ik=False, force_ik_off=True)
    # MediaPipe landmark order
    order = {
        "thumb": (1, 2, 3, 4),
        "index": (5, 6, 7, 8),
        "middle": (9, 10, 11, 12),
        "ring": (13, 14, 15, 16),
        "pinky": (17, 18, 19, 20),
    }
    out = {s: np.full((len(frames), 21, 3), np.nan) for s in ("l", "r")}
    for fi, f in enumerate(frames):
        rec = ev.frame(int(f))
        for side in ("l", "r"):
            rig = rigs[side]
            wrist = name2i[rig.wrist_name]
            pts = np.zeros((21, 3))
            pts[0] = rec["world_pos"][wrist]
            # world orientation of the wrist, then down each finger chain
            q_parent = rec["world_q"][wrist]
            for finger, idxs in order.items():
                chain = rig.bones.get(finger, [])
                if not chain:
                    continue
                qc = q_parent
                for k, bone in enumerate(chain):
                    bi = bone.index
                    pts[idxs[k]] = rec["world_pos"][bi]
                    q_local = rec["local_q"][bi]
                    qc = qnorm(qmul(qc, q_local))
                    if k == len(chain) - 1:
                        tip = (np.asarray(rec["world_pos"][bi])
                               + np.asarray(qrot(qc, bone.rest_dir)) * bone.length)
                        pts[idxs[k] + 1] = tip
            out[side][fi] = pts
    return out


def gt_hand_angles(model: Mapping, profile: Mapping, vmd: Mapping,
                   frames: Sequence[int]) -> np.ndarray:
    """(N,2,19) GT anatomical angles, same convention as run_hand angles3d."""
    from .hands import JOINT_INDEX, mp_joint_angles

    lm = _synthetic_landmarks(model, profile, vmd, frames)
    out = np.zeros((len(frames), 2, len(HAND_JOINT_KEYS)), np.float32)
    for si, side in enumerate(("l", "r")):
        for fi in range(len(frames)):
            p = lm[side][fi]
            if not np.all(np.isfinite(p)):
                continue
            a = mp_joint_angles(p)
            vec = out[fi, si]
            for j, key in enumerate(HAND_JOINT_KEYS):
                finger, joint = key.split(".")
                vec[j] = a.get(finger, {}).get(joint, 0.0)
    return out


# ---------------------------------------------------------------------------
# GT fingers: remap the source hand's anatomical directions onto a target rig
# ---------------------------------------------------------------------------
def _swing(a, b):
    a = vnorm(tuple(a))
    b = vnorm(tuple(b))
    dot = max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))
    cross = (a[1] * b[2] - a[2] * b[1],
             a[2] * b[0] - a[0] * b[2],
             a[0] * b[1] - a[1] * b[0])
    s = vlen(cross)
    if s < 1e-9:
        if dot > 0:
            return QID
        axis = (1.0, 0.0, 0.0) if abs(a[0]) < 0.9 else (0.0, 1.0, 0.0)
        c = vlen((a[1] * axis[2] - a[2] * axis[1],
                  a[2] * axis[0] - a[0] * axis[2],
                  a[0] * axis[1] - a[1] * axis[0]))
        axis = ((a[1] * axis[2] - a[2] * axis[1]) / c,
                (a[2] * axis[0] - a[0] * axis[2]) / c,
                (a[0] * axis[1] - a[1] * axis[0]) / c)
        return qaxis(axis, math.pi)
    return qaxis((cross[0] / s, cross[1] / s, cross[2] / s), math.atan2(s, dot))


def remap_fingers(src_model: Mapping, src_profile: Mapping, src_vmd: Mapping,
                  tgt_model: Mapping, tgt_profile: Mapping,
                  frames: Optional[Sequence[int]] = None,
                  src_rigs=None, tgt_rigs=None) -> Dict[str, List[Dict]]:
    """Copy the GT hand pose to another model in hand-anatomical coordinates."""
    from .hands import FINGER_ORDER, HandRig
    from .mmd.evaluate import VmdEvaluator

    src_rigs = src_rigs or {s: HandRig.from_profile(src_profile, s) for s in ("l", "r")}
    tgt_rigs = tgt_rigs or {s: HandRig.from_profile(tgt_profile, s) for s in ("l", "r")}
    src_ev = VmdEvaluator(src_model, src_vmd, solve_ik=False, force_ik_off=True)
    if frames is None:
        frames = list(range(int(src_vmd["max_frame"]) + 1))
    frames = list(frames)
    tracks: Dict[str, List[Dict]] = {}
    for side in ("l", "r"):
        sr, tg = src_rigs[side], tgt_rigs[side]
        u_s, l_s, n_s = sr.hand_basis
        u_t, l_t, n_t = tg.hand_basis
        for finger in FINGER_ORDER:
            chain_s = sr.bones.get(finger, [])
            chain_t = tg.bones.get(finger, [])
            if not chain_s or not chain_t:
                continue
            for name in tg.chains[finger]:
                tracks[name] = []
            prev = {name: None for name in tg.chains[finger]}
            for f in frames:
                rec = src_ev.frame(int(f))
                q_wrist = rec["world_q"][sr.wrist_index]
                # Express source directions in the hand's *current* frame, so
                # the wrist rotation is divided out and what remains is the
                # hand-local finger pose (the wrist is applied by FK later).
                u_cur = qrot(q_wrist, u_s)
                l_cur = qrot(q_wrist, l_s)
                n_cur = qrot(q_wrist, n_s)
                qc = q_wrist
                dirs = []
                for k, bone in enumerate(chain_s):
                    qc = qnorm(qmul(qc, rec["local_q"][bone.index]))
                    d = np.asarray(qrot(qc, bone.rest_dir), float)
                    dirs.append(np.array([float(np.dot(d, u_cur)),
                                          float(np.dot(d, l_cur)),
                                          float(np.dot(d, n_cur))]))
                parent_q = QID
                for k, bone in enumerate(chain_t):
                    # h_k is the hand-local direction (wrist rotation divided
                    # out); the local quat rotates rest_dir into it after the
                    # already-applied parent chain.
                    h = (dirs[k][0] * u_t + dirs[k][1] * l_t
                         + dirs[k][2] * n_t)
                    d_parent = qrot(qinv(parent_q), tuple(h))
                    q = qnorm(_swing(bone.rest_dir, d_parent))
                    if prev[bone.name] is not None and sum(
                            a * b for a, b in zip(q, prev[bone.name])) < 0:
                        q = tuple(-c for c in q)
                    prev[bone.name] = q
                    tracks[bone.name].append({
                        "frame": int(f), "pos": (0.0, 0.0, 0.0), "rot": q,
                        "curves": (DEFAULT_CURVE,) * 4})
                    parent_q = qnorm(qmul(parent_q, q))
    return tracks


# ---------------------------------------------------------------------------
# GT face per model: semantic -> the model's own morph name
# ---------------------------------------------------------------------------
def face_target_names(profile: Mapping) -> Dict[str, str]:
    from .face import MorphMap
    return dict(MorphMap.from_profile(profile).available)


TWIST_STD = ("forearm_twist_l", "forearm_twist_r",
             "wrist_twist_l", "wrist_twist_r")


def strip_twist_vmd(vmd: Mapping, profile: Mapping) -> Dict:
    """Copy without the arm twist helper tracks (teacher normalisation).

    MMD twist bones (腕捩/手捩) only roll the forearm mesh around its own axis:
    they do not move any joint and do not change any bone direction.  The
    predicted VMD has no such tracks, so including them in the GT local pose
    would ask the network to invent a roll that 2D evidence cannot recover.
    Joint positions and directions are unchanged by removing them.
    """
    smap = profile["standard_map"]
    drop = {smap[k] for k in TWIST_STD if k in smap}
    out = copy.deepcopy(dict(vmd))
    out["bones"] = {k: [dict(x) for x in v] for k, v in vmd["bones"].items()
                    if k not in drop}
    out["morphs"] = dict(vmd.get("morphs", {}))
    out["ik_frames"] = [dict(x) for x in vmd.get("ik_frames", [])]
    return out


# ---------------------------------------------------------------------------
# residual extraction / application
# ---------------------------------------------------------------------------
def _q_to_rotvec(q: Sequence[float]) -> np.ndarray:
    q = qnorm(tuple(float(x) for x in q))
    if q[3] < 0.0:
        q = (-q[0], -q[1], -q[2], -q[3])
    s = math.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2)
    if s < 1e-9:
        return np.zeros(3, np.float32)
    ang = 2.0 * math.atan2(s, q[3])
    return np.degrees(np.asarray(q[:3], np.float32) * (ang / s))


def _rotvec_to_q(v: Sequence[float]):
    """Rotation vector in **degrees** -> quaternion."""
    v = np.asarray(v, float)
    ang = float(np.linalg.norm(v))
    if ang < 1e-9:
        return QID
    return qaxis(tuple(v / ang), math.radians(ang))


def _clip_rotvec(v: np.ndarray, lim: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n > lim:
        v = v * (lim / n)
    return v


def extract_body_residual(gt: Mapping, pred: Mapping,
                          frames: Optional[Sequence[int]] = None,
                          bounds: Optional[Mapping[str, float]] = None
                          ) -> Dict[str, np.ndarray]:
    """(N, 41) root XZ + 13 local-rotation residuals (rotvec, parent frame)."""
    n = len(gt["frames"])
    out = np.zeros((n, 2 + 3 * len(BODY_FACTOR_NAMES)), np.float32)
    masks = np.ones_like(out)
    bound = dict(ROT_BOUND if bounds is None else bounds)
    # root: XZ trajectory difference relative to the first frame
    g = np.asarray(gt["world"]["root"], float)
    p = np.asarray(pred["world"]["root"], float)
    d = (g - p)[:, [0, 2]]
    d = d - d[0]
    out[:, 0:2] = np.clip(d, -ROOT_BOUND, ROOT_BOUND)
    # local rotations
    for bi, bone in enumerate(BODY_FACTOR_NAMES):
        for fi in range(n):
            dq = qmul(tuple(gt["local_q"][bone][fi]),
                      qinv(tuple(pred["local_q"][bone][fi])))
            rv = _q_to_rotvec(dq)
            out[fi, 2 + 3 * bi:2 + 3 * bi + 3] = np.clip(
                rv, -bound[bone], bound[bone])
    return {"target": out, "mask": masks}


def extract_hand_residual(gt_angles: np.ndarray, pred_angles: np.ndarray,
                          pred_level: np.ndarray) -> Dict[str, np.ndarray]:
    """(N, 38) angle residuals; only L0/L1 frames are supervised."""
    n = gt_angles.shape[0]
    out = np.zeros((n, 2 * len(HAND_JOINT_KEYS)), np.float32)
    mask = np.zeros_like(out)
    lim = HAND_BOUND_VEC.reshape(2, len(HAND_JOINT_KEYS))
    for s in range(2):
        diff = np.asarray(gt_angles[:, s], float) - np.asarray(pred_angles[:, s], float)
        diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
        diff = np.clip(diff, -lim[s], lim[s])
        ok = np.asarray(pred_level[:, s]) <= 1
        out[:, s * len(HAND_JOINT_KEYS):(s + 1) * len(HAND_JOINT_KEYS)] = diff
        mask[:, s * len(HAND_JOINT_KEYS):(s + 1) * len(HAND_JOINT_KEYS)] = \
            ok[:, None].astype(np.float32)
    return {"target": out, "mask": mask}


# M3's L1 geometry only drives blink (mouth landmarks are unreliable on anime
# faces and l1_mouth/l1_wink are off).  Supervising other channels at L1 would
# teach the network to invent morphs from evidence that does not exist.
L1_FACE_CHANNELS = (0,)          # 0 == SEMANTIC[0] == "blink"


def face_channel_mask(pred_level: np.ndarray,
                      gt_channels: Optional[np.ndarray] = None,
                      n_channels: int = 12) -> np.ndarray:
    lv = np.asarray(pred_level)
    out = np.zeros((len(lv), n_channels), np.float32)
    l0 = lv == 0
    if l0.any():
        out[l0] = 1.0
    l1 = lv == 1
    if l1.any():
        for j in L1_FACE_CHANNELS:
            out[l1, j] = 1.0
    if gt_channels is not None:
        out *= np.asarray(gt_channels, np.float32)[None, :]
    return out


def extract_face_residual(gt_weights: np.ndarray, pred_weights: np.ndarray,
                          pred_level: np.ndarray,
                          gt_channels: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """(N, 12) morph residuals, supervised only where M3 has usable evidence."""
    n = gt_weights.shape[0]
    diff = np.clip(np.asarray(gt_weights, float) - np.asarray(pred_weights, float),
                   -FACE_BOUND, FACE_BOUND)
    ok = face_channel_mask(pred_level, gt_channels, diff.shape[1])
    return {"target": diff.astype(np.float32), "mask": ok}


def combine_residuals(body: np.ndarray, hands: np.ndarray,
                      face: np.ndarray) -> np.ndarray:
    n = body.shape[0]
    out = np.zeros((n, RESIDUAL_DIM), np.float32)
    out[:, ROOT_SLICE] = body[:, :2]
    out[:, ROT_SLICE] = body[:, 2:]
    out[:, HAND_SLICE] = hands
    out[:, FACE_SLICE] = face
    return out


def split_residual(res: np.ndarray):
    return (res[:, ROOT_SLICE], res[:, ROT_SLICE],
            res[:, HAND_SLICE], res[:, FACE_SLICE])


# ---------------------------------------------------------------------------
# applying corrections
# ---------------------------------------------------------------------------
# Same values as the writer's default, but in the *parsed* VMD format
# (x1, y1, x2, y2) so a dict built here can be evaluated without a round-trip.
DEFAULT_CURVE = (20, 20, 107, 107)


def apply_body_residual(pred_vmd: Mapping, model: Mapping, profile: Mapping,
                        residual: np.ndarray,
                        frames: Optional[Sequence[int]] = None) -> Dict:
    """Return a copy of ``pred_vmd`` with the body/root residual applied.

    The residual is a local-rotation delta (parent frame) per bone plus a root
    XZ offset; hand/face tracks in ``pred_vmd`` are left untouched.
    """
    from .metrics.core import pose_track

    if frames is None:
        frames = list(range(int(pred_vmd["max_frame"]) + 1))
    frames = list(frames)
    tr = pose_track(model, pred_vmd, profile, frames=frames)
    smap = profile["standard_map"]
    root_name = smap["root"]
    out = copy.deepcopy(dict(pred_vmd))
    out["bones"] = {k: [dict(x) for x in v] for k, v in pred_vmd["bones"].items()}
    out.setdefault("morphs", dict(pred_vmd.get("morphs", {})))

    # ---- body local rotations ------------------------------------------
    for bi, bone in enumerate(BODY_FACTOR_NAMES):
        name = smap.get(bone)
        if name is None:
            continue
        q_new = []
        for fi, f in enumerate(frames):
            dq = _rotvec_to_q(residual[fi, 2 + 3 * bi:2 + 3 * bi + 3])
            q = qnorm(qmul(dq, tuple(tr["local_q"][bone][fi])))
            if q_new and sum(a * b for a, b in zip(q, q_new[-1])) < 0.0:
                q = tuple(-c for c in q)
            q_new.append(q)
        out["bones"][name] = [
            {"frame": int(f), "pos": (0.0, 0.0, 0.0), "rot": q,
             "curves": (DEFAULT_CURVE,) * 4}
            for f, q in zip(frames, q_new)]

    # ---- root XZ -------------------------------------------------------
    # The VMD position is the bone-local animation offset, not the world
    # position (the rest offset must not be added again).  Sample local_t
    # through the evaluator so sparse tracks are handled exactly.
    from .mmd.evaluate import VmdEvaluator
    d = np.asarray(residual[:, :2], float)
    if root_name in out["bones"]:
        name2i = {b["name"]: i for i, b in enumerate(model["bones"])}
        ri = name2i[root_name]
        ev = VmdEvaluator(model, pred_vmd, solve_ik=False, force_ik_off=True)
        new_keys = []
        for fi, f in enumerate(frames):
            rec = ev.frame(int(f))
            base_q = tuple(rec["local_q"][ri])
            base_p = tuple(rec["local_t"][ri])
            p = (base_p[0] + d[fi, 0], base_p[1], base_p[2] + d[fi, 1])
            new_keys.append({"frame": int(f), "pos": p, "rot": base_q,
                             "curves": (DEFAULT_CURVE,) * 4})
        out["bones"][root_name] = new_keys
    return out


def apply_hand_residual(pred_angles: np.ndarray,
                        residual: np.ndarray) -> np.ndarray:
    lim = residual_limits()[HAND_SLICE].reshape(2, len(HAND_JOINT_KEYS))
    out = np.array(pred_angles, float, copy=True)
    for s in range(2):
        r = residual[:, s * len(HAND_JOINT_KEYS):(s + 1) * len(HAND_JOINT_KEYS)]
        out[:, s] = np.clip(out[:, s] + np.clip(r, -lim[s], lim[s]), -360.0, 360.0)
    return out


def apply_face_residual(pred_weights: np.ndarray,
                        residual: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(pred_weights, float) + np.clip(residual, -FACE_BOUND,
                                                            FACE_BOUND), 0.0, 1.0)


# ---------------------------------------------------------------------------
# full VMD assembly (body + finger angles + face weights)
# ---------------------------------------------------------------------------
def assemble_vmd(body_vmd: Mapping, profile: Mapping, hand_angles: np.ndarray,
                 face_weights: np.ndarray, n_frames: Optional[int] = None) -> Dict:
    """Same contract as scripts/assemble_handface_vmd.py, as a library call."""
    from .face import MorphMap, compress_morph_keys, weights_to_morphs
    from .hands import HandRig, angles_to_local_quats, vector_to_measured

    n = int(body_vmd["max_frame"]) + 1 if n_frames is None else int(n_frames)
    rigs = {s: HandRig.from_profile(profile, s) for s in ("l", "r")}
    morph_map = MorphMap.from_profile(profile)
    bone_frames = {k: [dict(x) for x in v if int(x["frame"]) < n]
                   for k, v in body_vmd["bones"].items()}
    for s in ("l", "r"):
        rig = rigs[s]
        tracks = {name: [] for name in rig.finger_names()}
        prev = {name: None for name in tracks}
        for fi in range(n):
            measured = vector_to_measured(hand_angles[fi, 0 if s == "l" else 1])
            for name, q in angles_to_local_quats(rig, measured).items():
                q = qnorm(q)
                if prev[name] is not None and sum(a * b for a, b in zip(q, prev[name])) < 0:
                    q = tuple(-c for c in q)
                prev[name] = q
                tracks[name].append({"frame": fi, "pos": (0.0, 0.0, 0.0),
                                     "rot": q, "curves": (DEFAULT_CURVE,) * 4})
        bone_frames.update(tracks)
    morph_frames = {}
    for name, w in weights_to_morphs(face_weights[:n], morph_map).items():
        keys = compress_morph_keys(w)
        if keys:
            morph_frames[name] = keys
    return {"model_name": body_vmd.get("model_name", ""), "bones": bone_frames,
            "morphs": morph_frames,
            "ik_frames": [dict(x) for x in body_vmd.get("ik_frames", [])],
            "max_frame": n - 1}


# ---------------------------------------------------------------------------
# body-relative metrics
# ---------------------------------------------------------------------------
def _bone_dir(track: Mapping, bone: str) -> np.ndarray:
    child = BONE_CHILD[bone]
    d = np.asarray(track["world"][child], float) - np.asarray(track["world"][bone], float)
    n = np.linalg.norm(d, axis=1, keepdims=True)
    return d / np.maximum(n, 1e-9)


def body_relative_errors(gt: Mapping, pred: Mapping) -> Dict[str, float]:
    """Error of each factor after dividing out its reference bone's frame.

    ``upperarm_chest_deg`` / ``forearm_upperarm_deg`` are the M3.5 main metrics:
    they do not reward a correct root yaw or torso lean that cancels an arm
    error in world space.
    """
    out: Dict[str, float] = {}
    for bone, ref in METRIC_FACTORS:
        if bone not in gt["world"] or bone not in pred["world"]:
            continue
        if ref not in gt["world_q"] or ref not in pred["world_q"]:
            continue
        dg = _bone_dir(gt, bone)
        dp = _bone_dir(pred, bone)
        vg = np.asarray([qrot(tuple(qinv(tuple(gt["world_q"][ref][i]))), tuple(dg[i]))
                         for i in range(len(dg))], float)
        vp = np.asarray([qrot(tuple(qinv(tuple(pred["world_q"][ref][i]))), tuple(dp[i]))
                         for i in range(len(dp))], float)
        ang = np.degrees(np.arccos(np.clip(np.sum(vg * vp, axis=1), -1.0, 1.0)))
        out[f"{bone}_{ref}_deg"] = float(np.mean(ang))
    for g, keys in {
        "chest": ("chest_spine1_deg", "spine1_spine_deg", "spine_pelvis_deg"),
        "upperarm": ("upperarm_l_chest_deg", "upperarm_r_chest_deg"),
        "forearm": ("forearm_l_upperarm_l_deg", "forearm_r_upperarm_r_deg"),
        "thigh": ("thigh_l_pelvis_deg", "thigh_r_pelvis_deg"),
        "shin": ("shin_l_thigh_l_deg", "shin_r_thigh_r_deg"),
        "foot": ("foot_l_shin_l_deg", "foot_r_shin_r_deg"),
    }.items():
        vals = [out[k] for k in keys if k in out]
        if vals:
            out[f"group_{g}_deg"] = float(np.mean(vals))
    if all(k in out for k in ("group_upperarm_deg", "group_forearm_deg",
                              "group_chest_deg")):
        out["body_relative_main_deg"] = float(np.mean(
            [out["group_upperarm_deg"], out["group_forearm_deg"],
             out["group_chest_deg"]]))
    return out


# ---------------------------------------------------------------------------
# evidence-only inference features
# ---------------------------------------------------------------------------
KP_GROUPS = {
    "arm_l": (5, 7, 9), "arm_r": (6, 8, 10),
    "legs": (11, 12, 13, 14, 15, 16), "torso": (0, 1, 2, 3, 4),
}


def profile_vector(profile: Mapping) -> np.ndarray:
    """Static target-profile features (continuous; no unseen-model one-hot)."""
    by = {b["name"]: b for b in profile.get("bones", [])}
    smap = profile["standard_map"]
    h = float(profile.get("height") or 20.0)

    def dist(a, b):
        if a not in by or b not in by:
            return 0.0
        pa, pb = np.asarray(by[a]["pos"], float), np.asarray(by[b]["pos"], float)
        return float(np.linalg.norm(pa - pb))

    out = [h / 20.0,
           dist(smap.get("upperarm_l"), smap.get("forearm_l")) / h,
           dist(smap.get("forearm_l"), smap.get("hand_l")) / h,
           dist(smap.get("thigh_l"), smap.get("shin_l")) / h,
           dist(smap.get("shin_l"), smap.get("foot_l")) / h,
           dist(smap.get("hand_l"), profile.get("finger_map", {}).get("middle1_l", "")) / h]
    return np.asarray(out, np.float32)


def feature_names(profile: Mapping, feature_set: str = "v1") -> List[str]:
    names = ["root_height"]
    names += [f"pelvis_lq_{a}" for a in "xyzw"]
    names += [f"pelvis_dq_{a}" for a in "xyzw"]
    for b in FEATURE_DIR_BONES:
        names += [f"dir_{b}_{a}" for a in "xyz"]
    for b in FEATURE_DIR_BONES:
        names += [f"vel_{b}_{a}" for a in "xyz"]
    names += ["root_vx", "root_vz", "root_vy"]
    names += [f"kp{i}_{a}" for i in range(17) for a in ("x", "y", "c")]
    names += ["bbox_scale_h", "bbox_scale_img", "bbox_cx", "bbox_cy"]
    for s in ("l", "r"):
        names += [f"toe_{s}_y", f"toe_{s}_vy", f"toe_{s}_speed", f"contact_{s}"]
    for s in ("l", "r"):
        names += [f"hand_{s}_{k}" for k in HAND_JOINT_KEYS]
        names += [f"hand_{s}_L0", f"hand_{s}_L1", f"hand_{s}_score"]
    names += [f"face_{k}" for k in SEMANTIC]
    names += ["face_L01", "prof_height", "prof_upperarm", "prof_forearm",
              "prof_thigh", "prof_shin", "prof_hand"]
    for g in KP_GROUPS:
        names.append(f"conf_{g}")
    names += ["wrist_conf_l", "wrist_conf_r", "bbox_s", "adapter_scale",
              "hand_detect_l", "hand_detect_r", "face_detect"]
    if feature_set == "v2":
        # The network must see both the evidence stream and the baseline pose
        # it is asked to correct.  handgt/facegt replace the evidence with a
        # synthetic stream; without the baseline copy the hand/face head
        # cannot know which pose the residual is relative to.
        for s in ("l", "r"):
            names += [f"hand_base_{s}_{k}" for k in HAND_JOINT_KEYS]
        names += [f"face_base_{k}" for k in SEMANTIC]
    return names


def _resample_to(n: int, arr: Optional[np.ndarray], shape: Tuple[int, ...]) -> np.ndarray:
    out = np.zeros((n,) + shape, np.float32)
    if arr is not None and len(arr):
        m = min(n, len(arr))
        out[:m] = np.asarray(arr, np.float32)[:m]
    return out


def build_features(pred_track: Mapping, height: float,
                   kp2d: Optional[np.ndarray] = None,
                   bbx_xys: Optional[np.ndarray] = None,
                   hand_angles: Optional[np.ndarray] = None,
                   hand_level: Optional[np.ndarray] = None,
                   hand_score: Optional[np.ndarray] = None,
                   hand_detect: Optional[np.ndarray] = None,
                   face_weights: Optional[np.ndarray] = None,
                   face_level: Optional[np.ndarray] = None,
                   face_detect: Optional[np.ndarray] = None,
                   profile: Optional[Mapping] = None,
                   adapter_scale: float = 1.0,
                   img_wh: Tuple[int, int] = (720, 1280),
                   base_hand_angles: Optional[np.ndarray] = None,
                   base_face_weights: Optional[np.ndarray] = None,
                   feature_set: str = "v1") -> np.ndarray:
    """Evidence-only features (GT is not an argument of this function)."""
    n = len(pred_track["frames"])
    world = pred_track["world"]
    wq = pred_track["world_q"]
    root = np.asarray(world.get("root", np.zeros((n, 3))), float)
    q_pelvis = np.asarray(wq.get("pelvis", np.tile(QID, (n, 1))), float)

    dirs = np.zeros((n, len(FEATURE_DIR_BONES), 3), np.float32)
    for bi, bone in enumerate(FEATURE_DIR_BONES):
        child = BONE_CHILD[bone]
        if bone not in world or child not in world:
            continue
        d = np.asarray(world[child], float) - np.asarray(world[bone], float)
        d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        for fi in range(n):
            dirs[fi, bi] = qrot(tuple(qinv(tuple(q_pelvis[fi]))), tuple(d[fi]))
    vel = np.gradient(dirs, axis=0) if n > 1 else np.zeros_like(dirs)
    root_v = np.gradient(root, axis=0) if n > 1 else np.zeros_like(root)
    pelvis_lq = np.asarray(pred_track["local_q"].get(
        "pelvis", np.tile(QID, (n, 1))), np.float32)
    pelvis_dq = np.gradient(pelvis_lq, axis=0) if n > 1 else np.zeros_like(pelvis_lq)

    # 2D evidence
    kp = _resample_to(n, kp2d, (17, 3))
    bbx = _resample_to(n, bbx_xys, (3,))
    s = np.maximum(bbx[:, 2], 1e-6)
    kpn = np.zeros((n, 17, 3), np.float32)
    kpn[:, :, 0] = (kp[:, :, 0] - bbx[:, 0:1]) / s[:, None]
    kpn[:, :, 1] = (kp[:, :, 1] - bbx[:, 1:2]) / s[:, None]
    kpn[:, :, 2] = np.clip(kp[:, :, 2], 0.0, 1.0)

    # contact evidence from the predicted pose only
    toe_y = np.zeros((n, 2), np.float32)
    toe_v = np.zeros((n, 2), np.float32)
    toe_sp = np.zeros((n, 2), np.float32)
    for si, side in enumerate(("l", "r")):
        toe = np.asarray(world.get(f"toe_{side}", np.zeros((n, 3))), float)
        toe_y[:, si] = toe[:, 1] / max(height, 1e-9)
        toe_v[:, si] = (np.gradient(toe[:, 1]) / max(height, 1e-9)) if n > 1 else 0.0
        sp = np.zeros(n) if n < 2 else np.linalg.norm(
            np.gradient(toe[:, [0, 2]], axis=0), axis=1) / max(height, 1e-9)
        toe_sp[:, si] = sp
    ground = float(np.min(toe_y)) if n else 0.0
    contact = (toe_y <= ground + 0.02).astype(np.float32)

    # hands / face (predicted evidence, L2/L3 values are the fallback pose)
    ha = _resample_to(n, hand_angles, (2, len(HAND_JOINT_KEYS)))
    hl = _resample_to(n, hand_level, (2,))
    hs = _resample_to(n, hand_score, (2,))
    hd = _resample_to(n, hand_detect, (2,))
    fw = _resample_to(n, face_weights, (12,))
    fl = _resample_to(n, face_level, ())
    fd = _resample_to(n, face_detect, ())

    prof = profile_vector(profile) if profile is not None else np.zeros(6, np.float32)
    W, H = img_wh
    parts = [
        (root[:, 1:2] / max(height, 1e-9)).astype(np.float32),
        pelvis_lq, pelvis_dq,
        dirs.reshape(n, -1),
        vel.reshape(n, -1),
        (root_v[:, [0, 2]] / max(height, 1e-9)).astype(np.float32),
        (root_v[:, 1:2] / max(height, 1e-9)).astype(np.float32),
        kpn.reshape(n, -1),
        np.stack([s / max(height, 1e-9), s / max(H, 1),
                  bbx[:, 0] / max(W, 1) - 0.5, bbx[:, 1] / max(H, 1) - 0.5],
                 axis=1).astype(np.float32),
        toe_y, toe_v, toe_sp, contact,
        (ha.reshape(n, -1) / 90.0).astype(np.float32),
        (hl == 0).astype(np.float32), (hl == 1).astype(np.float32), hs,
        fw,
        (fl <= 1).astype(np.float32)[:, None],
        np.tile(prof[None, :], (n, 1)),
    ]
    for idxs in KP_GROUPS.values():
        parts.append(np.clip(kp[:, idxs, 2], 0, 1).mean(axis=1, keepdims=True))
    wrist = np.stack([np.clip(kp[:, 9, 2], 0, 1), np.clip(kp[:, 10, 2], 0, 1)], axis=1)
    parts += [wrist, (s / max(H, 1))[:, None],
              np.full((n, 1), float(adapter_scale), np.float32),
              (hd > 0).astype(np.float32), (fd > 0).astype(np.float32)[:, None]]
    if feature_set == "v2":
        hb = _resample_to(n, base_hand_angles if base_hand_angles is not None
                          else ha, (2, len(HAND_JOINT_KEYS)))
        fb = _resample_to(n, base_face_weights if base_face_weights is not None
                          else fw, (12,))
        parts += [(hb.reshape(n, -1) / 90.0).astype(np.float32), fb]
    feat = np.concatenate([np.asarray(p, np.float32).reshape(n, -1) for p in parts],
                          axis=1)
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# whole-segment splits (motion x model x camera)
# ---------------------------------------------------------------------------
TRAIN_MOTIONS = ("R1", "R2")
TEST_MOTIONS = ("R3",)
TRAIN_MODELS = ("si", "luoxi")
HELD_MODEL = "tiflos"
TRAIN_VIEWS = ("front", "three_quarter")
# M6 training-only viewpoints.  Evaluation stays on the fixed front /
# three_quarter / side clips so the held-out gate is unchanged.
EXTRA_TRAIN_VIEWS = ("front_left",)
HELD_VIEW = "side"
ALL_MODELS = ("si", "luoxi", "tiflos")
ALL_VIEWS = ("front", "three_quarter", "side")


def assign_split(motion: str, model: str, view: str) -> Optional[str]:
    """Return 'train' / 'val' / 'test' for a base clip, or None if not planned.

    R1/R2 split by model and camera: only si/luoxi x front/three_quarter are
    trained.  Every remaining R1/R2 combo is validation (held-out model and/or
    held-out camera on seen motions).  R3 is entirely test.
    """
    if motion in TEST_MOTIONS:
        return "test"
    if motion not in TRAIN_MOTIONS:
        return None
    if model in TRAIN_MODELS and view in (TRAIN_VIEWS + EXTRA_TRAIN_VIEWS):
        return "train"
    return "val"


def clip_id(motion: str, model: str, view: str, variant: str = "base") -> str:
    return f"{motion}_{model}_{view}" + ("" if variant == "base" else f"_{variant}")


def leakage_report(records: Iterable[Mapping]) -> Dict:
    """Whole-segment leakage audit for a list of clip records."""
    by = {"train": set(), "val": set(), "test": set()}
    motions = {k: set() for k in by}
    models = {k: set() for k in by}
    views = {k: set() for k in by}
    dup = []
    for r in records:
        cid = r["clip"]
        if cid in by[r["split"]]:
            dup.append(cid)
        by[r["split"]].add(cid)
        motions[r["split"]].add(r["motion"])
        models[r["split"]].add(r["model"])
        views[r["split"]].add(r["view"])
    inter = {a + "_" + b: sorted(by[a] & by[b])
             for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    checks = {
        "no_duplicate_clip": not dup,
        "no_clip_overlap": all(not v for v in inter.values()),
        "test_motion_unseen_in_train": not (motions["test"] & motions["train"]),
        "test_motion_unseen_in_val": not (motions["test"] & motions["val"]),
        "held_model_not_in_train": HELD_MODEL not in models["train"],
        "held_view_not_in_train": HELD_VIEW not in views["train"],
    }
    return {"ok": all(checks.values()), "checks": checks,
            "duplicates": dup, "overlap": inter,
            "motions": {k: sorted(v) for k, v in motions.items()},
            "models": {k: sorted(v) for k, v in models.items()},
            "views": {k: sorted(v) for k, v in views.items()}}
