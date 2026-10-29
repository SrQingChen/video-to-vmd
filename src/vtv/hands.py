# -*- coding: utf-8 -*-
"""Hand retargeting: MediaPipe landmarks -> PMX finger local rotations (M3).

Design notes
------------
* MediaPipe gives 21 landmarks per hand: wrist, 5 MCP joints, and 3 phalanges
  per finger.  We only trust *rotation-invariant* quantities from it:
  per-joint flexion magnitudes (angles between consecutive segments), MCP
  abduction (in-plane angle toward the index side) and thumb opposition.
  Sign-giving decisions (which way the palm faces, which way a finger may
  bend) come from the target PMX rest geometry, so fingers can never bend
  backwards even if MediaPipe's depth sign flips.
* The PMX hand frame is built from rest positions::

      u = normalise(middle MCP - wrist)        # along the hand
      l = normalise(index MCP - pinky MCP)  ⟂ u  # toward the thumb side
      n_palm = ±(l × u)                        # sign from the thumb base

  The thumb metacarpal base sits on the palmar side, which fixes the sign for
  both hands without hard-coding the model or the side.
* A finger bone's VMD rotation is local to its parent, whose rest axes are
  world-aligned (the convention verified in M1.5 against Blender/MMD Tools).
  For a target world direction ``d`` and parent world rotation ``Qp``::

      q_local = swing(rest_dir, Qp⁻¹ · d)

  The swing is the minimal rotation; no twist is invented.
* Confidence ladder (M3 spec):
      L0  3D landmarks, verified, temporally stable
      L1  2D landmarks only (or 3D unstable), damped
      L2  hand not detected but the body wrist is visible -> slight curl
      L3  hand/wrist not visible -> neutral, very light curl
  Levels are decided on whole segments, never on single frames.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .mmd.mathutil import qaxis, qinv, qmul, qnorm, qrot, qslerp, vcross, vdot, vlen, vnorm

# ---------------------------------------------------------------------------
# MediaPipe hand topology
# ---------------------------------------------------------------------------
WRIST = 0
THUMB = (1, 2, 3, 4)          # CMC, MCP, IP, TIP
INDEX = (5, 6, 7, 8)
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
PINKY = (17, 18, 19, 20)
FINGERS = {"thumb": THUMB, "index": INDEX, "middle": MIDDLE, "ring": RING, "pinky": PINKY}
FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
LONG_FINGERS = ("index", "middle", "ring", "pinky")

# Neutral (open hand) MediaPipe splay, degrees, positive toward the index.
# Used to turn the measured in-plane angle into a *delta* so the target's own
# rest splay is preserved.
MP_NEUTRAL_SPREAD = {"index": 5.0, "middle": 0.0, "ring": -5.0, "pinky": -9.0}
MP_NEUTRAL_THUMB = 45.0

# Joint limits (degrees).  Flexion is clamped at zero: a small hyperextension
# is not worth the risk of a visibly back-bent finger.
LIMITS = {
    "mcp": (0.0, 85.0),
    "pip": (0.0, 105.0),
    "dip": (0.0, 70.0),
    "spread": (-20.0, 20.0),
    "cmc": (-25.0, 25.0),
    "thumb_mcp": (0.0, 55.0),
    "thumb_ip": (0.0, 80.0),
}

# Fallback poses: (mcp, pip, dip) per long finger, thumb (cmc, mcp, ip).
L2_ANGLES = {"mcp": 13.0, "pip": 15.0, "dip": 7.0, "spread": 0.0,
             "cmc": 0.0, "thumb_mcp": 11.0, "thumb_ip": 7.0}
L3_ANGLES = {"mcp": 5.0, "pip": 6.0, "dip": 3.0, "spread": 0.0,
             "cmc": 0.0, "thumb_mcp": 4.0, "thumb_ip": 3.0}
FINGER_GAIN = {"index": 1.00, "middle": 1.05, "ring": 0.98, "pinky": 0.88}

LEVEL_L0, LEVEL_L1, LEVEL_L2, LEVEL_L3 = 0, 1, 2, 3
LEVEL_CONF = {LEVEL_L0: 1.0, LEVEL_L1: 0.7, LEVEL_L2: 0.3, LEVEL_L3: 0.1}

JOINT_KEYS = tuple(
    [f"{f}.spread" for f in LONG_FINGERS]
    + [f"{f}.mcp" for f in LONG_FINGERS]
    + [f"{f}.pip" for f in LONG_FINGERS]
    + [f"{f}.dip" for f in LONG_FINGERS]
    + ["thumb.cmc", "thumb.mcp", "thumb.ip"]
)
JOINT_INDEX = {k: i for i, k in enumerate(JOINT_KEYS)}


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros(3)


def _angle_deg(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    c = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return math.degrees(math.acos(c))


def _swing_q(d_from, d_to):
    """Minimal rotation (x, y, z, w) taking unit d_from onto unit d_to."""
    a = _unit(d_from)
    b = _unit(d_to)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    axis = np.cross(a, b)
    s = float(np.linalg.norm(axis))
    if s < 1e-9:
        if c > 0:
            return (0.0, 0.0, 0.0, 1.0)
        # 180 degrees: any axis perpendicular to a
        helper = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        axis = _unit(np.cross(a, helper))
        return qaxis(tuple(axis), math.pi)
    return qaxis(tuple(axis / s), math.atan2(s, c))


def _clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))


# ---------------------------------------------------------------------------
# target rig
# ---------------------------------------------------------------------------
@dataclass
class FingerBone:
    name: str
    parent: str
    index: int
    parent_index: int
    rest_pos: np.ndarray
    rest_dir: np.ndarray
    length: float


@dataclass
class HandRig:
    model_key: str
    side: str                       # 'l' or 'r'
    wrist_name: str
    wrist_index: int
    bones: Dict[str, List[FingerBone]]   # finger -> [proximal, middle, distal]
    chains: Dict[str, List[str]]
    hand_basis: Tuple[np.ndarray, np.ndarray, np.ndarray]  # u, l, n_palm (rest, world)
    rest_angles: Dict[str, Dict[str, float]] = field(default_factory=dict)
    rest_dirs: Dict[str, List[np.ndarray]] = field(default_factory=dict)

    # -- construction ----------------------------------------------------
    @classmethod
    def from_profile(cls, profile: Mapping, side: str) -> "HandRig":
        if side not in ("l", "r"):
            raise ValueError(f"side must be 'l' or 'r', got {side!r}")
        by_name = {b["name"]: b for b in profile["bones"]}
        finger_map = profile["finger_map"]
        std = profile["standard_map"]
        wrist_name = std[f"hand_{side}"]
        wrist = by_name[wrist_name]
        wrist_pos = np.asarray(wrist["pos"], float)

        def tail_pos(b):
            tb = b.get("tail_bone")
            if tb is not None and 0 <= tb < len(profile["bones"]):
                return np.asarray(profile["bones"][tb]["pos"], float)
            return np.asarray(b["pos"], float) + np.asarray(b.get("tail_offset") or (0, 1, 0), float)

        bones: Dict[str, List[FingerBone]] = {}
        chains: Dict[str, List[str]] = {}
        digits = [("thumb", 3), ("index", 3), ("middle", 3), ("ring", 3), ("pinky", 3)]
        for finger, count in digits:
            chain = []
            for k in range(count):
                key = f"{finger}{k + (0 if finger == 'thumb' else 1)}_{side}"
                if key not in finger_map:
                    continue
                name = finger_map[key]
                b = by_name[name]
                pos = np.asarray(b["pos"], float)
                tip = tail_pos(b)
                d = _unit(tip - pos)
                chain.append(FingerBone(
                    name=name,
                    parent=b.get("parent_name") or "",
                    index=int(b["index"]),
                    parent_index=int(b["parent"]),
                    rest_pos=pos,
                    rest_dir=d,
                    length=float(np.linalg.norm(tip - pos)),
                ))
            if chain:
                bones[finger] = chain
                chains[finger] = [b.name for b in chain]

        # hand basis from the proximal finger positions
        mid = np.asarray(by_name[finger_map[f"middle1_{side}"]]["pos"], float)
        idx = np.asarray(by_name[finger_map[f"index1_{side}"]]["pos"], float)
        pky = np.asarray(by_name[finger_map[f"pinky1_{side}"]]["pos"], float)
        thumb_base = np.asarray(by_name[finger_map[f"thumb0_{side}"]]["pos"], float)
        u = _unit(mid - wrist_pos)
        l = idx - pky
        l = _unit(l - u * float(np.dot(l, u)))
        n = _unit(np.cross(l, u))
        # the thumb metacarpal base lies on the palmar side of the hand plane
        if float(np.dot(thumb_base - wrist_pos, n)) < 0:
            n = -n

        rig = cls(model_key=str(profile.get("model_key", "")), side=side,
                  wrist_name=wrist_name, wrist_index=int(wrist["index"]),
                  bones=bones, chains=chains, hand_basis=(u, l, n))
        rig._compute_rest_angles()
        return rig

    def _compute_rest_angles(self):
        u, l, n = self.hand_basis
        for finger, chain in self.bones.items():
            dirs = [b.rest_dir for b in chain]
            self.rest_dirs[finger] = dirs
            if finger == "thumb":
                d0 = dirs[0]
                # in-plane angle toward the fingers (u) measured from l
                cmc = math.degrees(math.atan2(float(np.dot(d0, u)), float(np.dot(d0, l))))
                mcp = _angle_deg(dirs[0], dirs[1]) if len(dirs) > 1 else 0.0
                ip = _angle_deg(dirs[1], dirs[2]) if len(dirs) > 2 else 0.0
                self.rest_angles[finger] = {"cmc": cmc, "mcp": mcp, "ip": ip}
            else:
                d1 = dirs[0]
                inplane = math.degrees(math.atan2(float(np.dot(d1, l)), float(np.dot(d1, u))))
                mcp = _angle_deg(d1, dirs[1]) if len(dirs) > 1 else 0.0
                pip = _angle_deg(dirs[1], dirs[2]) if len(dirs) > 2 else 0.0
                self.rest_angles[finger] = {
                    "spread": inplane, "mcp": mcp, "pip": pip, "dip": 0.0,
                }

    # -- convenience -----------------------------------------------------
    def finger_names(self) -> List[str]:
        out = []
        for f in FINGER_ORDER:
            out.extend(self.chains.get(f, []))
        return out


def rigs_from_profile(profile: Mapping) -> Dict[str, HandRig]:
    return {side: HandRig.from_profile(profile, side) for side in ("l", "r")}


# ---------------------------------------------------------------------------
# MediaPipe landmarks -> anatomical angles
# ---------------------------------------------------------------------------
def mp_joint_angles(points: Sequence[Sequence[float]]) -> Dict[str, Dict[str, float]]:
    """Measure joint angles (degrees) from 21 landmarks (2D or 3D).

    Returned keys match the rig:
      long fingers: spread (deg, index-positive), mcp, pip, dip (unsigned)
      thumb:        cmc (in-plane opposition, deg), mcp, ip (unsigned)
    """
    p = np.asarray(points, float)
    if p.shape != (21, 3) and p.shape != (21, 2):
        raise ValueError(f"expected 21 landmarks, got {p.shape}")
    if p.shape[1] == 2:
        p = np.concatenate([p, np.zeros((21, 1))], axis=1)
    wrist = p[WRIST]
    u = p[MIDDLE[0]] - wrist
    l = p[INDEX[0]] - p[PINKY[0]]
    lu = float(np.dot(l, u) / max(np.dot(u, u), 1e-9))
    l = l - lu * u
    u_n = _unit(u)
    l_n = _unit(l)

    out: Dict[str, Dict[str, float]] = {}
    thumb = p[list(THUMB)]
    d0, d1, d2 = thumb[1] - thumb[0], thumb[2] - thumb[1], thumb[3] - thumb[2]
    out["thumb"] = {
        "cmc": math.degrees(math.atan2(float(np.dot(_unit(d0), u_n)),
                                       float(np.dot(_unit(d0), l_n)))),
        "mcp": _angle_deg(d0, d1),
        "ip": _angle_deg(d1, d2),
    }
    for finger, ids in (("index", INDEX), ("middle", MIDDLE), ("ring", RING), ("pinky", PINKY)):
        a, b, c, d = [p[i] for i in ids]
        d1 = b - a
        d2 = c - b
        d3 = d - c
        spread = math.degrees(math.atan2(float(np.dot(_unit(d1), l_n)),
                                        float(np.dot(_unit(d1), u_n))))
        out[finger] = {
            "spread": spread,
            "mcp": _angle_deg(a - wrist, d1),
            "pip": _angle_deg(d1, d2),
            "dip": _angle_deg(d2, d3),
        }
    return out


def angles_to_targets(rig: HandRig, measured: Mapping[str, Mapping[str, float]],
                      gain: float = 1.0) -> Dict[Tuple[str, int], np.ndarray]:
    """Turn measured MP angles into target world directions per phalanx.

    Joint angles are applied as *incremental* rotations on top of the target's
    rest bone directions.  This keeps the model's own rest splay/bend intact
    (a zero measurement reproduces the rest pose exactly) and makes the
    flexion direction anatomically fixed by the palm normal.
    """
    u, l, n = rig.hand_basis
    targets: Dict[Tuple[str, int], np.ndarray] = {}

    for finger in LONG_FINGERS:
        if finger not in rig.bones:
            continue
        m = measured.get(finger, {})
        spread = _clamp((m.get("spread", MP_NEUTRAL_SPREAD[finger])
                         - MP_NEUTRAL_SPREAD[finger]) * gain, *LIMITS["spread"])
        mcp = _clamp(m.get("mcp", 0.0) * gain, *LIMITS["mcp"])
        pip = _clamp(m.get("pip", 0.0) * gain, *LIMITS["pip"])
        dip = _clamp(m.get("dip", 0.0) * gain, *LIMITS["dip"])
        dirs = rig.rest_dirs[finger]
        # spread: rotate about d_rest x l so that +angle moves toward the index
        # (the sign of n x d flips between hands in MMD's left-handed frame)
        spread_axis = _unit(np.cross(dirs[0], l))
        r_spread = qaxis(tuple(spread_axis), math.radians(spread))
        d1_rest = qrot(r_spread, dirs[0])
        axis1 = _unit(np.cross(d1_rest, n))
        r1 = qmul(qaxis(tuple(axis1), math.radians(mcp)), r_spread)
        d1 = _unit(qrot(r1, dirs[0]))
        # middle/distal joints rotate about the parent-carried flexion axis
        axis2_rest = _unit(np.cross(dirs[1], n))
        axis2 = _unit(qrot(r1, axis2_rest))
        r2 = qmul(qaxis(tuple(axis2), math.radians(pip)), r1)
        d2 = _unit(qrot(r2, dirs[1]))
        axis3_rest = _unit(np.cross(dirs[2], n)) if len(dirs) > 2 else axis2_rest
        axis3 = _unit(qrot(r2, axis3_rest))
        r3 = qmul(qaxis(tuple(axis3), math.radians(dip)), r2)
        d3 = _unit(qrot(r3, dirs[2])) if len(dirs) > 2 else d2
        targets[(finger, 0)] = d1
        targets[(finger, 1)] = d2
        targets[(finger, 2)] = d3

    if "thumb" in rig.bones:
        m = measured.get("thumb", {})
        cmc = _clamp((m.get("cmc", MP_NEUTRAL_THUMB) - MP_NEUTRAL_THUMB) * gain, *LIMITS["cmc"])
        mcp = _clamp(m.get("mcp", 0.0) * gain, *LIMITS["thumb_mcp"])
        ip = _clamp(m.get("ip", 0.0) * gain, *LIMITS["thumb_ip"])
        dirs = rig.rest_dirs["thumb"]
        # opposition: +angle moves the thumb toward the fingers (+u)
        cmc_axis = _unit(np.cross(dirs[0], u))
        r_cmc = qaxis(tuple(cmc_axis), math.radians(cmc))
        d0 = _unit(qrot(r_cmc, dirs[0]))
        axis1_rest = _unit(np.cross(dirs[0], n))
        axis1 = _unit(qrot(r_cmc, axis1_rest))
        r1 = qmul(qaxis(tuple(axis1), math.radians(mcp)), r_cmc)
        d1 = _unit(qrot(r1, dirs[1]))
        if len(dirs) > 2:
            axis2_rest = _unit(np.cross(dirs[2], n))
            axis2 = _unit(qrot(r1, axis2_rest))
            r2 = qmul(qaxis(tuple(axis2), math.radians(ip)), r1)
            d2 = _unit(qrot(r2, dirs[2]))
        else:
            d2 = d1
        targets[("thumb", 0)] = d0
        targets[("thumb", 1)] = d1
        targets[("thumb", 2)] = d2
    return targets


def angles_to_local_quats(rig: HandRig, measured: Mapping[str, Mapping[str, float]],
                          wrist_world_q=None, gain: float = 1.0) -> Dict[str, Tuple[float, float, float, float]]:
    """Compute VMD-local rotations for every finger bone of one hand.

    The rotations are *hand-relative*: the wrist bone's own rotation is not
    involved, because finger bones are its children and follow it in FK.  The
    ``wrist_world_q`` argument is accepted for backward compatibility and
    ignored (target directions are built in the hand's rest frame).
    """
    targets = angles_to_targets(rig, measured, gain=gain)
    out: Dict[str, Tuple[float, float, float, float]] = {}
    for finger in FINGER_ORDER:
        chain = rig.bones.get(finger, [])
        if not chain:
            continue
        parent_q = (0.0, 0.0, 0.0, 1.0)
        for k, bone in enumerate(chain):
            d_target = targets.get((finger, k))
            if d_target is None:
                break
            d_local = qrot(qinv(parent_q), d_target)
            q = qnorm(_swing_q(bone.rest_dir, d_local))
            out[bone.name] = q
            parent_q = qnorm(qmul(parent_q, q))
    return out


# ---------------------------------------------------------------------------
# level assignment / temporal smoothing
# ---------------------------------------------------------------------------
@dataclass
class HandTrackConfig:
    # segment gating
    min_core_frames: int = 3
    core_valid_frac: float = 0.7
    core_step_p90: float = 30.0
    max_palm_cv: float = 0.22
    l0_max_agree: float = 28.0
    l1_max_agree: float = 35.0
    min_l1_frames: int = 3
    l1_valid_frac: float = 0.6
    l1_step_p90: float = 50.0
    # per-frame gating
    min_score: float = 0.25
    l0_min_score: float = 0.5
    # smoothing
    ema_alpha: float = 0.38
    ema_alpha_fallback: float = 0.22
    median_k: int = 3
    max_step_mcp: float = 22.0
    max_step_pip: float = 24.0
    max_step_dip: float = 16.0
    max_step_spread: float = 10.0
    max_gap_fill: int = 6
    # L1 2D compensation
    l1_gain: float = 1.35
    l1_damp: float = 0.85
    l1_spread_gain: float = 0.6


def _fallback_angles(level: int) -> np.ndarray:
    base = L2_ANGLES if level == LEVEL_L2 else L3_ANGLES
    arr = np.zeros(len(JOINT_KEYS), float)
    for f in LONG_FINGERS:
        g = FINGER_GAIN[f]
        arr[JOINT_INDEX[f"{f}.mcp"]] = base["mcp"] * g
        arr[JOINT_INDEX[f"{f}.pip"]] = base["pip"] * g
        arr[JOINT_INDEX[f"{f}.dip"]] = base["dip"] * g
        arr[JOINT_INDEX[f"{f}.spread"]] = base["spread"]
    arr[JOINT_INDEX["thumb.cmc"]] = base["cmc"]
    arr[JOINT_INDEX["thumb.mcp"]] = base["thumb_mcp"]
    arr[JOINT_INDEX["thumb.ip"]] = base["thumb_ip"]
    return arr


def _measured_vector(measured: Mapping[str, Mapping[str, float]], l1: bool,
                     cfg: HandTrackConfig) -> np.ndarray:
    arr = np.zeros(len(JOINT_KEYS), float)
    for f in LONG_FINGERS:
        m = measured.get(f, {})
        arr[JOINT_INDEX[f"{f}.spread"]] = m.get("spread", 0.0)
        arr[JOINT_INDEX[f"{f}.mcp"]] = m.get("mcp", 0.0)
        arr[JOINT_INDEX[f"{f}.pip"]] = m.get("pip", 0.0)
        arr[JOINT_INDEX[f"{f}.dip"]] = m.get("dip", 0.0)
    m = measured.get("thumb", {})
    arr[JOINT_INDEX["thumb.cmc"]] = m.get("cmc", MP_NEUTRAL_THUMB)
    arr[JOINT_INDEX["thumb.mcp"]] = m.get("mcp", 0.0)
    arr[JOINT_INDEX["thumb.ip"]] = m.get("ip", 0.0)
    if l1:
        # 2D foreshortens flexion; spread is noisy, damp it more.  A global
        # damp keeps a mis-detected 2D hand from producing an extreme pose.
        g = cfg.l1_gain * cfg.l1_damp
        for f in LONG_FINGERS:
            for k in ("mcp", "pip", "dip"):
                arr[JOINT_INDEX[f"{f}.{k}"]] *= g
            arr[JOINT_INDEX[f"{f}.spread"]] *= cfg.l1_spread_gain
        arr[JOINT_INDEX["thumb.mcp"]] *= g
        arr[JOINT_INDEX["thumb.ip"]] *= g
        arr[JOINT_INDEX["thumb.cmc"]] = (MP_NEUTRAL_THUMB
                                         + (arr[JOINT_INDEX["thumb.cmc"]]
                                            - MP_NEUTRAL_THUMB) * cfg.l1_spread_gain)
    return arr


def _limit_vector(arr: np.ndarray) -> np.ndarray:
    out = np.array(arr, float)
    for f in LONG_FINGERS:
        out[JOINT_INDEX[f"{f}.spread"]] = _clamp(out[JOINT_INDEX[f"{f}.spread"]], *LIMITS["spread"])
        for k in ("mcp", "pip", "dip"):
            out[JOINT_INDEX[f"{f}.{k}"]] = _clamp(out[JOINT_INDEX[f"{f}.{k}"]], *LIMITS[k])
    out[JOINT_INDEX["thumb.cmc"]] = _clamp(out[JOINT_INDEX["thumb.cmc"]], *LIMITS["cmc"])
    out[JOINT_INDEX["thumb.mcp"]] = _clamp(out[JOINT_INDEX["thumb.mcp"]], *LIMITS["thumb_mcp"])
    out[JOINT_INDEX["thumb.ip"]] = _clamp(out[JOINT_INDEX["thumb.ip"]], *LIMITS["thumb_ip"])
    return out


def flatten_angles(angles: Optional[Mapping[str, Mapping[str, float]]]) -> Dict[str, float]:
    if not angles:
        return {}
    out = {}
    for finger, joints in angles.items():
        for k, v in joints.items():
            out[f"{finger}.{k}"] = float(v)
    return out


def _agreement(rec: Optional[Mapping], gain: float) -> float:
    """Median |3D - gain*2D| over flexion joints; a reliability proxy."""
    if not rec:
        return 999.0
    a3 = flatten_angles(rec.get("angles3d"))
    a2 = flatten_angles(rec.get("angles2d"))
    if not a3 or not a2:
        return 999.0
    diffs = []
    for k in a3:
        if not (k.endswith(".mcp") or k.endswith(".pip") or k.endswith(".ip")):
            continue
        if k in a2:
            diffs.append(abs(a3[k] - a2[k] * gain))
    return float(np.median(diffs)) if diffs else 999.0


def _palm_cv(records: Sequence[Optional[Mapping]], seg: Sequence[int]) -> float:
    vals = [float(records[k]["palm_px"]) for k in seg
            if records[k] and records[k].get("valid") and records[k].get("palm_px")]
    if len(vals) < 2:
        return 0.0
    m = float(np.mean(vals))
    return float(np.std(vals) / m) if m > 1e-6 else 0.0


def _p90_step(records: Sequence[Optional[Mapping]], key: str) -> Optional[float]:
    steps = []
    prev = None
    for rec in records:
        if not rec or not rec.get("valid"):
            prev = None
            continue
        ang = flatten_angles(rec.get(key))
        if not ang:
            prev = None
            continue
        if prev is not None:
            steps.extend(abs(ang[k] - prev[k]) for k in ang if k in prev)
        prev = ang
    if not steps:
        return None
    return float(np.percentile(steps, 90))


def assign_levels(records: Sequence[Optional[Mapping]], wrist_conf: Sequence[float],
                  cfg: Optional[HandTrackConfig] = None) -> Tuple[np.ndarray, List[List[int]], Dict]:
    """Per-frame L0..L3 from per-frame detections (one side).

    ``records[i]`` is None or a dict with ``valid``, ``score``, ``angles3d``,
    ``angles2d`` and ``world_ok`` keys.  Levels are assigned per *segment* of
    consecutive detections; isolated frames never become L0/L1.
    """
    cfg = cfg or HandTrackConfig()
    n = len(records)
    levels = np.full(n, LEVEL_L3, dtype=np.int8)
    segments: List[List[int]] = []
    diagnostics: Dict[str, object] = {"segments": [], "rejected": 0}

    i = 0
    while i < n:
        if not records[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and records[j + 1]:
            j += 1
        seg = list(range(i, j + 1))
        valid = [k for k in seg if records[k].get("valid")]
        vfrac = len(valid) / len(seg)
        step3 = _p90_step([records[k] for k in seg], "angles3d")
        step2 = _p90_step([records[k] for k in seg], "angles2d")
        step = step3 if step3 is not None else step2
        palm_cv = _palm_cv(records, seg)
        info = {"start": i, "end": j, "len": len(seg), "valid": len(valid),
                "valid_frac": round(vfrac, 3),
                "palm_cv": round(palm_cv, 3),
                "step3_p90": None if step3 is None else round(step3, 2),
                "step2_p90": None if step2 is None else round(step2, 2)}

        core = (len(seg) >= cfg.min_core_frames and vfrac >= cfg.core_valid_frac
                and step is not None and step <= cfg.core_step_p90
                and palm_cv <= cfg.max_palm_cv)
        weak = (len(seg) >= cfg.min_l1_frames and vfrac >= cfg.l1_valid_frac
                and step is not None and step <= cfg.l1_step_p90
                and palm_cv <= cfg.max_palm_cv)
        if core or weak:
            accepted = 0
            for k in seg:
                rec = records[k]
                if not rec.get("valid") or rec.get("score", 0.0) < cfg.min_score:
                    continue
                agree = _agreement(rec, cfg.l1_gain)
                info.setdefault("agree", {})[k] = round(agree, 1)
                if agree > cfg.l1_max_agree:
                    continue
                if (core and agree <= cfg.l0_max_agree
                        and rec.get("angles3d") is not None
                        and rec.get("world_ok", False)
                        and rec.get("score", 0.0) >= cfg.l0_min_score):
                    levels[k] = LEVEL_L0
                else:
                    levels[k] = LEVEL_L1
                accepted += 1
            if accepted < 2:
                # too few trustworthy frames: leave the whole segment on fallback
                for k in seg:
                    levels[k] = LEVEL_L3
                info["level"] = "sparse"
            else:
                info["level"] = "core" if core else "weak"
        else:
            info["level"] = "rejected"
            diagnostics["rejected"] += len(valid)
        diagnostics["segments"].append(info)
        segments.append(seg)
        i = j + 1

    # L2 vs L3 on fallback frames: L2 if the body wrist is reasonably visible
    for k in range(n):
        if levels[k] >= LEVEL_L2:
            levels[k] = LEVEL_L2 if wrist_conf[k] >= 0.35 else LEVEL_L3
    return levels, segments, diagnostics


def _median_filter(x: np.ndarray, k: int, active: np.ndarray) -> np.ndarray:
    if k <= 1:
        return x
    out = np.array(x, float)
    half = k // 2
    n = len(x)
    for i in range(n):
        if not active[i]:
            continue
        lo, hi = max(0, i - half), min(n, i + half + 1)
        idx = [j for j in range(lo, hi) if active[j]]
        if len(idx) >= 2:
            out[i] = np.median(x[idx], axis=0)
    return out


def build_angle_track(records: Sequence[Optional[Mapping]], levels: np.ndarray,
                      cfg: Optional[HandTrackConfig] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Per-frame joint angles (N, 19), smoothed, with gaps filled.

    Returns (angles, target_before_smoothing).  Only L0/L1 frames carry measured
    angles; fallback frames start from the L2/L3 procedural pose.
    """
    cfg = cfg or HandTrackConfig()
    n = len(levels)
    target = np.zeros((n, len(JOINT_KEYS)), float)
    measured_mask = np.zeros(n, bool)
    for i in range(n):
        lvl = int(levels[i])
        if lvl in (LEVEL_L0, LEVEL_L1):
            rec = records[i]
            target[i] = _limit_vector(_measured_vector(rec["angles3d"] if lvl == LEVEL_L0
                                                      else rec["angles2d"], lvl == LEVEL_L1, cfg))
            measured_mask[i] = True
        else:
            target[i] = _fallback_angles(lvl)

    # linear fill of short holes between two measured runs
    active_fill = np.zeros(n, bool)
    i = 0
    while i < n:
        if measured_mask[i]:
            i += 1
            continue
        j = i
        while j < n and not measured_mask[j]:
            j += 1
        if i > 0 and j < n and (j - i) <= cfg.max_gap_fill:
            for k in range(i, j):
                t = (k - i + 1) / (j - i + 1)
                target[k] = target[i - 1] * (1 - t) + target[j] * t
            active_fill[i:j] = True
        i = j

    active = measured_mask | active_fill

    target = _median_filter(target, cfg.median_k, active)
    out = np.array(target, float)
    for i in range(1, n):
        alpha = cfg.ema_alpha if (active[i] and active[i - 1]) else cfg.ema_alpha_fallback
        prev = out[i - 1]
        cur = target[i]
        sm = prev * (1 - alpha) + cur * alpha
        step_lim = np.full(len(JOINT_KEYS), cfg.max_step_spread)
        for f in LONG_FINGERS:
            step_lim[JOINT_INDEX[f"{f}.mcp"]] = cfg.max_step_mcp
            step_lim[JOINT_INDEX[f"{f}.pip"]] = cfg.max_step_pip
            step_lim[JOINT_INDEX[f"{f}.dip"]] = cfg.max_step_dip
        step_lim[JOINT_INDEX["thumb.mcp"]] = cfg.max_step_mcp * 0.6
        step_lim[JOINT_INDEX["thumb.ip"]] = cfg.max_step_dip
        delta = np.clip(sm - prev, -step_lim, step_lim)
        out[i] = _limit_vector(prev + delta)
    out[0] = _limit_vector(target[0])
    return out, target


def vector_to_measured(vec: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Inverse of _measured_vector for feeding angles_to_local_quats."""
    out: Dict[str, Dict[str, float]] = {}
    for f in LONG_FINGERS:
        out[f] = {
            "spread": float(vec[JOINT_INDEX[f"{f}.spread"]]),
            "mcp": float(vec[JOINT_INDEX[f"{f}.mcp"]]),
            "pip": float(vec[JOINT_INDEX[f"{f}.pip"]]),
            "dip": float(vec[JOINT_INDEX[f"{f}.dip"]]),
        }
    out["thumb"] = {
        "cmc": float(vec[JOINT_INDEX["thumb.cmc"]]),
        "mcp": float(vec[JOINT_INDEX["thumb.mcp"]]),
        "ip": float(vec[JOINT_INDEX["thumb.ip"]]),
    }
    return out


# ---------------------------------------------------------------------------
# body FK: wrist world rotation per frame
# ---------------------------------------------------------------------------
def wrist_world_quats(model: Mapping, vmd: Mapping, wrist_name: str,
                      n_frames: int) -> List[Tuple[float, float, float, float]]:
    """Evaluate the body VMD on the PMX skeleton and return wrist world quats.

    IK is fully disabled (the body VMD is baked), but inherit-rotation bones
    (twist/D-chain helpers) still run, exactly like the renderer.
    """
    from .mmd.animation import Skeleton, Track

    skel = Skeleton(model)
    name_to_index = {b["name"]: i for i, b in enumerate(model["bones"])}
    tracks = {name: Track(keys) for name, keys in vmd["bones"].items()
              if name in name_to_index}
    wi = name_to_index[wrist_name]
    out = []
    for f in range(n_frames):
        sampled = {name_to_index[name]: tr.sample(f) for name, tr in tracks.items()}
        skel.pose(sampled, ik_enabled=set())
        q, _ = skel.world(wi)
        out.append(tuple(float(x) for x in q))
    return out


# ---------------------------------------------------------------------------
# detection plausibility (shared by run_hand / audit)
# ---------------------------------------------------------------------------
def hand_plausibility(points2d: Sequence[Sequence[float]],
                      world3d: Optional[Sequence[Sequence[float]]],
                      expected_wrist_xy: Sequence[float],
                      expected_palm_px: float) -> Tuple[bool, int, Dict]:
    """Geometric sanity check for a MediaPipe hand.

    The expected palm size comes from the body forearm length (foreshortened,
    so the size check is deliberately loose).  Checks are reported individually
    so the caller can debug rejections instead of just seeing ``False``.
    """
    p = np.asarray(points2d, float)
    if p.shape[0] != 21:
        return False, 99, {"error": "landmark count"}
    wrist = p[WRIST]
    palm = float(np.linalg.norm(p[MIDDLE[0]] - wrist))
    width = float(np.linalg.norm(p[INDEX[0]] - p[PINKY[0]]))
    if palm < 5.0:
        return False, 99, {"palm_px": palm}
    m = {"palm_px": palm, "width_px": width, "palm_ratio": width / palm}
    checks = {"palm_ratio": 0.30 <= m["palm_ratio"] <= 2.2}

    ratios = {}
    for finger, ids in (("thumb", THUMB), ("index", INDEX), ("middle", MIDDLE),
                        ("ring", RING), ("pinky", PINKY)):
        a, b, c, d = ids
        chain = (np.linalg.norm(p[b] - p[a]) + np.linalg.norm(p[c] - p[b])
                 + np.linalg.norm(p[d] - p[c]))
        ratios[finger] = float(chain / palm)
    m["finger_ratio"] = ratios
    long = [ratios[k] for k in LONG_FINGERS]
    checks["finger_ratio"] = all(0.40 <= v <= 2.2 for v in long)
    checks["finger_spread"] = (max(long) - min(long)) <= 1.0
    checks["thumb_ratio"] = 0.35 <= ratios["thumb"] <= 3.2
    # thumb base must be near the wrist (a hand-sized offset, not another limb)
    m["thumb_base_ratio"] = float(np.linalg.norm(p[THUMB[0]] - wrist) / palm)
    checks["thumb_base"] = 0.2 <= m["thumb_base_ratio"] <= 2.5

    world_ok = False
    if world3d is not None:
        w = np.asarray(world3d, float)
        if w.shape == (21, 3):
            scale = float(np.linalg.norm(w[MIDDLE[0]] - w[WRIST]))
            depth = float(np.ptp(w[:, 2]))
            m["world_scale_m"] = scale
            m["world_depth_m"] = depth
            checks["world_scale"] = 0.04 <= scale <= 0.22
            checks["world_depth"] = 0.005 <= depth <= 0.30
            world_ok = bool(checks["world_scale"] and checks["world_depth"])

    m["wrist_err_px"] = float(np.linalg.norm(wrist - np.asarray(expected_wrist_xy, float)))
    m["wrist_err_ratio"] = m["wrist_err_px"] / max(expected_palm_px, 1e-6)
    # the body wrist keypoint is often on the sleeve edge; allow up to ~3 palms
    checks["wrist_pos"] = m["wrist_err_ratio"] <= 3.0
    m["size_ratio"] = palm / max(expected_palm_px, 1e-6)
    # forearm foreshortening makes the expected size unreliable -> loose
    checks["size"] = 0.35 <= m["size_ratio"] <= 3.0

    m["checks"] = checks
    m["world_ok"] = world_ok
    fails = sum(1 for v in checks.values() if not v)
    return fails == 0, int(fails), m


def fit_l1_gain(records: Sequence[Optional[Mapping]],
                default: float = 1.35) -> Tuple[float, Dict]:
    """Fit the 2D->3D flexion gain from frames where both are available."""
    ratios = []
    for rec in records:
        if not rec or not rec.get("valid"):
            continue
        a3 = rec.get("angles3d")
        a2 = rec.get("angles2d")
        if not a3 or not a2:
            continue
        for finger in LONG_FINGERS:
            for k in ("mcp", "pip"):
                v2 = a2.get(finger, {}).get(k, 0.0)
                v3 = a3.get(finger, {}).get(k, 0.0)
                if 8.0 <= v2 <= 75.0 and v3 > v2:
                    ratios.append(v3 / v2)
    if len(ratios) < 8:
        return default, {"n": len(ratios), "gain": default, "source": "default"}
    gain = float(np.median(ratios))
    gain = _clamp(gain, 1.0, 1.5)
    return gain, {"n": len(ratios), "gain": gain, "source": "fit",
                  "p25": float(np.percentile(ratios, 25)),
                  "p75": float(np.percentile(ratios, 75))}
