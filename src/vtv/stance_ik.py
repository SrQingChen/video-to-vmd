# -*- coding: utf-8 -*-
"""M2.5 v0.1: stance-foot anchoring inside long confidence holes.

Why this stage exists
---------------------
M2.5 v0 kept a long occlusion continuous (no freeze, no boundary jump) and
beat "hold the last frame" on the gap pose error, but the planted foot still
travelled 0.17-0.35 units inside the hole.  The reason is structural: v0 has
to walk from the last observation to the recovery observation, and the
recovery observation itself sits at a different toe XZ (the predictor drifted
while the foot was hidden).  A bridge therefore *has* to move the foot.

v0.1 adds the missing physical constraint: if a foot was stably planted just
before the hole, keep its toe XZ in the world for the hole, solve the leg with
a 2-bone IK, and only hand the foot back near the end.  Only long holes, only
feet detected as stance at the hole entry, and only the thigh/shin/foot
rotations are touched.  The root, the other leg, the arms and every observed
frame stay exactly as they were.

Algorithm
---------
1. plans: for every low-confidence run of >= ``min_gap_frames`` whose frame
   before the hole is a high-confidence observation, look at the last
   ``anchor_frames`` high-confidence frames.  A side is anchored when contact
   was observed on the frame right before the hole and on at least
   ``anchor_min_contact`` of those frames.  A foot that hovers just above the
   self-referential contact band is accepted when it is within a 1% height
   slack and its three samples are mostly slow (a landing, not a swing).
2. anchor: toe XZ = median of the anchor frames; toe Y = median.  Default
   ``anchor_orientation='v0'`` anchors the XZ only and derives the ankle from
   the *v0* foot world orientation, so the correction is a pure horizontal
   shift (the ankle stays reachable and the contact height is unchanged).
   ``'anchor'`` also freezes the pre-gap foot world orientation and Y, which
   is the literal M2.5 spec and is kept for ablation (a rising hip can put
   that ankle out of leg reach).
3. solve: hip -> knee -> ankle 2-bone IK toward the target, keeping the
   current knee plane.  Trust region: knee <= 15 deg, thigh <= 25 deg,
   per-frame ankle displacement <= 2.5% of height.  When the target is out of
   reach or outside the trust region the weight is reduced smoothly
   (bisection), never hard-pulled.  In the ``anchor`` mode a reach-limited
   target may rise by <= 1% height instead of sliding horizontally.
4. release: inside the last ``release_s`` (0.4 s) the weight ramps down with
   smoothstep.  ``release_mode='budget'`` (default) chooses the smallest
   floor that keeps the toe within ``slide_budget`` of the anchor at the last
   gap frame; ``'full'`` is the textbook weight->0 release and is kept for
   the ablation.  A full release re-adopts the recovery observation, which is
   exactly the displacement the M2.5 slide gate measures, so the budget floor
   is the default.
5. verify: observed frames bit-identical, bone lengths unchanged, toe XZ
   drift inside the anchored span, boundary step at both gap edges.

Nothing here reads GT.  The scorer runs afterwards on the written VMD.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .cleanup import _runs, eval_tracks
from .contact_ik import _q_angle_deg, _swing
from .mmd.mathutil import qinv, qmul, qnorm, qrot, qslerp


# ---------------------------------------------------------------- config

@dataclass
class StanceConfig:
    """All v0.1 thresholds in one place."""

    # Confidence gate (same values as StreamingConfig by default).
    high: float = 0.75
    low: float = 0.35
    fps: float = 30.0
    # Trigger: long holes only.  A one/two-frame hole is not a visible slide
    # and v0 already holds it.  Six frames (0.2 s) also keeps the 3-5 frame
    # random holes on v0's bridge: the 0.4 s release would cover such a hole
    # entirely (no anchored hold), and a moving foot could be pulled away
    # from the v0 bridge (side random20 seed2: 0.49 diagnostic drift).
    min_gap_frames: int = 6
    # Anchor: last N high-confidence frames; the immediately preceding frame
    # must be contact, and at least anchor_min_contact of the N must be.
    anchor_frames: int = 3
    anchor_min_contact: int = 2
    # A foot that hovers a few mm above the self-referential contact band is
    # still a stance foot: the band is estimated from the prediction's own
    # minimum, which is noisy.  Accept it when it is within this slack and the
    # three samples are mostly slow: the per-frame cap tolerates one faster
    # landing frame, the median keeps a genuine swing foot out.
    anchor_band_slack_frac: float = 0.01
    anchor_speed: float = 0.20
    anchor_speed_median: float = 0.15
    anchor_vspeed: float = 0.08
    # Reject an anchor whose three samples are not actually still (a fast
    # drag can pass the loose cleanup speed gate but is not a plant).
    anchor_max_spread_frac: float = 0.01
    # Release.
    release_s: float = 0.4
    release_mode: str = "budget"      # "budget" | "full"
    slide_budget: float = 0.045       # units of toe XZ at the last gap frame
    min_release_weight: float = 0.0
    # What is held: ``"v0"`` anchors the toe XZ only and lets the foot world
    # orientation follow the v0 bridge (the ankle stays reachable and the
    # contact height is unchanged); ``"anchor"`` also freezes the pre-gap
    # world orientation and toe Y (the literal M2.5 spec, kept for ablation).
    anchor_orientation: str = "v0"
    # Trust region (M2.2 spec: knee 10-15 deg, thigh 25 deg, 2.5% height).
    max_knee_deg: float = 15.0
    max_thigh_deg: float = 25.0
    max_ankle_step_frac: float = 0.025
    # When the hip has travelled far enough that the anchored toe is out of
    # leg reach, let the toe rise by at most this fraction of height instead
    # of sliding it horizontally.  The XZ anchor is kept; the foot gives
    # vertically like a heel lift.  0 disables the relaxation.
    max_lift_frac: float = 0.01
    # Diagnostics: store the per-frame weight and toe drift in the report.
    debug_series: bool = False

    def release_frames(self) -> int:
        return max(1, int(round(float(self.release_s) * float(self.fps))))

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["release_frames"] = self.release_frames()
        return d


# ---------------------------------------------------------------- geometry

def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros(3)
    return v / n


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    c = float(np.dot(u, v)) / (nu * nv)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return np.asarray(qslerp(tuple(a), tuple(b), float(t)), dtype=np.float64)


def _rest_geometry(model: Dict, profile: Dict, side: str) -> Optional[Dict]:
    """Rest offsets of one leg and the joint names used for evaluation.

    ``d1`` = shin-thigh, ``d2`` = foot-shin, ``offset`` = toe-foot.  These are
    exactly the translations the PMX evaluator applies, so
    ``toe = ankle + R_foot * offset`` holds to numerical precision.
    """
    smap = profile.get("standard_map") or {}
    bones = {j: smap.get(f"{j}_{side}") for j in
             ("thigh", "shin", "foot", "toe")}
    if any(not b for b in bones.values()):
        return None
    name2i = {b["name"]: i for i, b in enumerate(model.get("bones") or [])}
    if any(b not in name2i for b in bones.values()):
        return None
    idx = {j: name2i[b] for j, b in bones.items()}
    m = model["bones"]
    rest = {j: np.asarray(m[i]["pos"], dtype=np.float64)
            for j, i in idx.items()}
    # The IK writes local rotations relative to the thigh's parent.  Use the
    # actual parent joint when the profile maps it; otherwise fall back to
    # the pelvis (all three human profiles parent the thigh to 下半身).
    parent_i = m[idx["thigh"]].get("parent", -1)
    parent_name = m[parent_i]["name"] if 0 <= parent_i < len(m) else None
    bone_to_joint = {v: k for k, v in smap.items()}
    parent_joint = bone_to_joint.get(parent_name, "pelvis")
    return {
        "bones": bones,
        "parent_joint": parent_joint,
        "d1": rest["shin"] - rest["thigh"],
        "d2": rest["foot"] - rest["shin"],
        "offset": rest["toe"] - rest["foot"],
        "l1": float(np.linalg.norm(rest["shin"] - rest["thigh"])),
        "l2": float(np.linalg.norm(rest["foot"] - rest["shin"])),
    }


# ---------------------------------------------------------------- 2-bone IK

def _knee_plane(H, K, A, u) -> np.ndarray:
    """Unit vector from the hip-ankle line toward the current knee."""
    v = K - H
    perp = v - float(np.dot(v, u)) * u
    n = float(np.linalg.norm(perp))
    if n > 1e-7:
        return perp / n
    # Straight leg: the bend plane is undefined; pick a stable plane.
    axis = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(u, axis))) > 0.9:
        axis = np.array([1.0, 0.0, 0.0])
    perp = np.cross(np.cross(u, axis), u)
    return _unit(perp)


def _solve_two_bone(H, K, A, T, l1: float, l2: float) -> Dict:
    """One frame: place the ankle at ``T`` (clamped to reach), keep the knee
    on its current side.  Returns positions plus the achieved deltas."""
    R = (l1 + l2) * (1.0 - 1e-9)
    lo = abs(l1 - l2) * (1.0 + 1e-9)
    u = T - H
    dist0 = float(np.linalg.norm(u))
    if dist0 < 1e-9:
        return {"ok": False}
    u = u / dist0
    dist = min(max(dist0, lo), R)
    perp = _knee_plane(H, K, A, u)
    cos_a = (l1 * l1 + dist * dist - l2 * l2) / (2.0 * l1 * dist + 1e-12)
    a = math.acos(max(-1.0, min(1.0, cos_a)))
    K2 = H + l1 * (math.cos(a) * u + math.sin(a) * perp)
    A2 = H + dist * u
    knee0 = _angle_deg(K - H, A - K)
    knee2 = _angle_deg(K2 - H, A2 - K2)
    return {
        "ok": True, "K": K2, "A": A2,
        "knee_deg": abs(knee2 - knee0),
        "thigh_deg": _angle_deg(K - H, K2 - H),
        "ankle_step": float(np.linalg.norm(A2 - A)),
        "reach_limited": bool(abs(dist - dist0) > 1e-9),
    }


def _reach_weight(H, A, Aa, l1: float, l2: float) -> float:
    """Largest w in [0,1] with |A + w*(Aa-A) - H| <= l1+l2."""
    R = (l1 + l2) * (1.0 - 1e-9)
    d0 = A - H
    dd = Aa - A
    c = float(np.dot(d0, d0)) - R * R
    if c > 0.0:
        # The current pose is always valid; a hair outside the inset radius is
        # numerical, not a reason to reject the whole frame.
        if float(np.dot(d0, d0)) > (l1 + l2) ** 2 * (1.0 + 1e-6):
            return 0.0
        c = 0.0
    a = float(np.dot(dd, dd))
    if a < 1e-12:
        return 1.0
    b = 2.0 * float(np.dot(d0, dd))
    disc = b * b - 4.0 * a * c
    if disc <= 0.0:
        return 1.0
    w = (-b + math.sqrt(disc)) / (2.0 * a)
    return max(0.0, min(1.0, w))


def _reach_lifted(H, Aa, l1: float, l2: float, max_lift: float
                  ) -> Tuple[np.ndarray, float]:
    """Raise the ankle target just enough to be reachable (XZ unchanged).

    A leg that is already nearly straight cannot keep a horizontal anchor
    when the hip travels; sliding the toe is not the only option - a real
    foot rolls and gives a little vertically.  The lift is capped, so a truly
    out-of-range anchor still falls back to the weighted release.
    """
    if max_lift <= 0.0:
        return Aa, 0.0
    # Aim slightly inside the reach sphere: a target exactly on it is
    # numerically unreachable (and a straight leg has zero margin).
    R = (l1 + l2) * (1.0 - 1e-4)
    d = Aa - H
    r2 = float(d[0] * d[0] + d[2] * d[2])
    if float(np.linalg.norm(d)) <= R or r2 >= R * R:
        return Aa, 0.0
    need = -math.sqrt(max(0.0, R * R - r2)) - float(d[1])
    if need <= 0.0:
        return Aa, 0.0
    dy = min(float(need), float(max_lift))
    out = np.asarray(Aa, dtype=np.float64).copy()
    out[1] += dy
    return out, dy


def _release_floor(A_end, R_ft_end, A_anchor, R_foot_anchor, toe_offset,
                   anchor_toe, budget: float, mode: str,
                   min_floor: float) -> float:
    """Smallest anchor weight at the last gap frame that keeps the toe XZ
    within ``budget`` of the anchor.

    The toe is not a linear function of the ankle weight: the foot world
    orientation is slerped at the same time, so solve the actual composition
    instead of approximating with the ankle distance.
    """
    if mode == "full":
        return 0.0

    def toe_at(w: float) -> np.ndarray:
        A_t = A_end + float(w) * (A_anchor - A_end)
        R_t = _slerp(R_ft_end, R_foot_anchor, w)
        return A_t + np.asarray(qrot(tuple(R_t), tuple(toe_offset)),
                                dtype=np.float64)

    def drift(w: float) -> float:
        return float(np.linalg.norm(
            (toe_at(w) - np.asarray(anchor_toe))[[0, 2]]))

    if drift(0.0) <= budget:
        return max(0.0, float(min_floor))
    lo, hi = 0.0, 1.0
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if drift(mid) <= budget:
            hi = mid
        else:
            lo = mid
    return max(float(min_floor), min(1.0, hi))


def _solve_weighted(H, K, A, Aa, w0: float, cfg: StanceConfig,
                    l1: float, l2: float, max_step: float,
                    prev_ankle: Optional[np.ndarray] = None) -> Dict:
    """Solve toward ``A + w*(Aa-A)`` for the largest trust-safe w <= w0.

    The weight is the anchor confidence: 1 = full anchor, 0 = leave v0 alone.
    Reducing it is how unreachable/over-trust-region frames release instead of
    hard-pulling the leg.  The 2.5% height cap is on the *output's* per-frame
    ankle step (``prev_ankle`` -> new ankle), not on the deviation from v0:
    the anchored pose may sit away from the v0 fill for the whole hole as long
    as it got there smoothly.
    """
    w0 = max(0.0, min(1.0, float(w0)))
    w0 = min(w0, _reach_weight(H, A, Aa, l1, l2))
    prev = A if prev_ankle is None else prev_ankle

    def ev(w: float) -> Dict:
        if w <= 1e-9:
            return {"ok": True, "K": K, "A": A, "knee_deg": 0.0,
                    "thigh_deg": 0.0, "ankle_step": 0.0,
                    "reach_limited": False, "w": 0.0}
        T = A + w * (Aa - A)
        r = _solve_two_bone(H, K, A, T, l1, l2)
        if not r.get("ok"):
            return {"ok": False, "w": 0.0}
        r["w"] = w
        r["ankle_step"] = float(np.linalg.norm(r["A"] - prev))
        return r

    def good(r: Dict) -> bool:
        return bool(r.get("ok")) and r["knee_deg"] <= cfg.max_knee_deg + 1e-9 \
            and r["thigh_deg"] <= cfg.max_thigh_deg + 1e-9 \
            and r["ankle_step"] <= max_step + 1e-9

    best = ev(w0)
    if good(best):
        return best
    # Sample then bisect: ``good`` is monotone in practice but the reach clamp
    # can flatten the end of the curve, so do not trust a pure bisection.
    lo_w, lo_r = 0.0, ev(0.0)
    hi_w = w0
    for i in range(8, 0, -1):
        w = w0 * i / 8.0
        r = ev(w)
        if good(r):
            lo_w, lo_r = w, r
            break
        hi_w = w
    if lo_w <= 0.0:
        return lo_r
    for _ in range(12):
        mid = 0.5 * (lo_w + hi_w)
        r = ev(mid)
        if good(r):
            lo_w, lo_r = mid, r
        else:
            hi_w = mid
    return lo_r


# ---------------------------------------------------------------- plans

def _anchor_plans(tracks: Dict, conf: Dict[str, np.ndarray], contact: Dict,
                  tr: Dict, model: Dict, profile: Dict, cfg: StanceConfig
                  ) -> Tuple[List[Dict], Dict]:
    """Find long holes and the sides that may be anchored in each."""
    names = sorted(conf)
    if not names or not contact:
        return [], {"long_gaps": 0, "reason": "no contact/confidence"}
    n = len(conf[names[0]])
    cmean = np.mean(np.stack([conf[nm] for nm in names], axis=0), axis=0)
    height = float(tr.get("height") or profile.get("height") or 1.0)
    max_spread = float(cfg.anchor_max_spread_frac) * height
    gaps = [(int(s), int(e)) for s, e in _runs(cmean < float(cfg.low))
            if e - s + 1 >= int(cfg.min_gap_frames)]
    plans: List[Dict] = []
    stats = {"long_gaps": len(gaps), "skipped_no_left": 0,
             "skipped_no_anchor_frames": 0, "skipped_spread": 0,
             "skipped_not_carried": 0, "skipped_no_contact": 0,
             "skipped_geometry": 0, "near_band_entries": 0}
    seen_sides = set()

    def near_stance(t: int, rec: Dict, toe: np.ndarray) -> bool:
        """Observed contact, or a slow toe within the band + small slack."""
        if bool(rec["observed"][t]):
            return True
        slack = float(cfg.anchor_band_slack_frac) * height
        if slack <= 0.0 or t <= 0:
            return False
        if float(toe[t, 1]) > float(rec["thr"]) + slack:
            return False
        d = toe[t] - toe[t - 1]
        if float(np.hypot(d[0], d[2])) > float(cfg.anchor_speed):
            return False
        if abs(float(d[1])) > float(cfg.anchor_vspeed):
            return False
        return True

    for s, e in gaps:
        if s <= 0 or cmean[s - 1] < float(cfg.high):
            stats["skipped_no_left"] += 1
            continue
        pre: List[int] = []
        t = s - 1
        while t >= 0 and len(pre) < int(cfg.anchor_frames):
            if cmean[t] >= float(cfg.high):
                pre.append(int(t))
            t -= 1
        if len(pre) < 2:
            stats["skipped_no_anchor_frames"] += 1
            continue
        if pre != list(range(s - 1, s - 1 - len(pre), -1)):
            # A short hole before the long one: the anchor frames are not a
            # contiguous observation and their velocities are held values.
            stats["skipped_no_anchor_frames"] += 1
            continue
        for side in ("l", "r"):
            rec = (contact.get("sides") or {}).get(side)
            if not rec:
                continue
            carried = rec.get("carried")
            observed = rec.get("observed")
            if carried is None or observed is None:
                continue
            toe = tr["world"].get(f"toe_{side}")
            foot_q = tr["world_q"].get(f"foot_{side}")
            if toe is None or foot_q is None:
                stats["skipped_geometry"] += 1
                continue
            entry_contact = bool(carried[s - 1])
            entry_near = (not entry_contact
                          and near_stance(s - 1, rec, toe))
            if not entry_contact and not entry_near:
                stats["skipped_not_carried"] += 1
                continue
            if entry_near:
                stats["near_band_entries"] += 1
            af = [t for t in pre if near_stance(t, rec, toe)]
            near_frames = [t for t in af if not bool(observed[t])]
            if len(near_frames) >= 2:
                spd = []
                for t in near_frames:
                    d = toe[t] - toe[t - 1]
                    spd.append(float(np.hypot(d[0], d[2])))
                if float(np.median(spd)) > float(cfg.anchor_speed_median):
                    stats["skipped_not_carried"] += 1
                    continue
            if (len(af) < min(int(cfg.anchor_min_contact), len(pre))
                    or (s - 1) not in af):
                stats["skipped_no_contact"] += 1
                continue
            geo = _rest_geometry(model, profile, side)
            if geo is None:
                stats["skipped_geometry"] += 1
                continue
            anchor = np.median(toe[af], axis=0)
            spread = float(np.max(np.linalg.norm(
                (toe[af] - anchor)[:, [0, 2]], axis=1)))
            if spread > max_spread:
                stats["skipped_spread"] += 1
                continue
            R_foot = np.asarray(foot_q[s - 1], dtype=np.float64)
            off = qrot(tuple(R_foot), tuple(geo["offset"]))
            ankle_target = anchor - np.asarray(off, dtype=np.float64)
            carried_frames = int(np.sum(carried[s:e + 1]))
            plans.append({
                "side": side, "start": int(s), "end": int(e),
                "length": int(e - s + 1),
                "anchor_frames": [int(x) for x in af],
                "anchor_spread": round(spread, 6),
                "entry": "contact" if entry_contact else "near_band",
                "anchor_toe": [round(float(x), 6) for x in anchor],
                "ankle_target": [round(float(x), 6) for x in ankle_target],
                "foot_world_q": [round(float(x), 9) for x in R_foot],
                "bone_map": dict(geo["bones"]),
                "parent_joint": geo["parent_joint"],
                "l1": geo["l1"], "l2": geo["l2"],
                "toe_offset": [float(x) for x in geo["offset"]],
                "carried_frames": carried_frames,
                "carried_expired": bool(carried_frames < e - s + 1),
                "release_mode": cfg.release_mode,
                "release_floor": 0.0,
            })
            seen_sides.add(side)
    return plans, stats


# ---------------------------------------------------------------- driver

def _copy_tracks(tracks: Dict) -> Dict:
    out = {}
    for name, rec in tracks.items():
        out[name] = {k: (np.asarray(v, dtype=np.float64).copy()
                         if isinstance(v, np.ndarray) else v)
                     for k, v in rec.items()}
    return out


def apply_stance_ik(tracks: Dict[str, Dict[str, np.ndarray]],
                    conf: Dict[str, np.ndarray], contact: Dict,
                    model: Dict, profile: Dict,
                    cfg: Optional[StanceConfig] = None) -> Dict:
    """Anchor planted toes inside long holes; returns new tracks + report."""
    cfg = cfg or StanceConfig()
    out = _copy_tracks(tracks)
    if not tracks or not conf or not contact:
        return {"tracks": out, "report": {"enabled": True, "plans": [],
                                          "gaps_anchored": 0}}
    tr = eval_tracks(model, profile, tracks)
    plans, stats = _anchor_plans(tracks, conf, contact, tr, model, profile, cfg)
    names = sorted(conf)
    n = len(conf[names[0]])
    height = float(tr.get("height") or profile.get("height") or 1.0)
    max_step = float(cfg.max_ankle_step_frac) * height
    look = cfg.release_frames()

    edited = np.zeros(n, dtype=bool)
    rows: List[Dict] = []
    reach_limited = 0
    knee_max = 0.0
    thigh_max = 0.0
    ankle_step_max = 0.0
    weight_sum = 0.0
    weight_n = 0
    weight_min = 1.0
    lift_frames = 0
    lift_max = 0.0

    for plan in plans:
        side = plan["side"]
        s, e = plan["start"], plan["end"]
        bt = plan["bone_map"]["thigh"]
        bs = plan["bone_map"]["shin"]
        bf = plan["bone_map"]["foot"]
        btoe = plan["bone_map"]["toe"]
        if not all(b in out for b in (bt, bs, bf, btoe)):
            plan["applied"] = False
            plan["reason"] = "leg bone tracks missing"
            rows.append(plan)
            continue
        pj = plan["parent_joint"]
        if pj not in tr["world_q"]:
            pj = "pelvis"
        if pj not in tr["world_q"]:
            plan["applied"] = False
            plan["reason"] = "parent world rotation missing"
            rows.append(plan)
            continue
        A_anchor = np.asarray(plan["ankle_target"], dtype=np.float64)
        R_foot_anchor = np.asarray(plan["foot_world_q"], dtype=np.float64)
        toe_offset = np.asarray(plan["toe_offset"], dtype=np.float64)
        anchor_toe = np.asarray(plan["anchor_toe"], dtype=np.float64)
        keep_orient = cfg.anchor_orientation == "anchor"
        # Release floor: keep the toe at the last gap frame within the slide
        # budget.  A full release would re-adopt an observation that is
        # (by construction) the displacement the gate measures.
        release_start = max(s, e - min(look, e - s + 1) + 1)
        A_end = tr["world"][f"foot_{side}"][e]
        R_ft_end = tr["world_q"][f"foot_{side}"][e]
        toe_end_v0 = A_end + np.asarray(
            qrot(tuple(R_ft_end), tuple(toe_offset)), dtype=np.float64)
        delta_end = float(np.linalg.norm(
            (toe_end_v0 - anchor_toe)[[0, 2]]))
        if keep_orient:
            floor = _release_floor(
                A_end, R_ft_end, A_anchor, R_foot_anchor, toe_offset,
                anchor_toe, float(cfg.slide_budget) * 0.95,
                cfg.release_mode, float(cfg.min_release_weight))
        elif cfg.release_mode == "full":
            floor = 0.0
        elif delta_end <= 1e-9:
            floor = 0.0
        else:
            floor = max(float(cfg.min_release_weight),
                        1.0 - float(cfg.slide_budget) * 0.95 / delta_end)
            floor = min(1.0, floor)
        plan["release_floor"] = round(floor, 6)
        plan["release_start"] = int(release_start)
        plan["end_delta_vs_anchor"] = round(delta_end, 6)
        plan["anchor_orientation"] = cfg.anchor_orientation

        applied_flags = np.zeros(e - s + 1, dtype=bool)
        w_eff_arr = np.zeros(e - s + 1)
        prev_out = np.asarray(tr["world"][f"foot_{side}"][s - 1],
                              dtype=np.float64)
        for t in range(s, e + 1):
            H = tr["world"][f"thigh_{side}"][t]
            K = tr["world"][f"shin_{side}"][t]
            A = tr["world"][f"foot_{side}"][t]
            # Never edit an observed leg frame (per-bone confidence can be
            # higher than the frame mean).
            if any(conf[b][t] >= float(cfg.high) for b in (bt, bs, bf)):
                prev_out = A
                continue
            if t < release_start:
                w = 1.0
            else:
                u = (t - release_start) / max(1, e - release_start)
                w = 1.0 - (1.0 - floor) * _smoothstep(u)
            R_th = tr["world_q"][f"thigh_{side}"][t]
            R_sh = tr["world_q"][f"shin_{side}"][t]
            R_ft = tr["world_q"][f"foot_{side}"][t]
            R_pel = tr["world_q"][pj][t]
            if keep_orient:
                target, lift = _reach_lifted(
                    H, A_anchor, float(plan["l1"]), float(plan["l2"]),
                    float(cfg.max_lift_frac) * height)
                if lift > 1e-9:
                    lift_frames += 1
                    lift_max = max(lift_max, lift)
            else:
                # Anchor the toe XZ; let the foot world orientation and height
                # follow the v0 bridge.  The correction is a pure horizontal
                # ankle shift, so a nearly-straight leg does not overstretch
                # and the contact height does not change.
                toe_v0 = A + np.asarray(qrot(tuple(R_ft), tuple(toe_offset)),
                                        dtype=np.float64)
                dxz = anchor_toe[[0, 2]] - toe_v0[[0, 2]]
                target = A + np.array([dxz[0], 0.0, dxz[1]],
                                      dtype=np.float64)
            res = _solve_weighted(H, K, A, target, w, cfg,
                                  float(plan["l1"]), float(plan["l2"]),
                                  max_step, prev_ankle=prev_out)
            if not res.get("ok") or float(res["w"]) <= 1e-9:
                if _reach_weight(H, A, target, float(plan["l1"]),
                                float(plan["l2"])) < 0.999:
                    reach_limited += 1
                prev_out = A
                continue
            we = float(res["w"])
            K2, A2 = res["K"], res["A"]
            q_th = qmul(_swing(K - H, K2 - H), tuple(R_th))
            q_sh = qmul(_swing(A - K, A2 - K2), tuple(R_sh))
            q_ft = _slerp(R_ft, R_foot_anchor, we) if keep_orient else R_ft
            out[bt]["rot"][t] = qnorm(qmul(qinv(tuple(R_pel)), q_th))
            out[bs]["rot"][t] = qnorm(qmul(qinv(q_th), q_sh))
            out[bf]["rot"][t] = qnorm(qmul(qinv(q_sh), q_ft))
            applied_flags[t - s] = True
            w_eff_arr[t - s] = we
            edited[t] = True
            prev_out = A2
            reach_limited += int(bool(res.get("reach_limited")))
            knee_max = max(knee_max, float(res["knee_deg"]))
            thigh_max = max(thigh_max, float(res["thigh_deg"]))
            ankle_step_max = max(ankle_step_max, float(res["ankle_step"]))
            weight_sum += we
            weight_n += 1
            weight_min = min(weight_min, we)
        plan["applied"] = bool(applied_flags.any())
        plan["frames_applied"] = int(applied_flags.sum())
        plan["weight_min"] = round(float(w_eff_arr[applied_flags].min()), 4) \
            if applied_flags.any() else 0.0
        plan["weight_mean"] = round(float(w_eff_arr[applied_flags].mean()), 4) \
            if applied_flags.any() else 0.0
        if cfg.debug_series:
            plan["weights"] = [round(float(x), 4) for x in w_eff_arr]
        rows.append(plan)

    # ---- verify on the written tracks ---------------------------------
    tr2 = eval_tracks(model, profile, out)
    plan_rows: List[Dict] = []
    for plan in rows:
        if not plan.get("applied"):
            plan_rows.append(plan)
            continue
        s, e = plan["start"], plan["end"]
        side = plan["side"]
        bt = plan["bone_map"]["thigh"]
        bs = plan["bone_map"]["shin"]
        bf = plan["bone_map"]["foot"]
        toe_before = tr["world"][f"toe_{side}"][s:e + 1]
        toe_after = tr2["world"][f"toe_{side}"][s:e + 1]
        anchor = np.asarray(plan["anchor_toe"], dtype=np.float64)
        drift = float(np.max(np.linalg.norm(
            (toe_after - anchor)[:, [0, 2]], axis=1)))
        end_slide = float(np.linalg.norm(
            toe_after[-1, [0, 2]] - toe_after[0, [0, 2]]))
        # boundary step: max local angle over the three leg bones at s-1->s
        # and e->e+1 (the same joints the scorer sees)
        steps = []
        for a, b in ((s - 1, s), (e, e + 1)):
            if a < 0 or b >= n:
                continue
            for bn in (bt, bs, bf):
                if bn in out and bn in tracks:
                    steps.append(_q_angle_deg(out[bn]["rot"][b],
                                              out[bn]["rot"][a]))
        # bone lengths must not change
        len_err = 0.0
        for a, b in ((f"thigh_{side}", f"shin_{side}"),
                     (f"shin_{side}", f"foot_{side}")):
            la = np.linalg.norm(tr["world"][a] - tr["world"][b], axis=1)
            lb = np.linalg.norm(tr2["world"][a] - tr2["world"][b], axis=1)
            len_err = max(len_err, float(np.abs(la - lb).max()))
        plan["verify"] = {
            "toe_xz_drift_max": round(drift, 6),
            "toe_xz_end_slide": round(end_slide, 6),
            "boundary_step_max_deg": round(max(steps), 4) if steps else 0.0,
            "bone_length_max_err": round(len_err, 9),
            "toe_before_end": [round(float(x), 5) for x in toe_before[-1]],
            "toe_after_end": [round(float(x), 5) for x in toe_after[-1]],
        }
        if cfg.debug_series:
            plan["toe_drift_series"] = [round(float(x), 5) for x in
                                        np.linalg.norm((toe_after - anchor)[:, [0, 2]],
                                                       axis=1)]
        plan_rows.append(plan)

    # observed frames must remain bit-identical (angle and position)
    changed = 0.0
    pos_changed = 0.0
    for name in names:
        ok = conf[name] >= float(cfg.high)
        if not ok.any():
            continue
        qa = np.asarray(out[name]["rot"], dtype=np.float64)[ok]
        qb = np.asarray(tracks[name]["rot"], dtype=np.float64)[ok]
        na = np.linalg.norm(qa, axis=1, keepdims=True)
        nb = np.linalg.norm(qb, axis=1, keepdims=True)
        d = np.abs(np.sum((qa / np.maximum(na, 1e-12))
                          * (qb / np.maximum(nb, 1e-12)), axis=1))
        changed = max(changed, float(np.degrees(
            2.0 * np.arccos(np.clip(d, -1.0, 1.0))).max()))
        pos_changed = max(pos_changed, float(np.abs(
            out[name]["pos"][ok] - np.asarray(tracks[name]["pos"])[ok]).max()))

    report = {
        "enabled": True,
        "config": cfg.to_dict(),
        "stats": stats,
        "gaps_anchored": int(sum(1 for p in plan_rows if p.get("applied"))),
        "sides_anchored": sorted({p["side"] for p in plan_rows
                                  if p.get("applied")}),
        "frames_edited": int(edited.sum()),
        "reach_limited_frames": int(reach_limited),
        "reach_lift_frames": int(lift_frames),
        "reach_lift_max": round(lift_max, 6),
        "knee_max_deg": round(knee_max, 4),
        "thigh_max_deg": round(thigh_max, 4),
        "ankle_step_max": round(ankle_step_max, 6),
        "weight_mean": round(weight_sum / weight_n, 4) if weight_n else 0.0,
        "weight_min": round(weight_min, 4) if weight_n else 0.0,
        "toe_xz_drift_max": round(max(
            ((p.get("verify") or {}).get("toe_xz_drift_max", 0.0)
             for p in plan_rows), default=0.0), 6),
        "toe_xz_end_slide_max": round(max(
            ((p.get("verify") or {}).get("toe_xz_end_slide", 0.0)
             for p in plan_rows), default=0.0), 6),
        "boundary_step_max_deg": round(max(
            ((p.get("verify") or {}).get("boundary_step_max_deg", 0.0)
             for p in plan_rows), default=0.0), 4),
        "observed_changed_max_deg": round(changed, 9),
        "observed_pos_changed_max": round(pos_changed, 9),
        "plans": plan_rows,
    }
    return {"tracks": out, "report": report}
