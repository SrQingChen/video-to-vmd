# -*- coding: utf-8 -*-
"""Contact-gated 2-bone leg IK for a VMD (M2.2).

Why this stage exists
---------------------
M2's selective cleanup lowered jerk and the side-view foot slide, but closed-loop
contact recall stayed at 0.46 / 0.70 / 0.98 because the predicted toe hovers
above the GT contact band.  ``scripts/diag_contact.py`` showed that this is not
a temporal glitch:

* front left toe sits **0.46 units** above GT on average (2.0% of height) with a
  std of only 0.13 - a *systematic* leg/hip placement offset;
* that same leg's predicted knee is over-flexed (30 deg vs GT 21), so part of the
  offset is recoverable by extending the leg;
* on 3/4 the leg is already at GT's extension and the remaining height error is
  the pelvis being ~0.4-0.5 units too high.  Leg IK cannot fix that without a
  root-Y move, which M2.2 forbids: it would fake contact and push the other foot
  through the floor.

So the solver only does what is physically justified: vertical 2-bone IK
(hip -> knee -> ankle) **inside contact intervals**, inside a trust region.
Everything it cannot fix is reported, never hidden.

Algorithm
---------
1. ground ``g`` = low percentile of both toe heights (the character stands on
   the floor at some point of the clip), clamped to y=0 because that is the
   scorer's floor.
2. per-foot contact probability from the height band, **vertical** speed and the
   optional 2D ankle confidence, with hysteresis + gap fill.  Horizontal speed is
   deliberately *not* part of ``p_contact``: a foot dragging along the floor is
   still in floor contact; horizontal speed gates XZ locking, which is cleanup's
   job (``cleanup.foot_lock``).
3. desired drop = how far the toe is above the mid-band, with a smooth "need"
   ramp that is exactly 0 inside the band (feet that are already fine are not
   touched - that is why the side view, already at 0.98 recall, barely moves).
4. exact vertical IK: target ankle = ankle - (0, drop, 0) in the current knee
   plane.  Leg reach and the per-frame knee/thigh rotation change are hard caps.
   The foot's **world orientation is preserved**, so the toe XZ does not move:
   this stage cannot make the closed-loop slide worse.
5. write thigh/shin/foot local rotations back.  The output is still a normal
   23-track IK-off VMD (no MMD IK track is written); Blender still bakes the
   inherited D-chain from it.

Nothing here reads GT.  ``scripts/solve_contact_ik.py --gt`` only scores the
output afterwards (synthetic closed loop); the real T5 path has no GT.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .cleanup import dense_tracks, eval_tracks, tracks_to_vmd, _runs, _moving_average
from .mmd.mathutil import QID, qinv, qmul, qnorm

# Contact band: identical to the closed-loop scorer's default margin
# (``metrics.core.DEFAULT_CONTACT_MARGIN`` = 0.02 of height at M2), so "inside
# the band" means "the scorer would call this tooth planted".
DEFAULT_BAND = 0.02
# The floor is the lowest point a foot reaches; a low percentile is more robust
# than the raw minimum (one glitch frame must not move the whole clip down).
DEFAULT_GROUND_PCT = 1.0
# Where inside the band a corrected toe is aimed.  0.5 minimises the expected
# height error when the GT position inside its own band is unknown.
DEFAULT_TARGET_FRAC = 0.5
# Hysteresis: enter contact below ``enter_frac * band`` above the floor, leave
# above ``exit_frac * band``.  ``need`` ramps to 1 half a band above the top.
DEFAULT_ENTER_FRAC = 1.5
DEFAULT_EXIT_FRAC = 2.0
# A planted foot does not move vertically fast: 0.10 unit/frame = ~2 unit/s.
DEFAULT_VSPEED = 0.10
# Trust region (M2.2 spec: single-frame knee 10-15 deg, displacement 2-3% h).
DEFAULT_MAX_KNEE_DEG = 15.0
DEFAULT_MAX_THIGH_DEG = 25.0
DEFAULT_MAX_DROP_FRAC = 0.025
# Contact islands shorter than this are flicker; holes up to this are bridged.
DEFAULT_MIN_LEN = 3
DEFAULT_FILL_GAP = 2
DEFAULT_RAMP = 3
DEFAULT_P_CONTACT_MIN = 0.8
# 2D ankle keypoints (ViTPose COCO-17): 15 = left ankle, 16 = right ankle.
_ANKLE_KP = {"l": 15, "r": 16}


# ---------------------------------------------------------------- quaternions

def _q_angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    d = abs(float(np.dot(np.asarray(a, dtype=np.float64),
                         np.asarray(b, dtype=np.float64))))
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, d))))


def _swing(u: np.ndarray, v: np.ndarray) -> Tuple[float, float, float, float]:
    """Shortest-arc quaternion rotating direction ``u`` onto ``v``."""
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < 1e-12 or nv < 1e-12:
        return QID
    u = u / nu
    v = v / nv
    d = float(np.dot(u, v))
    if d > 1.0 - 1e-12:
        return QID
    if d < -1.0 + 1e-12:
        # 180 deg: any perpendicular axis; pick one deterministically
        axis = np.cross(u, np.array([1.0, 0.0, 0.0]))
        if float(np.linalg.norm(axis)) < 1e-6:
            axis = np.cross(u, np.array([0.0, 1.0, 0.0]))
        axis = axis / (float(np.linalg.norm(axis)) + 1e-12)
        return (float(axis[0]), float(axis[1]), float(axis[2]), 0.0)
    axis = np.cross(u, v)
    q = (float(axis[0]), float(axis[1]), float(axis[2]), 1.0 + d)
    return qnorm(q)


def _rot_vec(q, v) -> np.ndarray:
    x, y, z, w = q
    tx = 2.0 * (y * v[2] - z * v[1])
    ty = 2.0 * (z * v[0] - x * v[2])
    tz = 2.0 * (x * v[1] - y * v[0])
    return np.array([v[0] + w * tx + y * tz - z * ty,
                     v[1] + w * ty + z * tx - x * tz,
                     v[2] + w * tz + x * ty - y * tx], dtype=np.float64)


# ---------------------------------------------------------------- contact

def ground_estimate(world: Dict[str, np.ndarray], pct: float = DEFAULT_GROUND_PCT,
                    toes: Sequence[str] = ("toe_l", "toe_r")) -> Dict:
    """Floor estimate from both toes; ``max(0, ..)`` matches the scorer's y=0."""
    vals = [world[t][:, 1] for t in toes if t in world]
    if not vals:
        return {"raw": 0.0, "value": 0.0, "pct": float(pct), "available": False}
    cat = np.concatenate(vals)
    raw = float(np.percentile(cat, float(pct)))
    return {"raw": round(raw, 4), "value": max(0.0, raw),
            "min": round(float(cat.min()), 4),
            "pct": float(pct), "available": True}


def contact_probability(toe_y: np.ndarray, ground: float, height: float,
                        band: float = DEFAULT_BAND,
                        enter_frac: float = DEFAULT_ENTER_FRAC,
                        exit_frac: float = DEFAULT_EXIT_FRAC,
                        vspeed: float = DEFAULT_VSPEED,
                        min_len: int = DEFAULT_MIN_LEN,
                        fill_gap: int = DEFAULT_FILL_GAP,
                        p2d: Optional[np.ndarray] = None) -> Dict:
    """Soft floor-contact probability for one toe track.

    Height, vertical speed and (optional) 2D ankle confidence are multiplied;
    hysteresis then turns the candidate into planted intervals.  The returned
    ``p`` is 0 outside those intervals, so a correction can never be scheduled
    on a frame the detector does not believe.
    """
    y = np.asarray(toe_y, dtype=np.float64)
    n = len(y)
    b = float(band) * float(height)
    h = y - float(ground)
    vy = np.abs(np.diff(y, prepend=y[0]))
    vok = vy < float(vspeed)
    enter = (h < float(enter_frac) * b) & vok
    hold = (h < float(exit_frac) * b) & vok

    cand = np.zeros(n, dtype=bool)
    state = False
    for t in range(n):
        state = bool(hold[t]) if state else bool(enter[t])
        cand[t] = state
    # bridge short holes, drop short islands (contact flicker must not move legs)
    for (_, b0), (a1, _) in zip(_runs(cand), _runs(cand)[1:]):
        if (a1 - b0 - 1) <= int(fill_gap):
            cand[b0 + 1:a1] = True
    for a, e in _runs(cand):
        if (e - a + 1) < int(min_len):
            cand[a:e + 1] = False

    span = max(1e-9, (float(exit_frac) - float(enter_frac)) * b)
    p_h = np.clip((float(exit_frac) * b - h) / span, 0.0, 1.0)
    p_v = np.clip(1.0 - (vy - float(vspeed)) / max(1e-9, float(vspeed)), 0.0, 1.0)
    p = p_h * p_v
    if p2d is not None:
        c = np.asarray(p2d, dtype=np.float64)
        if len(c) < n:
            c = np.concatenate([c, np.repeat(c[-1:], n - len(c))])
        p = p * np.clip(c[:n], 0.0, 1.0)
    p = np.where(cand, p, 0.0)
    return {"mask": cand, "p": p, "h": h, "vy": vy,
            "intervals": _runs(cand), "band_units": b,
            "contact_frames": int(cand.sum()),
            "contact_fraction": round(float(cand.mean()), 4) if n else 0.0,
            "p_mean": round(float(p.mean()), 4) if n else 0.0}


def _ramp_weights(mask: np.ndarray, ramp: int) -> np.ndarray:
    """Mask with a linear attack/decay of ``ramp`` frames at every interval."""
    m = np.asarray(mask, dtype=np.float64).copy()
    if ramp <= 0:
        return m
    for a, b in _runs(np.asarray(mask, dtype=bool)):
        span = b - a + 1
        r = min(int(ramp), max(0, (span - 1) // 2))
        for k in range(r):
            w = (k + 1) / (r + 1)
            m[a + k] = min(m[a + k], w) if m[a + k] > 0 else 0.0
            m[b - k] = min(m[b - k], w) if m[b - k] > 0 else 0.0
    return m


def desired_drops(toe_y: np.ndarray, ground: float, height: float,
                  contact: Dict, band: float = DEFAULT_BAND,
                  target_frac: float = DEFAULT_TARGET_FRAC,
                  ramp: int = DEFAULT_RAMP) -> np.ndarray:
    """Downward correction per frame (>=0): only feet above the band move."""
    y = np.asarray(toe_y, dtype=np.float64)
    b = float(band) * float(height)
    top = float(ground) + b
    target = float(ground) + float(target_frac) * b
    need = np.clip((y - top) / (0.5 * b), 0.0, 1.0)
    d = need * np.maximum(0.0, y - target)
    w = _ramp_weights(contact["mask"], ramp)
    return d * w


def _smoothed_min(y: np.ndarray, win: int = 3) -> float:
    """Lowest point of a toe track, robust to a single-frame glitch."""
    return float(np.min(_moving_average(np.asarray(y, dtype=np.float64), win)))


def hover_estimate(world: Dict[str, np.ndarray], height: float,
                   band: float = DEFAULT_BAND,
                   target_frac: float = 0.0,
                   hover_min_frac: float = 0.15,
                   hover_max_frac: float = 1.5,
                   toes: Sequence[str] = ("toe_l", "toe_r")) -> Dict:
    """Per-side constant vertical bias from the shared floor.

    A foot that never comes within ``hover_min_frac * band`` of the floor is
    treated as genuinely lifted (not corrected); a foot whose lowest point is
    above the floor by up to ``hover_max_frac * band`` is hovering and gets the
    constant offset removed.  The floor itself is the lower of the two smoothed
    toe minima - the character has at least one foot on the ground somewhere.
    """
    mins = {}
    for t in toes:
        if t in world:
            mins[t] = _smoothed_min(world[t][:, 1])
    if not mins:
        return {"floor": 0.0, "sides": {}, "available": False}
    floor = max(0.0, min(mins.values()))
    b = float(band) * float(height)
    base = floor + float(target_frac) * b
    sides = {}
    for t, m in mins.items():
        hover = max(0.0, m - base)
        if hover < float(hover_min_frac) * b:
            used, why = 0.0, "already reaches the floor"
        elif hover > float(hover_max_frac) * b:
            # too high above the floor to be a placement bias: a real lift
            used, why = 0.0, "lifted above hover_max_frac * band"
        else:
            used, why = hover, "hovering"
        sides[t] = {"min": round(m, 4), "hover": round(hover, 4),
                    "used": round(used, 4), "reason": why}
    return {"floor": round(floor, 4), "band_units": round(b, 4),
            "base": round(base, 4), "sides": sides, "available": True}


# ---------------------------------------------------------------- leg IK

def _dist_for_flex(l1: float, l2: float, theta: float) -> float:
    """Hip-ankle distance for a knee flexion ``theta`` (0 = straight)."""
    c = l1 * l1 + l2 * l2 + 2.0 * l1 * l2 * math.cos(theta)
    return math.sqrt(max(1e-12, c))


def _flex_for_dist(l1: float, l2: float, dist: float) -> float:
    c = (dist * dist - l1 * l1 - l2 * l2) / (2.0 * l1 * l2 + 1e-12)
    return math.acos(max(-1.0, min(1.0, c)))


def _solve_vertical(H, K, A, drop: float, max_knee_deg: float,
                    max_thigh_deg: float) -> Optional[Dict]:
    """One frame, one leg: exact vertical IK with knee/thigh/reach caps.

    Returns the new knee position, the achieved drop and the rotation deltas, or
    ``None`` when the vertical ray cannot be reached at all.
    """
    if drop <= 0.0:
        return {"drop": 0.0, "K": K, "T": A, "knee_deg": 0.0, "thigh_deg": 0.0,
                "limit": None}
    d1 = K - H
    d2 = A - K
    l1 = float(np.linalg.norm(d1))
    l2 = float(np.linalg.norm(d2))
    if l1 < 1e-9 or l2 < 1e-9:
        return None
    u = A - H
    uy = float(u[1])
    r2 = float(u[0] ** 2 + u[2] ** 2)
    dist0 = float(np.linalg.norm(u))
    theta0 = math.acos(max(-1.0, min(1.0, float(np.dot(d1, d2)) / (l1 * l2))))

    # reach: the ankle cannot go past a straight leg
    R = (l1 + l2) * (1.0 - 1e-4)
    if r2 >= R * R:
        return None
    drop_reach = uy + math.sqrt(R * R - r2)
    # trust: knee flexion may not change by more than max_knee_deg
    cap = math.radians(float(max_knee_deg))
    dist_hi = _dist_for_flex(l1, l2, max(0.0, theta0 - cap))   # straighter
    dist_lo = _dist_for_flex(l1, l2, min(math.pi, theta0 + cap))
    lo = max(dist_lo, abs(l1 - l2) * (1.0 + 1e-4))
    hi = min(dist_hi, R)
    if hi < lo:
        hi = lo
    dist_des = math.sqrt(r2 + (uy - drop) ** 2)
    dist = min(max(dist_des, lo), hi)
    if dist * dist < r2:
        return None
    # uy is negative (the ankle hangs below the hip): (uy - d)^2 = dist^2 - r2
    # with uy - d < 0 gives d = uy + sqrt(dist^2 - r2).
    d_eff = uy + math.sqrt(dist * dist - r2)
    d_eff = max(0.0, min(d_eff, drop, drop_reach))
    if d_eff <= 1e-9:
        return {"drop": 0.0, "K": K, "T": A, "knee_deg": 0.0, "thigh_deg": 0.0,
                "limit": "knee" if drop_reach > 1e-9 else "reach"}

    def solve(dd: float):
        T = A - np.array([0.0, dd, 0.0])
        v = T - H
        dist = float(np.linalg.norm(v))
        uh = v / max(1e-12, dist)
        nrm = np.cross(d1, d2)
        nn = float(np.linalg.norm(nrm))
        if nn < 1e-9:
            nrm = np.cross(uh, np.array([0.0, 0.0, 1.0]))
            nn = float(np.linalg.norm(nrm))
            if nn < 1e-9:
                nrm = np.cross(uh, np.array([1.0, 0.0, 0.0]))
                nn = float(np.linalg.norm(nrm))
        nrm = nrm / max(1e-12, nn)
        perp = np.cross(nrm, uh)
        # keep the knee on its current side of the hip-ankle line
        if float(np.dot(d1 - np.dot(d1, uh) * uh, perp)) < 0.0:
            perp = -perp
        cos_a = (l1 * l1 + dist * dist - l2 * l2) / (2.0 * l1 * dist + 1e-12)
        a = math.acos(max(-1.0, min(1.0, cos_a)))
        K2 = H + l1 * (math.cos(a) * uh + math.sin(a) * perp)
        th = float(np.linalg.norm(np.cross(d1, K2 - H)))
        knee2 = math.acos(max(-1.0, min(1.0, float(np.dot(K2 - H, T - K2))
                                     / (l1 * l2 + 1e-12))))
        thigh_deg = math.degrees(math.asin(max(-1.0, min(1.0, th / max(1e-12, l1 * l1)))))
        return K2, T, math.degrees(abs(knee2 - theta0)), abs(thigh_deg)

    K2, T, knee_deg, thigh_deg = solve(d_eff)
    limit = None
    if knee_deg > float(max_knee_deg) + 1e-6:
        limit = "knee"
    elif abs(d_eff - drop) > 1e-6:
        limit = "reach" if d_eff >= drop_reach - 1e-6 else "knee"
    if thigh_deg > float(max_thigh_deg) + 1e-6:
        # thigh swing is monotone in the drop: bisect to the cap
        limit = "thigh"
        lo_d, hi_d = 0.0, d_eff
        for _ in range(24):
            mid = 0.5 * (lo_d + hi_d)
            _, _, _, tg = solve(mid)
            if tg > float(max_thigh_deg):
                hi_d = mid
            else:
                lo_d = mid
        d_eff = lo_d
        K2, T, knee_deg, thigh_deg = solve(d_eff)
    return {"drop": float(d_eff), "K": K2, "T": T, "knee_deg": float(knee_deg),
            "thigh_deg": float(thigh_deg), "limit": limit, "reached": d_eff >= drop - 1e-6}


def _apply_side(tracks: Dict[str, Dict[str, np.ndarray]], tr: Dict, side: str,
                drops: np.ndarray, max_knee_deg: float,
                max_thigh_deg: float) -> Dict:
    """Solve and write thigh/shin/foot local rotations for one leg."""
    smap = tr["_map"] or {}
    # joint keys (standard_map keys) index eval output; values are VMD bone names
    thigh, shin, foot = f"thigh_{side}", f"shin_{side}", f"foot_{side}"
    bone_thigh = smap.get(thigh)
    bone_shin = smap.get(shin)
    bone_foot = smap.get(foot)
    if not (bone_thigh in tracks and bone_shin in tracks and bone_foot in tracks):
        return {"applied": False, "reason": "leg bone tracks missing"}
    H = tr["world"][thigh]
    K = tr["world"][shin]
    A = tr["world"][foot]
    R_pel = tr["world_q"]["pelvis"]
    R_th = tr["world_q"][thigh]
    R_sh = tr["world_q"][shin]
    R_ft = tr["world_q"][foot]

    applied = np.zeros(len(A))
    knee_deg = np.zeros(len(A))
    thigh_deg = np.zeros(len(A))
    limits = {"knee": 0, "thigh": 0, "reach": 0, "unreachable": 0}
    for t in range(len(A)):
        if drops[t] <= 1e-9:
            continue
        res = _solve_vertical(H[t], K[t], A[t], float(drops[t]),
                              max_knee_deg, max_thigh_deg)
        if res is None:
            limits["unreachable"] += 1
            continue
        if res["drop"] <= 1e-9:
            if res.get("limit"):
                limits[res["limit"]] += 1
            continue
        lim = res.get("limit")
        if lim in limits:
            limits[lim] += 1
        K2, T = res["K"], res["T"]
        q_th = qmul(_swing(K[t] - H[t], K2 - H[t]), tuple(R_th[t]))
        q_sh = qmul(_swing(A[t] - K[t], T - K2), tuple(R_sh[t]))
        tracks[bone_thigh]["rot"][t] = qnorm(qmul(qinv(tuple(R_pel[t])), q_th))
        tracks[bone_shin]["rot"][t] = qnorm(qmul(qinv(q_th), q_sh))
        # keep the foot's world orientation: only its parent changed
        tracks[bone_foot]["rot"][t] = qnorm(qmul(qinv(q_sh), tuple(R_ft[t])))
        applied[t] = res["drop"]
        knee_deg[t] = res["knee_deg"]
        thigh_deg[t] = res["thigh_deg"]
    sel = applied > 1e-9
    return {"applied": bool(sel.any()), "side": side,
            "bone_map": {"thigh": bone_thigh, "shin": bone_shin,
                         "foot": bone_foot},
            "joints": {"thigh": thigh, "shin": shin, "foot": foot},
            "applied_drop": applied, "knee_deg": knee_deg,
            "thigh_deg": thigh_deg, "limits": limits,
            "frames": int(sel.sum()),
            "drop_max": round(float(applied.max()), 4) if sel.any() else 0.0,
            "drop_mean": round(float(applied[sel].mean()), 4) if sel.any() else 0.0,
            "knee_max_deg": round(float(knee_deg[sel].max()), 3) if sel.any() else 0.0,
            "thigh_max_deg": round(float(thigh_deg[sel].max()), 3) if sel.any() else 0.0}


# ---------------------------------------------------------------- verify

def _verify(model, profile, tracks, before: Dict, after: Dict,
            plan: Dict, tol: float = 2e-5) -> Dict:
    """GT-free self-checks: lengths, foot orientation, toe XZ, untouched frames."""
    out = {"ok": True, "failures": []}

    def fail(msg):
        out["ok"] = False
        out["failures"].append(msg)

    for side, rec in plan.items():
        if not rec.get("applied"):
            continue
        bm = rec["joints"]
        for a, b in ((bm["thigh"], bm["shin"]), (bm["shin"], bm["foot"])):
            la = np.linalg.norm(before["world"][a] - before["world"][b], axis=1)
            lb = np.linalg.norm(after["world"][a] - after["world"][b], axis=1)
            err = float(np.abs(la - lb).max())
            if err > tol:
                fail(f"bone length {a}->{b} changed {err:.2e}")
        qa = before["world_q"][bm["foot"]]
        qb = after["world_q"][bm["foot"]]
        dq = [ _q_angle_deg(x, y) for x, y in zip(qa, qb) ]
        if max(dq) > 0.05:
            fail(f"foot world orientation moved {max(dq):.4f} deg")
        toe = f"toe_{side}"
        da = after["world"][toe] - before["world"][toe]
        sel = rec["applied_drop"] > 1e-9
        if sel.any():
            xz = float(np.abs(da[sel][:, [0, 2]]).max())
            dy = float(np.abs(da[sel][:, 1] + rec["applied_drop"][sel]).max())
            if xz > tol:
                fail(f"toe {side} XZ moved {xz:.2e} during contact IK")
            if dy > tol:
                fail(f"toe {side} vertical drop mismatch {dy:.2e}")
        rest = ~sel
        if rest.any():
            d = float(np.abs(da[rest]).max())
            if d > tol:
                fail(f"non-contact frames of {side} moved {d:.2e}")
        rec["verify"] = {"xz_max": float(np.abs(da[sel][:, [0, 2]]).max()) if sel.any() else 0.0,
                         "dy_err": float(np.abs(da[sel][:, 1] + rec["applied_drop"][sel]).max()) if sel.any() else 0.0,
                         "foot_rot_max_deg": float(max(dq)) if dq else 0.0}
    out["failures"] = out["failures"][:12]
    return out


# ---------------------------------------------------------------- driver

def _frame_caps(H, K, A, max_knee_deg: float, max_thigh_deg: float) -> np.ndarray:
    """Largest trust-safe vertical drop for every frame of one leg."""
    caps = np.zeros(len(A))
    for t in range(len(A)):
        r = _solve_vertical(H[t], K[t], A[t], 1e6, max_knee_deg, max_thigh_deg)
        if r is not None:
            caps[t] = r["drop"]
    return caps


def solve_contact_ik(vmd: Dict, model: Dict, profile: Dict, *,
                     mode: str = "bias",
                     band: float = DEFAULT_BAND,
                     ground_pct: float = DEFAULT_GROUND_PCT,
                     target_frac: float = 0.0,
                     enter_frac: float = DEFAULT_ENTER_FRAC,
                     exit_frac: float = DEFAULT_EXIT_FRAC,
                     vspeed: float = DEFAULT_VSPEED,
                     max_knee_deg: float = DEFAULT_MAX_KNEE_DEG,
                     max_thigh_deg: float = DEFAULT_MAX_THIGH_DEG,
                     max_drop_frac: float = DEFAULT_MAX_DROP_FRAC,
                     min_len: int = DEFAULT_MIN_LEN,
                     fill_gap: int = DEFAULT_FILL_GAP,
                     ramp: int = DEFAULT_RAMP,
                     p_contact_min: float = DEFAULT_P_CONTACT_MIN,
                     trust_pct: float = 100.0,
                     gain: float = 1.0,
                     hover_min_frac: float = 0.15,
                     hover_max_frac: float = 1.5,
                     kp2d: Optional[np.ndarray] = None,
                     ground: Optional[float] = None,
                     sides: Sequence[str] = ("l", "r")) -> Dict:
    """Apply contact-gated vertical leg IK; returns ``{vmd, frames, report}``.

    ``mode='bias'`` (default) removes one constant per-side vertical offset: the
    foot's lowest point is put on the estimated floor, which is what the
    diagnostic shows the error to be (a 0.46 mean offset with 0.13 std on the
    front left).  ``mode='band'`` instead pulls only the frames above the band,
    which flattens the peaks and adds jerk - kept for the ablation.
    """
    frames, tracks = dense_tracks(vmd)
    height = float(profile.get("height") or 1.0)
    tr = eval_tracks(model, profile, tracks)
    tr["_map"] = dict(profile.get("standard_map") or {})
    g_raw = ground_estimate(tr["world"], ground_pct)
    hov = hover_estimate(tr["world"], height, band, target_frac,
                         hover_min_frac, hover_max_frac)
    # one floor for the band, the mask and the bias: the lowest smoothed toe
    # minimum of either foot (a glitch frame cannot move it).
    gval = float(hov["floor"] if ground is None else ground)

    report = {"frames": len(frames), "height": round(height, 4), "mode": mode,
              "params": {"band": band, "ground_pct": ground_pct,
                         "target_frac": target_frac, "enter_frac": enter_frac,
                         "exit_frac": exit_frac, "vspeed": vspeed,
                         "max_knee_deg": max_knee_deg,
                         "max_thigh_deg": max_thigh_deg,
                         "max_drop_frac": max_drop_frac,
                         "p_contact_min": p_contact_min, "ramp": ramp,
                         "trust_pct": trust_pct, "gain": gain,
                         "hover_min_frac": hover_min_frac,
                         "hover_max_frac": hover_max_frac},
              "ground": {"smoothed_min": round(gval, 4),
                         "pct_min": g_raw.get("value"), "raw_min": g_raw.get("min"),
                         "override": ground is not None},
              "hover": hov,
              "band_units": round(band * height, 4),
              "sides": {}, "applied": False}
    plan = {}
    for side in sides:
        toe = f"toe_{side}"
        if toe not in tr["world"]:
            continue
        p2d = None
        if kp2d is not None:
            idx = _ANKLE_KP.get(side)
            arr = np.asarray(kp2d)
            if arr.ndim == 3 and arr.shape[1] > idx:
                conf = arr[:, idx, 2].astype(np.float64)
                # 0.2 is "invisible", 0.5+ is a usable ankle; linear between
                p2d = np.clip((conf - 0.2) / 0.3, 0.0, 1.0)
        y = tr["world"][toe][:, 1]
        side_hov = (hov.get("sides") or {}).get(toe) or {}
        y_ref = y
        const = 0.0
        if mode == "bias":
            const = float(side_hov.get("used") or 0.0) * float(gain)
            y_ref = y - const          # contact is judged on the intended track
        c = contact_probability(y_ref, gval, height, band,
                                enter_frac, exit_frac, vspeed, min_len,
                                fill_gap, p2d)
        gate = c["p"] >= float(p_contact_min)
        if mode == "bias":
            d_des = np.where(gate, const, 0.0) * _ramp_weights(c["mask"], ramp)
        else:
            d_des = desired_drops(y, gval, height, c, band, target_frac, ramp)
            d_des = np.where(gate, d_des, 0.0)
        d_des = np.minimum(d_des, float(max_drop_frac) * height)
        if mode == "bias" and const > 1e-9 and c["mask"].any():
            sel = c["mask"]
            cap = _frame_caps(tr["world"][f"thigh_{side}"],
                              tr["world"][f"shin_{side}"],
                              tr["world"][f"foot_{side}"],
                              max_knee_deg, max_thigh_deg)
            allow = float(np.percentile(cap[sel], float(trust_pct)))
            if allow < const:
                # keep one constant inside the trust region; report the shrink
                d_des = d_des * (allow / const)
                report.setdefault("trust_shrink", {})[side] = {
                    "hover": round(const, 4), "allowed": round(allow, 4),
                    "frame_cap_min": round(float(cap[sel].min()), 4),
                    "frame_cap_median": round(float(np.median(cap[sel])), 4)}
                const = allow
        rec = _apply_side(tracks, tr, side, d_des, max_knee_deg, max_thigh_deg)
        rec["constant_drop"] = round(float(const), 4)
        rec["hover"] = side_hov
        rec["contact"] = {"frames": c["contact_frames"],
                          "fraction": c["contact_fraction"],
                          "p_mean": c["p_mean"],
                          "intervals": [[int(a), int(b)] for a, b in c["intervals"]],
                          "enter_units": round(gval + enter_frac * band * height, 4),
                          "exit_units": round(gval + exit_frac * band * height, 4),
                          "band_top_units": round(gval + band * height, 4),
                          "target_units": round(gval + target_frac * band * height, 4)}
        rec["desired"] = {"frames": int((d_des > 1e-9).sum()),
                          "max": round(float(d_des.max()), 4),
                          "mean": round(float(d_des[d_des > 1e-9].mean()), 4)
                          if (d_des > 1e-9).any() else 0.0}
        plan[side] = rec
    report["applied"] = any(r.get("applied") for r in plan.values())

    # evaluate the result and build a JSON-safe report
    tr_clean = eval_tracks(model, profile, tracks)
    verify = _verify(model, profile, tracks, tr, tr_clean, plan)
    sides_report = {}
    for side, rec in plan.items():
        if not rec.get("applied"):
            sides_report[side] = {"applied": False,
                                  "contact": rec.get("contact"),
                                  "reason": rec.get("reason")}
            continue
        before_y = tr["world"][f"toe_{side}"][:, 1]
        after_y = tr_clean["world"][f"toe_{side}"][:, 1]
        sides_report[side] = {
            "applied": True,
            "constant_drop": rec.get("constant_drop"),
            "hover": rec.get("hover"),
            "contact": rec.get("contact"),
            "frames": rec.get("frames"),
            "drop_max": rec.get("drop_max"),
            "drop_mean": rec.get("drop_mean"),
            "knee_max_deg": rec.get("knee_max_deg"),
            "thigh_max_deg": rec.get("thigh_max_deg"),
            "limits": rec.get("limits"),
            "desired": rec.get("desired"),
            "toe_y": {"before_min": round(float(before_y.min()), 4),
                      "before_max": round(float(before_y.max()), 4),
                      "after_min": round(float(after_y.min()), 4),
                      "after_max": round(float(after_y.max()), 4),
                      "mean_abs_change": round(float(np.abs(after_y - before_y).mean()), 4)},
            "verify": rec.get("verify"),
        }
    report["sides"] = sides_report
    report["verify"] = verify
    return {"vmd": tracks_to_vmd(vmd, frames, tracks),
            "frames": frames, "report": report}
