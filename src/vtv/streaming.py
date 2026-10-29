# -*- coding: utf-8 -*-
"""M2.5 v0: streaming state + gap completion (no learned model).

Why this module exists
----------------------
``cleanup`` removes isolated spikes and ``contact_ik`` fixes the stance, but
neither has a notion of "this frame was not observed".  A real video pipeline
loses frames (detection failure, occlusion, low 2D confidence) and the adapter
then either holds the last pose or extrapolates without a bound.  M2.5 asks for
a causal state that keeps the motion going through those holes, blends back to
the observation without a jump, and can be refined offline.

State (per bone)
----------------
``s_t = (pose, velocity, contact, phase, uncertainty, confidence)``

* pose        local quaternion + local translation (VMD space)
* velocity    body angular velocity (rad/frame) and root linear velocity
* contact     per-side stance state carried from the last observed frame
* phase       contact age / gait phase estimate (never invents a beat)
* uncertainty deterministic growth during rollout, in degrees
* confidence  external per-frame (optionally per-bone) evidence in [0, 1]

The v0 policy is deliberately classical:

1. high confidence  -> adopt the observation exactly;
2. mid confidence   -> blend the AR prediction with the observation;
3. low confidence   -> bounded constant-velocity rollout with damping;
4. gap shorter than the lookahead (or offline with both anchors) -> SQUAD
   rotation bridge + cubic Hermite root bridge, so both boundaries match the
   observed pose *and* velocity;
5. longer gaps in streaming mode -> causal rollout, then a lookahead bridge
   over the last ``lookahead`` frames so the return to the observation is
   smooth while only low-confidence frames are edited;
6. offline refinement -> re-plan the whole gap from both ends and apply a
   small residual smoother inside the gap; observed frames are never changed.

Contact is *continued*, not re-invented: a stance that was observed just before
the gap is kept for a bounded time (``contact_hold``).  When both feet are
planted and the root is slow, the rollout damps root XZ and leg velocity so the
feet do not skate; when the character was travelling, it keeps the observed
velocity and reports the stance risk instead of freezing the clip.

This module never reads GT.  ``scripts/m25_dropout_bench.py`` is the only place
that scores the result against the synthetic GT.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .cleanup import (contact_intervals, dense_tracks, eval_tracks, _runs,
                      tracks_to_vmd)
from .mmd.mathutil import QID, qinv, qmul, qnorm, qslerp

FPS = 30.0


# ---------------------------------------------------------------- config

@dataclass
class StreamingConfig:
    """All thresholds in one place; every value is documented at M2.5."""

    fps: float = FPS
    # Confidence gate: >= high is an observation, <= low is a hole.
    high: float = 0.75
    low: float = 0.35
    # Latency (seconds) used by the streaming planner.  0.3-0.5 s is the
    # M2.5 range; offline mode ignores it and uses both gap ends.
    lookahead_s: float = 0.4
    # Bounded rollout: after this much missing time the velocity is damped
    # twice as fast instead of being extrapolated forever.
    max_rollout_s: float = 1.5
    velocity_half_life_s: float = 0.45
    capped_velocity_half_life_s: float = 0.18
    # Safety caps for the AR state.
    max_omega_deg_per_frame: float = 20.0
    max_root_speed: float = 0.8          # PMX units/frame (~6 cm/frame)
    root_y_margin: float = 0.8           # keep rollout inside observed Y range
    # Uncertainty growth (degrees) while a bone is not observed.
    sigma_rate_deg_per_frame: float = 1.0
    sigma_max_deg: float = 90.0
    # Contact continuation.
    contact_band: float = 0.02
    contact_horizontal_speed: float = 0.12
    contact_vertical_speed: float = 0.08
    contact_hold_s: float = 1.0
    contact_root_speed: float = 0.06     # units/frame; below this = standing
    contact_root_damping: float = 0.12   # keep 12% of the predicted XZ step
    contact_leg_damping: float = 0.15    # keep 15% of the leg angular step
    # Boundary bridge: 1.0 = full SQUAD/Hermite velocity matching, 0 = pure
    # slerp/linear.  The synthetic GT sweep showed 0.25 keeps the endpoint
    # tangent while avoiding the overshoot that a full tangent produces on a
    # noisy prediction or a long gap.
    bridge_velocity_scale: float = 0.25
    # How much of the bridge (vs. holding the last observation) to keep in
    # the middle of a visible gap.  0.25 was the best trade-off on R2:
    # it keeps the clip moving (no frozen frames), beats hold on gap error,
    # and adds less foot slide than a full bridge.  The last lookahead frames
    # always ramp to the bridge/observation so the return has no jump.
    bridge_blend: float = 0.25
    # A one-frame hole is not a visible freeze.  Holding the last observed
    # frame is the lowest-error fill there; longer holes are bridged.
    short_gap_hold: int = 1
    # Offline refinement.
    smooth_gaps: bool = True
    smooth_passes: int = 1
    # A gap longer than this is not bridged end-to-end even offline; the
    # causal rollout is kept and only the last lookahead is bridged.
    max_bridge_s: float = 4.0
    # M2.5 v0.1: anchor observed stance feet inside long holes (opt-in).
    # The anchor IK runs after the v0 fill and before the observed writeback,
    # touches only low-confidence thigh/shin/foot frames, and never moves the
    # root.  See ``stance_ik.py`` for the full policy.
    anchor_stance: bool = False

    def lookahead_frames(self) -> int:
        return max(1, int(round(float(self.lookahead_s) * float(self.fps))))

    def max_rollout_frames(self) -> int:
        return max(1, int(round(float(self.max_rollout_s) * float(self.fps))))

    def max_bridge_frames(self) -> int:
        return max(1, int(round(float(self.max_bridge_s) * float(self.fps))))

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["lookahead_frames"] = self.lookahead_frames()
        d["max_rollout_frames"] = self.max_rollout_frames()
        d["max_bridge_frames"] = self.max_bridge_frames()
        return d


# ---------------------------------------------------------------- quaternions

def _q_conj(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def _q_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz],
                    dtype=np.float64)


def _q_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array(QID, dtype=np.float64)
    return np.asarray(q, dtype=np.float64) / n


def _q_log(q: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle) of a unit quaternion, shortest path."""
    q = _q_normalize(q)
    if q[3] < 0.0:
        q = -q
    w = float(np.clip(q[3], -1.0, 1.0))
    v = q[:3]
    vn = float(np.linalg.norm(v))
    if vn < 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(vn, w)
    return v * (angle / vn)


def _q_exp(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    angle = float(np.linalg.norm(r))
    if angle < 1e-12:
        return _q_normalize(np.array([0.5 * r[0], 0.5 * r[1], 0.5 * r[2], 1.0]))
    half = 0.5 * angle
    s = math.sin(half) / angle
    return np.array([r[0] * s, r[1] * s, r[2] * s, math.cos(half)],
                    dtype=np.float64)


def _q_slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    a = _q_normalize(a)
    b = _q_normalize(b)
    d = float(np.dot(a, b))
    if d < 0.0:
        b = -b
        d = -d
    if d > 0.9995:
        return _q_normalize((1.0 - float(t)) * a + float(t) * b)
    th = math.acos(max(-1.0, min(1.0, d)))
    s = math.sin(th)
    wa = math.sin((1.0 - float(t)) * th) / s
    wb = math.sin(float(t) * th) / s
    return _q_normalize(wa * a + wb * b)


def _squad(q1: np.ndarray, q2: np.ndarray, q0: np.ndarray, q3: np.ndarray,
           t: float, alpha: float = 1.0) -> np.ndarray:
    """Spherical quadrangle: C1 interpolation using neighbour poses.

    ``alpha`` scales the tangent control points: 1 is the textbook SQUAD,
    0 reduces to slerp.  Damping to 0.25 avoids the overshoot that full
    endpoint tangents produce on noisy prediction velocities.
    """
    q1 = _q_normalize(q1)
    q2 = _q_normalize(q2)
    q0 = _q_normalize(q0) if q0 is not None else q1
    q3 = _q_normalize(q3) if q3 is not None else q2
    a = _q_mul(_q_conj(q1), q2)
    b = _q_mul(_q_conj(q1), q0)
    s1 = _q_mul(q1, _q_exp(-float(alpha) * (_q_log(a) + _q_log(b)) / 4.0))
    c = _q_mul(_q_conj(q2), q1)
    d = _q_mul(_q_conj(q2), q3)
    s2 = _q_mul(q2, _q_exp(-float(alpha) * (_q_log(c) + _q_log(d)) / 4.0))
    return _q_slerp(_q_slerp(q1, q2, t), _q_slerp(s1, s2, t),
                    2.0 * t * (1.0 - t))


def _q_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    d = abs(float(np.dot(_q_normalize(a), _q_normalize(b))))
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, d))))


def _hemisphere_rows(q: np.ndarray) -> np.ndarray:
    out = np.asarray(q, dtype=np.float64).copy()
    for i in range(1, len(out)):
        if float(np.dot(out[i], out[i - 1])) < 0.0:
            out[i] = -out[i]
    return out


def _clamp_vec(v: np.ndarray, limit: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n > limit > 0.0:
        return v * (limit / n)
    return v


def _smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def _hermite(p0: np.ndarray, p1: np.ndarray, v0: np.ndarray, v1: np.ndarray,
             t: float) -> np.ndarray:
    t = float(t)
    t2, t3 = t * t, t * t * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return h00 * p0 + h10 * v0 + h01 * p1 + h11 * v1


# ---------------------------------------------------------------- confidence

def normalize_confidence(confidence, tracks: Dict, n_frames: int
                         ) -> Dict[str, np.ndarray]:
    """Return ``{bone: (F,) float array}`` clipped to [0, 1].

    Accepts ``None`` (all observed), a ``(F,)`` or ``(F, J)`` array, or a dict.
    A NaN in the input is treated as "not observed" instead of propagating.
    """
    names = sorted(tracks)
    if confidence is None:
        ones = np.ones(n_frames, dtype=np.float64)
        return {n: ones.copy() for n in names}
    if isinstance(confidence, dict):
        out = {}
        for n in names:
            v = confidence.get(n)
            if v is None:
                out[n] = np.ones(n_frames, dtype=np.float64)
                continue
            v = np.asarray(v, dtype=np.float64).reshape(-1)
            if len(v) != n_frames:
                raise ValueError(f"confidence[{n}] has {len(v)} frames, "
                                 f"expected {n_frames}")
            out[n] = np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=0.0).clip(0, 1)
        return out
    arr = np.asarray(confidence, dtype=np.float64)
    if arr.ndim == 1:
        if len(arr) != n_frames:
            raise ValueError(f"confidence has {len(arr)} frames, "
                             f"expected {n_frames}")
        v = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0).clip(0, 1)
        return {n: v.copy() for n in names}
    if arr.ndim == 2:
        if arr.shape[0] != n_frames:
            raise ValueError(f"confidence has {arr.shape[0]} frames, "
                             f"expected {n_frames}")
        if arr.shape[1] != len(names):
            raise ValueError(f"confidence has {arr.shape[1]} columns, "
                             f"expected {len(names)} bones (sorted)")
        v = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0).clip(0, 1)
        return {n: v[:, j].copy() for j, n in enumerate(names)}
    raise ValueError("confidence must be None, dict, (F,) or (F,J)")


def heuristic_confidence(tracks: Dict[str, Dict[str, np.ndarray]],
                         spike_deg: float = 18.0,
                         accel_deg: float = 12.0) -> np.ndarray:
    """GT-free per-frame confidence from observation validity/smoothness.

    This is a *fallback* for a clip without an external 2D confidence track.
    It does not try to detect occlusion; it only down-weights frames whose
    local motion is a single-frame outlier (the adapter's failure mode).  The
    dropout benchmark passes an explicit mask instead.
    """
    names = sorted(tracks)
    if not names:
        return np.ones(0, dtype=np.float64)
    n = len(tracks[names[0]]["rot"])
    bad = np.zeros(n, dtype=np.float64)
    for name in names:
        q = np.asarray(tracks[name]["rot"], dtype=np.float64)
        if len(q) < 3:
            continue
        for t in range(1, len(q) - 1):
            mid = _q_slerp(q[t - 1], q[t + 1], 0.5)
            dev = _q_angle_deg(q[t], mid)
            step = max(_q_angle_deg(q[t - 1], q[t]),
                       _q_angle_deg(q[t], q[t + 1]))
            if dev > spike_deg and step > accel_deg:
                bad[t] = max(bad[t], min(1.0, (dev - spike_deg) / spike_deg))
    return np.clip(1.0 - bad, 0.0, 1.0)


# ---------------------------------------------------------------- contact

def _carry_contact(observed: np.ndarray, observed_conf: np.ndarray,
                   high: float, hold_frames: int) -> np.ndarray:
    """Carry a stance through unobserved frames, with a bounded hold."""
    out = np.zeros(len(observed), dtype=bool)
    state = False
    age = 0
    for t in range(len(observed)):
        if observed_conf[t] >= high:
            state = bool(observed[t])
            age = 0
        elif state:
            age += 1
            if age > hold_frames:
                state = False
        out[t] = state
    return out


def estimate_contact(model: Dict, profile: Dict, tracks: Dict,
                     conf: Dict[str, np.ndarray], cfg: StreamingConfig
                     ) -> Optional[Dict]:
    """Detect stance from the *observed* motion and carry it through holes.

    Contact is computed from the PMX toe trajectory (same rule as cleanup),
    then only accepted on high-confidence frames.  During a hole the last
    stance is continued for ``contact_hold_s``; after that it is released
    rather than inventing a new footfall.
    """
    if not model or not profile or not tracks:
        return None
    try:
        tr = eval_tracks(model, profile, tracks)
    except Exception:
        return None
    height = float(tr.get("height") or profile.get("height") or 1.0)
    frames = tr["frames"]
    n = len(frames)
    names = sorted(tracks)
    cmean = np.mean(np.stack([conf[nm] for nm in names], axis=0), axis=0)
    hold = int(round(cfg.contact_hold_s * cfg.fps))
    out: Dict[str, Dict] = {}
    for side in ("l", "r"):
        toe = tr["world"].get(f"toe_{side}")
        if toe is None:
            continue
        c = contact_intervals(
            toe, height, margin=cfg.contact_band,
            speed_thr=cfg.contact_horizontal_speed,
            vspeed_thr=cfg.contact_vertical_speed)
        observed = c["mask"] & (cmean >= cfg.high)
        carried = _carry_contact(observed, cmean, cfg.high, hold)
        out[side] = {
            "observed": observed,
            "carried": carried,
            "thr": float(c["thr"]),
            "observed_fraction": float(observed.mean()),
            "carried_fraction": float(carried.mean()),
            "intervals": [[int(a), int(b)] for a, b in c["intervals"]],
        }
    if not out:
        return None
    both = np.ones(n, dtype=bool)
    for side in out:
        both &= out[side]["carried"]
    return {"sides": out, "both": both, "height": height, "frames": n}


# ---------------------------------------------------------------- bone roles

def _bone_roles(profile: Optional[Dict], tracks: Dict
                ) -> Dict[str, Tuple[str, Optional[str]]]:
    """Map standard joints to (role, side) for contact damping."""
    roles: Dict[str, Tuple[str, Optional[str]]] = {}
    if not profile:
        return roles
    smap = profile.get("standard_map") or {}
    root = smap.get("root")
    if root in tracks:
        roles[root] = ("root", None)
    for side in ("l", "r"):
        for j in ("thigh", "shin", "foot", "toe"):
            bn = smap.get(f"{j}_{side}")
            if bn in tracks:
                roles[bn] = ("leg", side)
    return roles


# ---------------------------------------------------------------- causal pass

def _causal_bone(q_obs: np.ndarray, p_obs: np.ndarray, c: np.ndarray,
                 cfg: StreamingConfig, role: str = "", side: Optional[str] = None,
                 contact: Optional[Dict] = None, y_range: Optional[Tuple[float, float]] = None
                 ) -> Dict:
    """Causal state filter for one bone.

    Returns the filtered tracks plus per-frame velocity/uncertainty/diagnostics.
    """
    n = len(c)
    q_out = np.empty_like(q_obs)
    p_out = np.empty_like(p_obs)
    omega_hist = np.zeros((n, 3), dtype=np.float64)
    vel_hist = np.zeros((n, 3), dtype=np.float64)
    sigma = np.zeros(n, dtype=np.float64)
    low_age = np.zeros(n, dtype=np.int32)
    capped = 0
    damped = 0

    omega = np.zeros(3)
    vel = np.zeros(3)
    last_out_q: Optional[np.ndarray] = None
    last_out_p: Optional[np.ndarray] = None
    last_obs_q: Optional[np.ndarray] = None
    last_obs_p: Optional[np.ndarray] = None
    last_obs_t = -10
    sig = 0.0
    age = 0
    decay = math.exp(-1.0 / max(1e-6, cfg.velocity_half_life_s * cfg.fps))
    cap_decay = math.exp(-1.0 / max(1e-6,
                                   cfg.capped_velocity_half_life_s * cfg.fps))
    max_omega = math.radians(cfg.max_omega_deg_per_frame)

    for t in range(n):
        ct = float(c[t])
        if last_out_q is None:
            q_out[t] = _q_normalize(q_obs[t])
            p_out[t] = p_obs[t]
            last_out_q = q_out[t]
            last_out_p = p_out[t]
            if ct >= cfg.high:
                last_obs_q = q_obs[t].copy()
                last_obs_p = p_obs[t].copy()
                last_obs_t = t
            omega_hist[t] = omega
            vel_hist[t] = vel
            sigma[t] = sig
            continue

        # ---- predict (constant velocity in the local tangent space) ----
        q_pred = _q_mul(last_out_q, _q_exp(_clamp_vec(omega, max_omega)))
        p_pred = last_out_p + _clamp_vec(vel, cfg.max_root_speed)

        if ct >= cfg.high:
            # observation: adopt exactly, reset uncertainty
            q_out[t] = _q_normalize(q_obs[t])
            p_out[t] = p_obs[t]
            if last_obs_q is not None:
                dt = max(1, t - last_obs_t)
                omega = _clamp_vec(
                    _q_log(_q_mul(_q_conj(last_obs_q), q_out[t])) / dt,
                    max_omega)
            else:
                omega = _clamp_vec(
                    _q_log(_q_mul(_q_conj(last_out_q), q_out[t])), max_omega)
            if last_obs_p is not None:
                dt = max(1, t - last_obs_t)
                vel = (p_out[t] - last_obs_p) / dt
            else:
                vel = p_out[t] - last_out_p
            vel = _clamp_vec(vel, cfg.max_root_speed)
            sig = 0.0
            age = 0
            last_obs_q = q_out[t].copy()
            last_obs_p = p_out[t].copy()
            last_obs_t = t
        elif ct >= cfg.low:
            # mid confidence: smooth blend, velocity from the observation
            w = _smoothstep((ct - cfg.low) / max(1e-6, cfg.high - cfg.low))
            q_out[t] = _q_slerp(q_pred, q_obs[t], w)
            p_out[t] = (1.0 - w) * p_pred + w * p_obs[t]
            if last_obs_q is not None:
                dt = max(1, t - last_obs_t)
                omega_obs = _q_log(_q_mul(_q_conj(last_obs_q), q_obs[t])) / dt
            else:
                omega_obs = omega
            if last_obs_p is not None:
                dt = max(1, t - last_obs_t)
                vel_obs = (p_obs[t] - last_obs_p) / dt
            else:
                vel_obs = vel
            omega = _clamp_vec((1.0 - w) * omega + w * omega_obs, max_omega)
            vel = _clamp_vec((1.0 - w) * vel + w * vel_obs, cfg.max_root_speed)
            sig = (1.0 - w) * (sig + cfg.sigma_rate_deg_per_frame)
            age = 0
        else:
            # low confidence: bounded rollout
            q_out[t] = q_pred
            p_out[t] = p_pred
            age += 1
            low_age[t] = age
            omega *= decay
            vel *= decay
            if age > cfg.max_rollout_frames():
                omega *= cap_decay
                vel *= cap_decay
                sig = min(cfg.sigma_max_deg,
                          sig + cfg.sigma_rate_deg_per_frame * 1.5)
                capped += 1
            else:
                sig = min(cfg.sigma_max_deg,
                          sig + cfg.sigma_rate_deg_per_frame)
            # contact: a planted stance must not skate
            if contact is not None:
                both = bool(contact["both"][t]) if t < len(contact["both"]) else False
                side_contact = False
                if side is not None and side in contact.get("sides", {}):
                    side_contact = bool(contact["sides"][side]["carried"][t])
                if role == "root" and both:
                    step = p_out[t] - last_out_p
                    damp = cfg.contact_root_damping
                    p_out[t, 0] = last_out_p[0] + step[0] * damp
                    p_out[t, 2] = last_out_p[2] + step[2] * damp
                    vel = vel * damp
                    damped += 1
                elif role == "leg" and side_contact:
                    # Keep the stance leg close to its last observed pose even
                    # when the other foot is swinging; otherwise the toe can
                    # drift out of the contact band during a one-foot gap.
                    q_out[t] = _q_slerp(q_out[t], last_out_q,
                                      1.0 - cfg.contact_leg_damping)
                    omega = omega * cfg.contact_leg_damping
                    damped += 1
            if role == "root" and y_range is not None:
                lo, hi = y_range
                p_out[t, 1] = min(hi + cfg.root_y_margin,
                                  max(lo - cfg.root_y_margin, p_out[t, 1]))
        omega_hist[t] = omega
        vel_hist[t] = vel
        sigma[t] = sig
        last_out_q = q_out[t]
        last_out_p = p_out[t]
    return {"rot": q_out, "pos": p_out, "omega": omega_hist, "vel": vel_hist,
            "sigma": sigma, "low_age": low_age, "capped_frames": int(capped),
            "contact_damped_frames": int(damped)}


# ---------------------------------------------------------------- bridge

def _obs_y_range(q_obs: np.ndarray, p_obs: np.ndarray, c: np.ndarray,
                 high: float) -> Optional[Tuple[float, float]]:
    ok = c >= high
    if not ok.any():
        return None
    y = p_obs[ok, 1]
    return float(y.min()), float(y.max())


def _bridge_gap(q_in: np.ndarray, p_in: np.ndarray, q_fill: np.ndarray,
                p_fill: np.ndarray, c: np.ndarray, s: int, e: int,
                cfg: StreamingConfig, mode: str) -> Dict:
    """Fill one gap [s, e] using a SQUAD rotation and Hermite root bridge.

    ``q_fill/p_fill`` are the causal predictions (used as the left state when
    the gap is longer than the lookahead).  Observed frames are never touched.
    """
    n = len(c)
    b = e + 1
    left_ok = s - 1 >= 0 and c[s - 1] >= cfg.low
    right_ok = b < n and c[b] >= cfg.low
    look = cfg.lookahead_frames()
    gap_len = e - s + 1
    info = {"start": int(s), "end": int(e), "length": int(gap_len),
            "left_anchor": bool(left_ok), "right_anchor": bool(right_ok),
            "planned": "rollout"}
    if not right_ok:
        info["planned"] = "rollout_no_right_anchor"
        return info
    if not left_ok:
        # Leading hole: bridge from the input's first value to the first
        # observation.  q_in[s] is all we have on the left.
        lframe = s
        q1, p1 = q_in[s], p_in[s]
        q0 = q_in[s]
        v1 = np.zeros(3)
    elif mode == "offline" and gap_len <= cfg.max_bridge_frames():
        lframe = s - 1
        q1, p1 = q_in[lframe], p_in[lframe]
        q0 = q_in[lframe - 1] if lframe - 1 >= 0 else q1
        v1 = p1 - p_in[lframe - 1] if lframe - 1 >= 0 else np.zeros(3)
    elif mode == "offline":
        # Too long to bridge end-to-end; keep the causal rollout and blend the
        # last lookahead into the observation.
        split = max(s, e - look + 1)
        lframe = split - 1
        q1, p1 = q_fill[lframe], p_fill[lframe]
        q0 = q_fill[lframe - 1] if lframe - 1 >= 0 else q1
        v1 = p1 - p_fill[lframe - 1] if lframe - 1 >= 0 else np.zeros(3)
        info["planned"] = "rollout_then_bridge"
    else:
        # Streaming: a gap that fits in the lookahead can be bridged from the
        # last observation; a longer one rolls out and bridges its last
        # ``look`` frames from the causal state.
        if gap_len <= look:
            lframe = s - 1
            q1, p1 = q_in[lframe], p_in[lframe]
            q0 = q_in[lframe - 1] if lframe - 1 >= 0 else q1
            v1 = p1 - p_in[lframe - 1] if lframe - 1 >= 0 else np.zeros(3)
        else:
            split = max(s, e - look + 1)
            lframe = split - 1
            q1, p1 = q_fill[lframe], p_fill[lframe]
            q0 = q_fill[lframe - 1] if lframe - 1 >= 0 else q1
            v1 = p1 - p_fill[lframe - 1] if lframe - 1 >= 0 else np.zeros(3)
            info["planned"] = "rollout_then_bridge"
    q2, p2 = q_in[b], p_in[b]
    q3 = q_in[b + 1] if b + 1 < n and c[b + 1] >= cfg.low else q2
    if b + 1 < n and c[b + 1] >= cfg.low:
        v2 = p_in[b + 1] - p2
    elif b - 1 >= 0:
        v2 = p2 - p_in[b - 1]
    else:
        v2 = np.zeros(3)
    span = max(1, b - lframe)
    alpha = float(cfg.bridge_velocity_scale)
    d0 = alpha * v1 * span
    d1 = alpha * v2 * span
    for t in range(max(s, lframe + 1), e + 1):
        tau = (t - lframe) / span
        q_fill[t] = _squad(q1, q2, q0, q3, tau, alpha=alpha)
        p_fill[t] = _hermite(p1, p2, d0, d1, tau)
    info["planned"] = ("squad" if info["planned"] == "rollout"
                       else info["planned"])
    info["left_frame"] = int(lframe)
    info["span"] = int(span)
    return info


def _smooth_gap(q_fill: np.ndarray, p_fill: np.ndarray,
                s: int, e: int, passes: int = 1) -> None:
    """Smooth the inside of a gap; the anchors (s-1, e+1) stay fixed.

    A three-tap slerp/position average is deliberately weak: it removes the
    small kinks a SQUAD/Hermite bridge can leave on a fast clip without
    flattening an accent.  Only gap frames are written.
    """
    if e - s + 1 < 3:
        return
    n = len(q_fill)
    for _ in range(max(0, int(passes))):
        new_q = q_fill.copy()
        new_p = p_fill.copy()
        for t in range(s, e + 1):
            prev = max(0, t - 1)
            nxt = min(n - 1, t + 1)
            mid = _q_slerp(q_fill[prev], q_fill[nxt], 0.5)
            new_q[t] = _q_slerp(q_fill[t], mid, 0.35)
            new_p[t] = (0.25 * p_fill[prev] + 0.5 * p_fill[t]
                        + 0.25 * p_fill[nxt])
        q_fill[:] = new_q
        p_fill[:] = new_p


# ---------------------------------------------------------------- main fill

def fill_tracks(tracks: Dict[str, Dict[str, np.ndarray]],
                conf: Dict[str, np.ndarray],
                cfg: Optional[StreamingConfig] = None,
                mode: str = "offline",
                contact: Optional[Dict] = None,
                profile: Optional[Dict] = None) -> Dict:
    """Fill holes in dense tracks.  Returns new tracks + a report."""
    cfg = cfg or StreamingConfig()
    if mode not in ("stream", "offline"):
        raise ValueError("mode must be 'stream' or 'offline'")
    names = sorted(tracks)
    if not names:
        return {"tracks": {}, "report": {"frames": 0, "bones": 0}}
    n = len(tracks[names[0]]["rot"])
    for name in names:
        if len(tracks[name]["rot"]) != n:
            raise ValueError(f"track {name} length {len(tracks[name]['rot'])} "
                             f"!= {n}")
        if name not in conf:
            conf[name] = np.ones(n, dtype=np.float64)
    roles = _bone_roles(profile, tracks)
    out = {}
    gap_rows: List[Dict] = []
    total_capped = 0
    total_damped = 0
    sig_max = 0.0
    gating = {"high": 0, "mid": 0, "low": 0}
    for name in names:
        q_obs = np.asarray(tracks[name]["rot"], dtype=np.float64)
        p_obs = np.asarray(tracks[name]["pos"], dtype=np.float64)
        c = conf[name]
        gating["high"] += int((c >= cfg.high).sum())
        gating["mid"] += int(((c >= cfg.low) & (c < cfg.high)).sum())
        gating["low"] += int((c < cfg.low).sum())
        role, side = roles.get(name, ("", None))
        y_range = _obs_y_range(q_obs, p_obs, c, cfg.high) if role == "root" else None
        res = _causal_bone(q_obs, p_obs, c, cfg, role=role, side=side,
                           contact=contact, y_range=y_range)
        q_fill = res["rot"].copy()
        p_fill = res["pos"].copy()
        total_capped += res["capped_frames"]
        total_damped += res["contact_damped_frames"]
        sig_max = max(sig_max, float(res["sigma"].max()) if n else 0.0)

        gaps = _runs(c < cfg.low)
        for s, e in gaps:
            gap_len = e - s + 1
            if (gap_len <= cfg.short_gap_hold and s > 0
                    and c[s - 1] >= cfg.low):
                # A one/two-frame hole: holding the last observation is the
                # lowest-error fill and is not a visible freeze.
                q_fill[s:e + 1] = q_obs[s - 1]
                p_fill[s:e + 1] = p_obs[s - 1]
                info = {"start": int(s), "end": int(e), "length": int(gap_len),
                        "left_anchor": True, "right_anchor": True,
                        "planned": "hold_short", "bone": name}
                gap_rows.append(info)
                continue
            info = _bridge_gap(q_obs, p_obs, q_fill, p_fill, c, s, e, cfg, mode)
            info["bone"] = name
            if cfg.smooth_gaps and info["planned"] != "rollout_no_right_anchor":
                _smooth_gap(q_fill, p_fill, s, e, cfg.smooth_passes)
            # Hedge between "keep moving" and "hold the last observation":
            # the middle of a gap uses cfg.bridge_blend of the bridge, while
            # the last lookahead frames ramp to the bridge so the return to
            # the observation has no jump.
            if (cfg.bridge_blend < 1.0 and s > 0
                    and info["planned"] != "rollout_no_right_anchor"):
                q_hold = q_obs[s - 1]
                p_hold = p_obs[s - 1]
                look = cfg.lookahead_frames()
                for t in range(s, e + 1):
                    ramp = (0.0 if t <= e - look
                            else _smoothstep((t - (e - look)) / (look + 1)))
                    w = cfg.bridge_blend + (1.0 - cfg.bridge_blend) * ramp
                    q_fill[t] = _q_slerp(q_hold, q_fill[t], w)
                    p_fill[t] = (1.0 - w) * p_hold + w * p_fill[t]
                info["blend"] = round(float(cfg.bridge_blend), 4)
            gap_rows.append(info)

        # Observed frames are evidence: never edit them.
        ok = c >= cfg.high
        q_fill[ok] = q_obs[ok]
        p_fill[ok] = p_obs[ok]
        out[name] = {"rot": _hemisphere_rows(q_fill),
                     "pos": p_fill.copy(),
                     "sigma": res["sigma"]}

    # Contact-preserving root height clamp.  A bridge can lift the pelvis by a
    # few units*1e-2 while both feet are still planted, which is enough to take
    # a toe out of the scorer's band.  Hold the observed root height through
    # the stance part of the gap and blend to the bridge over the lookahead;
    # only low-confidence frames are touched.
    if contact is not None and profile:
        root_name = (profile.get("standard_map") or {}).get("root")
        if root_name in out and root_name in tracks:
            rp = out[root_name]["pos"]
            rp_obs = np.asarray(tracks[root_name]["pos"], dtype=np.float64)
            c_root = conf[root_name]
            any_contact = np.zeros(n, dtype=bool)
            for side in contact["sides"]:
                any_contact |= contact["sides"][side]["carried"]
            look = cfg.lookahead_frames()
            for s, e in _runs(c_root < cfg.low):
                if e - s + 1 < 3:
                    continue
                a = s - 1
                if a < 0 or not any_contact[a]:
                    continue
                # only when the character was effectively standing
                if a > 0:
                    speed = float(np.linalg.norm(
                        (rp_obs[a] - rp_obs[a - 1])[[0, 2]]))
                else:
                    speed = 0.0
                if speed > cfg.contact_root_speed:
                    continue
                y0 = float(rp_obs[a, 1])
                hold_end = e - min(look, e - s + 1)
                for t in range(s, e + 1):
                    if t <= hold_end:
                        rp[t, 1] = y0
                    else:
                        denom = float(e + 1 - hold_end)
                        w = _smoothstep((t - hold_end) / max(1e-6, denom))
                        rp[t, 1] = (1.0 - w) * y0 + w * float(rp[t, 1])
    # sign-invariant observed check (hemisphere may negate an observed quat)
    changed = 0.0
    pos_changed = 0.0
    for name in names:
        ok = conf[name] >= cfg.high
        if not ok.any():
            continue
        qa = out[name]["rot"][ok]
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
        "mode": mode,
        "config": cfg.to_dict(),
        "frames": int(n),
        "bones": int(len(names)),
        "gating_frames": gating,
        "gaps": gap_rows,
        "gaps_total": len(gap_rows),
        "max_sigma_deg": round(sig_max, 4),
        "rollout_capped_frames": int(total_capped),
        "contact_damped_frames": int(total_damped),
        "observed_changed_max_deg": round(changed, 9),
        "observed_pos_changed_max": round(pos_changed, 9),
    }
    if contact is not None:
        report["contact"] = {
            side: {k: v[k] for k in
                   ("observed_fraction", "carried_fraction", "thr")}
            for side, v in contact["sides"].items()
        }
    return {"tracks": out, "report": report}


def _visible_tracks(tracks: Dict[str, Dict[str, np.ndarray]],
                    conf: Dict[str, np.ndarray],
                    cfg: StreamingConfig) -> Dict[str, Dict[str, np.ndarray]]:
    """What a causal observer actually has: hold the last high-confidence value.

    Contact detection must not see the ground-truth values hidden by the
    dropout mask.  This view is used only for stance estimation; the filler
    itself still receives the full observation arrays and ignores the frames
    whose confidence is low.
    """
    out = {}
    for name, rec in tracks.items():
        c = conf[name]
        q = np.asarray(rec["rot"], dtype=np.float64).copy()
        p = np.asarray(rec["pos"], dtype=np.float64).copy()
        last_q = None
        last_p = None
        for t in range(len(c)):
            if c[t] >= cfg.high:
                last_q, last_p = q[t].copy(), p[t].copy()
            elif last_q is not None:
                q[t], p[t] = last_q, last_p
        out[name] = {"rot": q, "pos": p}
    return out


def fill_vmd(vmd: Dict, confidence=None, model: Optional[Dict] = None,
             profile: Optional[Dict] = None, cfg: Optional[StreamingConfig] = None,
             mode: str = "offline",
             anchor_stance: Optional[bool] = None,
             stance: Optional[Dict] = None) -> Dict:
    """Fill a VMD dict; returns ``{vmd, report, frames, tracks}``.

    ``anchor_stance`` enables the M2.5 v0.1 stance-foot IK (defaults to
    ``cfg.anchor_stance``).  ``stance`` is an optional ``StanceConfig``.
    """
    cfg = cfg or StreamingConfig()
    do_anchor = bool(cfg.anchor_stance if anchor_stance is None
                     else anchor_stance)
    frames, tracks = dense_tracks(vmd)
    n = len(frames)
    conf = normalize_confidence(confidence, tracks, n)
    if model is not None and profile is not None:
        visible = _visible_tracks(tracks, conf, cfg)
        contact = estimate_contact(model, profile, visible, conf, cfg)
    else:
        contact = None
    res = fill_tracks(tracks, conf, cfg=cfg, mode=mode, contact=contact,
                      profile=profile)
    stance_report = None
    if do_anchor:
        if model is None or profile is None or contact is None:
            stance_report = {"enabled": False,
                             "reason": "anchor_stance needs model/profile/contact"}
        else:
            from .stance_ik import StanceConfig, apply_stance_ik
            if stance is None:
                scfg = StanceConfig(high=cfg.high, low=cfg.low, fps=cfg.fps,
                                    release_s=cfg.lookahead_s)
            elif isinstance(stance, StanceConfig):
                scfg = stance
            else:
                scfg = StanceConfig(**dict(stance))
            sres = apply_stance_ik(res["tracks"], conf, contact, model,
                                   profile, cfg=scfg)
            res["tracks"] = sres["tracks"]
            stance_report = sres["report"]
            res["report"]["observed_changed_max_deg"] = max(
                float(res["report"].get("observed_changed_max_deg") or 0.0),
                float(stance_report.get("observed_changed_max_deg") or 0.0))
            res["report"]["observed_pos_changed_max"] = max(
                float(res["report"].get("observed_pos_changed_max") or 0.0),
                float(stance_report.get("observed_pos_changed_max") or 0.0))
    out_vmd = tracks_to_vmd(vmd, frames, res["tracks"])
    # tracks_to_vmd only emits the tracks it is given; keep the original
    # morph/IK sections (dense body tracks are the only thing that changed).
    out_vmd["morphs"] = vmd.get("morphs", {})
    out_vmd["ik_frames"] = vmd.get("ik_frames", [])
    out_vmd["max_frame"] = int(vmd.get("max_frame", frames[-1] if frames else 0))
    res["report"]["stance_ik"] = stance_report
    return {"vmd": out_vmd, "report": res["report"], "frames": frames,
            "tracks": res["tracks"], "confidence": conf, "contact": contact}


# ---------------------------------------------------------------- reporting

def summarize(report: Dict) -> Dict:
    """Aggregate a fill report for tables / JSON."""
    gaps = report.get("gaps") or []
    lengths = [g["length"] for g in gaps]
    planned = {}
    for g in gaps:
        planned[g["planned"]] = planned.get(g["planned"], 0) + 1
    out = {
        "mode": report.get("mode"),
        "frames": report.get("frames"),
        "gaps": len(gaps),
        "gap_frames": int(sum(lengths)),
        "gap_len_mean": round(float(np.mean(lengths)), 2) if lengths else 0.0,
        "gap_len_max": int(max(lengths)) if lengths else 0,
        "planned": planned,
        "max_sigma_deg": report.get("max_sigma_deg"),
        "rollout_capped_frames": report.get("rollout_capped_frames"),
        "contact_damped_frames": report.get("contact_damped_frames"),
        "observed_changed_max_deg": report.get("observed_changed_max_deg"),
    }
    st = report.get("stance_ik")
    if st:
        out["stance_ik"] = {k: st.get(k) for k in (
            "enabled", "gaps_anchored", "sides_anchored", "frames_edited",
            "toe_xz_drift_max", "toe_xz_end_slide_max",
            "boundary_step_max_deg", "reach_limited_frames",
            "reach_lift_frames", "observed_changed_max_deg",
            "observed_pos_changed_max")}
    return out
