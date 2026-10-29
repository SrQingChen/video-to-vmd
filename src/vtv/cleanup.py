# -*- coding: utf-8 -*-
"""Selective cleanup for a VMD (M2/F): spikes, contact, foot lock.

The rules follow v2 §4.6 and the M0 meter calibration:

* only **flagged** frames are modified; no global filter, so dance accents
  survive;
* local quaternion spikes are bridged with the *nearest unflagged* neighbours
  (this also handles a 2-4 frame burst, e.g. 108-110 on the R2 forearm);
* quaternion hemisphere is made continuous before and after the repair;
* foot lock is contact gated: a toe is considered planted only when it is
  inside the GT-style band (``min(toe_y) + margin * height``) **and** its
  horizontal speed is below a threshold, with hysteresis so contact flicker
  does not inject root motion;
* root X/Z is shifted by the average anchor error of the planted feet.  The
  correction is capped and smoothed; it never changes the vertical root
  except when ``vertical_lock`` is requested.

Everything is deterministic and model-aware only through the PMX evaluation
used for contact; the rotations themselves are edited in VMD space, so the
output is still a normal 23-track IK-off VMD (safe for MMD) and the Blender
renderer still bakes inherited bones from it.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .mmd.animation import Track
from .mmd.mathutil import QID, qnorm, qslerp
from .metrics.core import pose_track

# A frame is a spike when it deviates from the midpoint of its neighbours by
# more than this.  The R2 prediction has a median deviation of 0.65-0.9 deg on
# the forearms and 15-33 deg on the four bad bursts; 5 deg separates them with
# a wide margin and leaves the upper arms (max 4.5-9.8 deg) untouched.
DEFAULT_SPIKE_DEG = 5.0
# A single absolute threshold is wrong for a faster real clip: T5's genuine
# 10-15 deg/frame arm accents sit above R2's 4 deg, and bridging them raised
# jerk.  The effective threshold is max(absolute, median + k*MAD) of that
# bone's own neighbour-interpolation residual, so a fast clip only flags its
# true outliers.
DEFAULT_SPIKE_SIGMAS = 8.0
DEFAULT_MAX_RUN = 6
# Bridging a *real* half turn is worse than leaving it: the bridge preserves
# the endpoint rotation, but if the endpoints are more than ~120 deg apart the
# run is probably a genuine fast movement, not a prediction spike.
DEFAULT_MAX_BRIDGE_DEG = 120.0
DEFAULT_CONTACT_MARGIN = 0.02
DEFAULT_CONTACT_SPEED = 0.12   # units/frame; 0.12 * 0.08 m = 1 cm/frame
DEFAULT_CONTACT_VSPEED = 0.08  # units/frame; a planted foot does not bob
DEFAULT_ROOT_CAP = 2.0         # units (~16 cm) over the whole clip
# Only intervals whose *net* horizontal drift exceeds this are touched.  A
# prediction that is already stable (R2 front: 0.33 units) must not be moved
# by a lock that cannot know the GT interval boundaries.
DEFAULT_LOCK_DRIFT = 0.5       # units (~4 cm)
# A side whose predicted contact covers less than this is not trusted enough
# to move the root: on R2 three_quarter the right toe is low but passes
# through (70% contact) and locking it made the GT slide worse.
DEFAULT_LOCK_CONTACT_FRACTION = 0.8


# ---------------------------------------------------------------- quaternions

def _as_arrays(rot: np.ndarray) -> np.ndarray:
    q = np.asarray(rot, dtype=np.float64)
    n = np.linalg.norm(q, axis=1, keepdims=True)
    out = np.zeros_like(q)
    out[:, 3] = 1.0
    return np.divide(q, n, out=out, where=n > 1e-12)


def q_angle_deg(a, b) -> float:
    d = abs(float(np.dot(a, b)))
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, d))))


def hemisphere(q: np.ndarray) -> np.ndarray:
    """Make q[i] the closest representative to q[i-1] (shortest path)."""
    out = _as_arrays(q).copy()
    for i in range(1, len(out)):
        if float(np.dot(out[i], out[i - 1])) < 0.0:
            out[i] = -out[i]
    return out


def _slerp_np(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return np.asarray(qslerp(tuple(a), tuple(b), float(t)), dtype=np.float64)


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or len(x) < 2:
        return x.copy()
    pad = win // 2
    xp = np.concatenate([np.repeat(x[:1], pad, axis=0), x,
                         np.repeat(x[-1:], pad, axis=0)], axis=0)
    k = np.ones(win) / win
    if x.ndim == 1:
        return np.convolve(xp, k, mode="valid")
    return np.stack([np.convolve(xp[:, c], k, mode="valid")
                     for c in range(x.shape[1])], axis=1)


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    out, start, prev = [], int(idx[0]), int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i == prev + 1:
            prev = i
            continue
        out.append((start, prev))
        start = prev = i
    out.append((start, prev))
    return out


# ---------------------------------------------------------------- dense tracks

def dense_tracks(vmd: Dict) -> Tuple[List[int], Dict[str, Dict[str, np.ndarray]]]:
    """Sample every VMD bone at every integer frame (MMD interpolation).

    The pipeline VMDs are already dense; this also supports sparse VMDs and
    makes the spike/contact arithmetic frame-index based.
    """
    frames = list(range(int(vmd.get("max_frame", 0)) + 1))
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for name, keys in (vmd.get("bones") or {}).items():
        tr = Track(keys)
        rot = np.zeros((len(frames), 4), dtype=np.float64)
        pos = np.zeros((len(frames), 3), dtype=np.float64)
        for k, f in enumerate(frames):
            p, q = tr.sample(f)
            pos[k] = p
            rot[k] = q
        out[name] = {"rot": rot, "pos": pos}
    return frames, out


# ---------------------------------------------------------------- spike repair

def detect_spikes(rot: np.ndarray, threshold_deg: float = DEFAULT_SPIKE_DEG,
                  max_run: int = DEFAULT_MAX_RUN,
                  max_bridge_deg: float = DEFAULT_MAX_BRIDGE_DEG,
                  adaptive_sigmas: float = DEFAULT_SPIKE_SIGMAS) -> Dict:
    """Find non-smooth local quaternion bursts and bridge them.

    A frame is flagged when ``q[t]`` differs from ``slerp(q[t-1], q[t+1], .5)``
    by more than ``threshold_deg``.  Consecutive flags form a run; a run is
    repaired by slerping from the last clean frame before it to the first clean
    frame after it.  Runs longer than ``max_run`` or with an endpoint span above
    ``max_bridge_deg`` are reported but left alone (they are likely real).
    """
    q = hemisphere(rot)
    n = len(q)
    dev = np.zeros(n)
    for t in range(1, n - 1):
        mid = _slerp_np(q[t - 1], q[t + 1], 0.5)
        dev[t] = q_angle_deg(q[t], mid)
    inner = dev[1:-1] if n > 2 else dev
    med = float(np.median(inner)) if len(inner) else 0.0
    mad = float(np.median(np.abs(inner - med))) * 1.4826 if len(inner) else 0.0
    eff = max(float(threshold_deg), med + float(adaptive_sigmas) * mad)
    flagged = dev > eff
    if n >= 2:
        flagged[0] = flagged[-1] = False

    fixed = q.copy()
    repaired: List[Dict] = []
    skipped: List[Dict] = []
    for a, b in _runs(flagged):
        span = b - a + 1
        prev, nxt = a - 1, b + 1
        if span > max_run or prev < 0 or nxt >= n:
            skipped.append({"start": int(a), "end": int(b), "span": int(span),
                            "reason": "run too long or at clip edge"})
            continue
        bridge = q_angle_deg(q[prev], q[nxt])
        if bridge > float(max_bridge_deg):
            skipped.append({"start": int(a), "end": int(b), "span": int(span),
                            "reason": f"bridge {bridge:.1f} deg > limit",
                            "bridge_deg": round(bridge, 3)})
            continue
        for t in range(a, b + 1):
            fixed[t] = _slerp_np(q[prev], q[nxt], (t - prev) / (nxt - prev))
        repaired.append({"start": int(a), "end": int(b), "span": int(span),
                         "bridge_deg": round(bridge, 3),
                         "max_dev_deg": round(float(dev[a:b + 1].max()), 3)})
    return {"rot": fixed, "flags": flagged, "dev": dev,
            "repaired": repaired, "skipped": skipped,
            "threshold_deg": round(eff, 4), "dev_median_deg": round(med, 4),
            "dev_mad_deg": round(mad, 4)}


def repair_tracks(tracks: Dict[str, Dict[str, np.ndarray]],
                  threshold_deg: float = DEFAULT_SPIKE_DEG,
                  max_run: int = DEFAULT_MAX_RUN,
                  max_bridge_deg: float = DEFAULT_MAX_BRIDGE_DEG,
                  bones: Optional[Iterable[str]] = None,
                  adaptive_sigmas: float = DEFAULT_SPIKE_SIGMAS) -> Dict:
    """Run spike repair on every bone track; returns new tracks + report."""
    keep = set(bones) if bones is not None else None
    out = {n: {"rot": v["rot"].copy(), "pos": v["pos"].copy()}
           for n, v in tracks.items()}
    report = {}
    for name, v in tracks.items():
        if keep is not None and name not in keep:
            continue
        res = detect_spikes(v["rot"], threshold_deg, max_run, max_bridge_deg,
                            adaptive_sigmas)
        out[name]["rot"] = res["rot"]
        out[name]["rot"] = hemisphere(out[name]["rot"])
        report[name] = {
            "repaired": res["repaired"],
            "skipped": res["skipped"],
            "flagged_frames": int(res["flags"].sum()),
            "max_dev_deg": round(float(res["dev"].max()), 3) if len(res["dev"]) else 0.0,
            "threshold_deg": res["threshold_deg"],
            "dev_median_deg": res["dev_median_deg"],
            "dev_mad_deg": res["dev_mad_deg"],
        }
    return {"tracks": out, "report": report}


# ---------------------------------------------------------------- contact

def contact_intervals(toe: np.ndarray, height: float,
                      margin: float = DEFAULT_CONTACT_MARGIN,
                      speed_thr: float = DEFAULT_CONTACT_SPEED,
                      vspeed_thr: float = DEFAULT_CONTACT_VSPEED,
                      min_len: int = 3, fill_gap: int = 2) -> Dict:
    """Planted intervals for one toe track (world positions, MMD units).

    The band is the same self-referential one the closed-loop meter uses
    (``min(toe_y) + margin * height``), plus a horizontal-speed gate so a foot
    that is merely passing low is not mistaken for a plant.
    """
    y = toe[:, 1]
    xz = toe[:, [0, 2]]
    speed = np.linalg.norm(np.diff(xz, axis=0), axis=1)
    speed = np.concatenate([speed[:1], speed])
    vspeed = np.abs(np.diff(y))
    vspeed = np.concatenate([vspeed[:1], vspeed])
    thr = float(y.min()) + float(margin) * float(height)
    near = y < thr
    planted = near & (speed < float(speed_thr)) & (vspeed < float(vspeed_thr))
    # fill short holes, drop short islands: contact flicker must not move root
    runs = _runs(planted)
    for (_, b0), (a1, _) in zip(runs, runs[1:]):
        if (a1 - b0 - 1) <= fill_gap:
            planted[b0 + 1:a1] = True
    runs = _runs(planted)
    for a, b in runs:
        if (b - a + 1) < min_len:
            planted[a:b + 1] = False
    return {"mask": planted, "intervals": _runs(planted),
            "thr": thr, "min_y": float(y.min()),
            "speed": speed, "vspeed": vspeed}


def foot_lock(model: Dict, profile: Dict, tracks: Dict[str, Dict[str, np.ndarray]],
              height: float, margin: float = DEFAULT_CONTACT_MARGIN,
              speed_thr: float = DEFAULT_CONTACT_SPEED,
              vspeed_thr: float = DEFAULT_CONTACT_VSPEED,
              cap: float = DEFAULT_ROOT_CAP, iterations: int = 2,
              vertical_lock: bool = False,
              lock_drift: float = DEFAULT_LOCK_DRIFT,
              min_contact_fraction: float = DEFAULT_LOCK_CONTACT_FRACTION) -> Dict:
    """Shift the root so planted toes stop travelling horizontally.

    The anchors are taken from the *first* contact frame of each interval and
    kept fixed across iterations, so repeated passes converge instead of
    chasing their own correction.  When both feet are planted the average
    anchor error is applied (one root cannot satisfy two different targets).
    """
    root_name = (profile.get("standard_map") or {}).get("root")
    if not root_name or root_name not in tracks:
        return {"applied": False, "reason": "no root track"}
    sim = eval_tracks(model, profile, tracks)
    frames = sim["frames"]
    n = len(frames)
    height = float(height or sim.get("height") or 1.0)

    side_info = {}
    for side in ("l", "r"):
        toe = sim["world"].get(f"toe_{side}")
        if toe is None:
            continue
        c = contact_intervals(toe, height, margin, speed_thr, vspeed_thr)
        anchors = {}
        drifts = {}
        trusted = float(c["mask"].mean()) >= float(min_contact_fraction)
        if trusted:
            for a, b in c["intervals"]:
                # net drift of the smoothed interval endpoints; only a real
                # slide is worth touching, otherwise the lock injects its own
                p = _moving_average(toe[:, [0, 2]], 5)
                drift = float(np.linalg.norm(p[b] - p[a]))
                drifts[(a, b)] = drift
                if drift < float(lock_drift):
                    continue
                # robust anchor: median of the first 3 planted frames
                k = min(3, b - a + 1)
                anchors[(a, b)] = np.median(toe[a:a + k][:, [0, 2]], axis=0)
        side_info[side] = {"contact": c, "anchors": anchors,
                           "drifts": drifts, "trusted": trusted}

    shift = np.zeros((n, 3), dtype=np.float64)
    weight_sum = np.zeros(n, dtype=np.float64)
    target_y = None
    if vertical_lock:
        lows = [sim["world"][f"toe_{s}"][:, 1] for s in side_info]
        if lows:
            target_y = float(np.percentile(np.minimum.reduce(lows), 5.0))

    # weights, shared across iterations (contact masks are fixed).  Intervals
    # below the drift threshold stay at weight 0: they are already stable.
    for side, info in side_info.items():
        w = np.zeros(n, dtype=np.float64)
        for a, b in info["anchors"]:
            w[a:b + 1] = 1.0
            ramp = min(2, b - a + 1)
            for k in range(ramp):
                w[a + k] = max(w[a + k], (k + 1) / (ramp + 1))
                w[b - k] = max(w[b - k], (k + 1) / (ramp + 1))
        info["w"] = w

    for _ in range(max(1, int(iterations))):
        d = np.zeros((n, 3), dtype=np.float64)
        wsum = np.zeros(n, dtype=np.float64)
        contact_y = {side: np.full(n, np.nan) for side in side_info}
        for side, info in side_info.items():
            toe = sim["world"][f"toe_{side}"]
            w = info["w"]
            for (a, b), anchor in info["anchors"].items():
                for t in range(a, b + 1):
                    if w[t] <= 0.0:
                        continue
                    d[t, 0] += w[t] * (anchor[0] - toe[t, 0])
                    d[t, 2] += w[t] * (anchor[1] - toe[t, 2])
                    wsum[t] += w[t]
                    if vertical_lock:
                        contact_y[side][t] = toe[t, 1]
        if vertical_lock and target_y is not None:
            lows = np.nanmin(np.stack([contact_y[s] for s in side_info]), axis=0)
            ok = np.isfinite(lows) & (wsum > 1e-9)
            d[ok, 1] = target_y - lows[ok]
        good = wsum > 1e-9
        if not good.any():
            break
        corr = np.zeros_like(d)
        corr[good] = d[good] / wsum[good, None]
        corr[:, 0] = _moving_average(corr[:, 0], 5)
        corr[:, 2] = _moving_average(corr[:, 2], 5)
        if vertical_lock:
            corr[:, 1] = _moving_average(corr[:, 1], 5)
        # Smoothing can create a little motion outside contact; gate it back.
        corr[~good] *= 0.5
        prev = shift.copy()
        shift += corr
        total = np.linalg.norm(shift[:, [0, 2]], axis=1)
        if cap > 0 and total.max() > cap:
            shift *= cap / total.max()
        applied = shift - prev
        # apply the capped increment and re-evaluate for the next iteration
        tracks[root_name]["pos"] = tracks[root_name]["pos"] + applied
        sim = eval_tracks(model, profile, tracks)

    applied = shift.copy()
    locked_any = any(info["anchors"] for info in side_info.values())
    return {
        "applied": bool(locked_any),
        "root_bone": root_name,
        "lock_drift": float(lock_drift),
        "shift_units": {
            "max": [round(float(v), 4) for v in np.abs(applied).max(0)],
            "mean": [round(float(v), 4) for v in np.abs(applied).mean(0)],
        },
        "sides": {
            side: {
                "thr": round(info["contact"]["thr"], 4),
                "min_y": round(info["contact"]["min_y"], 4),
                "intervals": [[int(a), int(b)] for a, b in info["contact"]["intervals"]],
                "locked_intervals": [[int(a), int(b)] for a, b in info["anchors"]],
                "drift_units": [round(float(v), 4) for v in info["drifts"].values()],
                "contact_frames": int(info["contact"]["mask"].sum()),
                "contact_fraction": round(float(info["contact"]["mask"].mean()), 4),
                "trusted": bool(info["trusted"]),
            } for side, info in side_info.items()
        },
        "vertical_lock": bool(vertical_lock),
        "target_y": None if target_y is None else round(float(target_y), 4),
    }


def eval_tracks(model: Dict, profile: Dict, tracks: Dict[str, Dict[str, np.ndarray]]):
    """Evaluate dense tracks on the PMX skeleton (shared by cleanup/IK)."""
    frames = sorted({int(f) for v in tracks.values() for f in range(len(v["rot"]))})
    frames = list(range(max(frames) + 1))
    curve = (20, 20, 107, 107)
    vmd = {"max_frame": frames[-1] if frames else 0,
           "bones": {n: [{"frame": i, "pos": tuple(v["pos"][i]),
                          "rot": tuple(v["rot"][i]),
                          "curves": (curve, curve, curve, curve)}
                         for i in frames] for n, v in tracks.items()},
           "morphs": {}, "ik_frames": []}
    tr = pose_track(model, vmd, profile, frames=frames, solve_ik=False)
    return tr


# ---------------------------------------------------------------- assembly

def tracks_to_vmd(vmd: Dict, frames: List[int],
                  tracks: Dict[str, Dict[str, np.ndarray]]) -> Dict:
    """Write dense tracks back into a VMD dict (morphs/IK preserved)."""
    bones = {}
    curve = (20, 20, 107, 107)
    for name, v in tracks.items():
        bones[name] = [{"frame": int(f),
                        "pos": tuple(float(c) for c in v["pos"][k]),
                        "rot": tuple(float(c) for c in v["rot"][k]),
                        "curves": (curve, curve, curve, curve)}
                       for k, f in enumerate(frames)]
    return {"model_name": vmd.get("model_name", ""), "bones": bones,
            "morphs": vmd.get("morphs", {}), "ik_frames": vmd.get("ik_frames", []),
            "max_frame": int(vmd.get("max_frame", frames[-1] if frames else 0))}


def clean_vmd(vmd: Dict, model: Dict, profile: Dict,
              spike_deg: float = DEFAULT_SPIKE_DEG,
              adaptive_sigmas: float = DEFAULT_SPIKE_SIGMAS,
              max_run: int = DEFAULT_MAX_RUN,
              max_bridge_deg: float = DEFAULT_MAX_BRIDGE_DEG,
              contact_margin: float = DEFAULT_CONTACT_MARGIN,
              contact_speed: float = DEFAULT_CONTACT_SPEED,
              contact_vspeed: float = DEFAULT_CONTACT_VSPEED,
              root_cap: float = DEFAULT_ROOT_CAP,
              lock_drift: float = DEFAULT_LOCK_DRIFT,
              lock_contact_fraction: float = DEFAULT_LOCK_CONTACT_FRACTION,
              do_spikes: bool = True,
              do_footlock: bool = True,
              vertical_lock: bool = False,
              bones: Optional[Iterable[str]] = None) -> Dict:
    """Apply selective cleanup; returns ``{vmd, frames, report}``."""
    frames, tracks = dense_tracks(vmd)
    report = {"frames": len(frames),
              "spike_deg": spike_deg, "spike_sigmas": adaptive_sigmas,
              "contact_margin": contact_margin,
              "contact_speed": contact_speed}
    if do_spikes:
        res = repair_tracks(tracks, spike_deg, max_run, max_bridge_deg, bones,
                            adaptive_sigmas)
        tracks = res["tracks"]
        report["spikes"] = res["report"]
        report["spike_frames_total"] = int(sum(
            v["flagged_frames"] for v in res["report"].values()))
    if do_footlock:
        tr = eval_tracks(model, profile, tracks)
        report["foot_lock"] = foot_lock(
            model, profile, tracks, tr["height"], contact_margin,
            contact_speed, contact_vspeed, root_cap,
            vertical_lock=vertical_lock, lock_drift=lock_drift,
            min_contact_fraction=lock_contact_fraction)
    return {"vmd": tracks_to_vmd(vmd, frames, tracks),
            "frames": frames, "report": report}
