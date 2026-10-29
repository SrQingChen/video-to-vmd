# -*- coding: utf-8 -*-
"""M4 quality control: certificates, GT-free stability, change audits, gates.

Everything here works on an evaluated :func:`vtv.metrics.core.pose_track` or
on a parsed VMD, so the one-command pipeline, the segment fallback and the
standalone ``scripts/qc_report.py`` all use the same definitions.

Two metric layers are kept strictly apart:

* **Task gate** (what M4 must not regress): 53-track/IK-off certificate,
  GT-free jerk/foot-slide/penetration/finger-step/morph-step stability, the
  "default feedback must not touch root/legs/hands/face" audit, and - when a
  ground truth VMD is supplied - the M3.5 body-relative main metric plus the
  contact/timing checks.
* **Secondary** (reported, never silently hidden): absolute world-angle and
  3D-position error deltas.  They need GT.  T5 has none, so the T5 QC reports
  the *correction magnitude* and carries the held-out reference numbers
  instead of pretending to measure the same thing.

Legs/feet are deliberately excluded from the learned residual, so on the
default profile foot slide/penetration are structurally unchanged; the
checks still run so a future profile cannot skip them.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from . import feedback_data as FD
from .metrics.core import (
    BODY_JOINTS, _intervals, _jerk, _slide_drift, compute_metrics,
)

# same thresholds as scripts/m35_t5_eval.py (the M3.5 T5 gate)
JERK_TOL = 0.05
SLIDE_TOL = 0.05
SLIDE_ABS = 1e-3
PENETRATION_ABS = 0.02
# arm capsule proxy budget (M2.4c); used by the feedback window fallback too
ARM_TORSO_TOL = 0.02
FINGER_TOL = 0.10
FINGER_ABS = 0.1
MORPH_ABS = 0.05

# GT gate thresholds from scripts/eval_feedback.py
GT_RECALL_TOL = 0.02
GT_SLIDE_TOL = 0.05
GT_SLIDE_ABS = 0.01
GT_PENETRATION_ABS = 0.05
GT_JERK_TOL = 0.10

# secondary warning thresholds (M3.5 §6.2 / M4 guide §3.4)
WORLD_WARN_DEG = 2.0
POSITION_WARN_PCT_HEIGHT = 0.2

# canonical upper-body joints used by the "how much did the arms move" report
SECONDARY_WORLD_BONES = ("spine1", "chest", "upperarm_l", "upperarm_r",
                         "forearm_l", "forearm_r")
SECONDARY_POSITION_JOINTS = ("upperarm_l", "forearm_l", "hand_l",
                             "upperarm_r", "forearm_r", "hand_r")


# ---------------------------------------------------------------------------
# certificate
# ---------------------------------------------------------------------------
def _expected_body_names(profile: Mapping) -> set:
    smap = profile.get("standard_map", {})
    return {smap[j] for j in BODY_JOINTS if j in smap}


def _forbidden_names(profile: Mapping) -> set:
    """D-chain / twist main+helper / IK bones: must never be written."""
    out = set()
    for side in ("l", "r"):
        for b in profile.get("d_chain", {}).get(side, []):
            if b.get("name"):
                out.add(b["name"])
    for key, tw in profile.get("twist", {}).items():
        for k in ("main", "helper"):
            if tw.get(k):
                out.add(tw[k])
    for side in ("l", "r"):
        for kind in ("leg", "toe"):
            b = profile.get("ik", {}).get(side, {}).get(kind)
            if b and b.get("bone"):
                out.add(b["bone"])
    return out


def vmd_certificate(vmd: Mapping, profile: Mapping) -> Dict:
    """Structural certificate for a final VMD (53 tracks, IK off, no D/helper)."""
    finger_names = {n for n in profile.get("finger_map", {}).values()
                    if n}
    expected = _expected_body_names(profile) | finger_names
    written = set(vmd.get("bones", {}))
    forbidden = _forbidden_names(profile)
    morphs = set(vmd.get("morphs", {}))
    profile_morphs = {m["name"] for m in profile.get("morphs", [])}
    finger_keys = [k for n in finger_names for k in vmd.get("bones", {}).get(n, [])]
    all_keys = [k for keys in vmd.get("bones", {}).values() for k in keys]
    quat_err = max((abs(math.sqrt(sum(c * c for c in k["rot"])) - 1.0)
                    for k in all_keys), default=0.0)
    ik_frames = vmd.get("ik_frames", [])
    ik_off = bool(ik_frames) and all(
        not any(fr.get("ik", {}).values()) for fr in ik_frames)
    finger_pos = max((max(abs(float(c)) for c in k.get("pos", (0, 0, 0)))
                      for k in finger_keys), default=0.0)
    cert = {
        "bone_tracks": len(written),
        "expected_tracks": len(expected),
        "finger_tracks": sum(1 for n in finger_names if n in written),
        "missing_tracks": sorted(expected - written),
        "unexpected_tracks": sorted(written - expected),
        "forbidden_tracks": sorted(written & forbidden),
        "morph_tracks": sorted(morphs),
        "morphs_in_profile": all(m in profile_morphs for m in morphs),
        "ik_frames": len(ik_frames),
        "ik_off": ik_off,
        "finger_translation_max": round(float(finger_pos), 9),
        "quat_norm_max_err": float(quat_err),
    }
    cert["ok"] = bool(
        not cert["missing_tracks"] and not cert["unexpected_tracks"]
        and not cert["forbidden_tracks"]
        and cert["finger_tracks"] == len(finger_names)
        and cert["morphs_in_profile"]
        and cert["ik_off"]
        and cert["finger_translation_max"] == 0.0
        and cert["quat_norm_max_err"] < 1e-6
        and len(written) == len(expected))
    return cert


# ---------------------------------------------------------------------------
# per-frame / per-window stability
# ---------------------------------------------------------------------------
def finger_step_values(vmd: Mapping, profile: Mapping,
                       n: int) -> tuple:
    """(frames, degrees) for every consecutive finger key pair.

    The gate's p95 is over all bone/frame steps, not over the per-frame maximum
    (taking the max first and then a p95 over frames gives a much larger number
    and does not match ``eval_helpers._finger_step``).
    """
    names = sorted({n for n in profile.get("finger_map", {}).values()
                    if n in vmd.get("bones", {})})
    frames, vals = [], []
    for name in names:
        keys = sorted(vmd["bones"][name], key=lambda k: k["frame"])
        for a, b in zip(keys, keys[1:]):
            f = int(b["frame"])
            if f <= 0 or f >= int(n) or int(b["frame"]) - int(a["frame"]) != 1:
                continue
            dot = abs(sum(x * y for x, y in zip(a["rot"], b["rot"])))
            frames.append(f)
            vals.append(2.0 * math.degrees(math.acos(min(1.0, dot))))
    return np.asarray(frames, np.int32), np.asarray(vals, np.float64)


def finger_step_p95(vmd: Mapping, profile: Mapping, n: int) -> float:
    _, vals = finger_step_values(vmd, profile, n)
    return float(np.percentile(vals, 95)) if len(vals) else 0.0


def morph_step_series(vmd: Mapping, n: int) -> np.ndarray:
    """Max morph weight change (per frame) attributed to the later frame."""
    out = np.zeros(max(int(n), 1), np.float64)
    x = np.arange(int(n), dtype=float)
    for name, keys in vmd.get("morphs", {}).items():
        if len(keys) < 2:
            continue
        fr = np.asarray([k[0] for k in keys], float)
        w = np.asarray([k[1] for k in keys], float)
        d = np.interp(x, fr, w, left=w[0], right=w[-1])
        if len(d) > 1:
            out[1:len(d)] = np.maximum(out[1:len(d)], np.abs(np.diff(d)))
    return out


def _toe_world(track: Mapping) -> Dict[str, np.ndarray]:
    return {f"toe_{s}": np.asarray(track["world"][f"toe_{s}"], float)
            for s in ("l", "r") if f"toe_{s}" in track["world"]}


def window_metrics(track: Mapping, f0: int, f1: int, height: float,
                   finger_steps: Optional[np.ndarray] = None,
                   morph_steps: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Stability of one frame window, using the same definitions as the gate."""
    world = {k: np.asarray(v, float)[f0:f1 + 1]
             for k, v in track["world"].items()}
    jerk = _jerk(world)
    toes = _toe_world(track)
    slide = 0.0
    pen = 0.0
    if toes:
        ymin = min(float(v[f0:f1 + 1, 1].min()) for v in toes.values())
        thr = ymin + 0.02 * height
        for v in toes.values():
            seg = v[f0:f1 + 1]
            mask = seg[:, 1] < thr
            slide += _slide_drift(seg, _intervals(mask))
            pen = max(pen, float(np.maximum(0.0, -seg[:, 1]).max()))
    out = {
        "frames": int(f1 - f0 + 1),
        "jerk_rms": float(jerk["rms"]),
        "jerk_spikes": int(jerk["spike_count"]),
        "foot_slide": float(slide),
        "foot_penetration": float(pen),
    }
    if finger_steps is not None:
        fr, vals = finger_steps
        m = (np.asarray(fr) >= f0) & (np.asarray(fr) <= f1)
        seg = np.asarray(vals, float)[m]
        out["finger_step_p95"] = float(np.percentile(seg, 95)) if len(seg) else 0.0
        out["finger_step_max"] = float(seg.max()) if len(seg) else 0.0
    if morph_steps is not None:
        seg = np.asarray(morph_steps, float)[f0:f1 + 1]
        out["morph_step_max"] = float(seg.max()) if len(seg) else 0.0
    return out


def failing_windows(before: Mapping, after: Mapping, n: int, height: float,
                    window: int = 30,
                    finger_before: Optional[np.ndarray] = None,
                    finger_after: Optional[np.ndarray] = None,
                    morph_before: Optional[np.ndarray] = None,
                    morph_after: Optional[np.ndarray] = None,
                    jerk_floor: float = 0.0) -> List[Dict]:
    """Frame windows where the correction regresses a gate metric.

    Used by the automatic segment fallback: a failing window is reverted to the
    deterministic M3 output while the rest of the clip keeps the correction.
    ``jerk_floor`` suppresses *relative* jerk flags in near-static windows: a
    quiet intro can show a 10x ratio while its absolute jerk is still a small
    fraction of the clip's.  Pass e.g. 0.25 * whole-clip jerk RMS.
    """
    bad: List[Dict] = []
    if n <= 0:
        return bad
    window = max(8, int(window))
    for f0 in range(0, n, window):
        f1 = min(n - 1, f0 + window - 1)
        if f1 - f0 < 3:
            continue
        b = window_metrics(before, f0, f1, height, finger_before, morph_before)
        a = window_metrics(after, f0, f1, height, finger_after, morph_after)
        reasons = []
        if (a["jerk_rms"] > b["jerk_rms"] * (1.0 + JERK_TOL) + 1e-9
                and a["jerk_rms"] > float(jerk_floor)):
            reasons.append("jerk")
        if a["foot_slide"] > b["foot_slide"] * (1.0 + SLIDE_TOL) + SLIDE_ABS:
            reasons.append("foot_slide")
        if a["foot_penetration"] > b["foot_penetration"] + PENETRATION_ABS:
            reasons.append("penetration")
        if a.get("finger_step_p95", 0.0) > b.get("finger_step_p95", 0.0) * (
                1.0 + FINGER_TOL) + FINGER_ABS:
            reasons.append("finger_step")
        if a.get("morph_step_max", 0.0) > b.get("morph_step_max", 0.0) + MORPH_ABS:
            reasons.append("morph_step")
        if reasons:
            bad.append({"start": int(f0), "end": int(f1), "reasons": reasons,
                        "before": b, "after": a})
    return bad


# ---------------------------------------------------------------------------
# change audit: the conservative profile must not touch legs/root/hands/face
# ---------------------------------------------------------------------------
def _track_equal(a: Sequence[Mapping], b: Sequence[Mapping], tol=1e-9) -> bool:
    if len(a) != len(b):
        return False
    for ka, kb in zip(a, b):
        if int(ka["frame"]) != int(kb["frame"]):
            return False
        for field in ("pos", "rot"):
            if any(abs(float(x) - float(y)) > tol
                   for x, y in zip(ka[field], kb[field])):
                return False
        if list(map(tuple, ka.get("curves", []))) != list(
                map(tuple, kb.get("curves", []))):
            return False
    return True


def unchanged_audit(before: Mapping, after: Mapping, profile: Mapping,
                    allowed_canonical: Iterable[str] = FD.BODY_FACTOR_NAMES,
                    allow_root: bool = False, allow_hands: bool = False,
                    allow_face: bool = False) -> Dict:
    """Which tracks changed, and whether every change was permitted."""
    smap = profile.get("standard_map", {})
    allowed = {smap[j] for j in allowed_canonical if j in smap}
    if allow_root and "root" in smap:
        allowed.add(smap["root"])
    finger_names = {n for n in profile.get("finger_map", {}).values() if n}
    if allow_hands:
        allowed |= finger_names | {smap[j] for j in ("hand_l", "hand_r")
                                   if j in smap}
    changed: List[str] = []
    for name, keys in before.get("bones", {}).items():
        if name not in after.get("bones", {}) or not _track_equal(
                keys, after["bones"][name]):
            changed.append(name)
    unexpected = sorted(set(changed) - allowed)
    body_unchanged = {}
    for label, joints in (("root", ("root",)),
                          ("legs", ("thigh_l", "shin_l", "foot_l", "toe_l",
                                    "thigh_r", "shin_r", "foot_r", "toe_r")),
                          ("hands", ("hand_l", "hand_r"))):
        names = [smap[j] for j in joints if j in smap]
        if label == "hands":
            names += sorted(finger_names)
        body_unchanged[label] = all(
            n in after.get("bones", {}) and _track_equal(
                before["bones"][n], after["bones"][n])
            for n in names if n in before.get("bones", {}))
    face_equal = (set(before.get("morphs", {})) == set(after.get("morphs", {}))
                  and all(before["morphs"][n] == after["morphs"][n]
                          for n in before.get("morphs", {})))
    out = {
        "allowed_tracks": sorted(allowed),
        "changed_tracks": sorted(changed),
        "unexpected_changed": unexpected,
        "root_unchanged": body_unchanged["root"],
        "legs_unchanged": body_unchanged["legs"],
        "hands_unchanged": body_unchanged["hands"] and not allow_hands,
        "face_unchanged": face_equal and not allow_face,
        "morphs_equal": face_equal,
    }
    out["ok"] = (not unexpected and body_unchanged["root"]
                 and body_unchanged["legs"]
                 and (allow_hands or out["hands_unchanged"])
                 and (allow_face or out["face_unchanged"]))
    return out


# ---------------------------------------------------------------------------
# secondary report (correction magnitude without GT; real deltas with GT)
# ---------------------------------------------------------------------------
def _bone_dir(track: Mapping, bone: str) -> np.ndarray:
    child = FD.BONE_CHILD[bone]
    d = np.asarray(track["world"][child], float) - np.asarray(
        track["world"][bone], float)
    return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)


def correction_magnitude(before: Mapping, after: Mapping,
                         profile: Mapping) -> Dict[str, float]:
    """GT-free: how far the correction moves the upper body.

    This is *not* the M3.5 world/position trade-off (that one is an error
    delta against GT); it only makes the size of the correction visible when
    no GT exists.
    """
    height = float(profile.get("height") or 1.0)
    angs = []
    for bone in SECONDARY_WORLD_BONES:
        if bone in before["world"] and FD.BONE_CHILD[bone] in before["world"]:
            a = _bone_dir(before, bone)
            b = _bone_dir(after, bone)
            angs.append(np.degrees(np.arccos(np.clip(
                np.sum(a * b, axis=1), -1.0, 1.0))))
    joints = [j for j in SECONDARY_POSITION_JOINTS
              if j in before["world"] and j in after["world"]]
    d = [np.linalg.norm(np.asarray(before["world"][j], float)
                        - np.asarray(after["world"][j], float), axis=1)
         for j in joints]
    cat = np.concatenate(d) if d else np.array([0.0])
    return {
        "world_change_deg": round(float(np.concatenate(angs).mean()), 4)
        if angs else 0.0,
        "position_change_pct_height": round(100.0 * float(cat.mean()) / height, 4),
        "position_change_p99_pct_height": round(
            100.0 * float(np.percentile(cat, 99)) / height, 4) if len(cat) else 0.0,
        "joints": joints,
        "note": "correction magnitude, not an error against GT",
    }


def gt_secondary(gt_track: Mapping, before: Mapping, after: Mapping,
                 profile: Mapping) -> Dict:
    """GT-based absolute world-angle / position error deltas (M3.5 metrics)."""
    mb = compute_metrics(gt_track, before, profile)
    ma = compute_metrics(gt_track, after, profile)
    vb, va = mb["vector"], ma["vector"]
    world = round(float(va["world_angle_mean_deg"] - vb["world_angle_mean_deg"]), 5)
    pos = round(float(va["position_pct_height"] - vb["position_pct_height"]), 5)
    return {
        "available": True,
        "world_angle_delta_deg": world,
        "position_delta_pct_height": pos,
        "checks": {
            "world_angle_not_worse": world <= WORLD_WARN_DEG,
            "position_not_worse": pos <= POSITION_WARN_PCT_HEIGHT,
        },
        "before": {"world_angle_mean_deg": vb["world_angle_mean_deg"],
                   "position_pct_height": vb["position_pct_height"]},
        "after": {"world_angle_mean_deg": va["world_angle_mean_deg"],
                  "position_pct_height": va["position_pct_height"]},
    }


def gt_scores(gt_track: Mapping, before: Mapping, after: Mapping,
              profile: Mapping) -> Dict:
    """Task-gate metrics on a synthetic clip with ground truth."""
    mb = compute_metrics(gt_track, before, profile)
    ma = compute_metrics(gt_track, after, profile)
    rb = FD.body_relative_errors(gt_track, before)
    ra = FD.body_relative_errors(gt_track, after)
    keys = ("angle_mean_deg", "world_angle_mean_deg", "position_pct_height",
            "position_p99_pct_height", "contact_recall", "contact_precision",
            "contact_slide_delta", "contact_penetration_delta",
            "contact_height_mae_pct_height", "jerk_rms_ratio", "spike_delta")
    rec = {
        "main_before": rb.get("body_relative_main_deg"),
        "main_after": ra.get("body_relative_main_deg"),
        "chest_before": rb.get("group_chest_deg"),
        "chest_after": ra.get("group_chest_deg"),
        "upperarm_before": rb.get("group_upperarm_deg"),
        "upperarm_after": ra.get("group_upperarm_deg"),
        "forearm_before": rb.get("group_forearm_deg"),
        "forearm_after": ra.get("group_forearm_deg"),
        "before": {k: mb["vector"][k] for k in keys},
        "after": {k: ma["vector"][k] for k in keys},
    }
    b, a = rec["before"], rec["after"]
    rec["checks"] = {
        "body_relative_main_improves": (
            float(rec["main_after"] or 0.0) < float(rec["main_before"] or 0.0)),
        "contact_recall_not_worse": (
            (a["contact_recall"] or 0) >= (b["contact_recall"] or 0) - GT_RECALL_TOL),
        "contact_slide_not_worse": (
            a["contact_slide_delta"] <= b["contact_slide_delta"] * (1 + GT_SLIDE_TOL)
            + GT_SLIDE_ABS),
        "penetration_not_worse": (
            a["contact_penetration_delta"]
            <= b["contact_penetration_delta"] + GT_PENETRATION_ABS),
        "jerk_not_worse": a["jerk_rms_ratio"] <= b["jerk_rms_ratio"] * (
            1 + GT_JERK_TOL),
    }
    return rec


# ---------------------------------------------------------------------------
# full per-model QC record
# ---------------------------------------------------------------------------
def _round_dict(d: Mapping, nd=5) -> Dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, (int, np.integer)):
            out[k] = int(v)
        elif isinstance(v, (float, np.floating)):
            f = float(v)
            # do not round tiny values (verify errors ~1e-6, quat errors ~1e-8)
            out[k] = f if (f != 0.0 and abs(f) < 1e-4) else round(f, nd)
        elif isinstance(v, dict):
            out[k] = _round_dict(v, nd)
        elif isinstance(v, (list, tuple)):
            out[k] = [round(float(x), nd) if isinstance(x, (int, float, np.floating))
                      else x for x in v]
        else:
            out[k] = v
    return out


def evaluate_model(model_key: str, profile: Mapping, before_vmd: Mapping,
                   final_vmd: Mapping, before_track: Mapping,
                   final_track: Mapping, feedback: Optional[Mapping] = None,
                   render: Optional[Mapping] = None,
                   gt_track: Optional[Mapping] = None,
                   heldout: Optional[Mapping] = None,
                   n: Optional[int] = None,
                   model: Optional[Mapping] = None,
                   chunking: Optional[Mapping] = None) -> Dict:
    """Build the per-model QC JSON (task gate + secondary + warnings).

    ``chunking`` is the M5 long-video boundary report
    (:func:`vtv.longvideo.boundary_report`).  Its ``ok`` becomes a task-gate
    check so a merged clip cannot pass while a seam freezes or jumps.
    """
    n = int(final_vmd["max_frame"]) + 1 if n is None else int(n)
    cfg = dict((feedback or {}).get("config") or {})
    apply_hands = bool(cfg.get("apply_hands", False))
    apply_face = bool(cfg.get("apply_face", False))
    apply_root = bool(cfg.get("apply_root", False))

    cert = vmd_certificate(final_vmd, profile)
    audit = unchanged_audit(before_vmd, final_vmd, profile,
                            allow_root=apply_root, allow_hands=apply_hands,
                            allow_face=apply_face)

    # whole-clip GT-free stability (same code as the M3.5 T5 gate)
    f_before = finger_step_values(before_vmd, profile, n)
    f_after = finger_step_values(final_vmd, profile, n)
    m_before = morph_step_series(before_vmd, n)
    m_after = morph_step_series(final_vmd, n)
    sb = window_metrics(before_track, 0, n - 1, float(profile["height"]),
                        f_before, m_before)
    sa = window_metrics(final_track, 0, n - 1, float(profile["height"]),
                        f_after, m_after)
    checks = {
        "certificate": bool(cert["ok"]),
        "jerk_not_worse": sa["jerk_rms"] <= sb["jerk_rms"] * (1 + JERK_TOL) + 1e-9,
        "slide_not_worse": sa["foot_slide"] <= sb["foot_slide"] * (1 + SLIDE_TOL)
        + SLIDE_ABS,
        "penetration_not_worse": sa["foot_penetration"]
        <= sb["foot_penetration"] + PENETRATION_ABS,
        "finger_step_not_worse": sa["finger_step_p95"]
        <= sb["finger_step_p95"] * (1 + FINGER_TOL) + FINGER_ABS,
        "morph_step_not_worse": sa["morph_step_max"]
        <= sb["morph_step_max"] + MORPH_ABS,
        "root_unchanged": audit["root_unchanged"],
        "legs_unchanged": audit["legs_unchanged"],
        "hands_unchanged": audit["hands_unchanged"] if not apply_hands else True,
        "face_unchanged": audit["face_unchanged"] if not apply_face else True,
        "changed_tracks_allowed": not audit["unexpected_changed"],
    }
    gt = None
    gt_sec = None
    if gt_track is not None:
        gt = gt_scores(gt_track, before_track, final_track, profile)
        gt_sec = gt_secondary(gt_track, before_track, final_track, profile)
        checks["body_relative_main_improves"] = gt["checks"][
            "body_relative_main_improves"]
        for k in ("contact_recall_not_worse", "contact_slide_not_worse",
                  "penetration_not_worse"):
            checks[f"gt_{k}"] = gt["checks"][k]
        # the GT jerk gate is looser (10%) and applies to the GT ratio
        checks["gt_jerk_not_worse"] = gt["checks"]["jerk_not_worse"]
    if chunking is not None:
        checks["boundary_continuity"] = bool(chunking.get("ok"))
    secondary = {
        "correction": correction_magnitude(before_track, final_track, profile),
        "heldout_reference": heldout,
        "available": gt_sec is not None,
    }
    if gt_sec is not None:
        secondary.update({k: v for k, v in gt_sec.items() if k != "available"})
        secondary["checks"] = gt_sec["checks"]
    warnings: List[str] = []
    # arm capsule penetration (GT-free proxy collision) before vs after
    arm_pen = None
    if model is not None:
        try:
            from .arm_prior import capsule_penetrations
            pb = capsule_penetrations(before_track, model, profile, arm_arm=True)
            pa = capsule_penetrations(final_track, model, profile, arm_arm=True)
            arm_pen = {
                "before": {"max_arm_torso": pb["max_arm_torso"],
                           "max_arm_arm": pb["max_arm_arm"]},
                "after": {"max_arm_torso": pa["max_arm_torso"],
                          "max_arm_arm": pa["max_arm_arm"]},
                "checks": {
                    "arm_torso_not_worse": (
                        pa["max_arm_torso"] <= pb["max_arm_torso"] + ARM_TORSO_TOL),
                    "arm_arm_not_worse": (
                        pa["max_arm_arm"] <= pb["max_arm_arm"] + ARM_TORSO_TOL),
                },
                "note": ("PMX capsule proxies (arm_prior); arm-torso is a task "
                         "gate, arm-arm is diagnostic (strict per-pair overlap "
                         "can occur in GT; see M2.4c)"),
            }
            checks["arm_torso_penetration_not_worse"] = arm_pen["checks"][
                "arm_torso_not_worse"]
            if not arm_pen["checks"]["arm_arm_not_worse"]:
                warnings.append(
                    "arm-arm capsule proxy overlap increased "
                    f"{arm_pen['before']['max_arm_arm']:.4f}->"
                    f"{arm_pen['after']['max_arm_arm']:.4f} (diagnostic; "
                    "strict per-pair penetration is not a task gate)")
        except Exception as e:  # noqa: BLE001 - optional proxy check
            warnings.append(f"arm capsule penetration check unavailable: {e}")
    render_cert = None
    if render:
        d = render.get("d_chain", {})
        v = render.get("verify", {})
        render_cert = {
            "d_chain_pos_err": d.get("max_pos_err"),
            "d_chain_rot_err_deg": d.get("max_rot_err_deg"),
            "verify_max_err": v.get("max_head_error"),
            "scale_warning_count": render.get("scale_warning_count"),
            "frames_rendered": render.get("frames_rendered"),
            "ok": bool(render.get("ok")),
        }
        checks["render_certificate"] = render_cert["ok"]

    if not cert["ok"]:
        warnings.append("VMD certificate failed: see certificate block")
    if audit["unexpected_changed"]:
        warnings.append("feedback changed forbidden tracks: "
                        + ", ".join(audit["unexpected_changed"]))
    if feedback and feedback.get("segments_fallback"):
        warnings.append("segment fallback used: "
                        + str(len(feedback["segments_fallback"])) + " window(s)")
    if feedback and feedback.get("whole_clip_fallback"):
        warnings.append("whole clip fell back to feedback=off")
    if chunking is not None:
        for seam in (chunking.get("seams") or []):
            if not seam.get("ok"):
                failed = [k for k, v in (seam.get("checks") or {}).items() if not v]
                warnings.append(
                    f"chunk boundary {seam.get('frame')} failed {failed}: "
                    f"step {seam.get('max_step_deg')}°, freeze "
                    f"{seam.get('freeze_run')}, slide/frame "
                    f"{seam.get('slide_rate')}")
    if secondary["available"]:
        for k, ok in (secondary.get("checks") or {}).items():
            if not ok:
                warnings.append(f"secondary threshold exceeded: {k} "
                                f"(world Δ{secondary['world_angle_delta_deg']:+.3f}°, "
                                f"position Δ{secondary['position_delta_pct_height']:+.3f}%H)")
    elif heldout:
        sec = heldout.get("secondary") or {}
        if sec and not all(sec.values()):
            warnings.append(
                "held-out secondary cost (GT-based, not measurable on this "
                f"clip; {heldout.get('tag') or heldout.get('path')}): "
                f"world {heldout.get('world_angle_delta_deg'):+.2f}°, "
                f"position {heldout.get('position_delta_pct_height'):+.3f}%H")
    else:
        warnings.append("no GT and no held-out reference: secondary cost cannot "
                        "be quantified for this clip")

    task_checks = {k: v for k, v in checks.items()
                   if k != "render_certificate"}
    rec = {
        "model": model_key,
        "frames": n,
        "feedback": {
            "mode": (feedback or {}).get("mode"),
            "applied": bool((feedback or {}).get("applied")),
            "config": cfg,
            "ckpt": (feedback or {}).get("ckpt"),
            "ckpt_sha256": (feedback or {}).get("ckpt_sha256"),
            "segments_fallback": (feedback or {}).get("segments_fallback", []),
            "whole_clip_fallback": bool((feedback or {}).get("whole_clip_fallback")),
        },
        "certificate": cert,
        "stability": {"before": sb, "after": sa},
        "change_audit": audit,
        "checks": checks,
        "task_gate": bool(all(task_checks.values())),
        "secondary": secondary,
        "gt": gt,
        "chunking": chunking,
        "arm_penetration": arm_pen,
        "render": render_cert,
        "warnings": warnings,
    }
    rec["ok"] = bool(rec["task_gate"])
    return _round_dict(rec)
