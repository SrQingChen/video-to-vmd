# -*- coding: utf-8 -*-
"""M4 production application of the M3.5 conservative feedback.

The network and its evidence contract are unchanged from M3.5; this module is
the *application* layer that the one-command pipeline calls:

    body VMD (adapter + cleanup + contact IK)
      + M3 hand/face evidence (model-independent)
      -> evidence-only features -> 79-D bounded residual
      -> root/gain/gating/smoothing policy (default: torso .85, arm .20,
         smooth 3, hands/face/root off)
      -> corrected body + the *unchanged* M3 hand/face tracks -> final VMD

Safety rules enforced here:

* legs, feet and the root are never rotated/translated by the conservative
  profile (the residual layout has no leg factors, and root is zeroed);
* hand/face residual is only applied on M3 L0/L1 frames, never on fallback;
* any window whose correction regresses jerk / foot-slide / penetration falls
  back to the deterministic M3 output for that window (recorded, not silent);
* if hand/face corrections regress the finger/morph step gates, they are
  dropped entirely for the clip (recorded).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from . import feedback_data as FD
from . import qc
from .eval_helpers import gate_residual, load_net, predict, smooth_residual
from .metrics.core import pose_track

# Conservative profile = the M3.5 recommended / evaluated default.
MODES: Dict[str, Dict] = {
    "off": {"torso_gain": 0.0, "arm_gain": 0.0, "smooth": 1,
            "apply_root": False, "apply_hands": False, "apply_face": False},
    "default": {"torso_gain": 0.85, "arm_gain": 0.20, "smooth": 3,
                "apply_root": False, "apply_hands": False, "apply_face": False,
                "root_contact_joint": False},
    # Experimental only: T5 fails the finger/morph step gate (M3.5 §8).
    "full": {"torso_gain": 0.85, "arm_gain": 0.30, "smooth": 3,
             "apply_root": False, "apply_hands": True, "apply_face": True},
    # M6 v2: wider teacher bounds + baseline-pose features + expanded data.
    # v2 keeps hands/face off (T5 finger/morph exactly unchanged); v2_full is
    # the opt-in hand/face version (T5 finger p95 +8%, still inside the gate).
    "v2": {"torso_gain": 0.85, "arm_gain": 0.30, "smooth": 3,
           "apply_root": False, "apply_hands": False, "apply_face": False,
           "root_contact_joint": False},
    "v2_full": {"torso_gain": 0.85, "arm_gain": 0.30, "smooth": 3,
                "apply_root": False, "apply_hands": True,
                "apply_face": True, "root_contact_joint": False},
}
DEFAULT_FALLBACK = {
    "enabled": True, "window_frames": 30, "max_passes": 3, "ramp": 3,
}


def resolve_config(mode: str = "default",
                   modes: Optional[Mapping[str, Mapping]] = None,
                   overrides: Optional[Mapping] = None) -> Dict:
    """Merge a named mode with optional overrides; validate the fields."""
    modes = dict(MODES, **(modes or {}))
    if mode not in modes:
        raise ValueError(f"unknown feedback mode {mode!r}; have {sorted(modes)}")
    cfg = dict(modes[mode])
    cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
    cfg["mode"] = mode
    for k in ("torso_gain", "arm_gain"):
        if not 0.0 <= float(cfg[k]) <= 1.0:
            raise ValueError(f"{k} must be in [0,1], got {cfg[k]!r}")
    if int(cfg["smooth"]) < 1:
        raise ValueError("smooth must be >= 1")
    for k in ("apply_root", "apply_hands", "apply_face", "root_contact_joint"):
        cfg[k] = bool(cfg.get(k, False))
    return cfg


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_feedback_model(ckpt: str) -> Dict:
    """Load the M3.5 checkpoint once; keep the torch objects + provenance."""
    p = Path(ckpt)
    if not p.exists():
        raise FileNotFoundError(
            f"feedback checkpoint not found: {p}. The default M4 pipeline needs "
            "the M3.5 network (outputs/m35/net/m35_best.pt); use --feedback off "
            "to run without it.")
    net, ck = load_net(p)
    return {
        "net": net,
        "ck": ck,
        "path": str(p),
        "sha256": _sha256(p),
        "params": int(net.n_params()),
        "epoch": ck.get("epoch"),
        "cfg": ck.get("cfg"),
        "hand_supported": ck.get("hand_supported"),
        "face_supported": ck.get("face_supported"),
    }


# ---------------------------------------------------------------------------
# evidence at the VMD frame rate
# ---------------------------------------------------------------------------
def vmd_source_indices(n_out: int, slow_factor: float, n_src: int) -> np.ndarray:
    """Same sampling rule as ``adapter.canonical_from_smplx`` (frame k <- k/s)."""
    if slow_factor <= 0:
        raise ValueError("slow_factor must be > 0")
    return np.clip(np.round(np.arange(int(n_out)) / float(slow_factor)).astype(int),
                   0, int(n_src) - 1)


def _take(a, idx):
    if a is None:
        return None
    return np.asarray(a)[idx]


def resample_evidence(evidence: Mapping, n_out: int, slow_factor: float) -> Dict:
    """Resample all evidence arrays (kp2d, bbx, hand, face) to VMD frames."""
    kp = evidence.get("kp2d")
    n_src = len(kp) if kp is not None else int(evidence.get("n_src") or n_out)
    idx = vmd_source_indices(n_out, slow_factor, n_src)
    hand = evidence.get("hand") or {}
    face = evidence.get("face") or {}
    return {
        "kp2d": _take(kp, idx),
        "bbx_xys": _take(evidence.get("bbx_xys"), idx),
        "img_wh": tuple(evidence.get("img_wh") or (1080, 1920)),
        "adapter_scale": float(evidence.get("adapter_scale", 1.0)),
        "hand": {k: _take(hand.get(k), idx)
                 for k in ("angles", "level", "score", "detected")},
        "face": {k: _take(face.get(k), idx)
                 for k in ("weights", "level", "detected")},
    }


# ---------------------------------------------------------------------------
# residual policy
# ---------------------------------------------------------------------------
def shape_residual(raw: np.ndarray, hand_level: np.ndarray,
                   face_level: np.ndarray, profile: Mapping,
                   ck: Mapping, cfg: Mapping) -> np.ndarray:
    """Apply root/gain/hand-face policy, L0/L1 gating and smoothing."""
    res = np.array(raw, np.float32, copy=True)
    if not cfg.get("apply_root", False):
        res[:, FD.ROOT_SLICE] = 0.0
    for i, bone in enumerate(FD.BODY_FACTOR_NAMES):
        is_arm = bone.startswith(("clavicle", "upperarm", "forearm"))
        gain = cfg["arm_gain"] if is_arm else cfg["torso_gain"]
        res[:, 2 + 3 * i:5 + 3 * i] *= float(gain)
    if not cfg.get("apply_hands", False):
        res[:, FD.HAND_SLICE] = 0.0
    if not cfg.get("apply_face", False):
        res[:, FD.FACE_SLICE] = 0.0
    sups = (ck.get("hand_supported"), ck.get("face_supported"))
    names = FD.face_target_names(profile)
    res = gate_residual(res, hand_level, face_level, names, *sups)
    res = smooth_residual(res, hand_level, face_level,
                          win=int(cfg.get("smooth", 3)))
    res = gate_residual(res, hand_level, face_level, names, *sups)
    return res


def _smooth_body(res: np.ndarray, win: int) -> np.ndarray:
    """Edge-replicated moving average on the body/root slices only."""
    out = np.array(res, np.float32, copy=True)
    if win <= 1:
        return out
    k = np.ones(int(win), np.float64) / int(win)
    pad = int(win) // 2
    for sl in (FD.ROOT_SLICE, FD.ROT_SLICE):
        x = out[:, sl]
        xp = np.concatenate([np.repeat(x[:1], pad, axis=0), x,
                             np.repeat(x[-1:], pad, axis=0)], axis=0)
        out[:, sl] = np.stack([np.convolve(xp[:, c], k, mode="valid")
                               for c in range(x.shape[1])], axis=1)
    return out


def _slice_track(track: Mapping, f0: int, f1: int) -> Dict:
    """A contiguous frame slice of a pose_track (for window-local proxies)."""
    world = {k: np.asarray(v, np.float64)[f0:f1 + 1]
             for k, v in (track.get("world") or {}).items()}
    world_q = {k: np.asarray(v, np.float64)[f0:f1 + 1]
               for k, v in (track.get("world_q") or {}).items()}
    return {"frames": list(range(f0, f1 + 1)), "world": world, "world_q": world_q}


def _arm_torso_max(track: Mapping, model: Mapping, profile: Mapping) -> float:
    from .arm_prior import capsule_penetrations
    return float(capsule_penetrations(track, model, profile,
                                      arm_arm=False)["max_arm_torso"])


def _proxy_windows(before_track: Mapping, after_track: Mapping, model: Mapping,
                   profile: Mapping, n: int, window: int,
                   budget: float) -> list:
    """Windows where the arm-torso capsule proxy worsens beyond the QC budget."""
    bad = []
    for f0 in range(0, n, max(8, int(window))):
        f1 = min(n - 1, f0 + max(8, int(window)) - 1)
        if f1 - f0 < 3:
            continue
        b = _arm_torso_max(_slice_track(before_track, f0, f1), model, profile)
        a = _arm_torso_max(_slice_track(after_track, f0, f1), model, profile)
        if a > b + float(budget) + 1e-9:
            bad.append({"start": int(f0), "end": int(f1),
                        "reasons": ["arm_torso_penetration"],
                        "before": {"arm_torso": round(b, 6)},
                        "after": {"arm_torso": round(a, 6)}})
    return bad


def _body_fallback(body_vmd: Mapping, model: Mapping, profile: Mapping,
                   res: np.ndarray, n: int, cfg: Mapping,
                   fallback: Mapping) -> Tuple[np.ndarray, list, bool]:
    """Revert only the windows that make the *whole-clip* gate fail.

    The segment fallback is a localizer for a clip-level regression, not a
    second, stricter gate: while the whole clip passes (the M3.5 T5 gate), a
    locally larger ratio in a near-static window is not a visible artifact and
    the correction is kept.  When the whole clip fails, failing windows are
    masked out and the body residual is re-smoothed so the on/off transition
    cannot create new jerk.

    The gate covers every metric the QC task gate applies to the upper body:
    jerk, foot slide/penetration and the arm-torso capsule proxy.  Without the
    last one the M3.5 correction could regress the proxy in one 300-frame chunk
    while the QC marks that chunk failed (seen in the M5 long-video run).
    """
    if not fallback.get("enabled", True) or not np.any(res[:, FD.ROT_SLICE]):
        return res, [], False
    frames = list(range(n))
    before_track = pose_track(model, body_vmd, profile, frames=frames)
    height = float(profile.get("height") or 1.0)
    window = int(fallback.get("window_frames", 30))
    ramp = max(1, int(fallback.get("ramp", 3)))
    smooth_win = max(int(cfg.get("smooth", 3)), 2 * ramp + 1)
    global_jerk = qc.window_metrics(before_track, 0, n - 1, height)["jerk_rms"]
    # A quiet window can show a large *relative* jerk while its absolute jerk
    # is still a small fraction of the clip's: require significance too.
    jerk_floor = 0.25 * float(global_jerk)
    proxy_budget = float(fallback.get("arm_torso_budget",
                                      qc.ARM_TORSO_TOL))
    proxy_check = bool(fallback.get("arm_torso_check", True))
    jerk_tol = float(fallback.get("jerk_tol", qc.JERK_TOL))
    slide_tol = float(fallback.get("slide_tol", qc.SLIDE_TOL))
    pen_abs = float(fallback.get("penetration_abs", qc.PENETRATION_ABS))

    def body_ok(before_w, after_w):
        return (after_w["jerk_rms"] <= before_w["jerk_rms"] * (1 + jerk_tol)
                and after_w["foot_slide"]
                <= before_w["foot_slide"] * (1 + slide_tol) + qc.SLIDE_ABS
                and after_w["foot_penetration"]
                <= before_w["foot_penetration"] + pen_abs)

    b_all = qc.window_metrics(before_track, 0, n - 1, height)
    corrected = FD.apply_body_residual(body_vmd, model, profile, res)
    after_track = pose_track(model, corrected, profile, frames=frames)
    a_all = qc.window_metrics(after_track, 0, n - 1, height)
    b_proxy = _arm_torso_max(before_track, model, profile)
    a_proxy = _arm_torso_max(after_track, model, profile)
    if body_ok(b_all, a_all) and (
            not proxy_check or a_proxy <= b_proxy + proxy_budget + 1e-9):
        return res, [], False

    mask = np.ones(n, np.float32)
    applied: list = []
    for _ in range(max(1, int(fallback.get("max_passes", 3)))):
        corrected = FD.apply_body_residual(body_vmd, model, profile,
                                           res * mask[:, None])
        after_track = pose_track(model, corrected, profile, frames=frames)
        bad = qc.failing_windows(before_track, after_track, n, height,
                                 window=window, jerk_floor=jerk_floor)
        if proxy_check and _arm_torso_max(after_track, model,
                                          profile) > b_proxy + proxy_budget + 1e-9:
            bad = bad + _proxy_windows(before_track, after_track, model,
                                       profile, n, window, proxy_budget)
        if not bad:
            out = _smooth_body(res * mask[:, None], smooth_win)
            return out, applied, False
        for w in bad:
            mask[w["start"]:w["end"] + 1] = 0.0
        applied = applied + bad
    # still failing: fall back the whole clip rather than ship a regression
    return np.zeros_like(res), applied, True


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def apply_feedback(body_vmd: Mapping, model: Mapping, profile: Mapping,
                   feedback: Mapping, evidence: Mapping, cfg: Mapping,
                   fallback: Optional[Mapping] = None) -> Dict:
    """Apply the configured feedback to a 23-track body VMD.

    Returns ``vmd`` (53 tracks + morphs + IK off), the residual actually used,
    and the fallback audit.  ``evidence`` must already be at the VMD frame rate
    (use :func:`resample_evidence`).
    """
    fallback = dict(DEFAULT_FALLBACK, **(fallback or {}))
    n = int(body_vmd["max_frame"]) + 1
    hand = evidence.get("hand") or {}
    face = evidence.get("face") or {}
    ha = hand.get("angles")
    hl = hand.get("level")
    fw = face.get("weights")
    fl = face.get("level")
    if ha is None or hl is None or len(ha) < n:
        raise ValueError("hand evidence missing or shorter than the body VMD")
    if fw is None or fl is None or len(fw) < n:
        raise ValueError("face evidence missing or shorter than the body VMD")

    tr = pose_track(model, body_vmd, profile, frames=list(range(n)))
    feats = FD.build_features(
        tr, float(profile.get("height") or 1.0),
        kp2d=evidence.get("kp2d"), bbx_xys=evidence.get("bbx_xys"),
        hand_angles=ha[:n], hand_level=hl[:n],
        hand_score=hand.get("score"), hand_detect=hand.get("detected"),
        face_weights=fw[:n], face_level=fl[:n],
        face_detect=face.get("detected"), profile=profile,
        adapter_scale=float(evidence.get("adapter_scale", 1.0)),
        img_wh=tuple(evidence.get("img_wh") or (1080, 1920)),
        base_hand_angles=ha[:n], base_face_weights=fw[:n],
        feature_set=(feedback.get("ck") or {}).get("feature_set", "v1"))
    raw = predict(feedback["net"], feedback["ck"], feats)
    res = shape_residual(raw, hl[:n], fl[:n], profile, feedback["ck"], cfg)

    res, segments, whole = _body_fallback(body_vmd, model, profile, res, n,
                                          cfg, fallback)

    corr_body = FD.apply_body_residual(body_vmd, model, profile, res)
    if cfg.get("apply_root") and cfg.get("root_contact_joint"):
        # Root XZ moves planted feet; re-solve the deterministic contact IK on
        # the corrected body so the two corrections are jointly feasible.
        from .contact_ik import solve_contact_ik
        ik = solve_contact_ik(corr_body, model, profile,
                              kp2d=evidence.get("kp2d"))
        corr_body = ik["vmd"]
    ha_out = (FD.apply_hand_residual(ha[:n], res[:, FD.HAND_SLICE])
              if cfg.get("apply_hands") else np.array(ha[:n], copy=True))
    fw_out = (FD.apply_face_residual(fw[:n], res[:, FD.FACE_SLICE])
              if cfg.get("apply_face") else np.array(fw[:n], copy=True))
    vmd = FD.assemble_vmd(corr_body, profile, ha_out, fw_out, n_frames=n)

    # hand/face step gate: drop those corrections for the whole clip if they
    # regress the gates (recorded; the default profile never gets here).
    handface_fallback = False
    if cfg.get("apply_hands") or cfg.get("apply_face"):
        before_assembled = FD.assemble_vmd(
            body_vmd, profile, np.array(ha[:n]), np.array(fw[:n]), n_frames=n)
        before_finger = qc.finger_step_values(before_assembled, profile, n)
        after_finger = qc.finger_step_values(vmd, profile, n)
        before_morph = qc.morph_step_series(before_assembled, n)
        after_morph = qc.morph_step_series(vmd, n)
        before_p95 = (float(np.percentile(before_finger[1], 95))
                      if len(before_finger[1]) else 0.0)
        after_p95 = (float(np.percentile(after_finger[1], 95))
                     if len(after_finger[1]) else 0.0)
        finger_ok = after_p95 <= before_p95 * 1.10 + 0.1
        morph_ok = after_morph.max() <= before_morph.max() + qc.MORPH_ABS
        if not (finger_ok and morph_ok):
            res[:, FD.HAND_SLICE] = 0.0
            res[:, FD.FACE_SLICE] = 0.0
            ha_out = np.array(ha[:n], copy=True)
            fw_out = np.array(fw[:n], copy=True)
            corr_body = FD.apply_body_residual(body_vmd, model, profile, res)
            vmd = FD.assemble_vmd(corr_body, profile, ha_out, fw_out, n_frames=n)
            handface_fallback = True

    stats = {name: float(np.abs(res[:, sl]).mean())
             for name, sl in FD.residual_groups().items()}
    return {
        "vmd": vmd,
        "residual": res,
        "hand_out": ha_out,
        "face_out": fw_out,
        "applied": bool(np.any(res != 0.0)),
        "segments_fallback": segments,
        "whole_clip_fallback": bool(whole),
        "handface_fallback": bool(handface_fallback),
        "residual_mean_abs": stats,
        "features_shape": list(feats.shape),
        "raw_residual_mean_abs": {
            name: float(np.abs(raw[:, sl]).mean())
            for name, sl in FD.residual_groups().items()},
    }
