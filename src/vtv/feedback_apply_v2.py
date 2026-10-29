# -*- coding: utf-8 -*-
"""M6 two-pass residual application (apply -> recompute -> predict again).

Pass 1 is the M3.5/M6 one-shot refiner.  Pass 2 sees the corrected pose and
the corrected hand/face baseline, and predicts the remaining residual.  The
two residuals are composed in the parent frame (``q2 * q1``) so the final VMD
is mathematically the same as applying pass 1 and then pass 2 to the pose.

The safety layer is unchanged: root/hand/face policy, L0/L1 gating, smoothing,
window/whole-clip fallback and the hand/face step gate all run on the composed
residual.  Legs stay with contact IK; no leg factor exists in the layout.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np

from . import feedback_apply as FA
from . import feedback_data as FD
from . import qc
from .eval_helpers import gate_residual, predict
from .metrics.core import pose_track


def _compose(res1: np.ndarray, res2: np.ndarray, limits: np.ndarray) -> np.ndarray:
    out = np.array(res1, np.float32, copy=True)
    # root / hands / face: additive in their own coordinates
    for sl in (FD.ROOT_SLICE, FD.HAND_SLICE, FD.FACE_SLICE):
        out[:, sl] = np.clip(res1[:, sl] + res2[:, sl], -limits[sl], limits[sl])
    for i, bone in enumerate(FD.BODY_FACTOR_NAMES):
        a, b = 2 + 3 * i, 5 + 3 * i
        for fi in range(len(out)):
            q1 = FD._rotvec_to_q(res1[fi, a:b])
            q2 = FD._rotvec_to_q(res2[fi, a:b])
            rv = FD._q_to_rotvec(FD.qmul(q2, q1))
            out[fi, a:b] = np.clip(rv, -limits[a:b], limits[a:b])
    return out


def _features(body_vmd, model, profile, evidence, base_hand, base_face,
              feature_set):
    n = int(body_vmd["max_frame"]) + 1
    tr = pose_track(model, body_vmd, profile, frames=list(range(n)))
    hand = evidence.get("hand") or {}
    face = evidence.get("face") or {}
    return FD.build_features(
        tr, float(profile.get("height") or 1.0),
        kp2d=evidence.get("kp2d"), bbx_xys=evidence.get("bbx_xys"),
        hand_angles=hand.get("angles"), hand_level=hand.get("level"),
        hand_score=hand.get("score"), hand_detect=hand.get("detected"),
        face_weights=face.get("weights"), face_level=face.get("level"),
        face_detect=face.get("detected"), profile=profile,
        adapter_scale=float(evidence.get("adapter_scale", 1.0)),
        img_wh=tuple(evidence.get("img_wh") or (1080, 1920)),
        base_hand_angles=base_hand, base_face_weights=base_face,
        feature_set=feature_set)


def apply_feedback_v2(body_vmd: Mapping, model: Mapping, profile: Mapping,
                      feedback1: Mapping, feedback2: Optional[Mapping],
                      evidence: Mapping, cfg: Mapping,
                      fallback: Optional[Mapping] = None) -> Dict:
    fallback = dict(FA.DEFAULT_FALLBACK, **(fallback or {}))
    n = int(body_vmd["max_frame"]) + 1
    hand = evidence.get("hand") or {}
    face = evidence.get("face") or {}
    ha = np.asarray(hand.get("angles"))[:n]
    hl = np.asarray(hand.get("level"))[:n]
    fw = np.asarray(face.get("weights"))[:n]
    fl = np.asarray(face.get("level"))[:n]
    if ha is None or hl is None or len(ha) < n:
        raise ValueError("hand evidence missing or shorter than the body VMD")
    if fw is None or fl is None or len(fw) < n:
        raise ValueError("face evidence missing or shorter than the body VMD")

    ck1 = feedback1["ck"]
    feats1 = _features(body_vmd, model, profile, evidence, ha, fw,
                       ck1.get("feature_set", "v1"))
    raw1 = predict(feedback1["net"], ck1, feats1)
    res1 = FA.shape_residual(raw1, hl, fl, profile, ck1, cfg)

    body1 = FD.apply_body_residual(body_vmd, model, profile, res1)
    ha1 = (FD.apply_hand_residual(ha, res1[:, FD.HAND_SLICE])
           if cfg.get("apply_hands") else np.array(ha, copy=True))
    fw1 = (FD.apply_face_residual(fw, res1[:, FD.FACE_SLICE])
           if cfg.get("apply_face") else np.array(fw, copy=True))

    if feedback2 is not None:
        ck2 = feedback2["ck"]
        feats2 = _features(body1, model, profile, evidence, ha1, fw1,
                           ck2.get("feature_set", "v1"))
        raw2 = predict(feedback2["net"], ck2, feats2)
        res2 = FA.shape_residual(raw2, hl, fl, profile, ck2, cfg)
        g2 = float(cfg.get("pass2_gain", 1.0))
        if g2 != 1.0:
            res2 = res2 * g2
        limits = np.asarray(ck2.get("limits", FD.residual_limits("v2")))
        res = _compose(res1, res2, limits)
    else:
        res = res1

    res, segments, whole = FA._body_fallback(body_vmd, model, profile, res, n,
                                             cfg, fallback)
    corr_body = FD.apply_body_residual(body_vmd, model, profile, res)
    if cfg.get("apply_root") and cfg.get("root_contact_joint"):
        from .contact_ik import solve_contact_ik
        ik = solve_contact_ik(corr_body, model, profile,
                              kp2d=evidence.get("kp2d"))
        corr_body = ik["vmd"]
    ha_out = (FD.apply_hand_residual(ha, res[:, FD.HAND_SLICE])
              if cfg.get("apply_hands") else np.array(ha, copy=True))
    fw_out = (FD.apply_face_residual(fw, res[:, FD.FACE_SLICE])
              if cfg.get("apply_face") else np.array(fw, copy=True))
    vmd = FD.assemble_vmd(corr_body, profile, ha_out, fw_out, n_frames=n)

    handface_fallback = False
    if cfg.get("apply_hands") or cfg.get("apply_face"):
        before_assembled = FD.assemble_vmd(
            body_vmd, profile, np.array(ha), np.array(fw), n_frames=n)
        bf = qc.finger_step_values(before_assembled, profile, n)
        af = qc.finger_step_values(vmd, profile, n)
        bm = qc.morph_step_series(before_assembled, n)
        am = qc.morph_step_series(vmd, n)
        bp = float(np.percentile(bf[1], 95)) if len(bf[1]) else 0.0
        ap = float(np.percentile(af[1], 95)) if len(af[1]) else 0.0
        if not (ap <= bp * 1.10 + 0.1 and am.max() <= bm.max() + qc.MORPH_ABS):
            res[:, FD.HAND_SLICE] = 0.0
            res[:, FD.FACE_SLICE] = 0.0
            ha_out = np.array(ha, copy=True)
            fw_out = np.array(fw, copy=True)
            corr_body = FD.apply_body_residual(body_vmd, model, profile, res)
            vmd = FD.assemble_vmd(corr_body, profile, ha_out, fw_out,
                                  n_frames=n)
            handface_fallback = True

    return {
        "vmd": vmd, "residual": res, "residual_pass1": res1,
        "hand_out": ha_out, "face_out": fw_out,
        "applied": bool(np.any(res != 0.0)),
        "segments_fallback": segments,
        "whole_clip_fallback": bool(whole),
        "handface_fallback": bool(handface_fallback),
        "residual_mean_abs": {name: float(np.abs(res[:, sl]).mean())
                              for name, sl in FD.residual_groups().items()},
        "features_shape": list(feats1.shape),
        "raw_residual_mean_abs": {
            name: float(np.abs(raw1[:, sl]).mean())
            for name, sl in FD.residual_groups().items()},
    }
