# -*- coding: utf-8 -*-
"""Canonical 2D reprojection: a bounded, non-learned arm correction.

The source evidence is ViTPose COCO-17 in full-image pixels (saved as
``preprocess/vitpose.pt`` next to the GVHMR npz, and now also inside new
npz files).  GVHMR's ``smpl_params_incam`` is already in camera coordinates:
projecting the SMPL-X COCO-17 body with ``K_fullimg`` gives ~20-60 px error on
R2, so the 2D signal is real.

What this module deliberately does **not** do is fit every frame's keypoint
residual to zero.  On R2 the front-view upper-arm direction residual is only
3-5 deg while the 3D arm is 19-25 deg away from GT; that error is mostly in
depth, and a per-frame 2D fit reduces pixel error to ~6 px by moving the arms
12-25 deg away from the GT pose (M2 experiment, 2026-10-25).  The fit here is
therefore:

* a **single constant swing** per arm joint over the clip (mode="const"),
  optionally per-frame with a strong prior (mode="perframe", expected to be
  rejected by the trust region on depth-ambiguous views);
* 2D residuals only at the elbows/wrists, confidence gated;
* a 3D prior that keeps the corrected arm close to the GVHMR arm, plus a
  temporal smoothness term;
* a **trust region**: the correction is accepted only if the 2D arm median
  improves by at least ``min_improve``, no joint moves more than
  ``max_delta_deg``, and the mean 3D displacement stays under
  ``max_disp_pct`` of body height.  Otherwise the npz is written unchanged
  and the report says ``accepted: false``.

The output is a corrected GVHMR npz; ``run_adapter.py`` consumes it as usual,
so the adapter and GVHMR are untouched.  The report always gives both the 2D
improvement and the 3D displacement, so the trade-off is explicit.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# SMPL-X body_pose triples for the arm chain (joint index - 1).
UPPER_IDX = (15, 16)          # L/R shoulder (upper arm)
FORE_IDX = (17, 18)           # L/R elbow (forearm)
COCO_ARM = (7, 8, 9, 10)      # L/R elbow, L/R wrist
COCO_OTHER = (11, 12, 13, 14, 15, 16)  # hips/knees/ankles for 2D alignment
BODY_HEIGHT_M = 1.7


def _joints(mode: str) -> Tuple[int, ...]:
    if mode == "upper":
        return UPPER_IDX
    if mode == "fore":
        return FORE_IDX
    if mode == "both":
        return UPPER_IDX + FORE_IDX
    raise ValueError(f"unknown joints mode: {mode}")


def _resolve_kp(npz_path: Path, kp_path: Optional[str], data) -> np.ndarray:
    if kp_path:
        p = Path(kp_path)
    elif "kp2d" in data.files:
        return np.asarray(data["kp2d"], dtype=np.float32)
    else:
        p = npz_path.parent / "preprocess" / "vitpose.pt"
    if not p.exists():
        raise FileNotFoundError(
            f"2D keypoints not found: {p}. Pass --kp, or re-run run_gvhmr.py "
            "so the npz contains kp2d.")
    import torch
    return torch.load(str(p), map_location="cpu").float().numpy()


def _sim_align(uv, kp_xy, w, mode):
    """Per-frame 2D alignment from the non-arm joints; None = raw pixels."""
    if mode == "raw":
        return None, None
    ww = w[..., None]
    sw = ww.sum(1).clip(min=1e-6)
    mu = (ww * uv).sum(1) / sw
    mk = (ww * kp_xy).sum(1) / sw
    if mode == "trans":
        s = np.ones(len(uv), dtype=np.float64)
        return s, mk - mu
    a = uv - mu[:, None, :]
    b = kp_xy - mk[:, None, :]
    num = (ww * a * b).sum((1, 2))
    den = (ww * a * a).sum((1, 2)).clip(min=1e-6)
    s = np.clip(num / den, 0.7, 1.4)
    return s, mk - s[:, None] * mu


def gate_accept(improve: float, max_delta_deg: float, mean_disp_pct: float,
                min_improve: float, delta_limit: float,
                disp_limit: float) -> Tuple[bool, Dict]:
    """Trust region for the 2D correction (pure function, unit tested)."""
    checks = {
        "2d_improve": {"value": round(float(improve), 4),
                       "min": float(min_improve),
                       "ok": bool(improve >= min_improve)},
        "delta_max_deg": {"value": round(float(max_delta_deg), 3),
                          "max": float(delta_limit),
                          "ok": bool(max_delta_deg <= delta_limit)},
        "disp3d_mean_pct_height": {"value": round(float(mean_disp_pct), 3),
                                   "max": float(disp_limit),
                                   "ok": bool(mean_disp_pct <= disp_limit)},
    }
    return all(v["ok"] for v in checks.values()), checks


def _project(J, K):
    import torch
    uv = torch.einsum("fij,fkj->fki", K, J)
    return uv[..., :2] / uv[..., 2:3]


def _arm_errors(uv, kp, s, t, scale):
    """(F,4) pixel errors at the COCO elbows/wrists, aligned if s/t given."""
    uv_arm = uv[:, list(COCO_ARM)]
    if s is not None:
        uv_arm = s[:, None, None] * uv_arm + t[:, None, :]
    return (uv_arm - kp[:, list(COCO_ARM), :2]).norm(dim=-1), uv_arm


def fit(npz_path: str, kp_path: Optional[str] = None, joints: str = "upper",
        align: str = "raw", mode: str = "const",
        lam3d: float = 0.1, lampose: float = 0.01, lamtemp: float = 0.2,
        max_iter: int = 120, min_improve: float = 0.05,
        max_delta_deg: float = 15.0, max_disp_pct: float = 2.5,
        height_m: float = BODY_HEIGHT_M, seed: int = 0) -> Dict:
    """Fit and return a report plus the corrected 45-value arm slice.

    ``body_pose`` is the canonical local rotation; the same corrected values
    are written to both ``smpl_params_global`` and ``smpl_params_incam``
    (GVHMR predicts identical body_pose for both; verified on R2).
    """
    import sys
    import torch
    from hmr4d.utils.smplx_utils import make_smplx

    npz_path = Path(npz_path)
    d = np.load(npz_path)
    F = len(d["smpl_params_incam/body_pose"])
    body = torch.tensor(d["smpl_params_incam/body_pose"], dtype=torch.float32)
    go = torch.tensor(d["smpl_params_incam/global_orient"], dtype=torch.float32)
    tr = torch.tensor(d["smpl_params_incam/transl"], dtype=torch.float32)
    betas = torch.tensor(d["smpl_params_incam/betas"], dtype=torch.float32)
    K = torch.tensor(d["K_fullimg"], dtype=torch.float32)
    kp_np = _resolve_kp(npz_path, kp_path, d)
    kp = torch.tensor(kp_np, dtype=torch.float32)
    bbx = torch.tensor(d["bbx_xys"], dtype=torch.float32)
    scale = bbx[:, 2:3] / 2.0

    model = make_smplx("supermotion_coco17").eval()
    for p in model.parameters():
        p.requires_grad_(False)

    idx = _joints(joints)
    w0 = (kp[..., 2] > 0.3).float()

    def project_uv(J):
        return _project(J, K)

    def align_t(uv):
        with torch.no_grad():
            uv_other = uv[:, list(COCO_OTHER)].detach().numpy()
            kp_other = kp[:, list(COCO_OTHER), :2].detach().numpy()
            w_other = w0[:, list(COCO_OTHER)].detach().numpy()
            s_np, t_np = _sim_align(uv_other, kp_other, w_other, align)
        if s_np is None:
            return None, None
        return (torch.tensor(s_np, dtype=torch.float32),
                torch.tensor(t_np, dtype=torch.float32))

    if mode == "const":
        param = torch.zeros(1, len(idx), 3, requires_grad=True)

        def delta():
            return param.expand(F, -1, -1)
    elif mode == "perframe":
        param = torch.zeros(F, len(idx), 3, requires_grad=True)

        def delta():
            return param
    else:
        raise ValueError(f"unknown mode: {mode}")

    def forward(dl):
        bp = body.clone()
        for k, i in enumerate(idx):
            bp[:, i * 3:i * 3 + 3] += dl[:, k]
        return model(body_pose=bp, betas=betas, global_orient=go, transl=tr)

    def loss_fn():
        dl = delta()
        J = forward(dl)
        uv = project_uv(J)
        s, t = align_t(uv)
        err, uv_arm = _arm_errors(uv, kp, s, t, scale)
        w = w0[:, list(COCO_ARM)]
        r = err / scale
        ar = torch.abs(r)
        hub = torch.where(ar < 0.03, 0.5 * r * r, 0.03 * (ar - 0.015))
        lpos = (w * hub).sum() / w.sum().clamp(min=1)
        with torch.no_grad():
            J0 = model(body_pose=body, betas=betas, global_orient=go, transl=tr)
        l3 = ((J[:, list(COCO_ARM)] - J0[:, list(COCO_ARM)]) ** 2).sum(-1).mean() / height_m ** 2
        lp = (dl ** 2).mean()
        lt = ((dl[2:] - 2 * dl[1:-1] + dl[:-2]) ** 2).mean() if F > 2 else torch.tensor(0.0)
        return lpos + lam3d * l3 + lampose * lp + lamtemp * lt, lpos, l3

    opt = torch.optim.LBFGS([param], lr=0.5, max_iter=int(max_iter),
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        total, _, _ = loss_fn()
        total.backward()
        return total
    opt.step(closure)

    with torch.no_grad():
        dl = delta()
        J = forward(dl)
        uv = project_uv(J)
        s, t = align_t(uv)
        err_after, _ = _arm_errors(uv, kp, s, t, scale)
        J0 = model(body_pose=body, betas=betas, global_orient=go, transl=tr)
        uv0 = project_uv(J0)
        s0, t0 = align_t(uv0)
        err_before, _ = _arm_errors(uv0, kp, s0, t0, scale)
        disp = (J[:, list(COCO_ARM)] - J0[:, list(COCO_ARM)]).norm(dim=-1)
        delta_deg = torch.rad2deg(dl.norm(dim=-1))
        bp = body.clone()
        for k, i in enumerate(idx):
            bp[:, i * 3:i * 3 + 3] += dl[:, k]

    before_med = float(err_before.median())
    after_med = float(err_after.median())
    improve = (before_med - after_med) / max(before_med, 1e-9)
    max_delta = float(delta_deg.max())
    mean_disp_pct = 100.0 * float(disp.mean()) / float(height_m)
    max_disp_pct_val = 100.0 * float(disp.max()) / float(height_m)
    accepted, checks = gate_accept(improve, max_delta, mean_disp_pct,
                                   min_improve, max_delta_deg, max_disp_pct)
    report = {
        "npz": str(npz_path),
        "kp": "npz:kp2d" if (kp_path is None and "kp2d" in d.files)
              else str(kp_path or (npz_path.parent / "preprocess" / "vitpose.pt")),
        "frames": int(F),
        "mode": mode, "joints": joints, "align": align,
        "arm_coco": list(COCO_ARM),
        "before_2d_median_px": round(before_med, 3),
        "after_2d_median_px": round(after_med, 3),
        "before_2d_mean_px": round(float(err_before.mean()), 3),
        "after_2d_mean_px": round(float(err_after.mean()), 3),
        "improve_median": round(float(improve), 4),
        "delta_deg_mean": [round(float(v), 3) for v in
                           delta_deg.mean(0).tolist()],
        "delta_deg_max": [round(float(v), 3) for v in
                          delta_deg.max(0).values.tolist()],
        "delta_max_deg": round(max_delta, 3),
        "disp3d_mean_pct_height": round(mean_disp_pct, 3),
        "disp3d_max_pct_height": round(max_disp_pct_val, 3),
        "gate": {"min_improve": min_improve, "max_delta_deg": max_delta_deg,
                 "max_disp_pct": max_disp_pct, "checks": checks},
        "accepted": bool(accepted),
        "note": ("constant swing; 2D cannot resolve the arm depth ambiguity "
                 "when the view looks along the arm plane"),
    }
    if accepted:
        corrected = bp
    else:
        corrected = body
    d.close()
    return {"report": report, "body_pose": corrected,
            "delta": dl.detach().cpu().numpy(),
            "err_before": err_before.detach().cpu().numpy(),
            "err_after": err_after.detach().cpu().numpy(),
            "height_m": float(height_m)}


def write_npz(src: str, dst: str, body_pose, delta, accepted: bool,
              report: Dict, err_before=None, err_after=None) -> None:
    """Copy every array, replace both body_pose slices, add vtv_2d_fit/*."""
    import numpy as np
    src = Path(src)
    dst = Path(dst)
    with np.load(src) as d:
        arrays = {k: d[k] for k in d.files}
    bp = np.asarray(body_pose.detach().cpu() if hasattr(body_pose, "detach")
                    else body_pose, dtype=np.float32)
    arrays["smpl_params_global/body_pose"] = bp
    arrays["smpl_params_incam/body_pose"] = bp
    arrays["vtv_2d_fit/delta"] = np.asarray(delta, dtype=np.float32)
    arrays["vtv_2d_fit/accepted"] = np.array([1 if accepted else 0], dtype=np.int8)
    arrays["vtv_2d_fit/report_json"] = np.array(
        [json.dumps(report, ensure_ascii=False)])
    if err_before is not None:
        arrays["vtv_2d_fit/err_before"] = np.asarray(err_before, dtype=np.float32)
    if err_after is not None:
        arrays["vtv_2d_fit/err_after"] = np.asarray(err_after, dtype=np.float32)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **arrays)
