# -*- coding: utf-8 -*-
"""Multi-view arm upper bound (M2.3 experiment, not a production stage).

The question this answers: *if the depth of the arms were observable, how close
could the arms get?*  M2 showed the single-view 2D fit is depth-ambiguous: the
front-view upper-arm 2D direction residual is 3-5 deg while the 3D error is
19-25 deg, so a per-frame 2D fit moves the arms away from GT while the pixels
improve (34.97 -> 5.73 px, local angle 40.56 -> 41.66).

Here three synthetic cameras see the same R2 take.  Each view keeps its own
GVHMR camera (``K_fullimg``), root (``global_orient``/``transl``) and 2D
keypoints, and **one shared canonical arm ``body_pose``** is fitted against all
of them at once (``body_pose`` is a local joint rotation, so it is
camera-independent; only the root differs per view).  The fit is a single
constant swing per joint, exactly like the M2 stage, so the comparison is
apples-to-apples: same prior, same trust region, more evidence.

Known limitation, reported not hidden: each view's GVHMR root has its own
global-orientation error (M1.5: 13.4 / 35 / 84.4 deg for front / 3-4 / side).
A shared local pose cannot fix a per-view root error, so the fitted pose is
dragged by the view with the worst root.  ``--views`` lets the experiment be
run on front only, front+3-4, or all three, which is what makes that visible.

Nothing here is a production stage; it writes a *separate* npz per view so the
adapter and the closed loop can score it like any other candidate.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

UPPER_IDX = (15, 16)
FORE_IDX = (17, 18)
COCO_ARM = (7, 8, 9, 10)
COCO_OTHER = (11, 12, 13, 14, 15, 16)
BODY_HEIGHT_M = 1.7


def resolve_kp(npz_path: Path, kp_path: Optional[str]) -> np.ndarray:
    if kp_path:
        p = Path(kp_path)
        return np.load(p) if p.suffix == ".npy" else _torch_load(p)
    d = np.load(npz_path)
    if "kp2d" in d.files:
        return np.asarray(d["kp2d"], dtype=np.float32)
    return _torch_load(npz_path.parent / "preprocess" / "vitpose.pt")


def _torch_load(p: Path) -> np.ndarray:
    import torch
    t = torch.load(str(p), map_location="cpu")
    return (t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)).astype(np.float32)


def _joints(mode: str) -> Tuple[int, ...]:
    if mode == "upper":
        return UPPER_IDX
    if mode == "both":
        return UPPER_IDX + FORE_IDX
    if mode == "fore":
        return FORE_IDX
    raise ValueError(f"unknown joints mode: {mode}")


def _project(J, K):
    uv = np.einsum("fij,fkj->fki", K, J)
    return uv[..., :2] / np.maximum(uv[..., 2:3], 1e-6)


def _sim_align(uv, kp_xy, w):
    """Per-frame 2D scale+translation from the non-arm joints (None = raw)."""
    ww = w[..., None]
    sw = np.maximum(ww.sum(1), 1e-6)
    mu = (ww * uv).sum(1) / sw
    mk = (ww * kp_xy).sum(1) / sw
    a = uv - mu[:, None, :]
    b = kp_xy - mk[:, None, :]
    num = (ww * a * b).sum((1, 2))
    den = np.maximum((ww * a * a).sum((1, 2)), 1e-6)
    s = np.clip(num / den, 0.7, 1.4)
    return s, mk - s[:, None] * mu


def fit(views: Dict[str, str], kp: Optional[Dict[str, str]] = None,
        joints: str = "upper", mode: str = "perframe",
        lam3d: float = 0.1, lampose: float = 0.01,
        lamtemp: float = 0.0, max_iter: int = 120,
        min_improve: float = 0.03, max_delta_deg: float = 15.0,
        max_disp_pct: float = 2.5, height_m: float = BODY_HEIGHT_M,
        align: str = "raw") -> Dict:
    """Joint constant arm swing across ``views`` (name -> npz path)."""
    import sys
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shim"))
    from hmr4d.utils.smplx_utils import make_smplx

    names = list(views)
    data = {}
    for n in names:
        p = Path(views[n])
        d = np.load(p)
        kp_np = resolve_kp(p, (kp or {}).get(n))
        data[n] = {
            "body": np.asarray(d["smpl_params_incam/body_pose"], dtype=np.float32),
            "go": np.asarray(d["smpl_params_incam/global_orient"], dtype=np.float32),
            "tr": np.asarray(d["smpl_params_incam/transl"], dtype=np.float32),
            "betas": np.asarray(d["smpl_params_incam/betas"], dtype=np.float32),
            "K": np.asarray(d["K_fullimg"], dtype=np.float32),
            "kp": kp_np,
            "bbx": np.asarray(d["bbx_xys"], dtype=np.float32),
        }
        d.close()
    F = len(data[names[0]]["body"])
    for n in names:
        if len(data[n]["body"]) != F:
            raise ValueError(f"frame count mismatch: {n}")
    idx = _joints(joints)
    model = make_smplx("supermotion_coco17").eval()
    for p in model.parameters():
        p.requires_grad_(False)

    base = {}
    with torch.no_grad():
        for n, v in data.items():
            bp = torch.tensor(v["body"], dtype=torch.float32)
            J0 = model(body_pose=bp, betas=torch.tensor(v["betas"]),
                       global_orient=torch.tensor(v["go"]),
                       transl=torch.tensor(v["tr"]))
            base[n] = J0.detach()
    cols = [i * 3 + k for i in idx for k in range(3)]
    if mode == "const":
        # one shared *correction* on top of each view's own per-frame arms
        init = np.zeros((1, len(idx), 3), dtype=np.float32)
    elif mode == "perframe":
        # one shared *absolute* per-frame arm pose: the true multiview consensus
        init = np.mean([data[n]["body"][:, cols] for n in names],
                       axis=0).astype(np.float32).reshape(F, len(idx), 3)
    else:
        raise ValueError(f"unknown mode: {mode}")
    param = torch.tensor(init, dtype=torch.float32).requires_grad_(True)

    def forward(dl):
        out = {}
        for n, v in data.items():
            bp = torch.tensor(v["body"], dtype=torch.float32)
            for k, i in enumerate(idx):
                if mode == "const":
                    bp[:, i * 3:i * 3 + 3] = bp[:, i * 3:i * 3 + 3] + dl[0, k]
                else:
                    bp[:, i * 3:i * 3 + 3] = dl[:, k]
            out[n] = model(body_pose=bp, betas=torch.tensor(v["betas"]),
                           global_orient=torch.tensor(v["go"]),
                           transl=torch.tensor(v["tr"]))
        return out

    def view_err(J, v):
        uv = _project(J.detach().numpy(), v["K"])
        kp2 = v["kp"]
        w0 = (kp2[..., 2] > 0.3).astype(np.float32)
        w_other = w0[:, list(COCO_OTHER)]
        if align == "raw":
            s = np.ones(len(uv), dtype=np.float64)
            t = np.zeros((len(uv), 2), dtype=np.float64)
        else:
            s, t = _sim_align(uv[:, list(COCO_OTHER)],
                              kp2[:, list(COCO_OTHER), :2], w_other)
        uv_arm = s[:, None, None] * uv[:, list(COCO_ARM)] + t[:, None, :]
        err = np.linalg.norm(uv_arm - kp2[:, list(COCO_ARM), :2], axis=-1)
        return err, w0[:, list(COCO_ARM)], v["bbx"][:, 2:3] / 2.0

    def loss_fn():
        dl = param
        Js = forward(dl)
        total = 0.0
        for n, v in data.items():
            J = Js[n]
            Kt = torch.tensor(v["K"])
            uv = torch.einsum("fij,fkj->fki", Kt, J)
            uv = uv[..., :2] / uv[..., 2:3].clamp(min=1e-6)
            kp2 = torch.tensor(v["kp"])
            w0 = (kp2[..., 2] > 0.3).float()
            if align == "raw":
                s_np = np.ones(len(uv), dtype=np.float32)
                t_np = np.zeros((len(uv), 2), dtype=np.float32)
            else:
                with torch.no_grad():
                    uv_np = uv.detach().numpy()
                    s_np, t_np = _sim_align(uv_np[:, list(COCO_OTHER)],
                                            v["kp"][:, list(COCO_OTHER), :2],
                                            w0[:, list(COCO_OTHER)].numpy())
            s = torch.tensor(s_np, dtype=torch.float32)
            t = torch.tensor(t_np, dtype=torch.float32)
            uv_arm = s[:, None, None] * uv[:, list(COCO_ARM)] + t[:, None, :]
            r = (uv_arm - kp2[:, list(COCO_ARM), :2]).norm(dim=-1)
            r = r / torch.tensor(v["bbx"][:, 2:3] / 2.0)
            w = w0[:, list(COCO_ARM)]
            ar = torch.abs(r)
            hub = torch.where(ar < 0.03, 0.5 * r * r, 0.03 * (ar - 0.015))
            l2 = (w * hub).sum() / w.sum().clamp(min=1)
            d3 = ((J[:, list(COCO_ARM)] - base[n][:, list(COCO_ARM)]) ** 2).sum(-1).mean()
            total = total + l2 + lam3d * d3 / (height_m ** 2)
        if mode == "const":
            total = total + lampose * (dl[0] ** 2).mean()
        if lamtemp > 0 and F > 2:
            total = total + lamtemp * ((dl[2:] - 2 * dl[1:-1] + dl[:-2]) ** 2).mean()
        return total / len(names)

    opt = torch.optim.LBFGS([param], lr=0.5, max_iter=int(max_iter),
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        l = loss_fn()
        l.backward()
        return l
    opt.step(closure)

    with torch.no_grad():
        dl = param.detach()
        Js = forward(dl)
    report = {"views": names, "joints": joints, "frames": int(F),
              "fit": {"mode": mode, "lam3d": lam3d, "lampose": lampose,
                      "lamtemp": lamtemp, "align": align}, "per_view": {}}
    ok_any = True
    for n in names:
        v = data[n]
        err_b, w, scale = view_err(base[n], v)
        err_a, _, _ = view_err(Js[n], v)
        # per-view 3D displacement and per-joint correction
        disp = np.linalg.norm(Js[n][:, list(COCO_ARM)].numpy()
                              - base[n][:, list(COCO_ARM)].numpy(), axis=-1)
        delta_deg = []
        for k, i in enumerate(idx):
            if mode == "const":
                # the correction itself: axis-angle magnitude of the shared delta
                delta_deg.append(np.full(F, np.degrees(
                    np.linalg.norm(dl[0, k].numpy()))))
                continue
            aa0 = v["body"][:, i * 3:i * 3 + 3]
            aa1 = dl[:, k].numpy()
            R0 = _aa_to_R(aa0)
            R1 = _aa_to_R(aa1)
            tr = np.trace(R0.transpose(0, 2, 1) @ R1, axis1=1, axis2=2)
            delta_deg.append(np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))))
        delta_deg = np.concatenate(delta_deg)
        b_med = float(np.median(err_b))
        a_med = float(np.median(err_a))
        improve = (b_med - a_med) / max(b_med, 1e-9)
        checks = {
            "2d_improve": {"value": round(improve, 4), "min": min_improve,
                           "ok": bool(improve >= min_improve)},
            "delta_max_deg": {"value": round(float(delta_deg.max()), 3),
                              "max": max_delta_deg,
                              "ok": bool(delta_deg.max() <= max_delta_deg)},
            "disp3d_mean_pct_height": {
                "value": round(100.0 * float(disp.mean()) / height_m, 3),
                "max": max_disp_pct,
                "ok": bool(100.0 * float(disp.mean()) / height_m <= max_disp_pct)},
        }
        ok = all(c["ok"] for c in checks.values())
        ok_any = ok_any and ok
        report["per_view"][n] = {
            "npz": str(views[n]),
            "before_2d_median_px": round(b_med, 3),
            "after_2d_median_px": round(a_med, 3),
            "improve_median": round(improve, 4),
            "delta_max_deg": round(float(delta_deg.max()), 3),
            "disp3d_mean_pct_height": round(100.0 * float(disp.mean()) / height_m, 3),
            "disp3d_max_pct_height": round(100.0 * float(disp.max()) / height_m, 3),
            "checks": checks, "ok": bool(ok),
        }
    report["accepted"] = bool(ok_any)
    report["param"] = (dl[0] if mode == "const" else dl).numpy().round(5).tolist()
    report["note"] = ("shared arm fit on the listed views; each view keeps its own "
                      "root, so a per-view global-orientation error drags the "
                      "shared pose (M1.5: 13/35/84 deg)")
    return {"report": report, "param": (dl[0] if mode == "const" else dl).numpy(),
            "joints": idx, "frames": int(F), "mode": mode}


def _aa_to_R(aa: np.ndarray) -> np.ndarray:
    """Axis-angle (..., 3) -> rotation matrices (..., 3, 3)."""
    aa = np.asarray(aa, dtype=np.float64)
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = aa / np.maximum(th, 1e-9)
    K = np.zeros(aa.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    I = np.eye(3)
    s = np.sin(th)[..., None]
    c = (1.0 - np.cos(th))[..., None]
    return I + s * K + c * (K @ K)


def write_view_npz(src: str, dst: str, param: np.ndarray,
                   joints: Sequence[int], report: Dict,
                   mode: str = "perframe") -> None:
    """Write the shared arm slice into both body_pose copies of one view."""
    src, dst = Path(src), Path(dst)
    with np.load(src) as d:
        arrays = {k: d[k] for k in d.files}
    bp = np.array(arrays["smpl_params_incam/body_pose"], dtype=np.float32)
    for k, i in enumerate(joints):
        if mode == "const":
            bp[:, i * 3:i * 3 + 3] = (bp[:, i * 3:i * 3 + 3]
                                      + np.asarray(param[k], dtype=np.float32))
        else:
            bp[:, i * 3:i * 3 + 3] = np.asarray(param[:, k], dtype=np.float32)
    arrays["smpl_params_incam/body_pose"] = bp
    arrays["smpl_params_global/body_pose"] = bp.copy()
    arrays["vtv_multiview/report_json"] = np.array(
        [json.dumps(report, ensure_ascii=False)])
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **arrays)
