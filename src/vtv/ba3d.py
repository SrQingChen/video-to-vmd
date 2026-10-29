# -*- coding: utf-8 -*-
"""M2.4c multi-view bundle adjustment on the *target* PMX body.

Why this module exists
----------------------
M2.4b triangulated the 3D arm **direction** from three calibrated views.  The
direction target itself carried 15-25 deg of error (keypoint/bone offsets,
noise, and the line-to-direction sign ambiguity), and applying a GT
torso-relative direction to the **wrong predicted chest** added penetration.
M2.4c therefore asks a different question: if the torso and the arms are
estimated **jointly** as one skeleton, with the target PMX bone lengths, the
real camera extrinsics and a per-view root, can the multi-view 2D evidence
recover the torso-relative arm pose?

This is a *multi-view upper bound*: it needs three calibrated cameras and is
not a single-view prior.  The single-view (T5) path must not depend on it.

Model
-----
We use the canonical 23-joint body that the M1 adapter already speaks
(``vtv.canonical_body``), instantiated with the **target PMX rest joints**.
The forward kinematics is the same convention as ``adapter.fk_positions``:

    J_pelvis = root_pos
    J_j      = J_parent + qrot(G_parent, rest_j - rest_parent)
    G_j      = G_parent * L_j * exp(delta_j)

``L_j`` is the baseline local rotation (from the chosen initialization VMD),
``delta_j`` is the optimized local correction.  Bone lengths are fixed by the
PMX rest skeleton, so the solver cannot shorten an arm to fit the image.

The legs are **not** optimized: the pelvis and the whole leg chain keep their
baseline values, so contact recall / foot placement cannot be traded for an
arm fit.  The torso (spine/spine1/chest/neck), clavicles, upper arms,
forearms and hands are optimized.  The 2D leg observations still constrain the
root, but they do not move the legs.

The shared body pose is evaluated in all views; each view gets its own small
rigid root correction (``R_v``, ``t_v``) to absorb calibration/residual error.
Those corrections are heavily regularized, and they are *not* written into the
output VMD: the output is the shared pose.  A view whose root correction wants
to grow is a diagnostic that the shared pose/camera cannot explain that view.
"""
from __future__ import annotations

import copy
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import canonical_body as CB
from .adapter import (canonical_from_pmx, fk_positions, q_conj, q_from_axis_angle,
                      qmul, qrot)
from .cleanup import dense_tracks, eval_tracks
from .mmd.mathutil import qinv

# Canonical joints that are optimized.  The pelvis is deliberately excluded:
# rotating it would rotate the legs and change contact.  The head is excluded
# (it follows the neck); the legs are excluded for the same contact reason.
DEFAULT_OPT_JOINTS = (
    "spine", "spine1", "chest", "neck",
    "clavicle_l", "clavicle_r",
    "upperarm_l", "upperarm_r",
    "forearm_l", "forearm_r",
    "hand_l", "hand_r",
)

# COCO-17 keypoints used by the 2D term (index, confidence in channel 2).
# Arms are weighted higher: they are the quantity this stage is trying to
# estimate; the legs are only there to constrain the root.
VIEW_JOINTS = {
    "upperarm_l": (5, 2.0), "upperarm_r": (6, 2.0),
    "forearm_l": (7, 2.0), "forearm_r": (8, 2.0),
    "hand_l": (9, 1.5), "hand_r": (10, 1.5),
    "thigh_l": (11, 0.8), "thigh_r": (12, 0.8),
    "shin_l": (13, 0.8), "shin_r": (14, 0.8),
    "foot_l": (15, 0.8), "foot_r": (16, 0.8),
    "neck": (1, 0.5),
}


# --------------------------------------------------------------- torch helpers

def _torch():
    import torch  # local import: numpy-only callers should not need torch
    return torch


def qmul_t(a, b):
    """Hamilton product for (...,4) xyzw quaternions."""
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return _torch().stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dim=-1)


def qconj_t(q):
    return _torch().cat([-q[..., :3], q[..., 3:4]], dim=-1)


def qnorm_t(q):
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def qrot_t(q, v):
    """Rotate vector(s) ``v`` (...,3) by quaternion(s) ``q`` (...,4)."""
    torch = _torch()
    xyz = q[..., :3]
    w = q[..., 3:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def qexp_t(axis_angle):
    """Axis-angle (...,3) radians -> quaternion (...,4), stable at 0."""
    torch = _torch()
    ang = axis_angle.norm(dim=-1, keepdim=True)
    half = 0.5 * ang
    small = ang < 1e-8
    scale = torch.where(small, 0.5 - ang * ang / 48.0,
                        torch.sin(half) / ang.clamp_min(1e-12))
    xyz = axis_angle * scale
    w = torch.where(small, 1.0 - ang * ang / 8.0, torch.cos(half))
    return torch.cat([xyz, w], dim=-1)


def _as_t(x, device, dtype=None):
    torch = _torch()
    return torch.as_tensor(np.asarray(x, dtype=np.float64 if dtype is None else dtype),
                           dtype=dtype or torch.float64, device=device)


# --------------------------------------------------------------- body model

class PmxBodyBA:
    """Differentiable canonical PMX body used by the multi-view BA."""

    def __init__(self, motion: Dict, opt_joints: Sequence[str] = DEFAULT_OPT_JOINTS,
                 device: str = "cpu"):
        torch = _torch()
        self.motion = motion
        self.device = device
        self.opt_joints = list(opt_joints)
        self.joints = list(CB.ADAPTER_JOINTS)
        self.rest = {j: _as_t(motion["rest_joints"][j], device)
                     for j in self.joints if j in motion["rest_joints"]}
        self.parent = {j: CB.CANONICAL_PARENT.get(j) for j in self.joints}
        self.topo = self._topo()
        self.G0 = {j: _as_t(motion["world_rot"][j], device)
                   for j in self.joints if j in motion["world_rot"]}
        self.root0 = _as_t(motion["root_pos"], device)
        # Local rotation of every joint relative to its parent (pelvis is
        # relative to the fixed canonical root and is not optimized).
        self.L0 = {}
        for j in self.joints:
            if j == "root" or j not in self.G0:
                continue
            p = self.parent.get(j)
            if p in self.G0 and j != "pelvis":
                self.L0[j] = qnorm_t(qmul_t(qconj_t(self.G0[p]), self.G0[j]))
            else:
                self.L0[j] = qnorm_t(self.G0[j])

    def _topo(self) -> List[str]:
        out, seen = [], set()

        def visit(j):
            if j in seen or j not in self.joints:
                return
            p = self.parent.get(j)
            if p:
                visit(p)
            seen.add(j)
            out.append(j)

        for j in self.joints:
            visit(j)
        return out

    def forward(self, delta: Dict[str, object], dtrans):
        """Return world rotations and joint positions for the shared pose.

        ``delta`` maps joint -> (F,3) axis-angle local correction; missing
        joints are unchanged.  ``dtrans`` is the (F,3) pelvis translation
        correction (shared).
        """
        torch = _torch()
        G: Dict[str, object] = {}
        J: Dict[str, object] = {}
        zero = torch.zeros((self.root0.shape[0], 3), dtype=self.root0.dtype,
                           device=self.device)
        G["root"] = self.G0.get("root", _as_t([0.0, 0.0, 0.0, 1.0], self.device))
        for j in self.topo:
            if j not in self.G0:
                continue
            p = self.parent.get(j)
            if j == "pelvis" or p is None or p not in G:
                base = self.G0[j]
            else:
                base = qmul_t(G[p], self.L0.get(j, _as_t([0.0, 0.0, 0.0, 1.0],
                                                          self.device)))
            d = delta.get(j)
            if d is not None:
                base = qmul_t(base, qexp_t(d))
            G[j] = qnorm_t(base)
        J["pelvis"] = self.root0 + dtrans
        for j in self.topo:
            if j == "pelvis" or j not in G:
                continue
            p = self.parent.get(j)
            if p not in J or j not in self.rest or p not in self.rest:
                continue
            J[j] = J[p] + qrot_t(G[p], (self.rest[j] - self.rest[p]).expand_as(J[p]))
        return G, J


# --------------------------------------------------------------- cameras / 2D

def pmx_camera(cam: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a scene camera to the PMX world.

    ``arm3d.SCENE_TO_PMX`` is a reflection ``M`` (scene x,y,z -> pmx x,z,y).
    A camera->scene rotation ``R_s`` becomes camera->PMX ``M R_s``; the
    camera centre maps as ``M C``.  A PMX point projects with
    ``(P - C) @ (M R_s)``.
    """
    from .arm3d import SCENE_TO_PMX
    M = SCENE_TO_PMX
    return M @ np.asarray(cam["R"], dtype=np.float64), M @ np.asarray(cam["C"], dtype=np.float64)


def load_view(path_meta: str, path_kp: str) -> Dict:
    from .arm3d import load_blender_camera, load_keypoints
    cam = load_blender_camera(path_meta)
    kp = load_keypoints(path_kp)
    R, C = pmx_camera(cam)
    return {"cam": cam, "R": R, "C": C, "kp": kp,
            "fx": cam["fx"], "fy": cam["fy"], "cx": cam["cx"], "cy": cam["cy"],
            "meta": path_meta, "kp_path": path_kp}


def project_torch(P, view: Dict):
    """Project PMX points (F,K,3) to pixels; returns (F,K,2)."""
    torch = _torch()
    R = _as_t(view["R"], P.device)
    C = _as_t(view["C"], P.device)
    # R is camera->PMX, so camera coordinates are R^T (P-C); for row vectors
    # that is (P-C) @ R.
    Pc = torch.einsum("ji,fkj->fki", R, P - C)
    z = Pc[..., 2].clamp_min(1e-6)
    u = view["fx"] * Pc[..., 0] / z + view["cx"]
    v = view["fy"] * Pc[..., 1] / z + view["cy"]
    return torch.stack([u, v], dim=-1)


def joint_observations(view: Dict, frames: int):
    """Return canonical-name -> (obs (F,2), weight (F,)) for one view."""
    torch = _torch()
    kp = np.asarray(view["kp"], dtype=np.float64)
    F = min(frames, len(kp))
    obs, w = {}, {}
    for name, (idx, jw) in VIEW_JOINTS.items():
        o = np.zeros((frames, 2), dtype=np.float64)
        ww = np.zeros(frames, dtype=np.float64)
        o[:F] = kp[:F, idx, :2]
        conf = np.clip(kp[:F, idx, 2], 0.0, 1.0)
        ww[:F] = np.where(conf > 0.2, conf * jw, 0.0)
        obs[name] = torch.as_tensor(o, dtype=torch.float64, device=view.get("device", "cpu"))
        w[name] = torch.as_tensor(ww, dtype=torch.float64, device=view.get("device", "cpu"))
    return obs, w


# --------------------------------------------------------------- optimization

def _point_seg_dist(x, a, b):
    """Point-to-segment distance, broadcast over leading dims."""
    torch = _torch()
    ab = b - a
    den = (ab * ab).sum(-1, keepdim=True).clamp_min(1e-12)
    t = ((x - a) * ab).sum(-1, keepdim=True) / den
    t = t.clamp(0.0, 1.0)
    return (x - (a + t * ab)).norm(dim=-1)


def _segment_samples(p, q, n: int = 5):
    torch = _torch()
    ts = torch.linspace(0.0, 1.0, n, dtype=p.dtype, device=p.device)
    return p[..., None, :] + ts.view(1, 1, n, 1) * (q - p)[..., None, :]


def _collision_proxy(J, J0, margin: float = 0.01):
    """Baseline-relative arm/torso and arm/arm segment-distance penalty.

    The PMX capsule gate is not differentiable.  This proxy samples each bone
    segment and penalizes a candidate that brings an arm segment closer to a
    torso or opposite-arm segment than the *initialization* did.  It cannot
    certify the capsule gate, but it targets the specific failure M2.4c found:
    a 2D fit that pushes a forearm into spine1/chest.  The distance is in
    metres (the canonical skeleton is metric).
    """
    torch = _torch()
    torso = [("pelvis", "spine"), ("spine", "spine1"), ("spine1", "chest"),
             ("chest", "neck"), ("neck", "head")]
    arms = [("upperarm_l", "forearm_l"), ("forearm_l", "hand_l"),
            ("upperarm_r", "forearm_r"), ("forearm_r", "hand_r")]
    pen = 0.0
    count = 0
    # arm -> torso
    for a0, a1 in arms:
        if a0 not in J or a1 not in J:
            continue
        xa = _segment_samples(J[a0], J[a1])
        xa0 = _segment_samples(J0[a0], J0[a1])
        for t0, t1 in torso:
            if t0 not in J or t1 not in J:
                continue
            # pairwise point-segment distances, min over sampled arm points
            xb = J[t1] - J[t0]
            den = (xb * xb).sum(-1, keepdim=True).clamp_min(1e-12)
            t = ((xa - J[t0][:, None, :]) * xb[:, None, :]).sum(-1, keepdim=True) / den[:, None, :]
            t = t.clamp(0.0, 1.0)
            cp = J[t0][:, None, :] + t * xb[:, None, :]
            d = (xa - cp).norm(dim=-1).min(dim=-1).values
            xb0 = J0[t1] - J0[t0]
            den0 = (xb0 * xb0).sum(-1, keepdim=True).clamp_min(1e-12)
            t0v = ((xa0 - J0[t0][:, None, :]) * xb0[:, None, :]).sum(-1, keepdim=True) / den0[:, None, :]
            t0v = t0v.clamp(0.0, 1.0)
            cp0 = J0[t0][:, None, :] + t0v * xb0[:, None, :]
            d0 = (xa0 - cp0).norm(dim=-1).min(dim=-1).values
            pen = pen + torch.relu(d0 - d - margin).pow(2).mean()
            count += 1
    # left arm -> right arm
    left = [p for p in arms if p[0].endswith("_l")]
    right = [p for p in arms if p[0].endswith("_r")]
    for a0, a1 in left:
        if a0 not in J or a1 not in J:
            continue
        xa = _segment_samples(J[a0], J[a1])
        xa0 = _segment_samples(J0[a0], J0[a1])
        for b0, b1 in right:
            if b0 not in J or b1 not in J:
                continue
            xb = J[b1] - J[b0]
            den = (xb * xb).sum(-1, keepdim=True).clamp_min(1e-12)
            t = ((xa - J[b0][:, None, :]) * xb[:, None, :]).sum(-1, keepdim=True) / den[:, None, :]
            t = t.clamp(0.0, 1.0)
            cp = J[b0][:, None, :] + t * xb[:, None, :]
            d = (xa - cp).norm(dim=-1).min(dim=-1).values
            xb0 = J0[b1] - J0[b0]
            den0 = (xb0 * xb0).sum(-1, keepdim=True).clamp_min(1e-12)
            t0v = ((xa0 - J0[b0][:, None, :]) * xb0[:, None, :]).sum(-1, keepdim=True) / den0[:, None, :]
            t0v = t0v.clamp(0.0, 1.0)
            cp0 = J0[b0][:, None, :] + t0v * xb0[:, None, :]
            d0 = (xa0 - cp0).norm(dim=-1).min(dim=-1).values
            pen = pen + torch.relu(d0 - d - margin).pow(2).mean()
            count += 1
    return pen / max(count, 1)


def optimize_multiview(motion: Dict, views: Dict[str, Dict], *,
                       opt_joints: Sequence[str] = DEFAULT_OPT_JOINTS,
                       iters: int = 300, lr: float = 0.02,
                       prior_delta: float = 0.35,
                       prior_temporal: float = 1.5,
                       prior_root_rot: float = 3.0,
                       prior_root_trans: float = 20.0,
                       huber_px: float = 20.0,
                       w_2d: float = 1.0,
                       prior_collision: float = 0.0,
                       collision_margin: float = 0.01,
                       prior_vposer: float = 0.0,
                       vposer=None,
                       vposer_joints: Optional[Sequence[str]] = None,
                       vposer_mode: str = "kl",
                       freeze_shared_translation: bool = False,
                       device: str = "cpu",
                       verbose: bool = True) -> Dict:
    """Jointly fit a shared body pose + per-view roots to multi-view 2D.

    The loss is a robust 2D reprojection term (pixels / 100) plus weak priors
    that keep the local corrections near the initialization and temporally
    smooth.  No GT is used.

    ``vposer`` is an optional ``vtv.vposer_body.VPoserBodyPrior``.  With
    ``prior_vposer > 0`` the pose is additionally scored by VPoser v2 in
    SMPL-X body_pose space, with the joints outside ``opt_joints`` held at the
    source body_pose (so the prior measures the *torso-relative* change this
    stage makes, not the PMX foot convention).  The VPoser weights are frozen;
    this is a prior, not a trained head.
    """
    torch = _torch()
    body = PmxBodyBA(motion, opt_joints=opt_joints, device=device)
    F = int(np.asarray(motion["root_pos"]).shape[0])
    names = list(views)
    vpose_base = vpose_mask = None
    vposer_init = None
    vposer_joints = list(vposer_joints or opt_joints)
    if vposer is not None and prior_vposer > 0.0:
        vpose_base, vpose_mask = vposer.freeze_context(vposer_joints)
        with torch.no_grad():
            G_init = {j: torch.as_tensor(np.asarray(motion["world_rot"][j]),
                                         dtype=torch.float64, device=device)
                      for j in motion["world_rot"]}
            vposer_init = float(vposer.energy(G_init, base=vpose_base,
                                              mask=vpose_mask,
                                              mode=vposer_mode))
    # Optimization variables: local deltas for the selected joints, a shared
    # pelvis translation, and a small per-view rigid root correction.
    delta = {j: torch.zeros((F, 3), dtype=torch.float64, device=device,
                            requires_grad=True) for j in body.opt_joints}
    dtrans = torch.zeros((F, 3), dtype=torch.float64, device=device,
                         requires_grad=not freeze_shared_translation)
    root_rot = torch.zeros((F, len(names), 3), dtype=torch.float64,
                           device=device, requires_grad=True)
    root_trans = torch.zeros((F, len(names), 3), dtype=torch.float64,
                             device=device, requires_grad=True)
    obs = {}
    for v in names:
        views[v]["device"] = device
        obs[v] = joint_observations(views[v], F)
    params = list(delta.values()) + [root_rot, root_trans]
    if not freeze_shared_translation:
        params.append(dtrans)
    opt = torch.optim.Adam(params, lr=lr)
    hist = []
    if prior_collision > 0.0:
        with torch.no_grad():
            _, J0 = body.forward({}, torch.zeros((F, 3), dtype=torch.float64,
                                                device=device))
    else:
        J0 = None
    for it in range(iters):
        opt.zero_grad()
        G, J = body.forward(delta, dtrans)
        # shared positions in canonical-name order
        jnames = [j for j in VIEW_JOINTS if j in J]
        P = torch.stack([J[j] for j in jnames], dim=1)  # (F,K,3)
        Jp = J["pelvis"]
        loss_2d = 0.0
        loss_2d_views = {}
        for vi, v in enumerate(names):
            Rv = qexp_t(root_rot[:, vi, :])
            Q = qrot_t(Rv[:, None, :].expand(-1, P.shape[1], -1),
                       P - Jp[:, None, :]) + Jp[:, None, :] + root_trans[:, vi:vi + 1, :]
            uv = project_torch(Q, views[v])
            o, w = obs[v]
            O = torch.stack([o[j] for j in jnames], dim=1)
            W = torch.stack([w[j] for j in jnames], dim=1)[..., None]
            r = (uv - O) / 100.0
            # smooth L1 (Huber) in pixel units, then confidence weighted
            absr = r.abs()
            h = torch.where(absr * 100.0 < huber_px,
                            0.5 * r * r,
                            (huber_px / 100.0) * (absr - 0.5 * huber_px / 100.0))
            lv = (h * W).sum() / W.sum().clamp_min(1.0)
            loss_2d = loss_2d + lv
            loss_2d_views[v] = float(lv.detach())
        loss_2d = loss_2d / len(names)
        # priors
        lp = 0.0
        for j, d in delta.items():
            lp = lp + d.pow(2).mean()
        lp = lp / max(len(delta), 1)
        lt = 0.0
        for j, d in delta.items():
            if F > 1:
                lt = lt + (d[1:] - d[:-1]).pow(2).mean()
        lt = lt / max(len(delta), 1)
        lr_rot = root_rot.pow(2).mean()
        lr_tr = root_trans.pow(2).mean()
        lc = (_collision_proxy(J, J0, collision_margin)
              if J0 is not None else torch.zeros((), dtype=loss_2d.dtype,
                                                 device=device))
        lv = (vposer.energy(G, base=vpose_base, mask=vpose_mask,
                            mode=vposer_mode)
              if (vposer is not None and prior_vposer > 0.0)
              else torch.zeros((), dtype=loss_2d.dtype, device=device))
        loss = (w_2d * loss_2d + prior_delta * lp + prior_temporal * lt
                + prior_root_rot * lr_rot + prior_root_trans * lr_tr
                + prior_collision * lc + prior_vposer * lv)
        loss.backward()
        opt.step()
        if it % 25 == 0 or it == iters - 1:
            hist.append({"iter": it, "loss": float(loss.detach()),
                         "loss_2d": float(loss_2d.detach()),
                         "views": loss_2d_views,
                         "prior_delta": float(lp.detach()),
                         "prior_temporal": float(lt.detach()),
                         "prior_collision": float(lc.detach()),
                         "prior_vposer": float(lv.detach()),
                         "root_rot": float(lr_rot.detach()),
                         "root_trans": float(lr_tr.detach())})
            if verbose:
                print(f"  it {it:4d} loss {float(loss.detach()):.5f} "
                      f"2d {float(loss_2d.detach()):.5f} "
                      f"view {loss_2d_views} "
                      f"prior {float(lp.detach()):.4f}/{float(lt.detach()):.4f} "
                      f"vp {float(lv.detach()):.3f} "
                      f"coll {float(lc.detach()):.4f} "
                      f"root {float(lr_rot.detach()):.4f}/{float(lr_tr.detach()):.4f}")
    # final shared pose
    with torch.no_grad():
        G, J = body.forward(delta, dtrans)
        G_np = {j: q.detach().cpu().numpy() for j, q in G.items()}
        J_np = {j: p.detach().cpu().numpy() for j, p in J.items()}
        root_pos = J_np["pelvis"]
        root_rot_np = root_rot.detach().cpu().numpy()
        root_trans_np = root_trans.detach().cpu().numpy()
        delta_np = {j: d.detach().cpu().numpy() for j, d in delta.items()}
        # per-view 2D error in pixels
        err = {}
        for vi, v in enumerate(names):
            Rv = np.asarray([q_from_axis_angle(*root_rot_np[f, vi])
                             for f in range(F)])
            P = np.stack([J_np[j] for j in jnames], axis=1)
            Jp = J_np["pelvis"]
            Q = np.array([qrot(tuple(Rv[f]), tuple(P[f, k] - Jp[f])) + Jp[f]
                          + root_trans_np[f, vi] for f in range(F)
                          for k in range(P.shape[1])]).reshape(F, -1, 3)
            R = views[v]["R"]; C = views[v]["C"]
            Pc = np.einsum("ji,fkj->fki", R, Q - C)
            uv = np.stack([views[v]["fx"] * Pc[..., 0] / Pc[..., 2] + views[v]["cx"],
                           views[v]["fy"] * Pc[..., 1] / Pc[..., 2] + views[v]["cy"]],
                          axis=-1)
            o, w = obs[v]
            O = np.stack([o[j].cpu().numpy() for j in jnames], axis=1)
            W = np.stack([w[j].cpu().numpy() for j in jnames], axis=1)
            d = np.linalg.norm(uv - O, axis=-1)
            mask = W > 0
            err[v] = {"median_px": float(np.median(d[mask])) if mask.any() else None,
                      "mean_px": float(np.mean(d[mask])) if mask.any() else None,
                      "p90_px": float(np.percentile(d[mask], 90)) if mask.any() else None}
    vpose_stats = None
    if vposer is not None and prior_vposer > 0.0:
        G_final = {j: torch.as_tensor(q, dtype=torch.float64, device=device)
                   for j, q in G_np.items()}
        with torch.no_grad():
            vpose_stats = vposer.body_pose_stats(
                G_final, base=vpose_base, mask=vpose_mask)
        vpose_stats.update({"weight": float(prior_vposer),
                            "mode": str(vposer_mode),
                            "joints": list(vposer_joints),
                            "kl_init_prior": vposer_init})
    return {"body": body, "G": G_np, "J": J_np, "root_pos": root_pos,
            "delta": delta_np, "root_rot": root_rot_np,
            "root_trans": root_trans_np, "hist": hist, "view_error": err,
            "names": names, "opt_joints": list(body.opt_joints),
            "vposer": vpose_stats}


# --------------------------------------------------------------- VMD output

def motion_from_ba(motion: Dict, res: Dict) -> Dict:
    """Replace a canonical motion's pose with the BA shared pose."""
    out = dict(motion)
    out["world_rot"] = {j: np.asarray(q, dtype=np.float64)
                        for j, q in res["G"].items()}
    # The adapter expects every canonical joint; keep the baseline entries
    # for joints the BA did not carry (e.g. root, legs are in G already).
    for j, q in motion["world_rot"].items():
        if j not in out["world_rot"]:
            out["world_rot"][j] = np.asarray(q, dtype=np.float64)
    out["root_pos"] = np.asarray(res["root_pos"], dtype=np.float64)
    J = fk_positions(out["world_rot"], out["rest_joints"], out["root_pos"])
    reach = [f"toe_{s}" for s in ("l", "r") if f"toe_{s}" in J]
    if reach:
        out["toe_y"] = np.min(np.stack([J[j][:, 1] for j in reach], axis=0), axis=0)
    return out


def write_ba_vmd(motion: Dict, res: Dict, model: Dict, profile: Dict,
                 out_path: str, ground_calibrate: bool = False) -> Dict:
    """Adapt the BA canonical pose to the PMX and write a VMD."""
    from .adapter import adapt
    from .mmd.writer import write_vmd
    m = motion_from_ba(motion, res)
    ad = adapt(m, model, profile, ground_calibrate=ground_calibrate,
               stage_yaw_deg=0.0)
    write_vmd(out_path, model.get("name", ""), ad["bone_frames"], {},
              ad["ik_frames"])
    return {"motion": m, "adapt": ad}


# --------------------------------------------------------------- merge / gate

# The BA is allowed to change this chain.  The clavicle is included because
# the shoulder position is an arm DOF; the torso, root, legs and IK records are
# not.  The M2.4b verifier only listed upperarm/forearm/hand, so a BA that
# moves a shoulder would otherwise be reported as a non-arm change.  This
# module states the wider arm chain explicitly instead of silently widening
# the M2.4b gate.
ARM_CHAIN = (
    "clavicle_l", "clavicle_r",
    "upperarm_l", "upperarm_r",
    "forearm_l", "forearm_r",
    "hand_l", "hand_r",
)

# Torso joints that may be optimized jointly with the arms.  The pelvis is
# excluded (it would rotate the legs and change contact).
TORSO_CHAIN = ("spine", "spine1", "chest", "neck")

# M2.4d default: torso + arm chain **without the hands**.  The hand is a leaf
# with no palm/finger 2D evidence (M2.4b §5.4); letting the BA move it would
# add an unobservable DOF and inflate the "full 6 bone" mean.  Hand error is
# reported separately as a consequence of the forearm.
TORSO_ARM_CHAIN = TORSO_CHAIN + ("clavicle_l", "clavicle_r",
                                 "upperarm_l", "upperarm_r",
                                 "forearm_l", "forearm_r")

OPT_PRESETS = {
    "arms": ARM_CHAIN,
    "upper": ("upperarm_l", "upperarm_r", "forearm_l", "forearm_r",
              "hand_l", "hand_r"),
    "torso": DEFAULT_OPT_JOINTS,
    "torso_arms": TORSO_ARM_CHAIN,
}


def merge_pose_tracks(base_vmd: Dict, ba_vmd: Dict, model: Dict, profile: Dict,
                      joints: Sequence[str] = ARM_CHAIN) -> Tuple[Dict, Dict]:
    """Copy only the BA's optimized local tracks onto the baseline VMD.

    The baseline root, legs, IK records and morphs stay byte-identical; only
    ``joints`` comes from the BA.  This is what makes the candidate safe:
    contact/root metrics cannot change because the legs and root are untouched,
    and the BA's extra root differences (from ``adapt``) are not written.

    With ``joints=ARM_CHAIN`` this is the M2.4c arm merge; with
    ``joints=TORSO_ARM_CHAIN`` the torso locals move too, and the arm locals
    are still relative to the imported chest, so the BA pose is reproduced.
    """
    smap = profile["standard_map"]
    out = copy.deepcopy(base_vmd)
    copied, missing = [], []
    for j in joints:
        bn = smap.get(j)
        if bn and bn in ba_vmd.get("bones", {}) and bn in out.get("bones", {}):
            out["bones"][bn] = copy.deepcopy(ba_vmd["bones"][bn])
            copied.append(bn)
        else:
            missing.append(j)
    return out, {"copied": copied, "missing": missing}


# Backwards-compatible name used by M2.4c scripts/tests.
merge_arm_tracks = merge_pose_tracks


def _allowed_local_names(profile: Dict, joints: Sequence[str]) -> set:
    smap = profile["standard_map"]
    return {smap[j] for j in joints if smap.get(j)}


def _arm_local_names(profile: Dict) -> set:
    return _allowed_local_names(profile, ARM_CHAIN)


def verify_ba(base_vmd: Dict, out_vmd: Dict, model: Dict, profile: Dict,
              gt_vmd: Optional[Dict] = None,
              pair_tol: float = 0.02,
              opt_joints: Optional[Sequence[str]] = None) -> Dict:
    """Gate the merged BA VMD.

    Two penetration rules are reported:

    * ``penetration_pairs_strict`` — every pair must be <= baseline + tol.
      M2.4b used this.  **GT itself fails it on this clip** (right forearm /
      hand / upperarm vs chest and right forearm vs spine1), so it cannot be
      the only gate for a pose that is closer to GT.
    * ``penetration_pairs_budget`` — every pair must be <= max(baseline, GT) +
      tol.  This allows a contact that GT also has, but rejects a new contact
      that is worse than both the prediction and the target.  It is the gate
      used for the M2.4c report when GT is available (synthetic evaluation
      only; the solver never reads it).

    ``ok`` covers frames/bones/IK/arm lengths, non-arm tracks, hand local,
    temporal max/p99, global penetration, and — when ``gt_vmd`` is given — the
    GT-budget rule.  The strict pair rule is diagnostic, not in ``ok``.
    """
    from .arm3d import _q_close
    from .arm_prior import capsule_penetrations
    from .cleanup import dense_tracks, eval_tracks

    fin, tin = dense_tracks(base_vmd)
    fout, tout = dense_tracks(out_vmd)
    joints = list(opt_joints or ARM_CHAIN)
    allowed_names = _allowed_local_names(profile, joints)
    hand_names = {profile["standard_map"].get(j) for j in ("hand_l", "hand_r")}
    hands_optimized = any(j in joints for j in ("hand_l", "hand_r"))
    checks: Dict[str, Dict] = {}
    checks["frames"] = {"ok": fin == fout, "in": len(fin), "out": len(fout)}
    checks["same_bones"] = {"ok": set(tin) == set(tout),
                            "in": len(tin), "out": len(tout)}
    changed = []
    for bn in sorted(set(tin) & set(tout)):
        if bn in allowed_names:
            continue
        if not _q_close(tin[bn]["rot"], tout[bn]["rot"]) or not _q_close(
                tin[bn]["pos"], tout[bn]["pos"]):
            changed.append(bn)
    checks["non_arm_unchanged"] = {"ok": not changed, "changed": changed[:12],
                                   "optimized_chain": sorted(allowed_names),
                                   "note": "the pelvis, legs, feet, head and "
                                           "root are never optimized"}
    hand_changed = [bn for bn in sorted(hand_names & set(tin) & set(tout))
                    if not _q_close(tin[bn]["rot"], tout[bn]["rot"])]
    checks["hand_local_unchanged"] = {
        "ok": bool(hands_optimized or not hand_changed),
        "changed": hand_changed,
        "skipped": bool(hands_optimized),
        "reason": ("hands are in the optimized set; their world error is "
                   "reported separately" if hands_optimized else None)}
    checks["ik_frames"] = {
        "ok": base_vmd.get("ik_frames", []) == out_vmd.get("ik_frames", []),
        "count": len(out_vmd.get("ik_frames", []))}
    checks["track_count"] = {"ok": len(tout) == len(tin), "count": len(tout)}
    tr_in = eval_tracks(model, profile, tin)
    tr_out = eval_tracks(model, profile, tout)
    lbad = []
    for side in ("l", "r"):
        for a, b in (("upperarm", "forearm"), ("forearm", "hand")):
            ja, jb = f"{a}_{side}", f"{b}_{side}"
            li = np.linalg.norm(tr_in["world"][ja] - tr_in["world"][jb], axis=1)
            lo = np.linalg.norm(tr_out["world"][ja] - tr_out["world"][jb], axis=1)
            if float(np.max(np.abs(li - lo))) > 1e-6:
                lbad.append(ja)
    checks["arm_lengths"] = {"ok": not lbad, "changed": lbad}

    def _steps(tr):
        vals = []
        for side in ("l", "r"):
            for seg, child in (("upperarm", "forearm"), ("forearm", "hand")):
                j, c = f"{seg}_{side}", f"{child}_{side}"
                d = np.asarray(tr["world"][c]) - tr["world"][j]
                d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
                if len(d) > 1:
                    vals.append(np.degrees(np.arccos(np.clip(
                        (d[1:] * d[:-1]).sum(1), -1, 1.0))))
        return np.concatenate(vals) if vals else np.zeros(1)

    s_in, s_out = _steps(tr_in), _steps(tr_out)
    checks["temporal_smoothness"] = {
        "ok": (float(np.max(s_out)) <= float(np.max(s_in)) + 5.0
               and float(np.percentile(s_out, 99)) <=
               float(np.percentile(s_in, 99)) + 0.5),
        "max_step_in_deg": round(float(np.max(s_in)), 3),
        "max_step_out_deg": round(float(np.max(s_out)), 3),
        "p99_step_in_deg": round(float(np.percentile(s_in, 99)), 3),
        "p99_step_out_deg": round(float(np.percentile(s_out, 99)), 3),
        "note": "max may grow by <=5 deg/frame; p99 may grow by <=0.5 deg"}
    pin = capsule_penetrations(tr_in, model, profile, arm_arm=True)["summary"]
    pout = capsule_penetrations(tr_out, model, profile, arm_arm=True)["summary"]
    checks["arm_torso_penetration"] = {
        "ok": pout["arm_torso"]["max"] <= pin["arm_torso"]["max"] + 1e-9,
        "before": round(pin["arm_torso"]["max"], 4),
        "after": round(pout["arm_torso"]["max"], 4)}
    checks["arm_arm_penetration"] = {
        "ok": pout["arm_arm"]["max"] <= pin["arm_arm"]["max"] + 0.02,
        "before": round(pin["arm_arm"]["max"], 4),
        "after": round(pout["arm_arm"]["max"], 4)}

    def _pair_map(p):
        out = {}
        for g in ("arm_torso", "arm_arm"):
            for k, rec in (p[g].get("pairs_detail") or {}).items():
                out[(g, k)] = float(rec.get("max", 0.0))
        return out

    bin_pairs = _pair_map(pin)
    out_pairs = _pair_map(pout)
    gt_pairs = None
    if gt_vmd is not None:
        from .cleanup import dense_tracks as _dt
        gf, gt = _dt(gt_vmd)
        gt_pairs = _pair_map(capsule_penetrations(
            eval_tracks(model, profile, gt), model, profile, arm_arm=True)["summary"])
    strict = []
    for k, a in sorted(out_pairs.items()):
        b = bin_pairs.get(k, 0.0)
        if a > b + pair_tol:
            strict.append({"pair": f"{k[0]} {k[1]}", "before": round(b, 4),
                           "after": round(a, 4)})
    checks["penetration_pairs_strict"] = {
        "ok": not strict, "increased": strict[:12], "tolerance": pair_tol,
        "note": "M2.4b rule; GT itself fails it on R2 because the target has "
                "right-arm/chest contacts the baseline does not"}
    budget_bad = []
    for k, a in sorted(out_pairs.items()):
        b = bin_pairs.get(k, 0.0)
        g = (gt_pairs or {}).get(k, 0.0) if gt_pairs is not None else b
        allowed = max(b, g) + pair_tol
        if a > allowed + 1e-9:
            budget_bad.append({"pair": f"{k[0]} {k[1]}", "after": round(a, 4),
                               "budget": round(allowed, 4),
                               "baseline": round(b, 4),
                               "gt": round(g, 4)})
    checks["penetration_pairs_budget"] = {
        "ok": not budget_bad,
        "failed": budget_bad[:12],
        "tolerance": pair_tol,
        "rule": ("after <= max(baseline, GT) + tol; with no GT this is the "
                 "strict baseline rule")}
    # The strict baseline pair rule is diagnostic only: GT itself fails it on
    # R2, so it must not make ``ok`` false.  The GT-budget rule is the gate.
    gate_keys = [k for k in checks if k != "penetration_pairs_strict"]
    return {"ok": all(bool(checks[k].get("ok", True)) for k in gate_keys),
            "checks": checks,
            "penetration_summary": {"before": pin, "after": pout}}

