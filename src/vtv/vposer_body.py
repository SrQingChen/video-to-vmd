# -*- coding: utf-8 -*-
"""M2.4d: VPoser v2 prior on the *target PMX* body, without retraining.

Why this module exists
----------------------
M2.4c showed that freezing the torso limits the body-relative upper-arm /
forearm error to -6.18 deg, while freeing the torso makes the chest *worse*
(11.16 -> 19.35 deg) because the multi-view 2D term cannot determine the chest
orientation on its own.  The missing ingredient is a prior on the **torso /
torso-relative arm pose**.  This module connects the local VPoser v2 checkpoint
to the target PMX skeleton so that the prior can be used inside the multi-view
BA (``ba3d.optimize_multiview``) and as a torso-relative target.

Rig conversions (the part that is easy to get wrong)
----------------------------------------------------
The canonical layer stores **world rotations** rest-compensated to the PMX
rest skeleton.  VPoser speaks SMPL-X ``body_pose``: parent-relative local
rotations in the SMPL-X rest frame, with the SMPL-X model-frame convention.
The target adapter (``adapter.adapt``) retargets a canonical motion as

    G_pmx(j) = Yaw * D_j * G_src(j) * D_j^-1

with ``D_j = q_between(u_smplx(j), v_pmx(j))`` the per-joint rest-direction
compensation, ``Yaw`` the presentation/stage yaw (180 deg for the SMPL-X
sources used here), and ``D = identity`` for the feet (``adapter.NO_SWING``).
Inverting that line and applying the SMPL-X frame reflection gives the exact
``body_pose`` of the *pose the target adapter would have produced the VMD
from*.  This was verified on the R2 front clip: torso joints recover the
GVHMR ``body_pose`` to 0.00 deg, arms to 0.29 deg (the residual is the
cleanup smoothing), even though the raw rest-direction differences are
20-50 deg.  Ignoring ``Yaw`` or conjugating on the wrong side gives 30-77 deg
of error, so the mapping is not optional.

Two important limitations, both measured in the M2.4d report:

* **A prior is not an observation.**  VPoser rates the R2 GT *less* likely
  than the prediction (KL 116 vs 30, driven by the GT's folded arms).  On this
  clip the prior therefore cannot point at the GT; it can only act as a
  regularizer that stops the 2D term from "using the chest".
* **The VPoser autoencoder reconstructs too well to be a target.**  Decoding
  the encoded prediction changes the arm body_pose by ~0.1 deg (GT ~1 deg),
  so a "VPoser-projected target" is a near no-op.  The projection path is kept
  for completeness and reported, not assumed to help.

Nothing here trains a network or writes a new feed-forward model.
"""
from __future__ import annotations

import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import canonical_body as CB
from .adapter import q_between, q_conj, qmul, qnorm

DEFAULT_STAGE_YAW = 180.0
NO_SWING = ("foot_l", "foot_r", "toe_l", "toe_r")

# module-level cache: loading SMPL-X (for the rest skeleton) is the slow part
_SMPLX_RD_CACHE: Dict[Tuple[str, str, int, bytes], Dict[str, np.ndarray]] = {}


# --------------------------------------------------------------- rest frames

def smplx_rest_dirs(smplx_model_dir: Optional[str] = None,
                    gender: str = "NEUTRAL",
                    num_betas: int = 10,
                    betas: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Canonical-frame SMPL-X rest bone directions at zero pose.

    ``betas`` must be the clip's shape parameters when they are known: the
    pipeline's ``canonical_from_smplx`` builds its rest joints from the
    *shaped* SMPL-X skeleton, and using zero betas leaves a 5-7 deg systematic
    offset in the recovered body_pose (measured on R2).  The canonical frame is
    MMD-like (+X right, +Y up, +Z back), so the SMPL-X model-frame joints are
    reflected in Z, the same transform ``adapter.canonical_from_smplx``
    applies.  Results are cached because the body model load dominates runtime.
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model_dir = os.path.abspath(smplx_model_dir or os.path.join(root, "body_models"))
    b = np.zeros(num_betas, dtype=np.float64) if betas is None \
        else np.asarray(betas, dtype=np.float64).reshape(-1)[:num_betas]
    key = (model_dir, gender, int(num_betas), np.round(b, 6).tobytes())
    if key in _SMPLX_RD_CACHE:
        return _SMPLX_RD_CACHE[key]
    import smplx
    import torch

    bm = smplx.create(model_dir, model_type="smplx", gender=gender,
                      use_pca=False, num_betas=num_betas)
    with torch.no_grad():
        out = bm(betas=torch.as_tensor(b, dtype=torch.float32).reshape(1, -1),
                 global_orient=torch.zeros(1, 3),
                 body_pose=torch.zeros(1, bm.NUM_BODY_JOINTS * 3),
                 transl=torch.zeros(1, 3))
    J = out.joints[0].detach().cpu().numpy()
    rest = {j: np.array([J[i][0], J[i][1], -J[i][2]], dtype=np.float64)
            for j, i in CB.SMPLX_INDEX.items()}
    rd = {j: np.asarray(v, dtype=np.float64) for j, v in CB.rest_dirs(rest).items()}
    _SMPLX_RD_CACHE[key] = rd
    return rd


def frame_compensation(motion: Dict,
                       smplx_rd: Optional[Dict[str, np.ndarray]] = None,
                       no_swing: Sequence[str] = NO_SWING) -> Dict[str, np.ndarray]:
    """Per-joint ``D`` map taking SMPL-X rest directions to PMX rest directions.

    ``motion["rest_dir"]`` is the PMX rest direction table (canonical frame).
    The feet keep the identity because ``adapter.adapt`` deliberately does not
    swing-compensate the foot segment definitions (``NO_SWING``); using a real
    ``D`` there would mis-conjugate the ankle.
    """
    rd_sx = smplx_rd if smplx_rd is not None else smplx_rest_dirs()
    rd_pmx = motion.get("rest_dir") or {}
    out: Dict[str, np.ndarray] = {}
    for j in CB.BODY_JOINTS:
        if j in ("pelvis",) or j in no_swing:
            out[j] = np.array([0.0, 0.0, 0.0, 1.0])
        elif j in rd_sx and j in rd_pmx:
            out[j] = np.asarray(q_between(tuple(rd_sx[j]), tuple(rd_pmx[j])),
                                dtype=np.float64)
        else:
            out[j] = np.array([0.0, 0.0, 0.0, 1.0])
    return out


def yaw_quat(deg: float) -> np.ndarray:
    h = math.radians(float(deg)) * 0.5
    return np.array([0.0, math.sin(h), 0.0, math.cos(h)], dtype=np.float64)


# ------------------------------------------------------- numpy: pose -> VPoser

def _q_to_rotvec(q) -> np.ndarray:
    from scipy.spatial.transform import Rotation as Rot

    q = np.asarray(q, dtype=np.float64)
    x, y, z, w = q
    Rm = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return Rot.from_matrix(Rm).as_rotvec()


def _rotvec_to_q(v) -> np.ndarray:
    from scipy.spatial.transform import Rotation as Rot

    x, y, z, w = Rot.from_rotvec(np.asarray(v, dtype=np.float64)).as_quat()
    return np.array([x, y, z, w], dtype=np.float64)


def _reflect(q) -> np.ndarray:
    """SMPL-X model frame <-> canonical (MMD-like) frame on a rotation."""
    q = np.asarray(q, dtype=np.float64)
    return np.array([-q[0], -q[1], q[2], q[3]], dtype=np.float64)


def canonical_smplx_world(motion: Dict, D: Dict[str, np.ndarray],
                          stage_yaw_deg: float = DEFAULT_STAGE_YAW,
                          frames: Optional[Sequence[int]] = None) -> Dict[str, np.ndarray]:
    """Recover the SMPL-X-native world rotations from a PMX canonical motion.

    ``G_src = D^-1 * Yaw^-1 * G_pmx * D``; the inverse of the adapter's
    retargeting line.  Returns body joints only (root is not needed).
    """
    G = motion["world_rot"]
    n = len(motion["frames"]) if frames is None else len(frames)
    fmap = list(range(n)) if frames is None else list(frames)
    Y = yaw_quat(stage_yaw_deg)
    Yi = q_conj(tuple(Y))
    out: Dict[str, np.ndarray] = {}
    for j in CB.BODY_JOINTS:
        g = np.asarray(G[j])[fmap]
        d = np.asarray(D[j], dtype=np.float64)
        di = np.asarray(q_conj(tuple(d)))
        out[j] = np.asarray([
            qmul(qmul(qmul(di, Yi), tuple(g[f])), tuple(d)) for f in range(n)
        ])
    return out


def body_pose_from_motion(motion: Dict, D: Dict[str, np.ndarray],
                          stage_yaw_deg: float = DEFAULT_STAGE_YAW,
                          frames: Optional[Sequence[int]] = None) -> np.ndarray:
    """PMX canonical motion -> SMPL-X ``body_pose`` (F,21,3) axis-angle."""
    Gs = canonical_smplx_world(motion, D, stage_yaw_deg, frames)
    n = len(Gs["pelvis"])
    out = np.zeros((n, len(CB.BODY_POSE_INDEX), 3), dtype=np.float64)
    for j, bi in CB.BODY_POSE_INDEX.items():
        p = CB.CANONICAL_PARENT[j]
        loc = np.asarray([qmul(q_conj(tuple(Gs[p][f])), tuple(Gs[j][f]))
                          for f in range(n)])
        out[:, bi, :] = np.asarray([_q_to_rotvec(_reflect(q)) for q in loc])
    return out


def canonical_from_body_pose(body_pose: np.ndarray,
                             pelvis_world: np.ndarray,
                             D: Dict[str, np.ndarray],
                             stage_yaw_deg: float = DEFAULT_STAGE_YAW) -> Dict[str, np.ndarray]:
    """Inverse of :func:`body_pose_from_motion` with the pelvis fixed.

    ``pelvis_world`` is the PMX-world rotation of the pelvis (F,4), which is
    *not* part of ``body_pose``.  The returned dict contains every body joint
    in PMX canonical world rotations, ready for ``fk_positions``.
    """
    bp = np.asarray(body_pose, dtype=np.float64)
    n = len(bp)
    Y = yaw_quat(stage_yaw_deg)
    Yi = q_conj(tuple(Y))
    Gc: Dict[str, np.ndarray] = {}
    Gc["pelvis"] = np.asarray([qmul(tuple(Yi), tuple(pelvis_world[f]))
                               for f in range(n)])
    for j in CB.BODY_JOINTS:
        if j == "pelvis":
            continue
        bi = CB.BODY_POSE_INDEX[j]
        loc = np.asarray([_reflect(_rotvec_to_q(bp[f, bi])) for f in range(n)])
        p = CB.CANONICAL_PARENT[j]
        Gc[j] = np.asarray([qnorm(qmul(tuple(Gc[p][f]), tuple(loc[f])))
                            for f in range(n)])
    out: Dict[str, np.ndarray] = {}
    for j in CB.BODY_JOINTS:
        d = np.asarray(D[j], dtype=np.float64)
        di = np.asarray(q_conj(tuple(d)))
        out[j] = np.asarray([qnorm(qmul(qmul(qmul(tuple(Y), tuple(d)),
                                            tuple(Gc[j][f])), tuple(di)))
                             for f in range(n)])
    return out


# ------------------------------------------------------- torch: differentiable

def _torch():
    import torch
    return torch


def q_to_rotvec_t(q):
    """Quaternion (...,4) xyzw -> axis-angle (...,3), differentiable and stable."""
    torch = _torch()
    xyz = q[..., :3]
    w = q[..., 3:4]
    n = xyz.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(n, w)
    # shortest representation: w<0 -> angle in (pi,2pi] -> subtract 2*pi
    angle = torch.where(angle > math.pi, angle - 2.0 * math.pi, angle)
    small = n < 1e-8
    scale = torch.where(small, 2.0 + (n * n) / 3.0,
                        angle / n.clamp_min(1e-12))
    return xyz * scale


def reflect_t(q):
    torch = _torch()
    return torch.cat([-q[..., :2], q[..., 2:]], dim=-1)


def canonical_to_body_pose_t(G: Dict[str, object], D: Dict[str, np.ndarray],
                             stage_yaw_deg: float = DEFAULT_STAGE_YAW,
                             device: str = "cpu") -> object:
    """Differentiable ``G`` (torch quats) -> SMPL-X body_pose (F,21,3).

    ``G`` may carry the root/pelvis and every canonical body joint; only the
    body joints are read.  The returned tensor is float64; cast at the VPoser
    boundary.
    """
    torch = _torch()
    Y = torch.as_tensor(yaw_quat(stage_yaw_deg), dtype=torch.float64,
                        device=device)
    Gs: Dict[str, object] = {}
    for j in CB.BODY_JOINTS:
        d = torch.as_tensor(np.asarray(D[j], dtype=np.float64),
                            dtype=torch.float64, device=device)
        di = torch.stack([-d[0], -d[1], -d[2], d[3]])
        g = G[j]
        Gs[j] = qnorm_t(qmul_t(qmul_t(qmul_t(di.expand_as(g), Y.expand_as(g)),
                                      g), d.expand_as(g)))
    F = Gs["pelvis"].shape[0]
    out = torch.zeros((F, len(CB.BODY_POSE_INDEX), 3), dtype=torch.float64,
                      device=device)
    for j, bi in CB.BODY_POSE_INDEX.items():
        p = CB.CANONICAL_PARENT[j]
        loc = qmul_t(qconj_t(Gs[p]), Gs[j])
        out[:, bi, :] = q_to_rotvec_t(reflect_t(loc))
    return out


def qmul_t(a, b):
    return _torch().stack([
        a[..., 3] * b[..., 0] + a[..., 0] * b[..., 3]
        + a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
        a[..., 3] * b[..., 1] - a[..., 0] * b[..., 2]
        + a[..., 1] * b[..., 3] + a[..., 2] * b[..., 0],
        a[..., 3] * b[..., 2] + a[..., 0] * b[..., 1]
        - a[..., 1] * b[..., 0] + a[..., 2] * b[..., 3],
        a[..., 3] * b[..., 3] - a[..., 0] * b[..., 0]
        - a[..., 1] * b[..., 1] - a[..., 2] * b[..., 2],
    ], dim=-1)


def qconj_t(q):
    return _torch().cat([-q[..., :3], q[..., 3:4]], dim=-1)


def qnorm_t(q):
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _per_joint_geodesic_deg(a, b) -> np.ndarray:
    """Per-joint geodesic distance (F,21) between (F,63) axis-angle poses."""
    from scipy.spatial.transform import Rotation as Rot

    aa = np.asarray(a.detach().cpu().numpy(), dtype=np.float64).reshape(-1, 21, 3)
    bb = np.asarray(b.detach().cpu().numpy(), dtype=np.float64).reshape(-1, 21, 3)
    qa = Rot.from_rotvec(aa.reshape(-1, 3)).as_quat().reshape(len(aa), 21, 4)
    qb = Rot.from_rotvec(bb.reshape(-1, 3)).as_quat().reshape(len(bb), 21, 4)
    dot = np.abs(np.sum(qa * qb, axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))


# ------------------------------------------------------------------ prior

class VPoserBodyPrior:
    """VPoser v2 prior expressed in the target PMX canonical parameterization.

    Parameters
    ----------
    motion : canonical motion (from ``adapter.canonical_from_pmx``) used for
        the PMX rest directions and the pelvis frame.
    vposer : optional already-loaded ``vtv.vposer_prior.VPoserPrior``.
    stage_yaw_deg : the adapter stage yaw used to produce the VMDs (180 for
        the SMPL-X sources in this repo).
    source_body_pose : optional (F,63) SMPL-X body_pose from the source npz.
        When given, joints the BA does not optimize are taken from it (the
        legs/feet are the important case: the PMX foot convention keeps the
        identity compensation, so a PMX-derived foot body_pose is not the
        source's).

    The encoder weights are frozen: this is a prior, not a trainable module.
    """

    def __init__(self, motion: Dict, vposer=None,
                 stage_yaw_deg: float = DEFAULT_STAGE_YAW,
                 smplx_model_dir: Optional[str] = None,
                 gender: str = "NEUTRAL",
                 betas: Optional[np.ndarray] = None,
                 source_body_pose: Optional[np.ndarray] = None,
                 device: str = "cpu"):
        self.motion = motion
        self.device = device
        self.stage_yaw_deg = float(stage_yaw_deg)
        self.rd_smplx = smplx_rest_dirs(smplx_model_dir, gender=gender,
                                        betas=betas)
        self.D = frame_compensation(motion, self.rd_smplx)
        if vposer is None:
            from .vposer_prior import load_vposer_v2
            vposer = load_vposer_v2(device=device)
        self.vposer = vposer
        for p in self.vposer.model.parameters():
            p.requires_grad_(False)
        self.vposer.model.eval()
        F = len(motion["frames"])
        if source_body_pose is not None:
            src = np.asarray(source_body_pose, dtype=np.float64)
            if src.shape[1] != 63:
                raise ValueError(f"source_body_pose must be (F,63), got {src.shape}")
            if len(src) < F:
                pad = np.repeat(src[-1:], F - len(src), 0)
                src = np.concatenate([src, pad], 0)
            self.source_body_pose = src[:F]
        else:
            self.source_body_pose = None
        # baseline body pose of the initialization, used as the frozen context
        # when no source npz is supplied.
        self.base_body_pose = body_pose_from_motion(
            motion, self.D, self.stage_yaw_deg).reshape(F, 63)
        self.last_energy: Optional[Dict] = None

    # -- conversion ------------------------------------------------------
    def body_pose_t(self, G: Dict[str, object]):
        return canonical_to_body_pose_t(G, self.D, self.stage_yaw_deg,
                                       device=self.device)

    def body_pose_np(self, motion: Optional[Dict] = None) -> np.ndarray:
        return body_pose_from_motion(motion or self.motion, self.D,
                                     self.stage_yaw_deg)

    def freeze_context(self, joints: Iterable[str], source: bool = True):
        """Build the (F,63) context tensor + a mask of the free joints.

        Joints outside ``joints`` are held at the source (or baseline) value;
        the returned tensor is differentiable only through the free entries.
        """
        torch = _torch()
        base = self.source_body_pose if (source and self.source_body_pose is not None) \
            else self.base_body_pose
        base_t = torch.as_tensor(base, dtype=torch.float64, device=self.device)
        idx = sorted({CB.BODY_POSE_INDEX[j] for j in joints
                      if j in CB.BODY_POSE_INDEX})
        mask = torch.zeros(63, dtype=torch.bool, device=self.device)
        for bi in idx:
            mask[bi * 3:bi * 3 + 3] = True
        return base_t, mask

    # -- energy ----------------------------------------------------------
    def energy(self, G: Dict[str, object], joints: Optional[Iterable[str]] = None,
               base: Optional[object] = None, mask: Optional[object] = None,
               mode: str = "kl"):
        torch = _torch()
        x = self.body_pose_t(G).reshape(-1, 63)
        if base is not None:
            b = base.to(dtype=x.dtype, device=x.device)
            if mask is None:
                m = torch.ones(63, dtype=torch.bool, device=x.device)
            else:
                m = mask.to(device=x.device)
            x = torch.where(m, x, b)
        x32 = x.float()
        q = self.vposer.encode(x32)
        if mode == "kl":
            return self.vposer.kl(q).mean()
        if mode == "recon":
            out = self.vposer.decode(q.mean)["pose_body"].reshape(-1, 63)
            return ((out - x32) ** 2).mean()
        raise ValueError(f"unknown VPoser energy mode: {mode}")

    def body_pose_stats(self, G: Dict[str, object],
                        base: Optional[object] = None,
                        mask: Optional[object] = None) -> Dict:
        """KL / reconstruction of the recovered pose, with and without context.

        ``kl_full`` is the KL of the pose exactly as recovered from the PMX
        skeleton; ``kl_prior`` replaces the frozen joints with the source
        body_pose first (the quantity the BA actually penalizes).  The feet are
        the reason the two differ: the adapter keeps the identity compensation
        there, so a PMX-derived foot body_pose is not the source's.
        """
        torch = _torch()
        x = self.body_pose_t(G).reshape(-1, 63)
        out: Dict = {"frames": int(x.shape[0])}
        with torch.no_grad():
            for tag, xx in (("full", x), ("prior", x)):
                if tag == "prior" and base is not None:
                    b = base.to(dtype=xx.dtype, device=xx.device)
                    m = (torch.ones(63, dtype=torch.bool, device=xx.device)
                         if mask is None else mask.to(device=xx.device))
                    xx = torch.where(m, xx, b)
                x32 = xx.float()
                q = self.vposer.encode(x32)
                kl = self.vposer.kl(q)
                dec = self.vposer.decode(q.mean)["pose_body"].reshape(-1, 63)
                rec = dec - x32
                geo = _per_joint_geodesic_deg(x32, dec)
                out[f"kl_{tag}_mean"] = float(kl.mean())
                out[f"kl_{tag}_median"] = float(kl.median())
                out[f"recon_{tag}_mse"] = float((rec ** 2).mean())
                out[f"recon_{tag}_deg_mean"] = float(geo.mean())
                out[f"recon_{tag}_arm_deg_mean"] = float(
                    geo[:, [15, 16, 17, 18, 19, 20]].mean())
                out[f"recon_{tag}_torso_deg_mean"] = float(
                    geo[:, [2, 5, 8, 11]].mean())
        return out

    # -- projection ------------------------------------------------------
    def project_body_pose(self, body_pose: Optional[np.ndarray] = None,
                          mode: str = "full") -> np.ndarray:
        """VPoser autoencoder projection of a (F,21,3) or (F,63) body pose."""
        torch = _torch()
        bp = self.base_body_pose.reshape(-1, 63) if body_pose is None \
            else np.asarray(body_pose, dtype=np.float64).reshape(-1, 63)
        with torch.no_grad():
            x = torch.as_tensor(bp, dtype=torch.float32)
            q = self.vposer.encode(x)
            out = self.vposer.decode(q.mean)["pose_body"].reshape(-1, 63).numpy()
        return out

    def reconstructed_world(self, body_pose: np.ndarray,
                           motion: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        m = motion or self.motion
        return canonical_from_body_pose(
            np.asarray(body_pose, dtype=np.float64).reshape(-1, 21, 3),
            np.asarray(m["world_rot"]["pelvis"]), self.D, self.stage_yaw_deg)


# --------------------------------------------------------------- evaluation

def body_relative_from_world(G: Dict[str, np.ndarray],
                             J: Dict[str, np.ndarray],
                             joints: Optional[Dict[str, str]] = None) -> Dict[str, np.ndarray]:
    """Arm directions in the chest frame, from world rotations + positions.

    Same convention as ``arm3d.body_relative_errors`` but without a VMD;
    used for the VPoser projection diagnostics.
    """
    from .adapter import qrot

    qc = np.asarray(G["chest"])
    out: Dict[str, np.ndarray] = {}
    for side in ("l", "r"):
        for a, b in (("upperarm", "forearm"), ("forearm", "hand")):
            j, c = f"{a}_{side}", f"{b}_{side}"
            if j not in J or c not in J:
                continue
            d = np.asarray(J[c]) - np.asarray(J[j])
            d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
            out[f"{a}_{side}"] = np.asarray([
                qrot(tuple(q_conj(tuple(qc[f]))), tuple(d[f]))
                for f in range(len(d))])
    return out
