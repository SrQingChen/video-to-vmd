# -*- coding: utf-8 -*-
"""Minimal PyTorch3D transforms shim (Apache/BSD-compatible reimplementation).

Only the functions GVHMR imports at inference time are provided.  Semantics
follow PyTorch3D: quaternions are (w, x, y, z), rotation matrices act on
column vectors, and rotation_6d is the first two matrix rows (row-major).
"""
import math

import torch
import torch.nn.functional as F


def _normalize(v, eps=1e-12):
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1).clamp(min=1e-12)
    o = torch.stack((
        1 - two_s * (j * j + k * k),
        two_s * (i * j - k * r),
        two_s * (i * k + j * r),
        two_s * (i * j + k * r),
        1 - two_s * (i * i + k * k),
        two_s * (j * k - i * r),
        two_s * (i * k - j * r),
        two_s * (j * k + i * r),
        1 - two_s * (i * i + j * j),
    ), -1)
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Shepperd's method; returns (w, x, y, z)."""
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError("Invalid rotation matrix shape.")
    m = matrix
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]

    q0 = torch.zeros_like(m00)
    q1 = torch.zeros_like(m00)
    q2 = torch.zeros_like(m00)
    q3 = torch.zeros_like(m00)

    tr = m00 + m11 + m22
    mask = tr > 0
    if mask.any():
        s = torch.sqrt(tr[mask] + 1.0) * 2
        q0[mask] = 0.25 * s
        q1[mask] = (m21[mask] - m12[mask]) / s
        q2[mask] = (m02[mask] - m20[mask]) / s
        q3[mask] = (m10[mask] - m01[mask]) / s
    mask2 = (~mask) & (m00 > m11) & (m00 > m22)
    if mask2.any():
        s = torch.sqrt(1.0 + m00[mask2] - m11[mask2] - m22[mask2]) * 2
        q0[mask2] = (m21[mask2] - m12[mask2]) / s
        q1[mask2] = 0.25 * s
        q2[mask2] = (m01[mask2] + m10[mask2]) / s
        q3[mask2] = (m02[mask2] + m20[mask2]) / s
    mask3 = (~mask) & (~mask2) & (m11 > m22)
    if mask3.any():
        s = torch.sqrt(1.0 + m11[mask3] - m00[mask3] - m22[mask3]) * 2
        q0[mask3] = (m02[mask3] - m20[mask3]) / s
        q1[mask3] = (m01[mask3] + m10[mask3]) / s
        q2[mask3] = 0.25 * s
        q3[mask3] = (m12[mask3] + m21[mask3]) / s
    mask4 = (~mask) & (~mask2) & (~mask3)
    if mask4.any():
        s = torch.sqrt(1.0 + m22[mask4] - m00[mask4] - m11[mask4]) * 2
        q0[mask4] = (m10[mask4] - m01[mask4]) / s
        q1[mask4] = (m02[mask4] + m20[mask4]) / s
        q2[mask4] = (m12[mask4] + m21[mask4]) / s
        q3[mask4] = 0.25 * s
    return _normalize(torch.stack([q0, q1, q2, q3], dim=-1))


def _skew(v):
    zero = torch.zeros_like(v[..., 0])
    return torch.stack((
        torch.stack((zero, -v[..., 2], v[..., 1]), -1),
        torch.stack((v[..., 2], zero, -v[..., 0]), -1),
        torch.stack((-v[..., 1], v[..., 0], zero), -1),
    ), -2)


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    theta = axis_angle.norm(dim=-1, keepdim=True)
    k = axis_angle / theta.clamp(min=1e-12)
    K = _skew(k)
    I = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    sin = torch.sin(theta)[..., None]
    cos = torch.cos(theta)[..., None]
    R = I + sin * K + (1 - cos) * (K @ K)
    # small-angle series (theta^3 terms), avoids 0/0 direction noise
    small = theta < 1e-6
    if small.any():
        R_small = I + K * theta[..., None]
        R = torch.where(small[..., None], R_small, R)
    return R


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    q = matrix_to_quaternion(matrix)
    return quaternion_to_axis_angle(q)


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    q = _normalize(quaternions)
    # shortest arc: w >= 0
    q = torch.where((q[..., 0] < 0).unsqueeze(-1), -q, q)
    w = q[..., 0].clamp(-1.0, 1.0)
    sin_half = torch.sqrt((1.0 - w * w).clamp(min=1e-24))
    angle = 2.0 * torch.acos(w)
    axis = q[..., 1:] / sin_half.unsqueeze(-1)
    return axis * angle.unsqueeze(-1)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    batch_dims = matrix.shape[:-2]
    return matrix[..., :2, :].clone().reshape(*batch_dims, 6)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def so3_exp_map(log_rot: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    return axis_angle_to_matrix(log_rot)


def so3_log_map(matrix: torch.Tensor) -> torch.Tensor:
    return matrix_to_axis_angle(matrix)


def _axis_matrix(axis: str, angle: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(angle), torch.sin(angle)
    z, o = torch.zeros_like(c), torch.ones_like(c)
    if axis == "X":
        rows = ((o, z, z), (z, c, -s), (z, s, c))
    elif axis == "Y":
        rows = ((c, z, s), (z, o, z), (-s, z, c))
    elif axis == "Z":
        rows = ((c, -s, z), (s, c, z), (z, z, o))
    else:
        raise ValueError(f"Invalid axis {axis}")
    return torch.stack([torch.stack(r, -1) for r in rows], -2)


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str = "XYZ") -> torch.Tensor:
    """Intrinsic rotations applied in the given convention order.

    Mirrors PyTorch3D's convention (R = R_c0 @ R_c1 @ R_c2 for extrinsic
    interpretation; GVHMR only uses this for optional visualization paths).
    """
    if len(convention) != 3 or any(c not in "XYZ" for c in convention):
        raise ValueError("Invalid convention")
    mats = [_axis_matrix(convention[i], euler_angles[..., i])
            for i in range(3)]
    return mats[0] @ mats[1] @ mats[2]


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    angle = axis_angle.norm(dim=-1, keepdim=True)
    axis = axis_angle / angle.clamp(min=1e-12)
    half = angle * 0.5
    return torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1)
