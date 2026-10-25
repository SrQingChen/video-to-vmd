"""Numerical tests for shim/pytorch3d/transforms.py (run with the E: venv)."""
import os
import sys
import unittest

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "shim"))

from pytorch3d.transforms import (  # noqa: E402
    axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_quaternion,
    matrix_to_rotation_6d, quaternion_to_axis_angle, quaternion_to_matrix,
    rotation_6d_to_matrix, so3_exp_map, so3_log_map,
)


def random_rotations(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, 4, generator=g)
    return quaternion_to_matrix(q / q.norm(dim=-1, keepdim=True))


class TestShim(unittest.TestCase):
    def test_quat_matrix_roundtrip(self):
        R = random_rotations()
        q = matrix_to_quaternion(R)
        R2 = quaternion_to_matrix(q)
        self.assertLess((R - R2).abs().max().item(), 1e-5)
        # q and -q are the same rotation
        q2 = matrix_to_quaternion(R2)
        dot = (q * q2).sum(-1).abs()
        self.assertLess((1 - dot).abs().max().item(), 1e-5)

    def test_axis_angle_roundtrip(self):
        R = random_rotations()
        aa = matrix_to_axis_angle(R)
        R2 = axis_angle_to_matrix(aa)
        self.assertLess((R - R2).abs().max().item(), 1e-5)

    def test_so3(self):
        g = torch.Generator().manual_seed(1)
        aa = torch.randn(32, 3, generator=g) * 1.5
        R = so3_exp_map(aa)
        aa2 = so3_log_map(R)
        self.assertLess((R - axis_angle_to_matrix(aa2)).abs().max().item(), 1e-5)

    def test_6d_roundtrip(self):
        R = random_rotations()
        d = matrix_to_rotation_6d(R)
        R2 = rotation_6d_to_matrix(d)
        self.assertLess((R - R2).abs().max().item(), 1e-5)

    def test_quat_axis_angle(self):
        g = torch.Generator().manual_seed(2)
        aa = torch.randn(16, 3, generator=g)
        q = torch.cat([torch.cos(aa.norm(dim=-1, keepdim=True) / 2),
                       aa / aa.norm(dim=-1, keepdim=True) *
                       torch.sin(aa.norm(dim=-1, keepdim=True) / 2)], -1)
        aa2 = quaternion_to_axis_angle(q)
        R1 = axis_angle_to_matrix(aa)
        R2 = axis_angle_to_matrix(aa2)
        self.assertLess((R1 - R2).abs().max().item(), 1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
