# -*- coding: utf-8 -*-
"""2D reprojection plumbing: trust region + npz replacement.

No torch / GVHMR / body model is needed here; the heavy integration run is
documented in docs/M2_执行结果_选择性清理与2D重投影.md.
"""
import json
import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

from vtv.reproj2d import gate_accept, write_npz


def test_gate_accept():
    ok, checks = gate_accept(0.20, 8.0, 1.5, 0.05, 15.0, 2.5)
    assert ok and all(v["ok"] for v in checks.values())
    ok, _ = gate_accept(0.01, 8.0, 1.5, 0.05, 15.0, 2.5)
    assert not ok
    ok, _ = gate_accept(0.30, 20.0, 1.5, 0.05, 15.0, 2.5)
    assert not ok
    ok, _ = gate_accept(0.30, 8.0, 3.0, 0.05, 15.0, 2.5)
    assert not ok


def test_write_npz_replaces_both_body_pose_slices():
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "in.npz")
        dst = os.path.join(td, "out.npz")
        bp = np.zeros((4, 63), dtype=np.float32)
        np.savez_compressed(src, **{
            "smpl_params_global/body_pose": bp,
            "smpl_params_incam/body_pose": bp,
            "smpl_params_global/transl": np.arange(12, dtype=np.float32).reshape(4, 3),
            "bbx_xys": np.ones((4, 3), dtype=np.float32),
        })
        new = bp.copy()
        new[:, 45:48] = 0.25
        report = {"accepted": True, "note": "unit"}
        write_npz(src, dst, new, np.zeros((1, 2, 3)), True, report,
                  err_before=np.ones((4, 4)), err_after=np.zeros((4, 4)))
        with np.load(dst) as d:
            assert np.allclose(d["smpl_params_global/body_pose"], new)
            assert np.allclose(d["smpl_params_incam/body_pose"], new)
            # untouched arrays survive
            assert np.allclose(d["smpl_params_global/transl"].reshape(-1),
                               np.arange(12, dtype=np.float32))
            assert int(d["vtv_2d_fit/accepted"][0]) == 1
            assert json.loads(str(d["vtv_2d_fit/report_json"][0]))["accepted"] is True


if __name__ == "__main__":
    test_gate_accept()
    test_write_npz_replaces_both_body_pose_slices()
    print("test_reproj2d: OK")
