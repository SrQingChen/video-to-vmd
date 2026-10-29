"""Unit tests for src/vtv/feedback_net.py (M3.5).

Run:  python tests/test_feedback_net.py
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from vtv import feedback_data as FD  # noqa: E402
from vtv.feedback_net import (  # noqa: E402
    FeedbackNet, NetConfig, group_losses, make_windows,
)


def _cfg(**kw):
    base = dict(feat_dim=24, out_dim=FD.RESIDUAL_DIM, hidden=64, layers=1,
                dropout=0.0, arch="gru", bidirectional=True, window=9)
    base.update(kw)
    return NetConfig(**base)


def test_param_budget():
    cfg = NetConfig(feat_dim=219, out_dim=FD.RESIDUAL_DIM, hidden=256, layers=2,
                    dropout=0.1, arch="gru", bidirectional=True, window=31)
    net = FeedbackNet(cfg, FD.residual_limits())
    n = net.n_params()
    assert 1_000_000 <= n <= 5_000_000, n


def test_bounded_output():
    lim = FD.residual_limits()
    net = FeedbackNet(_cfg(), lim).eval()
    x = torch.randn(2, 12, 24) * 100.0
    with torch.no_grad():
        y = net(x)
    assert y.shape == (2, 12, FD.RESIDUAL_DIM)
    assert torch.isfinite(y).all()
    assert (y.abs() <= torch.tensor(lim) + 1e-6).all()


def test_windows():
    n, f, d = 11, 9, 4
    x = np.arange(n * f, dtype=np.float32).reshape(n, f)
    t = np.arange(n * d, dtype=np.float32).reshape(n, d)
    m = np.ones((n, d), np.float32)
    W, T, M = make_windows(x, t, m, 5)
    assert W.shape == (n, 5, f)
    assert np.array_equal(W[0, 0], x[0])          # edge replication
    assert np.array_equal(W[0, 2], x[0])
    assert np.array_equal(W[-1, -1], x[-1])
    assert np.array_equal(W[5, 2], x[5])


def test_mask_ignores_unmasked_targets():
    limits = torch.tensor(FD.residual_limits())
    groups = FD.residual_groups()
    pred = torch.zeros(1, 4, FD.RESIDUAL_DIM)
    target = torch.zeros(1, 4, FD.RESIDUAL_DIM)
    mask = torch.zeros(1, 4, FD.RESIDUAL_DIM)
    mask[..., FD.HAND_SLICE] = 1.0
    t2 = target.clone()
    t2[..., FD.FACE_SLICE] = 5.0                 # masked -> must not matter
    l1, _ = group_losses(pred, target, mask, limits, groups, {})
    l2, _ = group_losses(pred, t2, mask, limits, groups, {})
    assert abs(float(l1) - float(l2)) < 1e-7
    # changing a supervised component must change the loss
    t3 = target.clone(); t3[..., FD.HAND_SLICE] = 0.5
    l3, parts = group_losses(pred, t3, mask, limits, groups, {})
    assert float(l3) > float(l1)
    assert float(parts["hands"]) > 0.0


def test_causal_output_does_not_see_future():
    net = FeedbackNet(_cfg(bidirectional=False), FD.residual_limits()).eval()
    x = torch.randn(1, 10, 24)
    x2 = x.clone(); x2[:, 6:] += 10.0
    with torch.no_grad():
        a = net(x)
        b = net(x2)
    assert (a[:, :6] - b[:, :6]).abs().max() == 0.0
    assert (a[:, 6:] - b[:, 6:]).abs().max() > 0.0


def test_smooth_residual_gating():
    from vtv.eval_helpers import smooth_residual
    n = 9
    res = np.ones((n, FD.RESIDUAL_DIM), np.float32)
    level = np.zeros((n, 2), np.int8)
    level[:3] = 2
    level[3:6, 1] = 2
    sm = smooth_residual(res, level, win=3)
    assert np.allclose(sm[:, FD.ROT_SLICE], 1.0)          # dense body keeps 1
    # both hands invalid for 0..2; right hand also invalid for 3..5
    assert np.allclose(sm[:3, FD.HAND_SLICE.start:FD.HAND_SLICE.start + 19], 0.0)
    assert np.allclose(sm[3:, FD.HAND_SLICE.start:FD.HAND_SLICE.start + 19], 1.0)
    assert np.allclose(sm[:6, FD.HAND_SLICE.start + 19:FD.HAND_SLICE.stop], 0.0)
    assert np.allclose(sm[6:, FD.HAND_SLICE.start + 19:FD.HAND_SLICE.stop], 1.0)
    assert np.allclose(smooth_residual(res, level, win=1), res)
    fl = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0], np.int8)
    smf = smooth_residual(res, level, fl, win=3)
    assert np.allclose(smf[fl > 1, FD.FACE_SLICE], 0.0)
    assert np.allclose(smf[fl <= 1, FD.FACE_SLICE], 1.0)


def test_mlp_arch():
    net = FeedbackNet(_cfg(arch="mlp"), FD.residual_limits()).eval()
    with torch.no_grad():
        y = net(torch.randn(3, 7, 24))
    assert y.shape == (3, 7, FD.RESIDUAL_DIM)
    assert torch.isfinite(y).all()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"test_feedback_net: OK ({len(tests)} tests)")
