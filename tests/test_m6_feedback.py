"""M6 v2 tests: wider teacher bounds, v2 features, arches and two-pass compose.

Run:  python tests/test_m6_feedback.py
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from vtv import feedback_data as FD  # noqa: E402
from vtv import feedback_apply as FA  # noqa: E402
from vtv.feedback_net import FeedbackNet, NetConfig  # noqa: E402


def test_v2_limits():
    v1 = FD.residual_limits()
    v2 = FD.residual_limits("v2")
    assert v1.shape == v2.shape == (FD.RESIDUAL_DIM,)
    # only the shoulder chain widens
    i = FD.BODY_FACTOR_NAMES.index("clavicle_l")
    assert v2[2 + 3 * i] > v1[2 + 3 * i]
    assert np.all(v2[FD.ROOT_SLICE] == v1[FD.ROOT_SLICE])
    assert np.all(v2[FD.HAND_SLICE] == v1[FD.HAND_SLICE])
    assert np.all(v2[FD.FACE_SLICE] == v1[FD.FACE_SLICE])


def test_extra_train_view():
    assert FD.assign_split("R1", "si", "front_left") == "train"
    assert FD.assign_split("R1", "luoxi", "front_left") == "train"
    assert FD.assign_split("R1", "tiflos", "front_left") == "val"
    recs = [{"clip": "R1_si_front_left", "motion": "R1", "model": "si",
             "view": "front_left", "split": "train"},
            {"clip": "R1_si_side", "motion": "R1", "model": "si",
             "view": "side", "split": "val"},
            {"clip": "R3_si_front", "motion": "R3", "model": "si",
             "view": "front", "split": "test"}]
    rep = FD.leakage_report(recs)
    assert rep["ok"], rep
    assert "side" not in rep["views"]["train"]


def test_v2_feature_names():
    v1 = FD.feature_names(None, "v1")
    v2 = FD.feature_names(None, "v2")
    assert len(v2) == len(v1) + 2 * len(FD.HAND_JOINT_KEYS) + 12
    assert "hand_base_l_index.mcp" in v2
    assert "face_base_blink" in v2


def _cfg(**kw):
    base = dict(feat_dim=289, out_dim=FD.RESIDUAL_DIM, hidden=64, layers=2,
                dropout=0.0, arch="gru", bidirectional=True, window=9,
                heads="shared", causal=False)
    base.update(kw)
    return NetConfig(**base)


def test_arches_forward():
    lim = FD.residual_limits("v2")
    x = torch.randn(2, 9, 289)
    for arch in ("gru", "mlp", "tcn", "transformer", "mrn", "graph"):
        for heads in ("shared", "factor"):
            net = FeedbackNet(_cfg(arch=arch, heads=heads), lim).eval()
            with torch.no_grad():
                y = net(x)
            assert y.shape == (2, 9, FD.RESIDUAL_DIM), (arch, heads)
            assert torch.isfinite(y).all()
            assert (y.abs() <= torch.tensor(lim) + 1e-6).all()


def test_causal_tcn_and_transformer():
    lim = FD.residual_limits("v2")
    x = torch.randn(1, 10, 289)
    x2 = x.clone(); x2[:, 6:] += 10.0
    for arch in ("tcn", "transformer"):
        net = FeedbackNet(_cfg(arch=arch, causal=True, bidirectional=False),
                          lim).eval()
        with torch.no_grad():
            a, b = net(x), net(x2)
        assert (a[:, :6] - b[:, :6]).abs().max() == 0.0, arch
        assert (a[:, 6:] - b[:, 6:]).abs().max() > 0.0, arch


def test_v2_modes():
    v2 = FA.resolve_config("v2")
    assert v2["arm_gain"] == 0.3
    assert not v2["apply_hands"] and not v2["apply_face"]
    full = FA.resolve_config("v2_full")
    assert full["apply_hands"] and full["apply_face"]
    assert full["torso_gain"] == 0.85


def test_compose_two_passes():
    from vtv.feedback_apply_v2 import _compose
    lim = FD.residual_limits("v2")
    r1 = np.zeros((2, FD.RESIDUAL_DIM), np.float32)
    r2 = np.zeros((2, FD.RESIDUAL_DIM), np.float32)
    i = FD.BODY_FACTOR_NAMES.index("upperarm_l")
    r1[:, 2 + 3 * i:5 + 3 * i] = [3.0, 0.0, 0.0]
    r2[:, 2 + 3 * i:5 + 3 * i] = [4.0, 0.0, 0.0]
    out = _compose(r1, r2, lim)
    # same axis -> additive
    assert np.allclose(out[:, 2 + 3 * i], 7.0, atol=1e-4)
    # root adds
    r1[:, 0] = 0.2
    r2[:, 0] = 0.3
    assert np.isclose(_compose(r1, r2, lim)[0, 0], 0.5, atol=1e-6)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"test_m6_feedback: OK ({len(tests)} tests)")
