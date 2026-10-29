"""Optional VPoser v2 prior wrapper for M2.4d.

The official VPoser v2 model code and weights are MPG-licensed,
research/non-commercial assets. They are kept outside the GPL-3.0-only core:

  models/vposer/V02_05/           # weights + yaml (git-ignored)
  models/vposer/code/             # official v2 model code (git-ignored)

This module only contains loader/adaptor code (GPL-3.0-only). It imports the
official local model class when available. If the files are absent, users must
download VPoser v2 from https://smpl-x.is.tue.mpg.de/ and fetch the official
model code from https://github.com/nghorbani/human_body_prior.

Usage:
    from vtv.vposer_prior import load_vposer_v2
    vp = load_vposer_v2()
    q = vp.encode(body_pose_aa)       # (N,63) or (N,21,3)
    out = vp.decode(q.mean)           # {'pose_body': (N,21,3), ...}
    kl = vp.kl(q)                     # (N,)
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Optional

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CKPT = os.path.join(
    ROOT, "models", "vposer", "V02_05", "snapshots",
    "V02_05_epoch=13_val_loss=0.03.ckpt",
)
DEFAULT_CODE_DIR = os.path.join(ROOT, "models", "vposer", "code")


class VPoserPrior:
    """Thin wrapper around the official VPoser v2 nn.Module."""

    def __init__(self, model, latent_dim: int = 32, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.latent_dim = latent_dim
        self.device = device

    def encode(self, pose_body):
        x = torch.as_tensor(pose_body, dtype=torch.float32, device=self.device)
        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)
        if x.shape[-1] != 63:
            raise ValueError(f"VPoser v2 expects (N,63) body_pose, got {tuple(x.shape)}")
        return self.model.encode(x)

    def decode(self, z):
        z = torch.as_tensor(z, dtype=torch.float32, device=self.device)
        return self.model.decode(z)

    def kl(self, q):
        """KL(q(z|x) || N(0,I)) per sample, shape (N,)."""
        var = q.scale ** 2
        return 0.5 * (q.mean ** 2 + var - 1.0 - torch.log(var)).sum(-1)

    def prior_nll(self, pose_body):
        """Latent KL(q(z|pose) || N(0,I)); this is a prior energy, not a pose likelihood."""
        q = self.encode(pose_body)
        return self.kl(q)


def load_vposer_v2(
    ckpt: Optional[str] = None,
    code_dir: Optional[str] = None,
    device: str = "cpu",
    num_neurons: int = 512,
    latent_dim: int = 32,
) -> VPoserPrior:
    """Load the local VPoser v2 checkpoint into the official v2 model class."""
    ckpt = os.path.abspath(ckpt or DEFAULT_CKPT)
    code_dir = os.path.abspath(code_dir or DEFAULT_CODE_DIR)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"VPoser v2 checkpoint not found: {ckpt}. "
            "Download VPoser v2 from https://smpl-x.is.tue.mpg.de/ and place it under models/vposer/."
        )
    model_file = os.path.join(code_dir, "vposer_model.py")
    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"VPoser v2 model code not found: {model_file}. "
            "Fetch human_body_prior/models/ from https://github.com/nghorbani/human_body_prior."
        )

    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    from vposer_model import VPoser  # type: ignore

    obj = torch.load(ckpt, map_location="cpu")
    state = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    prefix = "vp_model."
    state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not state:
        raise ValueError("no vp_model.* weights found in VPoser checkpoint")

    ps = SimpleNamespace(model_params=SimpleNamespace(num_neurons=num_neurons, latentD=latent_dim))
    model = VPoser(ps)
    model.load_state_dict(state, strict=True)
    return VPoserPrior(model, latent_dim=latent_dim, device=device)
