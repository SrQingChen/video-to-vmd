# -*- coding: utf-8 -*-
"""Bounded residual feedback networks (M3.5 baseline + M6 variants).

The network consumes evidence features only and predicts a bounded correction
in the factor space defined by :mod:`vtv.feedback_data`:

    root XZ (2) + 9 upper-body local rotations (27)
    + 2x19 hand angles (38) + 12 face channels = 79 dims.

Every output is ``tanh * limit`` so a failure mode is a small correction, not a
broken pose.

Architectures (M6 comparison):
  * ``gru``        - M3.5 BiGRU / causal GRU baseline
  * ``mlp``        - frame-wise control
  * ``tcn``        - dilated temporal convolutions (causal or bidirectional)
  * ``transformer``- pre-norm Transformer encoder (causal mask optional)
  * ``mrn``        - MotionRefineNet-style temporal + frequency branches
  * ``graph``      - kinematic bone graph over the 15 direction bones + global

``heads="factor"`` gives root/body/hands/face independent MLP heads; the
default is one shared head.  The checkpoint config is the source of truth for
feature/output dimensions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - torch is a runtime dependency
    torch = None
    nn = object
    F = None

from . import feedback_data as FD


@dataclass
class NetConfig:
    feat_dim: int = 217
    out_dim: int = 91
    hidden: int = 256
    layers: int = 2
    dropout: float = 0.10
    arch: str = "gru"          # gru | mlp | tcn | transformer | mrn | graph
    bidirectional: bool = True
    window: int = 31
    heads: str = "shared"      # shared | factor
    causal: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------
class _TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float,
                 causal: bool):
        super().__init__()
        self.causal = causal
        self.dilation = dilation
        self.conv1 = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)

    def _conv(self, conv, x):
        # x: (B,C,T)
        if self.causal:
            x = F.pad(x, (2 * self.dilation, 0))
            return conv(x)
        return conv(F.pad(x, (self.dilation, self.dilation), mode="replicate"))

    def forward(self, x):                         # (B,C,T)
        h = self._conv(self.conv1, x)
        h = self.norm1(h.transpose(1, 2)).transpose(1, 2)
        h = self.drop(F.gelu(h))
        h = self._conv(self.conv2, h)
        h = self.norm2(h.transpose(1, 2)).transpose(1, 2)
        return x + self.drop(F.gelu(h))


class _TCN(nn.Module):
    def __init__(self, feat_dim: int, hidden: int, layers: int,
                 dropout: float, causal: bool):
        super().__init__()
        self.inp = nn.Conv1d(feat_dim, hidden, 1)
        self.blocks = nn.ModuleList([
            _TCNBlock(hidden, 2 ** i, dropout, causal) for i in range(layers)])
        self.out = nn.Conv1d(hidden, hidden, 1)

    def forward(self, x):                         # (B,T,F)
        h = self.inp(x.transpose(1, 2))
        for b in self.blocks:
            h = b(h)
        return self.out(h).transpose(1, 2)


class _PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float()
                        * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):                         # (B,T,D)
        return x + self.pe[:, :x.shape[1]].to(x.dtype)


def _causal_mask(length: int, device):
    return torch.triu(torch.full((length, length), float("-inf"),
                                 device=device), diagonal=1)


class _TransformerEncoder(nn.Module):
    def __init__(self, feat_dim: int, hidden: int, layers: int,
                 dropout: float, causal: bool):
        super().__init__()
        self.causal = causal
        self.inp = nn.Linear(feat_dim, hidden)
        self.pos = _PositionalEncoding(hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=hidden * 2,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, x):
        z = self.pos(self.inp(x))
        mask = _causal_mask(z.shape[1], z.device) if self.causal else None
        return self.enc(z, mask=mask)


class _MRN(nn.Module):
    """Temporal TCN branch + length-agnostic frequency branch.

    The frequency branch applies a shared per-bin MLP to the real/imaginary
    parts of the rFFT, so the same weights work for a 31-frame training window
    and a full 500+ frame inference sequence.
    """

    def __init__(self, feat_dim: int, hidden: int, layers: int,
                 dropout: float):
        super().__init__()
        self.temporal = _TCN(feat_dim, hidden, layers, dropout, causal=False)
        self.freq_in = nn.Linear(2 * feat_dim, hidden)
        self.freq_out = nn.Linear(hidden, 2 * feat_dim)
        self.freq_proj = nn.Linear(feat_dim, hidden)
        self.fuse = nn.Linear(2 * hidden, hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        zt = self.temporal(x)
        spec = torch.fft.rfft(x, dim=1)           # (B,Ff,F) complex
        s = torch.cat([spec.real, spec.imag], dim=-1)
        s = self.drop(F.gelu(self.freq_in(s)))
        s = self.freq_out(s)
        spec2 = torch.complex(s[..., :x.shape[-1]], s[..., x.shape[-1]:])
        zf = torch.fft.irfft(spec2, n=x.shape[1], dim=1)
        zf = self.freq_proj(zf)
        z = self.fuse(torch.cat([zt, zf], dim=-1))
        return self.drop(F.gelu(z))


class _GraphConv(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.w = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, adj):                    # (B,T,N,D), (N,N)
        out = torch.einsum("ij,btjd->btid", adj, self.w(h))
        return h + self.drop(F.gelu(self.norm(out)))


class _BoneGraph(nn.Module):
    """Kinematic graph over ``FEATURE_DIR_BONES`` + a global feature branch."""

    def __init__(self, feat_dim: int, hidden: int, layers: int,
                 dropout: float):
        super().__init__()
        self.bones = FD.FEATURE_DIR_BONES
        self.nb = len(self.bones)
        self.dir_start = 9                     # root 1 + pelvis q 4 + dq 4
        self.dir_len = 3 * self.nb
        self.vel_start = self.dir_start + self.dir_len
        self.vel_len = 3 * self.nb
        self.bone_dim = 6
        self.inp = nn.Linear(self.bone_dim, hidden)
        self.blocks = nn.ModuleList([_GraphConv(hidden, dropout)
                                     for _ in range(layers)])
        rest_dim = feat_dim - self.dir_len - self.vel_len
        self.global_in = nn.Linear(rest_dim, hidden)
        self.fuse = nn.Linear(2 * hidden, hidden)
        adj = torch.eye(self.nb)
        from .canonical_body import CANONICAL_PARENT
        idx = {b: i for i, b in enumerate(self.bones)}
        for b, p in CANONICAL_PARENT.items():
            if b in idx and p in idx:
                adj[idx[b], idx[p]] = 1.0
                adj[idx[p], idx[b]] = 1.0
        adj = adj / adj.sum(dim=1, keepdim=True).clamp_min(1.0)
        self.register_buffer("adj", adj)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape
        d = x[:, :, self.dir_start:self.dir_start + self.dir_len]
        v = x[:, :, self.vel_start:self.vel_start + self.vel_len]
        h = torch.cat([d.reshape(B, T, self.nb, 3),
                       v.reshape(B, T, self.nb, 3)], dim=-1)
        h = self.inp(h)
        for blk in self.blocks:
            h = blk(h, self.adj.to(h.dtype))
        g = self.global_in(torch.cat([
            x[:, :, :self.dir_start], x[:, :, self.vel_start + self.vel_len:]],
            dim=-1))
        z = self.fuse(torch.cat([h.mean(dim=2), g], dim=-1))
        return self.drop(F.gelu(z))


# ---------------------------------------------------------------------------
# main model
# ---------------------------------------------------------------------------
class FeedbackNet(nn.Module):
    def __init__(self, cfg: NetConfig, limits: np.ndarray):
        super().__init__()
        self.cfg = cfg
        self.norm = nn.LayerNorm(cfg.feat_dim)
        if cfg.arch == "gru":
            self.encoder = nn.GRU(
                cfg.feat_dim, cfg.hidden, num_layers=cfg.layers,
                batch_first=True,
                bidirectional=cfg.bidirectional and not cfg.causal,
                dropout=cfg.dropout if cfg.layers > 1 else 0.0)
            h = cfg.hidden * (2 if (cfg.bidirectional and not cfg.causal) else 1)
        elif cfg.arch == "mlp":
            self.encoder = nn.Sequential(
                nn.Linear(cfg.feat_dim, cfg.hidden), nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden, cfg.hidden), nn.GELU())
            h = cfg.hidden
        elif cfg.arch == "tcn":
            self.encoder = _TCN(cfg.feat_dim, cfg.hidden, cfg.layers,
                                cfg.dropout, cfg.causal)
            h = cfg.hidden
        elif cfg.arch == "transformer":
            self.encoder = _TransformerEncoder(
                cfg.feat_dim, cfg.hidden, cfg.layers, cfg.dropout,
                cfg.causal)
            h = cfg.hidden
        elif cfg.arch == "mrn":
            self.encoder = _MRN(cfg.feat_dim, cfg.hidden, cfg.layers,
                                cfg.dropout)
            h = cfg.hidden
        elif cfg.arch == "graph":
            self.encoder = _BoneGraph(cfg.feat_dim, cfg.hidden, cfg.layers,
                                      cfg.dropout)
            h = cfg.hidden
        else:
            raise ValueError(f"unknown arch {cfg.arch}")

        self.heads_kind = getattr(cfg, "heads", "shared")
        if self.heads_kind == "factor":
            slices = FD.residual_groups()
            self.heads = nn.ModuleDict({
                name: nn.Sequential(
                    nn.Linear(h, max(64, h // 2)), nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(max(64, h // 2), sl.stop - sl.start))
                for name, sl in slices.items()})
            self.group_slices = slices
        else:
            self.head = nn.Sequential(
                nn.Linear(h, cfg.hidden), nn.GELU(), nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden, cfg.out_dim))
        self.register_buffer("limits", torch.tensor(limits, dtype=torch.float32))

    def forward(self, x):                     # (B,T,F)
        z = self.norm(x)
        if self.cfg.arch == "gru":
            z, _ = self.encoder(z)
        else:
            z = self.encoder(z)
        if self.heads_kind == "factor":
            parts = []
            for name, sl in self.group_slices.items():
                parts.append((name, sl, self.heads[name](z)))
            out = torch.zeros(*z.shape[:2], self.cfg.out_dim,
                              device=z.device, dtype=z.dtype)
            for _, sl, v in parts:
                out[..., sl] = v
        else:
            out = self.head(z)
        return torch.tanh(out) * self.limits

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def make_windows(features: np.ndarray, targets: np.ndarray, masks: np.ndarray,
                 window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centred windows with edge replication; one sample per frame."""
    n = len(features)
    half = window // 2
    pad = np.pad(features, ((half, window - 1 - half), (0, 0)), mode="edge")
    win = np.stack([pad[i:i + window] for i in range(n)], axis=0)
    return win.astype(np.float32), targets.astype(np.float32), masks.astype(np.float32)


def group_losses(pred, target, mask, limits, group_slices, group_weights,
                 smooth_weight=0.05, mag_weight=0.01):
    """Masked Huber per factor group + temporal/magnitude regularisers."""
    err = (pred - target) / limits
    huber = torch.where(err.abs() < 1.0, 0.5 * err ** 2, err.abs() - 0.5)
    total = 0.0
    parts = {}
    for name, sl in group_slices.items():
        m = mask[..., sl]
        denom = m.sum().clamp_min(1.0)
        val = (huber[..., sl] * m).sum() / denom
        parts[name] = val
        total = total + group_weights.get(name, 1.0) * val
    if smooth_weight > 0 and pred.dim() == 3 and pred.shape[1] > 1:
        d = (pred[:, 1:] - pred[:, :-1]) / limits
        parts["smooth"] = d.pow(2).mean()
        total = total + smooth_weight * parts["smooth"]
    if mag_weight > 0:
        parts["magnitude"] = (pred / limits).pow(2).mean()
        total = total + mag_weight * parts["magnitude"]
    return total, parts


def select_device():
    if torch is None:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def masked_rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
                limits: np.ndarray) -> Dict[str, float]:
    from .feedback_data import residual_groups
    out = {}
    e = (pred - target) / limits
    for name, sl in residual_groups().items():
        m = mask[:, sl] > 0
        if not m.any():
            out[name] = float("nan")
            continue
        out[name] = float(np.sqrt(np.mean(e[:, sl][m] ** 2)))
    return out


def causal_mask(batch: int, length: int, device) -> "torch.Tensor":
    """Upper-triangular mask for the causal ablation (not used by the offline net)."""
    return torch.triu(torch.ones(length, length, device=device), diagonal=1).bool()
