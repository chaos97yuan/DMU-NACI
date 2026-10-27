"""Uncertainty-specific sequence encoders (Stage-U).

Four encoders map the raw standardized window x in [B, 72, 5] to a unified
risk representation H_sigma in [B, 32]. All of them are trained ONLY through
the shared Gaussian heteroscedastic NLL on the frozen point-prediction
residuals; none of them touches the point-prediction path.

  U0 RawMLPEncoder           : flatten -> Linear(360,32) -> GELU
                               (bitwise equivalent architecture of the Stage-D1
                               R1 NoiseHead(360) first layers)
  U1 TCNUncertaintyEncoder   : causal dilated TCN (src/models/tcn_blocks.py,
                               channels=[32]x6, kernel=3, dilations=[1,2,4,8,
                               16,32], receptive field 127 >= 72), last step
  U2 GRUUncertaintyEncoder   : unidirectional 1-layer GRU(hidden=32), final h_n
  U3 PatchTSTUncertaintyEncoder: patch_len=12, stride=6 -> 11 patches x 5 vars
                               -> shared Linear(12->32) patch embedding + learnable
                               positional & variable embeddings -> 2-layer
                               Transformer encoder (d_model=32, nhead=4, d_ff=64,
                               dropout=0.1) -> mean pooling

All encoders are followed by the SAME scale head Linear(32, 1) and the SAME
sigma transform softplus(a) + NOISE_FLOOR (variance, scaled target space),
identical to src/models/bscn_heteroscedastic.py::NoiseHead.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.models.tcn_blocks import TemporalConvNet, compute_receptive_field

NOISE_FLOOR = 1e-6
H_SIGMA_DIM = 32


class RawMLPEncoder(nn.Module):
    """U0: flattened raw window -> Linear(360,32) -> GELU -> [B,32]."""

    def __init__(self, seq_len: int = 72, n_vars: int = 5, hidden_dim: int = H_SIGMA_DIM) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.n_vars = int(n_vars)
        self.flat_dim = self.seq_len * self.n_vars
        self.net = nn.Sequential(nn.Linear(self.flat_dim, hidden_dim), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.n_vars):
            raise ValueError(f"RawMLPEncoder expects [B,{self.seq_len},{self.n_vars}], got {tuple(x.shape)}.")
        return self.net(x.reshape(x.shape[0], -1))


class TCNUncertaintyEncoder(nn.Module):
    """U1: dedicated causal-dilated TCN for the scale task -> last step [B,32]."""

    def __init__(self, seq_len: int = 72, n_vars: int = 5, hidden_dim: int = H_SIGMA_DIM, kernel_size: int = 3,
                 n_blocks: int = 6, dropout: float = 0.1) -> None:
        super().__init__()
        dilations = [2 ** i for i in range(n_blocks)]
        self.seq_len = int(seq_len)
        self.tcn = TemporalConvNet(num_inputs=n_vars, channels=[hidden_dim] * n_blocks,
                                   kernel_size=kernel_size, dropout=dropout, dilations=dilations)
        self.receptive_field = compute_receptive_field(kernel_size, dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"TCNUncertaintyEncoder expects [B, C, T]-ready [B,72,5], got {tuple(x.shape)}.")
        h = self.tcn(x.transpose(1, 2))  # [B, 32, 72]
        return h[:, :, -1]


class GRUUncertaintyEncoder(nn.Module):
    """U2: unidirectional 1-layer GRU(hidden=32) -> final hidden state [B,32]."""

    def __init__(self, seq_len: int = 72, n_vars: int = 5, hidden_dim: int = H_SIGMA_DIM, num_layers: int = 1) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.gru = nn.GRU(input_size=n_vars, hidden_size=hidden_dim, num_layers=num_layers,
                          batch_first=True, bidirectional=False)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"GRUUncertaintyEncoder expects [B,72,5], got {tuple(x.shape)}.")
        _, h_n = self.gru(x)
        return h_n[-1]


class PatchTSTUncertaintyEncoder(nn.Module):
    """U3: lightweight PatchTST-style encoder -> mean-pooled [B,32].

    patch_len=12 (1 h at 5-min resolution), stride=6 -> 11 patches per variable;
    channel-independent patch embedding (shared Linear), learnable positional and
    variable embeddings, tokens = vars x patches; 2 TransformerEncoder layers;
    fixed mean pooling over tokens.
    """

    def __init__(self, seq_len: int = 72, n_vars: int = 5, patch_len: int = 12, stride: int = 6,
                 d_model: int = H_SIGMA_DIM, n_heads: int = 4, e_layers: int = 2,
                 d_ff: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        if (seq_len - patch_len) % stride != 0:
            raise ValueError("seq_len - patch_len must be divisible by stride.")
        self.seq_len = seq_len
        self.n_vars = n_vars
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (seq_len - patch_len) // stride + 1
        self.d_model = d_model

        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(self.n_patches, d_model))
        self.var_embed = nn.Parameter(torch.zeros(n_vars, d_model))
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.norm = nn.LayerNorm(d_model)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.var_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.n_vars):
            raise ValueError(f"PatchTSTUncertaintyEncoder expects [B,{self.seq_len},{self.n_vars}], got {tuple(x.shape)}.")
        B = x.shape[0]
        # [B, n_vars, n_patches, patch_len]
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride).transpose(1, 2)
        tokens = self.patch_embed(patches)                       # [B, n_vars, n_patches, d]
        tokens = tokens + self.pos_embed[None, None, :, :] + self.var_embed[None, :, None, :]
        tokens = self.dropout(tokens).reshape(B, self.n_vars * self.n_patches, self.d_model)
        h = self.encoder(tokens)
        return self.norm(h).mean(dim=1)                          # fixed mean pooling -> [B,32]


ENCODER_FACTORY = {
    "RawMLP": RawMLPEncoder,
    "TCN": TCNUncertaintyEncoder,
    "GRU": GRUUncertaintyEncoder,
    "PatchTST": PatchTSTUncertaintyEncoder,
}


class ScaleModel(nn.Module):
    """encoder -> H_sigma[32] -> shared Linear(32,1) -> softplus+1e-6 variance.

    The sigma transform and noise floor are IDENTICAL for every encoder and
    identical to Stage-D1 / src/models/bscn_heteroscedastic.py::NoiseHead.
    """

    def __init__(self, encoder_name: str, seq_len: int = 72, n_vars: int = 5) -> None:
        super().__init__()
        if encoder_name not in ENCODER_FACTORY:
            raise ValueError(f"unknown encoder {encoder_name}")
        self.encoder = ENCODER_FACTORY[encoder_name](seq_len=seq_len, n_vars=n_vars)
        self.scale_head = nn.Linear(H_SIGMA_DIM, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def noise_variance(self, x: torch.Tensor) -> torch.Tensor:
        """v = softplus(scale_raw) + 1e-6 (VARIANCE, scaled target space)."""
        return nn.functional.softplus(self.scale_head(self.encoder(x)).squeeze(-1)) + NOISE_FLOOR

    def parameter_report(self) -> dict:
        enc = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        head = sum(p.numel() for p in self.scale_head.parameters() if p.requires_grad)
        return {"encoder_params": int(enc), "scale_head_params": int(head),
                "total_params": int(enc + head), "h_sigma_dim": H_SIGMA_DIM}
