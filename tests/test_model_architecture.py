"""Architecture checks against the paper's stated parameter counts.

Paper (Table S1 / Section 2.2): Multi-scale TCN 36,705 params; GRU scale model
3,777 params. If either architecture changes, this test must be updated
together with the paper -- that is the point.
"""

from __future__ import annotations

import torch

from src.models.multiscale_tcn import MultiScaleTCN, count_trainable_parameters
from src.models.uncertainty_encoders import ScaleModel
from src.data.schema import FEATURE_COLUMNS, HISTORY_STEPS


def test_multiscale_tcn_param_count_and_shapes():
    torch.manual_seed(0)
    model = MultiScaleTCN()
    assert count_trainable_parameters(model) == 36_705
    x = torch.randn(4, HISTORY_STEPS, len(FEATURE_COLUMNS))
    h = model.forward_features(x)
    assert h.shape == (4, 64)
    y = model(x)
    assert y.shape == (4,)
    assert torch.isfinite(y).all()


def test_gru_scale_model_param_count_and_variance():
    model = ScaleModel("GRU")
    rep = model.parameter_report()
    assert rep["total_params"] == 3_777
    assert rep["h_sigma_dim"] == 32
    x = torch.randn(6, HISTORY_STEPS, len(FEATURE_COLUMNS))
    v = model.noise_variance(x)
    assert v.shape == (6,)
    assert (v > 0).all(), "noise variance must be strictly positive (softplus + 1e-6)"


def test_cqr_head_matches_table5_protocol():
    import torch.nn as nn

    class CQRNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = MultiScaleTCN(dropout=0.1)
            self.head = nn.Linear(64, 3)

        def forward(self, x):
            return self.head(self.trunk.forward_features(x))

    model = CQRNet()
    x = torch.randn(2, HISTORY_STEPS, len(FEATURE_COLUMNS))
    q = model(x)
    assert q.shape == (2, 3)
