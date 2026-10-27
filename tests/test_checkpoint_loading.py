"""Every archived checkpoint must load into the exact architecture it was
trained with (verified by state-dict key/shape match), and the stored scaler
fingerprints must match the recomputation from the stored scaler arrays.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments.rigorous_v2 import common_v2 as C
from src.models.multiscale_tcn import MultiScaleTCN
from src.models.uncertainty_encoders import ScaleModel

ROOT = Path(__file__).resolve().parents[1]
CK = ROOT / "checkpoints"

SEEDS = [42, 123, 2026, 3407, 8888]


def _load(p: Path) -> dict:
    return torch.load(p, weights_only=False, map_location="cpu")


def _site18_scaler() -> dict:
    raw = json.loads((CK / "site18" / "site18_scaler.json").read_text())
    return {"mean": raw["mean"], "std": raw["std"],
            "target_mean": raw["target_mean"], "target_std": raw["target_std"]}


# --------------------------------------------------------------- Site 18 (5.2)
def test_site18_mean_tcns():
    arch = MultiScaleTCN(dropout=0.1)
    for seed in SEEDS:
        ck = _load(CK / "site18" / f"final_mean_seed{seed}.pt")
        arch.load_state_dict(ck["state_dict"])
        assert ck["seed"] == seed
        assert ck["cfg"] == {"dropout": 0.1, "lr": 1e-3}


def test_site18_scaler_fingerprint_matches_checkpoints():
    import numpy as np
    raw = json.loads((CK / "site18" / "site18_scaler.json").read_text())
    sc = {"mean": np.asarray(raw["mean"], float), "std": np.asarray(raw["std"], float),
          "target_mean": float(raw["target_mean"]), "target_std": float(raw["target_std"])}
    fp = C.scaler_fingerprint(sc)
    for seed in SEEDS:
        ck = _load(CK / "site18" / f"final_mean_seed{seed}.pt")
        assert ck["scaler_fp"] == fp, f"scaler drift for seed {seed}"


def test_site18_gru_checkpoints_registry_matches_freeze_manifest():
    manifest = json.loads((ROOT / "results" / "rigorous_v2" / "FREEZE_MANIFEST.json").read_text())
    gru = manifest["gru_checkpoints"]
    for name, sha in gru.items():
        p = CK / "site18" / name
        assert p.exists(), f"missing archived GRU checkpoint {name}"
        assert C.sha256_file(p) == sha, f"sha mismatch for {name}"


def test_site18_mean_checkpoints_registry_matches_freeze_manifest():
    manifest = json.loads((ROOT / "results" / "rigorous_v2" / "FREEZE_MANIFEST.json").read_text())
    for name, sha in manifest["mean_checkpoints"].items():
        p = CK / "site18" / name
        assert p.exists(), f"missing archived mean checkpoint {name}"
        assert C.sha256_file(p) == sha, f"sha mismatch for {name}"


@pytest.mark.parametrize("seed", SEEDS)
def test_site18_gru_loads(seed):
    arch = ScaleModel("GRU")
    for variant in ("T1p", "T2p", "T3p", "T1in", "T3match", "V1", "V2", "V3"):
        ck = _load(CK / "site18" / f"gru_{variant}_seed{seed}.pt")
        arch.load_state_dict(ck["state_dict"])


@pytest.mark.parametrize("fold", ["F1", "F2", "F3", "F4"])
@pytest.mark.parametrize("seed", SEEDS)
def test_site18_oof_fold_checkpoints_load(fold, seed):
    ck = _load(CK / "site18" / f"oof_{fold}_seed{seed}.pt")
    MultiScaleTCN(dropout=0.1).load_state_dict(ck["state_dict"])
    assert ck["fold"] == fold and ck["seed"] == seed


@pytest.mark.parametrize("tag", ["M22", "M23", "M24"])
@pytest.mark.parametrize("seed", SEEDS)
def test_site18_vintage_checkpoints_load(tag, seed):
    ck = _load(CK / "site18" / f"{tag}_seed{seed}.pt")
    MultiScaleTCN(dropout=0.1).load_state_dict(ck["state_dict"])
    assert ck["tag"] == tag and ck["seed"] == seed


def test_site18_cqr_checkpoints_load():
    from experiments.site1b_validation.p8b_site18_cqr import CQRNet
    arch = CQRNet()
    for seed in SEEDS:
        ck = _load(CK / "site18" / f"site18_cqr_seed{seed}.pt")
        arch.load_state_dict(ck["state_dict"])


# --------------------------------------------------------------- Site 1B (23.4)
@pytest.mark.parametrize("seed", SEEDS)
def test_site1b_mean_and_gru_load(seed):
    mean_ck = _load(CK / "site1b" / f"site1b_mean_checkpoint_seed{seed}.pt")
    MultiScaleTCN(dropout=0.1).load_state_dict(mean_ck["state_dict"])
    gru_ck = _load(CK / "site1b" / f"site1b_gru_T3p_seed{seed}.pt")
    ScaleModel("GRU").load_state_dict(gru_ck["state_dict"])


@pytest.mark.parametrize("fold", ["F1", "F2", "F3", "F4"])
@pytest.mark.parametrize("seed", SEEDS)
def test_site1b_oof_checkpoints_load(fold, seed):
    ck = _load(CK / "site1b" / f"site1b_oof_{fold}_seed{seed}.pt")
    MultiScaleTCN(dropout=0.1).load_state_dict(ck["state_dict"])


def test_site1b_cqr_checkpoints_load():
    from experiments.site1b_validation.common_1b import train_tcn_cqr  # noqa: F401
    import torch.nn as nn
    from src.models.multiscale_tcn import MultiScaleTCN

    class CQRNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = MultiScaleTCN(dropout=0.1)
            self.head = nn.Linear(64, 3)

        def forward(self, x):
            return self.head(self.trunk.forward_features(x))

    arch = CQRNet()
    for seed in SEEDS:
        ck = _load(CK / "site1b" / f"site1b_cqr_seed{seed}.pt")
        arch.load_state_dict(ck["state_dict"])
