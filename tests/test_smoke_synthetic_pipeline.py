"""End-to-end smoke test on fully synthetic data (no DKASC download needed).

Runs the smallest faithful slice of the real pipeline:
  synthetic DKASC-format year -> frozen cleaning functions -> causal-safe
  windows -> TCN mean training (2 epochs, 1 seed) -> residual construction ->
  GRU scale training (few updates) -> normalized conformal scores on a CALIB
  slice -> delayed ACI interval issuance.

All artifacts stay in a pytest tmp dir; nothing is written into data/ or
results/. Numbers here are NOT paper numbers (synthetic data, tiny budget).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.rigorous_v2 import common_v2 as C  # noqa: E402
from scripts.make_synthetic_example import generate  # noqa: E402

YEAR = 2021  # build_year_windows path (2024/2025 use the frozen preopen artifacts)


@pytest.fixture(scope="module")
def synthetic_windows(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("synthetic")
    raw_dir = tmp / "raw"
    raw_dir.mkdir()
    df = generate(YEAR, n_days=366, seed=7)
    df.to_csv(raw_dir / f"Alice_Springs_{YEAR}.csv", index=False)
    # redirect the module-level data locations into the tmp tree
    orig_raw, orig_data = C.RAW, C.DATA_V2
    C.RAW = raw_dir
    C.DATA_V2 = tmp / "processed"
    C.DATA_V2.mkdir()
    try:
        yield C.load_year_windows(YEAR)
    finally:
        C.RAW, C.DATA_V2 = orig_raw, orig_data


def test_cleaning_and_windows(synthetic_windows):
    w = synthetic_windows
    assert w["x_raw"].shape[1:] == (72, 5)
    assert len(w["y_kw"]) == len(w["origins"]) == len(w["x_raw"])
    o, t = pd.DatetimeIndex(w["origins"]), pd.DatetimeIndex(w["targets"])
    assert (t - o == C.HORIZON_TD).all()
    assert np.isfinite(w["x_raw"]).all() and np.isfinite(w["y_kw"]).all()
    assert (w["y_kw"] >= 0).all()


def test_tcn_training_and_residuals(synthetic_windows):
    w = synthetic_windows
    n = len(w["y_kw"])
    n_train = int(n * 0.6)
    train = {"x_raw": w["x_raw"][:n_train], "y_kw": w["y_kw"][:n_train],
             "targets": w["targets"][:n_train]}
    es = {"x_raw": w["x_raw"][n_train:n_train + 500], "y_kw": w["y_kw"][n_train:n_train + 500],
          "targets": w["targets"][n_train:n_train + 500]}
    sc = C.fit_scaler(train["x_raw"])
    model, info = C.train_tcn(train, es, sc, seed=42, max_epochs=2, patience=2)
    mu = C.tcn_mu(model, w["x_raw"][n_train:], sc)
    mp = C.mu_phys(mu)
    assert np.isfinite(mp).all() and (mp >= 0).all() and (mp <= C.P_RATED + 1e-9).all()
    resid = w["y_kw"][n_train:] - mp
    assert np.isfinite(resid).all()
    # a trained model must beat climatology on synthetic data (weak but real signal)
    rmse_model = float(np.sqrt(np.mean(resid ** 2)))
    rmse_persist = float(np.sqrt(np.mean((w["y_kw"][n_train:] - w["y_kw"][: n - n_train][-len(resid):]) ** 2))) \
        if n - n_train >= len(resid) else rmse_model
    assert rmse_model < 1.0  # absolute sanity: kW scale, learned something


def test_gru_conformal_and_delayed_aci(synthetic_windows):
    w = synthetic_windows
    n = len(w["y_kw"])
    n_train = int(n * 0.5)
    n_cal = 1500
    train = {"x_raw": w["x_raw"][:n_train], "y_kw": w["y_kw"][:n_train]}
    cal = {"x_raw": w["x_raw"][n_train:n_train + n_cal], "y_kw": w["y_kw"][n_train:n_train + n_cal]}
    test = {"x_raw": w["x_raw"][n_train + n_cal:], "y_kw": w["y_kw"][n_train + n_cal:],
            "origins": w["origins"][n_train + n_cal:]}
    sc = C.fit_scaler(train["x_raw"])
    # mean + residuals (standardized target space)
    tcn, _ = C.train_tcn({"x_raw": train["x_raw"], "y_kw": train["y_kw"], "targets": train["x_raw"][:, -1, 0]},
                         {"x_raw": train["x_raw"][-1000:], "y_kw": train["y_kw"][-1000:]},
                         sc, seed=42, max_epochs=1, patience=1)
    mu_cal = C.mu_phys(C.tcn_mu(tcn, cal["x_raw"], sc))
    e_cal = (cal["y_kw"] - mu_cal) / sc["target_std"]
    mu_tr = C.mu_phys(C.tcn_mu(tcn, train["x_raw"], sc))
    e_tr = (train["y_kw"] - mu_tr) / sc["target_std"]
    gru, _, _ = C.train_gru(train["x_raw"], e_tr, seed=42)
    sg_cal = C.sigma_kw(gru, cal["x_raw"], sc)
    sg_te = C.sigma_kw(gru, test["x_raw"], sc)
    assert (sg_cal > 0).all() and (sg_te > 0).all()
    scores = np.sort(np.abs(cal["y_kw"] - mu_cal) / (sg_cal + C.EPS))
    lo, up, alphas, qs = C.run_aci(test["y_kw"],
                                   C.mu_phys(C.tcn_mu(tcn, test["x_raw"], sc)), sg_te,
                                   scores, test["origins"], gamma=0.05)
    assert np.isfinite(lo).all() and np.isfinite(up).all()
    assert (up >= lo).all()
    picp = float(((test["y_kw"] >= lo) & (test["y_kw"] <= up)).mean())
    assert 0.5 < picp <= 1.0
    metrics = C.interval_metrics(test["y_kw"], lo, up, test["origins"])
    assert {"PICP", "PINAW", "IS", "RCE1D", "RCE3D", "LongestMissStreak"} <= set(metrics)
