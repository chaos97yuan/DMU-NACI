"""Site 1B cross-system validation -- shared adapter over rigorous_v2.

Design rule: every algorithmic decision (architecture, optimizer, training loop,
ACI, conformal quantile, bootstrap, role boundaries, OOF folds, seeds) is IMPORTED
from experiments/rigorous_v2.common_v2 so that Site 18 and Site 1B differ only in
data, rated capacity and retrained weights. Only capacity-dependent quantities
(mu_phys, PINAW/NIS normalization) and the Site 1B data loader live here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # release: repo-relative (was absolute path)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location(
    "rigorous_v2_common", PROJECT_ROOT / "experiments" / "rigorous_v2" / "common_v2.py")
V2 = importlib.util.module_from_spec(spec)
sys.modules["rigorous_v2_common"] = V2
spec.loader.exec_module(V2)

# ---- frozen site-specific facts (asserted everywhere) -------------------------
SITE_ID = "1B"
P_RATED_1B = 23.4
assert P_RATED_1B == 23.4
INPUT_DIM, SAMPLING_MIN, HISTORY, HORIZON = 5, 5, 72, 3
assert (INPUT_DIM, SAMPLING_MIN, HISTORY, HORIZON) == (5, 5, 72, 3)

RAW_COLS = {
    "power": "87_DKA_M9_A_C_Phases_Active_Power",
    "GHI": "101_DKA_WeatherStation_Global_Horizontal_Radiation",
    "DHI": "101_DKA_WeatherStation_Diffuse_Horizontal_Radiation",
    "T": "101_DKA_WeatherStation_Weather_Temperature_Celsius",
    "RH": "101_DKA_WeatherStation_Weather_Relative_Humidity",
}
FEAT_ORDER = ["power", "GHI", "DHI", "T", "RH"]  # == Site 18 order

OUT = PROJECT_ROOT / "results" / "site1b_val"
CLEAN_CSV = PROJECT_ROOT / "data" / "processed" / "site1b_5var_5min.csv"
RADIATION_SMALL_NEG = 5.0   # Site 18 frozen rule
RADIATION_HARD_MAX = 1500.0
POWER_CAP_KW = 25.0         # 1.07 x rated; removes the 2025-02-04 16:30 spike

SEEDS = V2.SEEDS
ALPHA = V2.ALPHA
EPS = V2.EPS


def mu_phys(mu_raw: np.ndarray) -> np.ndarray:
    return np.clip(mu_raw, 0.0, P_RATED_1B)


# ---------------------------------------------------------------------------
# Phase 2: clean dataset (no interpolation anywhere)
# ---------------------------------------------------------------------------

def build_clean(force: bool = False) -> pd.DataFrame:
    if CLEAN_CSV.exists() and not force:
        return pd.read_csv(CLEAN_CSV, parse_dates=["timestamp"])
    frames, counts = [], {"sentinel": 0, "power_outlier": 0, "radiation_neg_deep": 0,
                          "radiation_high": 0, "rh_out": 0, "temp_out": 0}
    for year in (2022, 2023, 2024, 2025):
        raw = pd.read_csv(PROJECT_ROOT / f"data/raw/Alice_Springs_{year}.csv",
                          usecols=["timestamp"] + list(RAW_COLS.values()),
                          on_bad_lines="skip")
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
        raw = raw.dropna(subset=["timestamp"]).drop_duplicates("timestamp")
        grid = pd.date_range(f"{year}-01-01 00:00", f"{year}-12-31 23:55", freq="5min")
        df = raw.set_index("timestamp").reindex(grid)
        df.index.name = "timestamp"
        ren = {v: k for k, v in RAW_COLS.items()}
        df = df.rename(columns=ren)
        for c in FEAT_ORDER:
            v = pd.to_numeric(df[c], errors="coerce")
            counts["sentinel"] += int(((v > 90000) | (v < -1e8) | np.isinf(v.astype(float))).sum())
            v[(v > 90000) | (v < -1e8)] = np.nan
            df[c] = v
        counts["power_outlier"] += int((df["power"] > POWER_CAP_KW).sum())
        df.loc[df["power"] > POWER_CAP_KW, "power"] = np.nan          # keep night tare
        for c in ("GHI", "DHI"):
            deep = df[c] < -RADIATION_SMALL_NEG
            counts["radiation_neg_deep"] += int(deep.sum())
            df.loc[deep, c] = np.nan
            df.loc[df[c] < 0, c] = 0.0                                 # small neg -> 0
            hi = df[c] > RADIATION_HARD_MAX
            counts["radiation_high"] += int(hi.sum())
            df.loc[hi, c] = np.nan
        counts["temp_out"] += int(((df["T"] < -50) | (df["T"] > 60)).sum())
        df.loc[(df["T"] < -50) | (df["T"] > 60), "T"] = np.nan
        counts["rh_out"] += int(((df["RH"] < 0) | (df["RH"] > 100)).sum())
        df.loc[(df["RH"] < 0) | (df["RH"] > 100), "RH"] = np.nan
        frames.append(df)
    clean = pd.concat(frames).reset_index()
    clean = clean.rename(columns={"power": "active_power_kw", "GHI": "ghi_wm2",
                                  "DHI": "dhi_wm2", "T": "temperature_c",
                                  "RH": "relative_humidity_pct"})
    clean["valid_flag"] = (clean[["active_power_kw", "ghi_wm2", "dhi_wm2",
                                  "temperature_c", "relative_humidity_pct"]]
                           .notna().all(axis=1)).astype(int)
    clean.to_csv(CLEAN_CSV, index=False)
    clean.attrs["counts"] = counts
    return clean


def load_year_grid(year: int) -> pd.DataFrame:
    """Clean rows for one year on the full 5-min grid, FEAT_ORDER columns."""
    assert year != 2025 or _allow_2025[0], "2025 may only be loaded by p8_eval (after freeze)"
    df = pd.read_csv(CLEAN_CSV, parse_dates=["timestamp"])
    df = df[df["timestamp"].dt.year == year].set_index("timestamp")
    ren = {"active_power_kw": "power", "ghi_wm2": "GHI", "dhi_wm2": "DHI",
           "temperature_c": "T", "relative_humidity_pct": "RH"}
    return df.rename(columns=ren)[FEAT_ORDER]


_allow_2025 = [False]


def open_2025() -> None:
    _allow_2025[0] = True


def windows(df: pd.DataFrame, origin_mask=None) -> dict:
    """Causal windows: 72 consecutive valid rows ending at origin + valid target
    15 min ahead (target row inside the same contiguous valid run)."""
    ts = pd.DatetimeIndex(df.index)
    feats = df[FEAT_ORDER].to_numpy(dtype=np.float32)
    ok = np.isfinite(feats).all(axis=1)
    # contiguous-block id: a NEW block starts at every ok-state transition, so a
    # NaN row is never in the same block as the valid rows that follow it
    state = ok.astype(np.int8)
    block = np.cumsum(np.r_[0, np.diff(state)] != 0)
    run = block
    # position within the current contiguous valid run (0-based), via per-run cumcount
    pos = (pd.Series(np.arange(len(ok))).groupby(run).cumcount().to_numpy())
    tgt = np.zeros(len(ok), dtype=bool)
    tgt[: len(ok) - HORIZON] = ok[HORIZON:] & (run[HORIZON:] == run[:-HORIZON])
    sel = ok & (pos >= HISTORY - 1) & tgt
    if origin_mask is not None:
        sel = sel & np.asarray(origin_mask, dtype=bool)
    origins = np.flatnonzero(sel)
    idx = origins[:, None] - (HISTORY - 1) + np.arange(HISTORY)[None, :]
    x_raw = feats[idx]                                   # [N, 72, 5]
    y_kw = feats[origins + HORIZON, 0].astype(np.float64)
    return {"x_raw": x_raw, "y_kw": y_kw,
            "origins": ts[origins], "targets": ts[origins + HORIZON],
            "origin_GHI": feats[origins, 1].astype(np.float64)}


def span_on_origins(w: dict, lo: str, hi: str) -> np.ndarray:
    o = pd.DatetimeIndex(w["origins"])
    return np.asarray((o >= pd.Timestamp(lo)) & (o < pd.Timestamp(hi)), dtype=bool)


def span_on_targets(w: dict, lo: str, hi: str) -> np.ndarray:
    """Training-row selection: the TARGET (which the model consumes) must lie in
    [lo, hi) -- mirrors rigorous_v2's target_mask convention for training rows."""
    t = pd.DatetimeIndex(w["targets"])
    return np.asarray((t >= pd.Timestamp(lo)) & (t < pd.Timestamp(hi)), dtype=bool)


def target_span_mask(df: pd.DataFrame, lo: str, hi: str) -> np.ndarray:
    t = pd.DatetimeIndex(df.index) + pd.Timedelta(minutes=HORIZON * SAMPLING_MIN)
    return np.asarray((t >= pd.Timestamp(lo)) & (t < pd.Timestamp(hi)), dtype=bool)


# ---------------------------------------------------------------------------
# capacity-normalized metrics
# ---------------------------------------------------------------------------

def interval_metrics_1b(y, lo, up, origins=None) -> dict:
    cov = (y >= lo) & (y <= up)
    width = np.maximum(up - lo, 0)
    isv = (up - lo) + (2.0 / ALPHA) * np.maximum(lo - y, 0) + (2.0 / ALPHA) * np.maximum(y - up, 0)
    out = {"PICP": float(cov.mean()), "PINAW": float(width.mean() / P_RATED_1B),
           "IS": float(isv.mean()), "NIS": float(isv.mean() / P_RATED_1B)}
    if origins is not None:
        o = pd.DatetimeIndex(origins)
        d1 = pd.Series(cov.astype(float)).groupby(o.normalize()).mean()
        out["RCE1D"] = float(np.abs(d1 - (1 - ALPHA)).mean())
        out["RCE3D"] = float(np.abs(d1.rolling(3, min_periods=1).mean() - (1 - ALPHA)).mean())
        streak = best = 0
        for m in (~cov):
            streak = streak + 1 if m else 0
            best = max(best, streak)
        out["LongestMissStreak"] = int(best)
    return out


# ---------------------------------------------------------------------------
# CQR (canonical conformalized quantile regression), frozen protocol
# ---------------------------------------------------------------------------

CQR_QUANTILES = (0.05, 0.50, 0.95)


def train_tcn_cqr(grad_rows: dict, es_rows: dict, sc: dict, seed: int,
                  dropout: float = 0.1, lr: float = 1e-3, max_epochs: int = 100,
                  patience: int = 5):
    """Canonical CQR: the frozen MultiScaleTCN trunk (forward_features, identical
    architecture/optimizer/ES protocol) with a linear 3-quantile head {0.05,0.5,
    0.95}; ES on the median quantile's kW RMSE over the internal early-stop tail."""
    from src.models.multiscale_tcn import MultiScaleTCN

    class CQRNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = MultiScaleTCN(dropout=dropout)
            self.head = torch.nn.Linear(64, 3)

        def forward(self, x):
            return self.head(self.trunk.forward_features(x))

    torch.manual_seed(int(seed) % (2 ** 31))
    model = CQRNet().to(V2.DEVICE)
    x = torch.as_tensor(V2.apply_scaler(grad_rows["x_raw"], sc), dtype=torch.float32,
                        device=V2.DEVICE)
    y = torch.as_tensor(((grad_rows["y_kw"] - sc["target_mean"]) / sc["target_std"])
                        .astype(np.float32), device=V2.DEVICE)
    xv = torch.as_tensor(V2.apply_scaler(es_rows["x_raw"], sc), dtype=torch.float32,
                         device=V2.DEVICE)
    yv_np = np.asarray(es_rows["y_kw"], dtype=np.float64)
    n = len(y)
    rng = np.random.default_rng(58003)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    qs = torch.tensor(CQR_QUANTILES, device=V2.DEVICE).view(1, -1)

    @torch.no_grad()
    def es_rmse() -> float:
        model.eval()
        pred = model(xv).detach().cpu().numpy().astype(np.float64)  # [N,3]
        model.train()
        med = pred[:, 1] * sc["target_std"] + sc["target_mean"]
        return float(np.sqrt(np.mean((med - yv_np) ** 2)))

    best, best_state, best_epoch, no_improve = float("inf"), None, 0, 0
    for epoch in range(max_epochs):
        perm = rng.permutation(n)
        for s0 in range(0, n, 512):
            idx = perm[s0:s0 + 512]
            opt.zero_grad()
            pred = model(x[idx])                                    # [B,3]
            err = y[idx].unsqueeze(1) - pred
            loss = torch.mean(torch.mean(
                torch.maximum(qs * err, (qs - 1) * err), dim=1))
            loss.backward()
            opt.step()
        vr = es_rmse()
        if vr < best - 1e-6:
            best, best_epoch, no_improve = vr, epoch + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"best_epoch": best_epoch, "best_es_rmse_kw": best}


@torch.no_grad()
def cqr_quantiles(model, x_raw: np.ndarray, sc: dict, batch: int = 4096) -> np.ndarray:
    xs = V2.apply_scaler(x_raw, sc)
    outs = []
    for s0 in range(0, len(xs), batch):
        xb = torch.as_tensor(xs[s0:s0 + batch], dtype=torch.float32, device=V2.DEVICE)
        outs.append(model(xb).detach().float().cpu().numpy().astype(np.float64))
    pred = np.concatenate(outs)                                     # [N,3] standardized
    return pred * sc["target_std"] + sc["target_mean"]


def cqr_qhat(y_cal, q_cal) -> float:
    s = np.sort(np.maximum(q_cal[:, 0] - y_cal, y_cal - q_cal[:, 2]))
    return V2.conformal_q(s)


def sha16(path: Path) -> str:
    return V2.sha256_file(path)
