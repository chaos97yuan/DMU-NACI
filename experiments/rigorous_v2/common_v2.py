"""rigorous_v2 shared library: strict-chronology residual experiment (DMU-NACI).

Replaces the Stage-8 window experiment with a provenance-controlled protocol:

  PROVENANCE labels (every residual row carries exactly one):
    IN_SAMPLE   fitted residual of the final mean on its own training rows
    OOF         rolling-origin expanding-window refit inside 2024-01-01..08-07
    REPLAY_OOS  frozen final mean replayed on archived 2022/2023 inputs
    VINTAGE_OOS year-ahead vintage model forecasting its designated year

Frozen role boundaries (2024, on forecast ORIGINS for evaluation spans,
on forecast TARGETS for anything that feeds gradient/early-stop):
    MEAN_GRAD  2024-01-01 00:00 .. 2024-07-17 00:00   (final-mean gradient rows)
    MEAN_ES    2024-07-17 00:00 .. 2024-08-08 00:00   (final-mean early-stop tail)
    SELECT     2024-08-08 00:00 .. 2024-09-16 00:00   (window/gamma/encoder selection)
    CALIB      2024-09-16 00:00 .. 2024-10-02 00:00   (conformal q ONLY)
    SAME       2024-10-02 00:00 .. 2025-01-01 00:00   (same-year test, report-only)
    CROSS      2025-01-01 00:00 .. 2026-01-01 00:00   (final prequential ACI test)

2025 is loaded by s7 ONLY, after the s6 freeze manifest is written.

This module never mutates Stage 1-8 artifacts.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

V2 = PROJECT_ROOT / "results" / "rigorous_v2"
DATA_V2 = V2 / "data"
CKPT_V2 = V2 / "checkpoints"

SEEDS = [42, 123, 2026, 3407, 8888]
P_RATED = 5.2
EPS = 1e-6
ALPHA = 0.10
GAMMA_GRID = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
ALPHA_MIN, ALPHA_MAX = 0.001, 0.5
ACI_DELAY_MIN = 15           # 3 steps x 5 min, matured feedback only
N_BOOTSTRAP = 2000
DAYLIGHT_GHI_THRESHOLD = 20.0  # frozen: scripts/04_build_windows.py + historical_familiarity
HISTORY_STEPS, HORIZON_STEPS, SAMPLING_MIN = 72, 3, 5
HORIZON_TD = pd.Timedelta(minutes=HORIZON_STEPS * SAMPLING_MIN)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- frozen role boundaries (half-open [lo, hi)) --------------------------------
# MEAN spans select ROWS BY TARGET (training consumes the target y);
# SELECT/CALIB/SAME select rows BY ORIGIN (a forecast belongs to the span in which
# it is issued); hi is trimmed to 23:40 so that no span's target (origin+15 min)
# crosses the next boundary (3 boundary rows per span fall in no role).
MEAN_GRAD = ("2024-01-01 00:00", "2024-07-17 00:00")
MEAN_ES = ("2024-07-17 00:00", "2024-08-08 00:00")
SELECT = ("2024-08-08 00:00", "2024-09-15 23:40")
CALIB = ("2024-09-16 00:00", "2024-10-01 23:40")
SAME = ("2024-10-02 00:00", "2024-12-31 23:40")
CROSS = ("2025-01-01 00:00", "2026-01-01 00:00")
EVAL_SPANS = {"select": SELECT, "calib": CALIB, "same": SAME, "cross": CROSS}

# --- rolling OOF folds inside the 2024 development window -----------------------
# training rows are selected by target < train_end; ES tail by es_start<=target<train_end;
# forecast block by forecast origin in [fc_lo, fc_hi).
OOF_FOLDS = [
    {"fold": "F1", "es_start": "2024-04-10 00:00", "train_end": "2024-05-01 00:00",
     "fc": ("2024-05-01 00:00", "2024-06-01 00:00")},
    {"fold": "F2", "es_start": "2024-05-11 00:00", "train_end": "2024-06-01 00:00",
     "fc": ("2024-06-01 00:00", "2024-07-01 00:00")},
    {"fold": "F3", "es_start": "2024-06-10 00:00", "train_end": "2024-07-01 00:00",
     "fc": ("2024-07-01 00:00", "2024-08-01 00:00")},
    {"fold": "F4", "es_start": "2024-07-11 00:00", "train_end": "2024-08-01 00:00",
     "fc": ("2024-08-01 00:00", "2024-08-08 00:00")},
]

# --- year-ahead vintage models ---------------------------------------------------
VINTAGE = [
    {"tag": "M22", "years": [2021], "fc_year": 2022, "es_start": "2021-12-10 00:00"},
    {"tag": "M23", "years": [2021, 2022], "fc_year": 2023, "es_start": "2022-12-10 00:00"},
    {"tag": "M24", "years": [2021, 2022, 2023], "fc_year": 2024, "es_start": "2023-12-10 00:00"},
]

ALLOWED_PROVENANCE = {"IN_SAMPLE", "OOF", "REPLAY_OOS", "VINTAGE_OOS"}
PROV_OOF, PROV_REPLAY, PROV_VINTAGE, PROV_INSAMPLE = "OOF", "REPLAY_OOS", "VINTAGE_OOS", "IN_SAMPLE"


def load_module(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Release note: the original file loaded experiments/diagnose_multiyear_uncertainty.py
# here (only for its load_year_preopen helper). That module's import cascade pulls the
# legacy Stage-D1 diagnostics, which are not part of this release, so the verbatim
# helper was extracted to experiments/rigorous_v2/preopen.py (see CHANGELOG.md).
from experiments.rigorous_v2 import preopen as M3  # noqa: E402
from src.models.multiscale_tcn import MultiScaleTCN  # noqa: E402
from src.models.uncertainty_encoders import ScaleModel  # noqa: E402
from src.data.schema import (  # noqa: E402
    COLUMN_MAP, FEATURE_COLUMNS, NUMERIC_COLUMNS, OUTPUT_COLUMNS, REQUIRED_RAW_COLUMNS,
)

CLEAN = load_module("stage_clean", "scripts/02_clean_data.py")
CAUSAL = load_module("stage_causal", "src/eval/historical_interpolation_causality_audit.py")

RAW = PROJECT_ROOT / "data" / "raw"


def wjson(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True,
                               default=str) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# assertions (the contract of this experiment)
# ---------------------------------------------------------------------------

def assert_role_boundaries_disjoint() -> None:
    spans = [MEAN_GRAD, MEAN_ES, SELECT, CALIB, SAME, CROSS]
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            (a0, a1), (b0, b1) = spans[i], spans[j]
            assert pd.Timestamp(a1) <= pd.Timestamp(b0) or pd.Timestamp(b1) <= pd.Timestamp(a0), \
                f"role spans overlap: {spans[i]} vs {spans[j]}"
    for f in OOF_FOLDS:
        assert pd.Timestamp(f["es_start"]) < pd.Timestamp(f["train_end"]) <= pd.Timestamp(f["fc"][0]), \
            f"fold {f['fold']} violates chronology"
        assert pd.Timestamp(f["fc"][1]) <= pd.Timestamp(MEAN_ES[1]), "fold forecast exceeds dev window"


def assert_no_2025_in(origins, targets, what: str) -> None:
    o, t = pd.DatetimeIndex(origins), pd.DatetimeIndex(targets)
    assert not ((o >= pd.Timestamp(CROSS[0])) & (o < pd.Timestamp(CROSS[1]))).any(), \
        f"{what}: 2025 origins present"
    assert not ((t >= pd.Timestamp(CROSS[0])) & (t < pd.Timestamp(CROSS[1]))).any(), \
        f"{what}: 2025 targets present"


def assert_training_precedes(train_targets, forecast_targets, what: str) -> None:
    tmax = pd.DatetimeIndex(train_targets).max()
    fmin = pd.DatetimeIndex(forecast_targets).min()
    assert tmax < fmin, f"{what}: max train target {tmax} not < min forecast target {fmin}"


def assert_provenance(labels) -> None:
    bad = set(np.unique(labels)) - ALLOWED_PROVENANCE
    assert not bad, f"unknown provenance labels: {bad}"


# ---------------------------------------------------------------------------
# data: clean a raw year -> full-grid core/flags (frozen 02_clean_data functions)
# ---------------------------------------------------------------------------

def build_year_core(year: int) -> dict:
    """Clean one raw DKASC year with the frozen cleaning functions (37rh-style)."""
    core_p = DATA_V2 / f"{year}_core.csv"
    flags_p = DATA_V2 / f"{year}_flags.csv"
    if core_p.exists() and flags_p.exists():
        return {"core": pd.read_csv(core_p), "flags": pd.read_csv(flags_p)}
    import logging
    lg = logging.getLogger(f"v2_clean_{year}")
    raw = pd.read_csv(RAW / f"Alice_Springs_{year}.csv", usecols=REQUIRED_RAW_COLUMNS)
    df = raw.rename(columns=COLUMN_MAP).loc[:, OUTPUT_COLUMNS].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in NUMERIC_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    start, end = pd.Timestamp(f"{year}-01-01 00:00"), pd.Timestamp(f"{year}-12-31 23:55")
    df = df.loc[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
    assert not df["timestamp"].duplicated().any(), f"{year}: duplicate timestamps"
    timeline = pd.date_range(start, end, freq=f"{SAMPLING_MIN}min")
    reid = df.sort_values("timestamp").set_index("timestamp").reindex(timeline)
    reid.index.name = "timestamp"
    reid = reid.reset_index()
    flags = pd.DataFrame({"timestamp": reid["timestamp"]})
    flags["timestamp_was_missing"] = flags["timestamp"].isin(
        timeline.difference(pd.DatetimeIndex(df["timestamp"]))).astype(int)
    flags = CLEAN.initialize_quality_flags(flags)
    cleaned, flags, _ = CLEAN.clean_active_power(reid, flags)
    cleaned, flags, _ = CLEAN.clean_radiation_values(cleaned, flags)
    cleaned, flags, _ = CLEAN.clean_humidity(cleaned, flags, lg)
    cleaned, flags, _ = CLEAN.clean_rainfall(cleaned, flags)
    cleaned, flags, _ = CLEAN.apply_variable_specific_interpolation(cleaned, flags)
    cleaned, flags = CLEAN.build_segment_ids(cleaned, flags)
    core = CLEAN.build_core_dataset(cleaned)
    assert len(core) == len(timeline), f"{year}: core is not a complete grid"
    core_p.parent.mkdir(parents=True, exist_ok=True)
    core.to_csv(core_p, index=False)
    flags.to_csv(flags_p, index=False)
    return {"core": core, "flags": flags}


def build_year_windows(year: int) -> dict:
    """Causal-safe windows for one year, target attached (finite, non-imputed kept).

    Convention identical to scripts/37rh_2024_preopen.py: input = 72 steps ending
    at origin; target = origin + 15 min; histories must be single-segment, QC-clean,
    and free of future-interpolation endpoint dependency.
    """
    X_p, meta_p = DATA_V2 / f"{year}_X.npy", DATA_V2 / f"{year}_window_manifest.csv"
    if X_p.exists() and meta_p.exists():
        return {"x_raw": np.load(X_p), "meta": pd.read_csv(meta_p)}
    parts = build_year_core(year)
    core, flags = parts["core"], parts["flags"]
    ts = pd.DatetimeIndex(pd.to_datetime(core["timestamp"]))
    lineage = CAUSAL.reconstruct_lineage(flags, core)
    candidates, _ = CAUSAL.audit_candidates(flags, lineage)
    future_dep = candidates["future_endpoint_dependent"].to_numpy(dtype=bool)
    eligible = candidates["history_qc_eligible"].to_numpy(dtype=bool)
    feats = core.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    power = core["Active_Power"].to_numpy(dtype=float)
    imp = flags["power_imputed"].fillna(0).to_numpy(dtype=np.int8) if "power_imputed" in flags \
        else np.zeros(len(core), dtype=np.int8)
    ghi = core["Global_Horizontal_Radiation"].to_numpy(dtype=float)
    lo = pd.Timestamp(f"{year}-01-01 00:00")
    hi = pd.Timestamp(f"{year + 1}-01-01 00:00")
    rows, meta_rows = [], []
    n_c = len(candidates)
    for ci in range(n_c):
        origin_idx = ci + HISTORY_STEPS - 1
        target_idx = origin_idx + HORIZON_STEPS
        o_ts, t_ts = ts[origin_idx], ts[target_idx]
        if not (lo <= o_ts < hi and lo <= t_ts < hi):
            continue
        if not eligible[ci] or future_dep[ci]:
            continue
        if not np.isfinite(power[target_idx]) or imp[target_idx] != 0:
            continue  # target must be an observed (non-imputed) finite value
        rows.append(feats[origin_idx - HISTORY_STEPS + 1: origin_idx + 1].astype(np.float32))
        meta_rows.append({
            "sample_index": len(meta_rows), "forecast_origin_timestamp": o_ts,
            "target_timestamp": t_ts, "y_kw": power[target_idx],
            "origin_GHI": float(ghi[origin_idx]),
            "provenance": None,  # assigned by the pool builder
        })
    X = np.stack(rows)
    meta = pd.DataFrame(meta_rows)
    np.save(X_p, X)
    meta.to_csv(meta_p, index=False)
    wjson(DATA_V2 / f"{year}_windows_audit.json", {
        "year": year, "candidates": int(n_c), "retained": int(len(meta)),
        "daylight_share": float((meta["origin_GHI"] > DAYLIGHT_GHI_THRESHOLD).mean()),
        "raw_sha": sha256_file(RAW / f"Alice_Springs_{year}.csv"),
    })
    return {"x_raw": X, "meta": meta}


def load_year_windows(year: int) -> dict:
    """Unified loader: v2-built (2021, 2022) or existing verified artifacts (2023..2025)."""
    if year in (2021, 2022):
        d = build_year_windows(year)
        meta = d["meta"]
        return {"x_raw": d["x_raw"], "origins": pd.DatetimeIndex(meta["forecast_origin_timestamp"]),
                "targets": pd.DatetimeIndex(meta["target_timestamp"]), "y_kw": meta["y_kw"].to_numpy(float),
                "origin_GHI": meta["origin_GHI"].to_numpy(float)}
    if year == 2023:
        meta = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "external_2023" /
                           "external_window_metadata.csv")
        fl = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "external_2023" /
                         "site18_2023_quality_flags.csv")
        X = np.load(PROJECT_ROOT / "data" / "processed" / "external_2023" / "external_windows_X.npy",
                    mmap_mode="r")
        core = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "external_2023" /
                           "site18_2023_clean_core.csv", usecols=["timestamp", "Active_Power"])
        fl["timestamp"] = pd.to_datetime(fl["timestamp"])
        fl = fl.set_index("timestamp")
        origins = pd.to_datetime(meta["forecast_origin_timestamp"])
        tidx = origins + HORIZON_TD
        y = np.asarray(core.set_index(pd.to_datetime(core["timestamp"]))["Active_Power"]
                       .reindex(tidx).to_numpy(float))
        t_imp = fl["power_imputed"].reindex(tidx).fillna(1).to_numpy(np.int8) \
            if "power_imputed" in fl else np.ones(len(tidx), np.int8)
        keep = np.isfinite(y) & (t_imp == 0)
        return {"x_raw": np.asarray(X[keep], dtype=np.float32),
                "origins": origins[keep], "targets": tidx[keep], "y_kw": y[keep],
                "origin_GHI": np.asarray(X[:, -1, 1], dtype=float)[keep]}
    # 2024 / 2025 preopen artifacts (x/y/origins/states; mu column IGNORED)
    yv = M3.load_year_preopen(year, SEEDS)
    yv["targets"] = pd.DatetimeIndex(yv["origins"]) + HORIZON_TD
    yv["origin_GHI"] = np.asarray(yv["x_raw"][:, -1, 1], dtype=float)
    return yv


def span_mask(origins, lo: str, hi: str):
    o = pd.DatetimeIndex(origins)
    return np.asarray((o >= pd.Timestamp(lo)) & (o < pd.Timestamp(hi)), dtype=bool)


def target_mask(targets, lo: str, hi: str):
    t = pd.DatetimeIndex(targets)
    return np.asarray((t >= pd.Timestamp(lo)) & (t < pd.Timestamp(hi)), dtype=bool)


def sub(part: dict, mask: np.ndarray) -> dict:
    out = {k: (v[mask] if isinstance(v, np.ndarray) else v) for k, v in part.items()
           if k in ("x_raw", "y_kw", "origin_GHI", "N_group", "hard_DF", "year", "name")}
    out["origins"] = pd.DatetimeIndex(part["origins"])[mask]
    if "targets" in part:
        out["targets"] = pd.DatetimeIndex(part["targets"])[mask]
    return out


# ---------------------------------------------------------------------------
# scaler (single common reference for the whole experiment)
# ---------------------------------------------------------------------------

def fit_scaler(x_rows: np.ndarray) -> dict:
    flat = x_rows.reshape(-1, x_rows.shape[2]).astype(np.float64)
    mean = flat.mean(axis=0)
    std = np.maximum(flat.std(axis=0, ddof=0), 1e-8)
    return {"mean": mean, "std": std, "target_mean": float(mean[0]), "target_std": float(std[0])}


def apply_scaler(x: np.ndarray, sc: dict) -> np.ndarray:
    return ((x.astype(np.float64) - sc["mean"]) / sc["std"]).astype(np.float32)


def scaler_fingerprint(sc: dict) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(np.asarray(sc["mean"], float).tobytes())
    h.update(np.asarray(sc["std"], float).tobytes())
    h.update(np.float64(sc["target_mean"]).tobytes())
    h.update(np.float64(sc["target_std"]).tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# models: TCN (mean) and GRU (scale)
# ---------------------------------------------------------------------------

def train_tcn(grad_rows: dict, es_rows: dict, sc: dict, seed: int, dropout: float = 0.1,
              lr: float = 1e-3, max_epochs: int = 100, patience: int = 5):
    """Stage-6 protocol (AdamW 1e-3/1e-4, batch 512, rng 58003) with correct-scale
    early stopping: ES RMSE computed on de-standardized kW predictions vs kW targets
    (the legacy stage6 helper compared standardized predictions against kW targets;
    fixed here because every v2 model is retrained anyway).

    grad_rows / es_rows: dicts with x_raw [N,72,5] and y_kw [N] (kW).
    Causality contract: max(grad target, es target) < min(forecast target of the
    span this model will be applied to -- asserted by the callers).
    """
    torch.manual_seed(int(seed) % (2 ** 31))
    model = MultiScaleTCN(dropout=dropout).to(DEVICE)
    x = torch.as_tensor(apply_scaler(grad_rows["x_raw"], sc), dtype=torch.float32, device=DEVICE)
    y = torch.as_tensor(((grad_rows["y_kw"] - sc["target_mean"]) / sc["target_std"]).astype(np.float32),
                        dtype=torch.float32, device=DEVICE)
    xv = torch.as_tensor(apply_scaler(es_rows["x_raw"], sc), dtype=torch.float32, device=DEVICE)
    yv_np = np.asarray(es_rows["y_kw"], dtype=np.float64)
    n = len(y)
    rng = np.random.default_rng(58003)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    @torch.no_grad()
    def es_rmse_kw() -> float:
        model.eval()
        pred = model(xv).detach().cpu().numpy().astype(np.float64)
        model.train()
        return float(np.sqrt(np.mean((pred * sc["target_std"] + sc["target_mean"] - yv_np) ** 2)))

    best, best_state, best_epoch, no_improve = float("inf"), None, 0, 0
    curve = []
    t0 = time.perf_counter()
    for epoch in range(max_epochs):
        perm = rng.permutation(n)
        for s0 in range(0, n, 512):
            idx = perm[s0:s0 + 512]
            opt.zero_grad()
            loss = torch.mean((model(x[idx]) - y[idx]) ** 2)
            loss.backward()
            opt.step()
        vr = es_rmse_kw()
        curve.append({"epoch": epoch + 1, "es_rmse_kw": vr})
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
    info = {"best_epoch": best_epoch, "best_es_rmse_kw": best, "epochs_ran": len(curve),
            "train_seconds": round(time.perf_counter() - t0, 1),
            "curve_tail": curve[-5:], "seed": seed, "dropout": dropout, "lr": lr}
    return model, info


@torch.no_grad()
def tcn_mu(model, x_raw: np.ndarray, sc: dict, batch: int = 4096) -> np.ndarray:
    """mu_raw in kW (no clipping here)."""
    xs = apply_scaler(x_raw, sc)
    outs = []
    for s0 in range(0, len(xs), batch):
        xb = torch.as_tensor(xs[s0:s0 + batch], dtype=torch.float32, device=DEVICE)
        outs.append(model(xb).detach().float().cpu().numpy().astype(np.float64))
    return np.concatenate(outs) * sc["target_std"] + sc["target_mean"]


def mu_phys(mu_raw: np.ndarray) -> np.ndarray:
    return np.clip(mu_raw, 0.0, P_RATED)


def train_gru(x_train_std, e_train_std, seed: int, total_updates: int | None = None):
    """Stage-3/8 GRU protocol verbatim: 3 x 50 epochs with a fresh AdamW per outer
    restart, batch 512, rng 58003, no early stopping. total_updates (sensitivity
    arm) stops at a fixed optimizer-update count to equalize training compute
    across pools of different size."""
    torch.manual_seed(int(seed) % (2 ** 31))
    model = ScaleModel("GRU").to(DEVICE)
    x_all = torch.as_tensor(x_train_std, dtype=torch.float32, device=DEVICE)
    e_all = torch.as_tensor(e_train_std.astype(np.float32), dtype=torch.float32, device=DEVICE)
    n = len(e_all)
    rng = np.random.default_rng(58003)
    steps_done = 0
    t0 = time.perf_counter()
    model.train()
    for outer in range(3):
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(50):
            perm = rng.permutation(n)
            for s0 in range(0, n, 512):
                idx = perm[s0:s0 + 512]
                v = model.noise_variance(x_all[idx])
                loss = 0.5 * torch.mean(torch.log(v) + e_all[idx] ** 2 / v)
                opt.zero_grad()
                loss.backward()
                opt.step()
                steps_done += 1
                if total_updates is not None and steps_done >= total_updates:
                    model.eval()
                    return model, steps_done, round(time.perf_counter() - t0, 1)
    model.eval()
    return model, steps_done, round(time.perf_counter() - t0, 1)


@torch.no_grad()
def sigma_kw(model, x_raw: np.ndarray, sc: dict, batch: int = 4096) -> np.ndarray:
    outs = []
    xs = apply_scaler(x_raw, sc)
    for s0 in range(0, len(xs), batch):
        xb = torch.as_tensor(xs[s0:s0 + batch], dtype=torch.float32, device=DEVICE)
        outs.append(model.noise_variance(xb).cpu().numpy().astype(np.float64))
    return np.sqrt(np.concatenate(outs)) * sc["target_std"]


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def gaussian_nll(y, mu, s):
    z = (y - mu) / (s + EPS)
    return float(np.mean(0.5 * np.log(2 * np.pi * (s + EPS) ** 2) + 0.5 * z ** 2))


def conformal_q(scores_sorted: np.ndarray, alpha: float = ALPHA) -> float:
    n = len(scores_sorted)
    k = min(max(int(np.ceil((n + 1) * (1 - alpha))), 1), n)
    return float(scores_sorted[k - 1])


def interval_score(y, lo, up, alpha: float = ALPHA) -> np.ndarray:
    isv = (up - lo) + (2.0 / alpha) * np.maximum(lo - y, 0) + (2.0 / alpha) * np.maximum(y - up, 0)
    return isv


def scale_metrics(y, mu, s) -> dict:
    from scipy.stats import spearmanr, pearsonr
    z = (y - mu) / (s + EPS)
    az = np.abs(z)
    return {"NLL": gaussian_nll(y, mu, s), "Std_z": float(np.std(z)), "Mean_z": float(np.mean(z)),
            "Median_abs_z": float(np.median(az)), "Q90_abs_z": float(np.quantile(az, 0.90)),
            "Q95_abs_z": float(np.quantile(az, 0.95)),
            "Spearman": float(spearmanr(s, np.abs(y - mu)).statistic),
            "Pearson": float(pearsonr(s, np.abs(y - mu)).statistic),
            "Median_sigma_kw": float(np.median(s))}


def interval_metrics(y, lo, up, origins=None) -> dict:
    cov = (y >= lo) & (y <= up)
    out = {"PICP": float(cov.mean()), "PINAW": float(np.maximum(up - lo, 0).mean() / P_RATED),
           "IS": float(interval_score(y, lo, up).mean())}
    if origins is not None:
        o = pd.DatetimeIndex(origins)
        day = o.normalize()
        uniq, code = np.unique(day, return_inverse=True)
        rce1, rce3 = [], []
        d1 = pd.Series(cov.astype(float)).groupby(day).mean()
        out["RCE1D"] = float(np.abs(d1 - (1 - ALPHA)).mean())
        roll3 = d1.rolling(3, min_periods=1).mean()
        out["RCE3D"] = float(np.abs(roll3 - (1 - ALPHA)).mean())
        # longest miss streak
        miss = ~cov
        streak = best = 0
        for m in miss:
            streak = streak + 1 if m else 0
            best = max(best, streak)
        out["LongestMissStreak"] = int(best)
    return out


def state_split(n_group, hard_df):
    ng = np.asarray(n_group).astype(str)
    hd = np.asarray(hard_df).astype(bool)
    return {"NORMAL": (~hd) & (ng == "N"), "HIGH_N": (~hd) & (ng != "N"), "HARD_DF": hd}


def run_aci(y, mu, eff_sigma, cal_scores_sorted, origins, gamma, alpha0: float = ALPHA):
    """Delayed ACI with 15-min matured feedback (Stage-4/5/7/8 protocol, verbatim
    semantics incl. the no-future-update assert)."""
    import math
    n = len(y)
    dev = np.abs(y - mu)
    k_fn = lambda a: min(max(int(math.ceil((len(cal_scores_sorted) + 1) * (1 - a))), 1),
                         len(cal_scores_sorted))
    lower = np.empty(n)
    upper = np.empty(n)
    alphas = np.empty(n)
    qs = np.empty(n)
    alpha = alpha0
    queue: list[tuple[pd.Timestamp, float]] = []
    qi = 0
    last_update = pd.Timestamp("1900-01-01")
    o_all = pd.DatetimeIndex(origins)
    for i in range(n):
        o = o_all[i]
        while qi < len(queue) and queue[qi][0] <= o:
            alpha = min(max(alpha + gamma * (ALPHA - queue[qi][1]), ALPHA_MIN), ALPHA_MAX)
            last_update = queue[qi][0]
            qi += 1
        assert last_update <= o, "ACI update from the future detected"
        q = cal_scores_sorted[k_fn(alpha) - 1]
        half = q * eff_sigma[i]
        lower[i], upper[i] = mu[i] - half, mu[i] + half
        alphas[i], qs[i] = alpha, q
        e = 0.0 if (y[i] >= lower[i] and y[i] <= upper[i]) else 1.0
        queue.append((o + pd.Timedelta(minutes=ACI_DELAY_MIN), e))
    return lower, upper, alphas, qs


def select_gamma(y_v, mu_v, sg_v, o_v, use_sigma: bool):
    """Stage-7 validation-only rule (SELECT span): feasibility |PICP-.9|<=.02,
    then min IS, then min gamma."""
    e_val = np.abs(y_v - mu_v)
    sc = np.sort(e_val / (sg_v + EPS)) if use_sigma else np.sort(e_val)
    cands = []
    for gamma in GAMMA_GRID:
        lo, up, _, _ = run_aci(y_v, mu_v, sg_v if use_sigma else np.ones(len(y_v)),
                               sc, o_v, gamma)
        cov = float(((y_v >= lo) & (y_v <= up)).mean())
        isv = float(interval_score(y_v, lo, up).mean())
        cands.append((abs(cov - 0.90), isv, gamma))
    feas = [c for c in cands if c[0] <= 0.02]
    best = (min(feas, key=lambda c: (c[1], c[2])) if feas
            else min(cands, key=lambda c: (c[0], c[2])))
    detail = {float(c[2]): {"abs_picp_err": c[0], "IS": c[1]} for c in cands}
    return best[2], detail


def paired_day_bootstrap(metric_a, metric_b, origins, arrs_a: list, arrs_b: list,
                         n_boot: int = N_BOOTSTRAP, seed: int = 0) -> dict:
    """Paired day-block bootstrap: resample unique days with replacement and apply
    the SAME row indices to every per-sample array of both arms, then report
    metric_b(arrs_b[idx]) - metric_a(arrs_a[idx]). Closures may bake in per-arm
    scalar constants (e.g. the conformal q). All arrays must be row-aligned with
    origins."""
    o = pd.DatetimeIndex(origins)
    n = len(o)
    assert all(len(a) == n for a in arrs_a + arrs_b), "bootstrap arrays misaligned"
    uniq, code = np.unique(o.normalize(), return_inverse=True)
    day_rows = [np.flatnonzero(code == d) for d in range(len(uniq))]
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([day_rows[d] for d in pick])
        diffs[b] = metric_b(*[a[idx] for a in arrs_b]) - metric_a(*[a[idx] for a in arrs_a])
    return {"mean_diff": float(np.mean(diffs)), "median_diff": float(np.median(diffs)),
            "ci_lower": float(np.quantile(diffs, 0.025)),
            "ci_upper": float(np.quantile(diffs, 0.975))}


def seasons_southern(month: np.ndarray) -> np.ndarray:
    s = np.empty(len(month), dtype=object)
    s[np.isin(month, [12, 1, 2])] = "DJF"
    s[np.isin(month, [3, 4, 5])] = "MAM"
    s[np.isin(month, [6, 7, 8])] = "JJA"
    s[np.isin(month, [9, 10, 11])] = "SON"
    return s
