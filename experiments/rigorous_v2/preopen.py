"""Loader for the frozen 2024/2025 "preopen" window universes (Site 18).

`load_year_preopen` is copied verbatim from the original project file
``experiments/diagnose_multiyear_uncertainty.py`` (only the module-level path
constants were carried over). It is extracted here so that ``common_v2`` does
not need to import that diagnostic module (whose import cascade pulls the full
legacy Stage-D1 tooling, which is not part of this release).

The loaded artifacts are produced by the legacy preopen builders
``legacy/preopen_builders/37rh_2024_preopen.py`` (2024) and
``37rj_2025_preopen.py`` (2025) from the raw DKASC downloads:

  data/processed/external_{year}_preopen/{year}_core.csv
  data/processed/external_{year}_preopen/{year}_context_preopen.csv
  data/processed/external_{year}_preopen/X_{year}_preopen.npy
  outputs/predictions/stage37r_{year}_preopen/{year}_preopen_seed{seed}.csv

The ``mu`` column of the preopen CSV is read only to keep the row alignment of
the original pipeline; it is ignored downstream (the deployed TCN is re-run on
X by s2/s7).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXT = PROJECT_ROOT / "data" / "processed"


def load_year_preopen(year: int, seeds: list) -> dict:
    pre = pd.read_csv(PROJECT_ROOT / "outputs" / "predictions" / f"stage37r_{year}_preopen" /
                      f"{year}_preopen_seed{seeds[0]}.csv",
                      usecols=["sample_index", "forecast_origin_timestamp", "target_timestamp", "mu"])
    ctx = pd.read_csv(EXT / f"external_{year}_preopen" / f"{year}_context_preopen.csv",
                      usecols=["sample_index", "N_group", "HARD_DF"])
    assert (pre["sample_index"].to_numpy() == ctx["sample_index"].to_numpy()).all()
    assert (pre["sample_index"].to_numpy() == np.arange(len(pre))).all(), "X row order must equal sample_index"
    core = pd.read_csv(EXT / f"external_{year}_preopen" / f"{year}_core.csv",
                       usecols=["timestamp", "Active_Power"])
    y_map = pd.Series(core["Active_Power"].to_numpy(float), index=pd.to_datetime(core["timestamp"]))
    y = np.asarray(y_map.reindex(pd.to_datetime(pre["target_timestamp"])).to_numpy(float))
    finite = np.isfinite(y)
    return {
        "x_raw": np.asarray(np.load(EXT / f"external_{year}_preopen" / f"X_{year}_preopen.npy",
                                    mmap_mode="r")[finite], dtype=np.float32),
        "y_kw": y[finite],
        "mu_kw": pre["mu"].to_numpy(float)[finite],
        "origins": pd.to_datetime(pre["forecast_origin_timestamp"]).iloc[np.flatnonzero(finite)].reset_index(drop=True),
        "N_group": ctx["N_group"].astype(str).to_numpy()[finite],
        "hard_DF": ctx["HARD_DF"].astype(bool).to_numpy()[finite],
        "year": year,
    }
