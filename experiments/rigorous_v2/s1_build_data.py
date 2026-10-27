"""rigorous_v2 STEP 1: build 2021 + full-2022 cleaned cores and causal-safe windows.

The old chain's 2022 artifacts stop at 2022-11-07; this rebuilds the FULL year from
raw with the frozen cleaning functions (scripts/02_clean_data.py) and the causal-safe
window machinery (src/eval/historical_interpolation_causality_audit.py), so that
E22_replay covers Nov-Dec 2022 as well. 2021 is built for the vintage arm.

Outputs -> results/rigorous_v2/data/{year}_core.csv, {year}_flags.csv,
           {year}_X.npy, {year}_window_manifest.csv, {year}_windows_audit.json
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.rigorous_v2 import common_v2 as C  # noqa: E402


def summarize(year: int, d: dict) -> dict:
    o = pd.DatetimeIndex(d["origins"])
    months = o.month.to_numpy()
    return {
        "year": year,
        "n_windows": int(len(d["y_kw"])),
        "first_origin": str(o.min()),
        "last_origin": str(o.max()),
        "months_covered": sorted(set(int(m) for m in np.unique(months))),
        "daylight_share": float((d["origin_GHI"] > C.DAYLIGHT_GHI_THRESHOLD).mean()),
        "y_kw_mean": float(np.mean(d["y_kw"])),
        "y_kw_std": float(np.std(d["y_kw"])),
        "x_finite": bool(np.isfinite(d["x_raw"]).all()),
        "y_finite": bool(np.isfinite(d["y_kw"]).all()),
        "dtype": str(d["x_raw"].dtype),
    }


def main() -> None:
    C.DATA_V2.mkdir(parents=True, exist_ok=True)
    C.assert_role_boundaries_disjoint()
    rows = []
    for year in (2021, 2022):
        d = C.load_year_windows(year)
        # chronology sanity: all origins/targets inside the year, target = origin+15min
        o, t = pd.DatetimeIndex(d["origins"]), pd.DatetimeIndex(d["targets"])
        assert (t - o == C.HORIZON_TD).all(), f"{year}: target != origin + 15 min"
        assert (o >= pd.Timestamp(f"{year}-01-01")).all() and (o < pd.Timestamp(f"{year + 1}-01-01")).all()
        rows.append(summarize(year, d))
        print(rows[-1])
    C.wjson(C.DATA_V2 / "s1_data_audit.json", {"windows": rows})
    print("s1 done")


if __name__ == "__main__":
    main()
