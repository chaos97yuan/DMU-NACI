"""Generate synthetic DKASC-format PV data for smoke tests and format examples.

This is NOT real observations and must never be used for paper numbers. It emits
the exact raw schema the pipeline expects (src/data/schema.py REQUIRED_RAW_COLUMNS)
so that the cleaning/windowing machinery can be exercised end to end without the
DKASC downloads:

  timestamp + 71_DKA_M2_C_Phase_Active_Power + 101_DKA_WeatherStation_*
  (Global_Horizontal_Radiation, Diffuse_Horizontal_Radiation,
   Radiation_Global_Tilted, Radiation_Diffuse_Tilted, Weather_Temperature_Celsius,
   Weather_Relative_Humidity, Wind_Speed, Wind_Direction, Weather_Daily_Rainfall)

Model: deterministic clear-sky bell curve (solar elevation proxy) with a seeded
multiplicative cloud process, plus diurnal temperature/RH cycles. Fully
self-contained (numpy only).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "timestamp",
    "71_DKA_M2_C_Phase_Active_Power",
    "101_DKA_WeatherStation_Global_Horizontal_Radiation",
    "101_DKA_WeatherStation_Diffuse_Horizontal_Radiation",
    "101_DKA_WeatherStation_Radiation_Global_Tilted",
    "101_DKA_WeatherStation_Radiation_Diffuse_Tilted",
    "101_DKA_WeatherStation_Weather_Temperature_Celsius",
    "101_DKA_WeatherStation_Weather_Relative_Humidity",
    "101_DKA_WeatherStation_Wind_Speed",
    "101_DKA_WeatherStation_Wind_Direction",
    "101_DKA_WeatherStation_Weather_Daily_Rainfall",
]


def generate(year: int, n_days: int, seed: int = 7, rated_kw: float = 5.2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{year}-01-01 00:00", periods=n_days * 288, freq="5min")
    minute = (idx.hour * 60 + idx.minute).to_numpy(float)
    doy = idx.dayofyear.to_numpy(float)
    # solar elevation proxy: peak at 12:30 local, longer days toward DOY 172 (shift
    # by 6 months so either hemisphere looks reasonable for a synthetic demo)
    daylight = np.clip(660 + 120 * np.sin(2 * np.pi * (doy - 81) / 365.25), 640, 780)
    elev = np.sin(np.pi * (minute - (720 - daylight / 2)) / daylight)
    elev = np.clip(elev, 0.0, None)
    cloud = 0.75 + 0.25 * rng.random(len(idx))
    slow = 0.9 + 0.1 * np.sin(2 * np.pi * np.arange(len(idx)) / (288 * 3))  # 3-day weather cycle
    ghi = 1050.0 * elev ** 1.15 * cloud * slow
    dhi = 0.18 * ghi + 12.0 * elev
    ghi = np.maximum(ghi, 0.0)
    dhi = np.maximum(dhi, 0.0)
    power = rated_kw * (ghi / 1050.0) ** 1.1 * (0.97 - 0.004 * np.maximum(ghi / 1050, 0) * 30)
    power = np.clip(power * (0.98 + 0.02 * rng.random(len(idx))), 0.0, rated_kw)
    temp = 22.0 + 9.0 * elev - 4.0 * np.sin(2 * np.pi * (doy - 15) / 365.25) + rng.normal(0, 0.4, len(idx))
    rh = np.clip(92.0 - 45.0 * elev - 10.0 * np.sin(2 * np.pi * (doy - 15) / 365.25) + rng.normal(0, 2, len(idx)), 5, 100)
    wind = np.clip(2.5 + 1.5 * np.sin(2 * np.pi * np.arange(len(idx)) / 288.0) + rng.normal(0, 0.6, len(idx)), 0, None)
    wind_dir = (200 + 60 * np.sin(2 * np.pi * np.arange(len(idx)) / (288 * 2)) + rng.normal(0, 8, len(idx))) % 360
    rain = np.zeros(len(idx))
    df = pd.DataFrame({
        "timestamp": idx,
        "71_DKA_M2_C_Phase_Active_Power": np.round(power, 4),
        "101_DKA_WeatherStation_Global_Horizontal_Radiation": np.round(ghi, 4),
        "101_DKA_WeatherStation_Diffuse_Horizontal_Radiation": np.round(dhi, 4),
        "101_DKA_WeatherStation_Radiation_Global_Tilted": np.round(ghi * 1.05, 4),
        "101_DKA_WeatherStation_Radiation_Diffuse_Tilted": np.round(dhi, 4),
        "101_DKA_WeatherStation_Weather_Temperature_Celsius": np.round(temp, 3),
        "101_DKA_WeatherStation_Weather_Relative_Humidity": np.round(rh, 3),
        "101_DKA_WeatherStation_Wind_Speed": np.round(wind, 3),
        "101_DKA_WeatherStation_Wind_Direction": np.round(wind_dir, 2),
        "101_DKA_WeatherStation_Weather_Daily_Rainfall": rain,
    })
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--days", type=int, default=3, help="3 -> repo example file; 366 -> full synthetic year for tests")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    df = generate(args.year, args.days, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df):,} rows x {len(df.columns)} cols -> {args.out}")


if __name__ == "__main__":
    main()
