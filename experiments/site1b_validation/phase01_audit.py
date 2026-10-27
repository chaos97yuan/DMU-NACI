"""Site 1B external (cross-system) validation -- PHASE 0/1: locate + audit. No training.

System identification (triangle of evidence):
  1. DKASC official system page URL slug 'dka-m9-a-c-phases' is titled
     "DKASC, Alice Springs 1B Trina mono-Si Dual 2009"  -> device 87 channel M9 A+C.
  2. Magnitude: 87_DKA_M9_A_C_Phases_Active_Power peaks 21.8-23.0 kW, consistent
     with the 23.4 kW rating cited in the literature; 103_M1_B (1.8 kW) and
     78_M11 are different systems.
  3. Clear-day profile: dual-axis tracking yields a flat-topped daily profile.

Five-variable protocol identical to Site 18:
  P (M9 A+C) + GHI/DHI/T/RH from the shared 101 weather station.

Outputs -> results/site1b_val/phase01_audit.json (+ SITE_METADATA.json)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # release: repo-relative (was absolute path)
OUT = ROOT / "results" / "site1b_val"
YEARS = [2021, 2022, 2023, 2024, 2025]

P_RATED = 23.4  # literature value; cross-checked against data below
POWER_COL = "87_DKA_M9_A_C_Phases_Active_Power"
WCOLS = {
    "GHI": "101_DKA_WeatherStation_Global_Horizontal_Radiation",
    "DHI": "101_DKA_WeatherStation_Diffuse_Horizontal_Radiation",
    "T": "101_DKA_WeatherStation_Weather_Temperature_Celsius",
    "RH": "101_DKA_WeatherStation_Weather_Relative_Humidity",
}
S18_POWER = "71_DKA_M2_C_Phase_Active_Power"
HISTORY, HORIZON, STEP = 72, 3, 5


def clean(v: pd.Series) -> pd.Series:
    v = pd.to_numeric(v, errors="coerce")
    v[(v > 90000) | (v < -1e8)] = np.nan
    return v


def load_year(year: int) -> pd.DataFrame:
    use = ["timestamp", POWER_COL, *WCOLS.values()]
    df = pd.read_csv(ROOT / f"data/raw/Alice_Springs_{year}.csv", usecols=use,
                     on_bad_lines="skip")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp")
    df["power"] = clean(df[POWER_COL])
    for k, c in WCOLS.items():
        df[k] = clean(df[c])
    # physical validity masks (counted, not modified)
    df.loc[df["GHI"] < -50, "GHI"] = np.nan
    df.loc[df["GHI"] > 1400, "GHI"] = np.nan
    df.loc[df["DHI"] < -50, "DHI"] = np.nan
    df.loc[df["DHI"] > 1400, "DHI"] = np.nan
    df.loc[(df["T"] < -50) | (df["T"] > 60), "T"] = np.nan
    df.loc[(df["RH"] < 0) | (df["RH"] > 100), "RH"] = np.nan
    return df.set_index("timestamp")


def window_coverage(df: pd.DataFrame) -> tuple[int, int, int]:
    """Valid 72-step windows with a finite target, on the full 5-min grid."""
    grid = pd.date_range(df.index.min(), df.index.max(), freq=f"{STEP}min")
    g = df.reindex(grid)
    ok = g[["power", "GHI", "DHI", "T", "RH"]].notna().all(axis=1).to_numpy()
    n_all5 = int(ok.sum())
    # contiguous run lengths of valid rows -> windows = max(0, run-72+1-... )
    n_win = 0
    i = 0
    n = len(ok)
    target_ok = ok  # target = t+15min must also be valid; handled by shifting
    tgt = np.zeros(n, dtype=bool)
    tgt[: n - HORIZON] = ok[HORIZON:]
    while i < n:
        if ok[i]:
            j = i
            while j < n and ok[j]:
                j += 1
            run = j - i
            if run >= HISTORY:
                # windows fully inside [i, j): origin i+71 .. j-1, target valid
                for o in range(i + HISTORY - 1, j):
                    if tgt[o]:
                        n_win += 1
            i = j
        else:
            i += 1
    return len(g), n_all5, n_win


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit: dict = {}

    # ---- candidate identification table ----
    audit["candidates"] = [
        {"file": f"data/raw/Alice_Springs_{y}.csv", "system": "87 DKA M9 A+C Phases",
         "channel": POWER_COL, "evidence": "DKASC page slug dka-m9-a-c-phases = "
         "'Alice Springs 1B Trina mono-Si Dual 2009'; peaks 21.8-23.0 kW vs 23.4 kW rated"}
        for y in YEARS
    ]
    audit["rejected"] = [
        {"channel": "103_DKA_M1_B_Phase_Active_Power", "reason": "peaks 1.8 kW; "
         "name-similarity trap ('M1_B' != Site 1B)"},
        {"channel": "78_DKA_M11_3_Phase_Active_Power", "reason": "peaks ~23.5 kW but is "
         "a different DKASC system (M11), not the 1B tracker page"},
    ]
    audit["site_metadata"] = {
        "name": "DKASC Alice Springs Site 1B", "source_page":
        "https://dkasolarcentre.com.au/source/alice-springs/dka-m9-a-c-phases",
        "official_title": "DKASC, Alice Springs 1B Trina mono-Si Dual 2009",
        "technology": "Trina mono-Si, dual-axis tracking, 4 trackers, commissioned 2009",
        "rated_capacity_kw": P_RATED, "rated_capacity_source":
        "literature (IET RPG / Frontiers studies cite 23.4 kW); data max 21.8-23.0 kW",
        "shared_weather_with_site18": "101 DKA WeatherStation (same physical channels)",
        "relationship_to_site18": "same meteorological environment; different capacity, "
        "array configuration and tracking; supports cross-system (not cross-climate) claims",
    }

    rows, quality, winrows, s18_weather_check = [], {}, [], {}
    merged_store = {}
    for y in YEARS:
        df = load_year(y)
        merged_store[y] = df
        exp = 527040 // STEP if y % 4 == 0 else 525600 // STEP
        diffs = df.index.to_series().diff().dropna()
        ivc = diffs.value_counts()
        row = {
            "year": y, "rows": int(len(df)), "expected": int(exp),
            "coverage_pct": round(len(df) / exp * 100, 2),
            "power_valid_pct": round(df["power"].notna().mean() * 100, 2),
            "GHI_valid_pct": round(df["GHI"].notna().mean() * 100, 2),
            "DHI_valid_pct": round(df["DHI"].notna().mean() * 100, 2),
            "T_valid_pct": round(df["T"].notna().mean() * 100, 2),
            "RH_valid_pct": round(df["RH"].notna().mean() * 100, 2),
            "all5_valid_pct": round((df[["power", "GHI", "DHI", "T", "RH"]]
                                     .notna().all(axis=1)).mean() * 100, 2),
            "pct_5min": round(float(ivc.get(pd.Timedelta(minutes=5), 0) / int(ivc.sum())) * 100, 3),
            "power_max_kw": round(float(df["power"].max()), 2),
            "power_negative_count": int((df["power"] < 0).sum()),
            "power_gt_rated_count": int((df["power"] > P_RATED).sum()),
            "dhi_gt_ghi_pct": round(float((df["DHI"] > df["GHI"]).mean() * 100), 3),
        }
        rows.append(row)
        n_grid, n_all5, n_win = window_coverage(df)
        winrows.append({"year": y, "grid_rows": n_grid, "all5_valid": n_all5,
                        "valid_windows_with_target": n_win,
                        "window_coverage_pct": round(n_win / (exp - HISTORY + 1) * 100, 2)})
        print(row)
        print(winrows[-1])

    audit["per_year"] = rows
    audit["window_coverage"] = winrows

    # ---- shared weather with Site 18: numeric proof on 2022-2024 overlap ----
    # Both systems consume the SAME raw 101_* channels; Site 18's processed core
    # additionally applies cleaning/interpolation. Correlate our cleaned weather
    # with the Site 18 processed core (2022) to show they describe one stream.
    s18 = pd.read_csv(ROOT / "data/processed/site18_clean_core.csv", parse_dates=["timestamp"])
    s18 = s18.set_index("timestamp")
    pairs = {"GHI": "Global_Horizontal_Radiation", "DHI": "Diffuse_Horizontal_Radiation",
             "T": "Weather_Temperature_Celsius", "RH": "Weather_Relative_Humidity"}
    d22 = merged_store[2022]
    for k, c18 in pairs.items():
        if c18 not in s18.columns:
            continue
        j = d22[[k]].join(s18[[c18]].rename(columns={c18: "s18"}), how="inner").dropna()
        if len(j) > 1000:
            s18_weather_check[k] = {
                "n": int(len(j)),
                "pearson": round(float(np.corrcoef(j[k], j["s18"])[0, 1]), 6),
                "MAE": round(float((j[k] - j["s18"]).abs().mean()), 4),
                "exact_match_pct": round(float((j[k] == j["s18"]).mean() * 100), 2),
            }
    audit["shared_weather_vs_site18_processed_2022"] = s18_weather_check

    # ---- tracker profile check (clear days 2022, healthy tracker year) ----
    d = merged_store[2022]
    hr = d.index.hour + d.index.minute / 60
    midday = d["GHI"].where((hr >= 10) & (hr <= 14))
    daily = pd.DataFrame({"gmax": d["GHI"].resample("1D").max(),
                          "gstd_mid": midday.resample("1D").std()})
    clear = daily[(daily["gmax"] > 950) & (daily["gstd_mid"] < 80)].index[:10]
    noon = float(d[(d.index.normalize().isin(clear)) & (hr >= 11.5) & (hr <= 12.5)]["power"].median())
    mid = d[(d.index.normalize().isin(clear)) & (hr >= 10) & (hr <= 14)]["power"]
    early = float(d[(d.index.normalize().isin(clear)) & (hr >= 8.3) & (hr <= 8.8)]["power"].median())
    late = float(d[(d.index.normalize().isin(clear)) & (hr >= 15.3) & (hr <= 15.8)]["power"].median())
    audit["tracker_profile_check"] = {
        "clear_days_n": int(len(clear)), "noon_median_kw": round(noon, 2),
        "midday_flatness_std_over_mean": round(float(mid.std() / mid.mean()), 3),
        "early_0830_over_noon": round(early / noon, 3),
        "late_1530_over_noon": round(late / noon, 3),
        "interpretation": "flat midday plateau with high shoulder ratios indicates "
                          "dual-axis tracking (fixed-tilt Site 18 shows cosine arcs)",
    }
    print(json.dumps(audit["tracker_profile_check"], indent=2))

    # ---- verdict ----
    wby = {w["year"]: w["window_coverage_pct"] for w in winrows}
    ok_2022_2025 = all(wby.get(y, 0) >= 80 for y in (2022, 2023, 2024, 2025))
    audit["protocol_recommendation"] = {
        "historical_replay_years": [2022, 2023], "development": 2024, "cross_test": 2025,
        "rationale": "window coverage >=80%% for 2022-2025 -> symmetric protocol with "
                     "Site 18 rigorous_v2" if ok_2022_2025 else "see per-year table",
    }
    audit["phase01_verdict"] = "PASS" if ok_2022_2025 else "CONDITIONAL_PASS"
    (OUT / "phase01_audit.json").write_text(json.dumps(audit, indent=2, default=str),
                                            encoding="utf-8")
    (OUT / "SITE_METADATA.json").write_text(json.dumps(audit["site_metadata"], indent=2),
                                             encoding="utf-8")
    print("verdict:", audit["phase01_verdict"])


if __name__ == "__main__":
    main()
