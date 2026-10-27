"""Stage 9.5 Fig.1 (Site 18 data characteristics): PRE-REGISTERED typical-week
selection + 5-panel overview.

Selection rule (fixed before plotting; uses ONLY raw cleaned 2024 series,
no model output involved):
  universe  = 2024 development span 2024-01-01 00:00 .. 2024-08-08 00:00
             (the MEAN_GRAD + MEAN_ES window; data/processed/external_2024_preopen/2024_core.csv)
  unit      = ISO week (Mon 00:00 .. Sun 23:55); eligible if >=95% of the 5-min
             grid has finite Active_Power
  features  = E = weekly median of daily PV energy (kWh, positive part)
              V = weekly median of daily daytime GHI flicker index
                  v_d = mean(|diff(GHI)| over GHI>20 W/m2 samples) / max(GHI_d)
  choice    = eligible week minimizing sqrt(z_E^2 + z_V^2), z = standardized
             feature over eligible weeks (closest to the median training state)
Outputs: figures/Fig1_data_overview.png, figure_source_data/
         Fig1_data_overview_source.csv + week_selection.csv,
         01_DATA_OVERVIEW_SELECTION.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # release: repo-relative
sys.path.insert(0, str(ROOT / "figures" / "figure_scripts"))
import figstyle as FS  # noqa: E402

CORE = ROOT / "data" / "processed" / "external_2024_preopen" / "2024_core.csv"
STAGE = ROOT / "figures"
SRC = STAGE / "figure_source_data"

DEV_END = pd.Timestamp("2024-08-08 00:00")
DAYLIGHT = 20.0  # W/m2, frozen pipeline threshold


def week_features(core: pd.DataFrame) -> pd.DataFrame:
    ts = pd.DatetimeIndex(core["timestamp"])
    core = core.assign(_d=ts.normalize(), _w=ts.to_period("W-SUN").start_time.normalize())
    rows = []
    for w, g in core.groupby("_w"):
        if not (w + pd.Timedelta(days=7) <= DEV_END):
            # only full weeks fully inside the development span
            continue
        days, energies, flickers = [], [], []
        for d, gd in g.groupby("_d"):
            p = gd["Active_Power"].to_numpy(float)
            ghi = gd["Global_Horizontal_Radiation"].to_numpy(float)
            days.append(d)
            energies.append(float(np.clip(p, 0, None).sum() * 5.0 / 60.0))  # kWh
            m = ghi > DAYLIGHT
            both = m[:-1] & m[1:]  # successive full-resolution steps inside daytime
            if m.sum() > 10 and np.nanmax(ghi[m]) > 50 and both.sum() > 5:
                flick = float(np.nanmean(np.abs(np.diff(ghi))[both]) / np.nanmax(ghi[m]))
            else:
                flick = np.nan
            flickers.append(flick)
        finite_share = float(np.isfinite(g["Active_Power"].to_numpy(float)).mean())
        rows.append({"week_start": w, "n_days": len(days),
                     "finite_share": finite_share,
                     "median_daily_energy_kwh": float(np.nanmedian(energies)),
                     "median_daily_flicker": float(np.nanmedian(flickers))})
    return pd.DataFrame(rows)


def main() -> None:
    SRC.mkdir(parents=True, exist_ok=True)
    core = pd.read_csv(CORE)
    core["timestamp"] = pd.to_datetime(core["timestamp"])
    core = core[core["timestamp"] < DEV_END]

    wf = week_features(core)
    elig = wf[(wf["finite_share"] >= 0.95) & np.isfinite(wf["median_daily_flicker"])].copy()
    zE = (elig["median_daily_energy_kwh"] - elig["median_daily_energy_kwh"].mean()) / elig["median_daily_energy_kwh"].std()
    zV = (elig["median_daily_flicker"] - elig["median_daily_flicker"].mean()) / elig["median_daily_flicker"].std()
    elig["z_energy"], elig["z_flicker"] = zE, zV
    elig["distance"] = np.sqrt(zE ** 2 + zV ** 2)
    elig = elig.sort_values("distance")
    chosen = elig.iloc[0]
    ws, we = chosen["week_start"], chosen["week_start"] + pd.Timedelta(days=7)
    print(f"chosen week: {ws.date()} .. {we.date()}  (dist={chosen['distance']:.3f})")

    wf.to_csv(SRC / "Fig1_week_selection_full.csv", index=False)
    elig.to_csv(SRC / "Fig1_week_selection_eligible_ranked.csv", index=False)

    # ---- figure ----
    m = (core["timestamp"] >= ws) & (core["timestamp"] < we)
    wk = core.loc[m]
    FS.setup(7.0, 6.4)
    fig, axes = FS.plt.subplots(5, 1, sharex=True, figsize=(7.0, 6.4))
    panels = [
        ("Active_Power", "Active power (kW)", FS.C_A3, "Power"),
        ("Global_Horizontal_Radiation", r"GHI (W m$^{-2}$)", FS.C_T2, "GHI"),
        ("Diffuse_Horizontal_Radiation", r"DHI (W m$^{-2}$)", FS.C_T2, "DHI"),
        ("Weather_Temperature_Celsius", r"Air temperature (°C)", FS.C_T1, "Temp"),
        ("Weather_Relative_Humidity", "Relative humidity (%)", FS.C_T3, "RH"),
    ]
    for ax, (col, lab, colr, _) in zip(axes, panels):
        ax.plot(wk["timestamp"], wk[col], color=colr, lw=0.55)
        ax.set_ylabel(lab)
        ax.margins(x=0.01)
    axes[-1].set_xlabel("Time (5-min resolution, local time)")
    import matplotlib.dates as mdates
    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    for ax, letter in zip(axes, "abcde"):
        FS.panel_label(ax, letter, dx=-0.135, dy=1.10)
    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=0.35)

    src = wk.rename(columns={"Active_Power": "power_kw",
                             "Global_Horizontal_Radiation": "ghi_wm2",
                             "Diffuse_Horizontal_Radiation": "dhi_wm2",
                             "Weather_Temperature_Celsius": "temp_c",
                             "Weather_Relative_Humidity": "rh_pct"})
    src.to_csv(SRC / "Fig1_data_overview_source.csv", index=False)
    FS.save(fig, "Fig1_data_overview")

    # ---- selection doc ----
    top = elig.head(8).copy()
    top["week_start"] = top["week_start"].dt.strftime("%Y-%m-%d")
    md = f"""# 01_DATA_OVERVIEW_SELECTION.md

Purpose: choose the "typical week" shown in Fig. 1 (Site 18 data characteristics)
by a rule that is independent of any model output.

## Pre-registered rule (fixed before any plotting)

- Universe: cleaned 5-min series of the 2024 development span
  (2024-01-01 00:00 to 2024-08-08 00:00 = MEAN_GRAD + MEAN_ES),
  `data/processed/external_2024_preopen/2024_core.csv` — the same cleaned core the
  pipeline consumes.
- Unit: full ISO weeks (Mon 00:00 .. Sun 23:55) entirely inside the span.
- Eligibility: >= 95% of 5-min timestamps with finite Active_Power.
- Features (raw series only):
  - E = weekly median of daily PV energy (kWh; positive part of power x 5 min).
  - V = weekly median of the daily daytime GHI flicker index
    v_d = mean(|Delta GHI| over samples with GHI > 20 W/m2) / max(GHI_d).
- Choice: minimize sqrt(z_E^2 + z_V^2) over eligible weeks, z = feature
  standardized across eligible weeks (i.e. the week closest to the median
  training state in energy-flicker space).
- No model prediction, residual, interval, or metric enters the rule.

## Result

- Eligible weeks: {len(elig)} of {len(wf)} candidate weeks.
- Chosen week: **{ws.strftime('%Y-%m-%d')} to {(we - pd.Timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M')}**
  (z_E = {chosen['z_energy']:+.3f}, z_V = {chosen['z_flicker']:+.3f},
  distance = {chosen['distance']:.3f}).

Top-ranked weeks (smaller distance = more typical):

```
{top[['week_start', 'median_daily_energy_kwh', 'median_daily_flicker', 'z_energy', 'z_flicker', 'distance']].round(3).to_string(index=False)}
```

Full tables: `figure_source_data/Fig1_week_selection_full.csv`,
`Fig1_week_selection_eligible_ranked.csv`.

## Independence statement

The selection uses only `2024_core.csv` raw channels (power, GHI) of the
development span. No TCN/GRU output, no residual pool, no calibration artifact,
and no 2025 data were accessed while selecting the week. The rule, features,
and eligibility threshold were fixed in this script before the figure was
rendered.

## 2023 low-power regime

Per the Stage-9.5 brief, the anomalous 2023 operational regime is NOT shown in
the main overview; it is documented in manuscript Section 3.2 (Site 1B) and in
the replay-pool statistics (2023 replay MAE 0.351 kW vs 2022 0.450 kW).
"""
    (STAGE / "01_DATA_OVERVIEW_SELECTION.md").write_text(md, encoding="utf-8")
    print("wrote 01_DATA_OVERVIEW_SELECTION.md")


if __name__ == "__main__":
    main()
