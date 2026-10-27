"""Layout repair of Fig.1 (data overview): panel letters moved INSIDE each
axes (they used to sit on top of the rotated y-axis labels). Selection logic
and data identical to the frozen pre-registered script; this variant writes
only the figure into manuscript/latex_v4/figs (no writes to frozen dirs).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # release: repo-relative
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as FS  # noqa: E402

CORE = ROOT / "data" / "processed" / "external_2024_preopen" / "2024_core.csv"

DEV_END = pd.Timestamp("2024-08-08 00:00")
DAYLIGHT = 20.0  # W/m2, frozen pipeline threshold


def week_features(core: pd.DataFrame) -> pd.DataFrame:
    ts = pd.DatetimeIndex(core["timestamp"])
    core = core.assign(_d=ts.normalize(), _w=ts.to_period("W-SUN").start_time.normalize())
    rows = []
    for w, g in core.groupby("_w"):
        if not (w + pd.Timedelta(days=7) <= DEV_END):
            continue
        days, energies, flickers = [], [], []
        for d, gd in g.groupby("_d"):
            p = gd["Active_Power"].to_numpy(float)
            ghi = gd["Global_Horizontal_Radiation"].to_numpy(float)
            days.append(d)
            energies.append(float(np.clip(p, 0, None).sum() * 5.0 / 60.0))
            m = ghi > DAYLIGHT
            both = m[:-1] & m[1:]
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
    print(f"chosen week: {ws.date()} .. {we.date()}  (must be 2024-05-06 .. 2024-05-12)")

    m = (core["timestamp"] >= ws) & (core["timestamp"] < we)
    wk = core.loc[m]
    FS.setup(7.0, 6.4)
    fig, axes = FS.plt.subplots(5, 1, sharex=True, figsize=(7.0, 6.4))
    panels = [
        ("Active_Power", "Active power (kW)", FS.C_A3),
        ("Global_Horizontal_Radiation", r"GHI (W m$^{-2}$)", FS.C_T2),
        ("Diffuse_Horizontal_Radiation", r"DHI (W m$^{-2}$)", FS.C_T2),
        ("Weather_Temperature_Celsius", r"Air temperature (°C)", FS.C_T1),
        ("Weather_Relative_Humidity", "Relative humidity (%)", FS.C_T3),
    ]
    for ax, (col, lab, colr) in zip(axes, panels):
        ax.plot(wk["timestamp"], wk[col], color=colr, lw=0.55)
        ax.set_ylabel(lab)
        ax.margins(x=0.01)
    axes[-1].set_xlabel("Time (5-min resolution, local time)")
    import matplotlib.dates as mdates
    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    for ax, letter in zip(axes, "abcde"):
        ax.text(0.012, 0.985, f"({letter})", transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top", ha="left")
    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=0.35)
    FS.save(fig, "Fig1_data_overview")


if __name__ == "__main__":
    main()
