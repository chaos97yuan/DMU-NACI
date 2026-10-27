"""Layout repair of Fig.7 (interval case study). Fixes vs the frozen render:
- panel letters moved INSIDE each axes (they used to collide with column
  titles from the neighbouring panel);
- case row labels wrapped to three short lines (they used to overlap
  neighbouring rows);
- re-rendered from the frozen replay parquet (all three pre-registered case
  days present with 288 target times each; the frozen PNG had empty panels for
  cases 1-2).
Data inputs unchanged: Fig7_case_days.csv + intervals_cross_seed42.parquet +
frozen 2025 HARD_DF labels.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # release: repo-relative
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
import figstyle as FS  # noqa: E402

SRC = ROOT / "figures" / "figure_source_data"

CASE_TITLES = {
    "C1_clear_stable": "Case 1\nclear,\nstable day",
    "C2_cloud_variable": "Case 2\ncloud-induced\nvariable day",
    "C3_ramp_harddf": "Case 3\nramp /\nHARD_DF-rich",
}
METHODS = [("A0", FS.C_A0, "A0  absolute fixed conformal"),
           ("A3", FS.C_A3, "A3  normalized delayed ACI (DMU-NACI)"),
           ("CQR", FS.C_CQR, "CQR")]


def main() -> None:
    days = pd.read_csv(SRC / "Fig7_case_days.csv")
    iv = pd.read_parquet(SRC / "intervals_cross_seed42.parquet")
    iv["target_time"] = pd.DatetimeIndex(iv["origin"]) + pd.Timedelta(minutes=15)

    from experiments.rigorous_v2 import common_v2 as C
    y25 = C.load_year_windows(2025)
    hd = np.asarray(y25["hard_DF"]).astype(bool)

    # NOTE: sharex must be OFF — each row shows a different day, so every row
    # needs its own x-window (sharex="col" let the last row's set_xlim hide the
    # data of the rows above — the exact bug that emptied the frozen PNG).
    fig, axes = FS.plt.subplots(3, 3, figsize=(7.0, 6.6), sharex=False)
    for r, (_, case) in enumerate(days.iterrows()):
        d = pd.Timestamp(case["day"])
        m = (iv["target_time"] >= d) & (iv["target_time"] < d + pd.Timedelta(days=1))
        g = iv.loc[m]
        assert len(g) > 0, f"no replay rows for case {case['case']} ({case['day']})"
        hdm = hd[np.asarray(m)]
        for cidx, (nm, colr, lab) in enumerate(METHODS):
            ax = axes[r, cidx]
            ax.fill_between(g["target_time"], g[f"{nm}_lo"], g[f"{nm}_up"],
                            color=colr, alpha=0.30, lw=0)
            ax.plot(g["target_time"], g["y_kw"], color=FS.C_OBS, lw=0.7)
            ax.plot(g["target_time"], g["mu_phys"], color="#555555", lw=0.7, ls="--")
            if hdm.any():
                tt = pd.DatetimeIndex(g["target_time"])[hdm]
                ax.plot(tt, np.full(len(tt), -0.34), ls="none", marker="|",
                        ms=3.5, color="#999999")
            if r == 0:
                ax.set_title(lab, fontsize=7.2)
            if cidx == 0:
                ax.set_ylabel("Power (kW)")
            ax.set_ylim(-0.55, 5.6)
            ax.set_xlim(d + pd.Timedelta(hours=5), d + pd.Timedelta(hours=21))
            ax.margins(x=0.01)
            if r < 2:
                ax.tick_params(labelbottom=False)
            ax.text(0.03, 0.96, f"({'abcdefghi'[r * 3 + cidx]})", transform=ax.transAxes,
                    fontsize=8.5, fontweight="bold", va="top", ha="left")
        axes[r, 0].annotate(CASE_TITLES[case["case"]], xy=(-0.30, 0.5),
                            xycoords="axes fraction", rotation=90,
                            va="center", ha="center", fontsize=7.2, linespacing=1.4)
    import matplotlib.dates as mdates
    for ax in axes[-1]:
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[6, 10, 14, 18]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_xlabel("Target time (local)")
    handles = [FS.plt.Line2D([], [], color=FS.C_OBS, lw=0.8, label="Observed power"),
               FS.plt.Line2D([], [], color="#555555", lw=0.8, ls="--", label="Point forecast"),
               FS.plt.Rectangle((0, 0), 1, 1, fc="#888888", alpha=0.3, label="90% prediction interval"),
               FS.plt.Line2D([], [], color="#999999", ls="none", marker="|", ms=4,
                             label="HARD_DF origins (frozen label)")]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=6.5,
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.2, handletextpad=0.5)
    fig.tight_layout(h_pad=0.5, w_pad=0.4, rect=[0.035, 0, 1, 0.94])
    FS.save(fig, "Fig7_interval_examples")


if __name__ == "__main__":
    main()
