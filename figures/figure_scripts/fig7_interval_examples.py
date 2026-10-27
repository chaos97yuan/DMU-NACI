"""Stage 9.5 Fig.7: prediction-interval case study (3 cases x A0/A3/CQR).

Reads ONLY the pre-registered case days (Fig7_case_days.csv, chosen by
fig7_case_selection.py without access to any method output) and the verified
frozen replay intervals (intervals_cross_seed42.parquet, replay_verification
<1e-6). x-axis = forecast target time (issue origin + 15 min).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # release: repo-relative
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "figures" / "figure_scripts"))
import figstyle as FS  # noqa: E402

SRC = ROOT / "figures" / "figure_source_data"

CASE_TITLES = {
    "C1_clear_stable": "Case 1: clear, stable day",
    "C2_cloud_variable": "Case 2: cloud-induced variable day",
    "C3_ramp_harddf": "Case 3: ramp / HARD_DF-rich day",
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

    fig, axes = FS.plt.subplots(3, 3, figsize=(7.0, 6.6), sharex="col")
    for r, (_, case) in enumerate(days.iterrows()):
        d = pd.Timestamp(case["day"])
        m = (iv["target_time"] >= d) & (iv["target_time"] < d + pd.Timedelta(days=1))
        g = iv.loc[m]
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
                ax.set_title(lab, fontsize=8)
            if cidx == 0:
                ax.set_ylabel("Power (kW)")
            ax.set_ylim(-0.55, 5.6)
            ax.set_xlim(d + pd.Timedelta(hours=5), d + pd.Timedelta(hours=21))
            ax.margins(x=0.01)
        axes[r, 0].annotate(CASE_TITLES[case["case"]], xy=(-0.62, 0.5),
                            xycoords="axes fraction", rotation=90,
                            va="center", ha="center", fontsize=8.5)
    import matplotlib.dates as mdates
    for ax in axes[-1]:
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[6, 10, 14, 18]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_xlabel("Target time (local)")
    for ax, letter in zip(axes.ravel(), "abcdefghi"):
        FS.panel_label(ax, letter, dx=-0.16, dy=1.06)
    handles = [FS.plt.Line2D([], [], color=FS.C_OBS, lw=0.8, label="Observed power"),
               FS.plt.Line2D([], [], color="#555555", lw=0.8, ls="--", label="Point forecast"),
               FS.plt.Rectangle((0, 0), 1, 1, fc="#888888", alpha=0.3, label="90% prediction interval"),
               FS.plt.Line2D([], [], color="#999999", ls="none", marker="|", ms=4,
                             label="HARD_DF origins (frozen label)")]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=6.5,
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.2, handletextpad=0.5)
    fig.tight_layout(h_pad=0.5, w_pad=0.4, rect=[0, 0, 1, 0.93])
    FS.save(fig, "Fig7_interval_examples")


if __name__ == "__main__":
    main()
    # persist source data after figure
    import pandas as pd
    days = pd.read_csv(SRC / "Fig7_case_days.csv")
    iv = pd.read_parquet(SRC / "intervals_cross_seed42.parquet")
    iv["target_time"] = pd.DatetimeIndex(iv["origin"]) + pd.Timedelta(minutes=15)
    out = []
    for _, case in days.iterrows():
        d = pd.Timestamp(case["day"])
        m = (iv["target_time"] >= d) & (iv["target_time"] < d + pd.Timedelta(days=1))
        g = iv.loc[m, ["target_time", "y_kw", "mu_phys", "sigma",
                       "A0_lo", "A0_up", "A3_lo", "A3_up", "CQR_lo", "CQR_up"]].copy()
        g.insert(0, "case", case["case"])
        out.append(g)
    pd.concat(out).to_csv(SRC / "Fig7_interval_examples_source.csv", index=False)
    print("saved Fig7_interval_examples_source.csv")
