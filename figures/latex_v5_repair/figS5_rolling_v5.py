"""v5 Supplementary Fig. S5: rolling 1-day / 3-day PICP with CQR+ACI added.

Data: frozen Fig8_rolling_coverage_source.csv (A0/A1/A2/A3/CQR, seed-median
daily coverage) + the controlled CQR-ACI run artifacts
(results/rigorous_v2/s9_cqr_aci/cqr_aci_intervals_seed*.csv, per-seed daily
coverage -> seed median; picp3d = 3-day rolling mean, mirroring Fig8).
Writes manuscript/latex_v5/figs/figS5_rolling.png (+300dpi) only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # release: repo-relative
sys.path.insert(0, str(ROOT / "figures" / "latex_v4_repair"))
import figstyle as FS  # noqa: E402

SRC = ROOT / "figures" / "figure_source_data"
ACI = ROOT / "results" / "rigorous_v2" / "s9_cqr_aci"
OUT = ROOT / "figures" / "output_v5"
SEEDS = [42, 123, 2026, 3407, 8888]
COLORS = {"A0": FS.C_A0, "A1": FS.C_A1, "A2": FS.C_T2, "A3": FS.C_A3,
          "CQR": FS.C_CQR, "CQR+ACI": "#0072B2"}
SHORT = {"A0": "A0", "A1": "A1", "A2": "A2", "A3": "A3 (proposed)",
         "CQR": "CQR", "CQR+ACI": "CQR+ACI"}


def main() -> None:
    roll = pd.read_csv(SRC / "Fig8_rolling_coverage_source.csv", parse_dates=["day"])

    # CQR-ACI: per-seed daily coverage from the frozen controlled-run artifacts
    frames = []
    for s in SEEDS:
        d = pd.read_csv(ACI / f"cqr_aci_intervals_seed{s}.csv",
                        parse_dates=["target_time"])
        cov = ((d["y_true"] >= d["lower"]) & (d["y_true"] <= d["upper"])).astype(float)
        day = pd.DatetimeIndex(d["target_time"]).normalize()
        frames.append(pd.Series(cov.to_numpy(), index=day).groupby(level=0).mean())
    df = pd.concat(frames, axis=1)
    med = df.median(axis=1)
    cqraci = pd.DataFrame({"day": med.index, "picp1d": med.to_numpy(),
                           "picp3d": med.rolling(3, min_periods=1).mean().to_numpy()})

    FS.setup()
    fig, axes = FS.plt.subplots(1, 2, figsize=(7.2, 2.7))
    for k, (key, ttl) in enumerate((("picp1d", "1-day rolling PICP"),
                                    ("picp3d", "3-day rolling PICP"))):
        ax = axes[k]
        ax.axhline(0.90, color=FS.C_REF, ls="--", lw=0.9, zorder=1)
        for m in ["A0", "A1", "A2", "A3", "CQR"]:
            ax.plot(roll["day"], roll[f"{m}_{key}"], color=COLORS[m], lw=1.0,
                    zorder=3, label=SHORT[m])
        col = "picp1d" if k == 0 else "picp3d"
        ax.plot(cqraci["day"], cqraci[col], color=COLORS["CQR+ACI"], lw=1.0,
                zorder=3, label=SHORT["CQR+ACI"])
        ax.set_ylim(0.70, 1.00)
        ax.set_ylabel(ttl)
        import matplotlib.dates as mdates
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.tick_params(axis="x", labelsize=6.8)
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.04), ncol=6, fontsize=6.0,
                  columnspacing=0.8, handletextpad=0.35)
        FS.panel_label(ax, "ab"[k], dx=-0.16, dy=1.16)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for dpi, tag in ((600, ""), (300, "_300dpi")):
        fig.savefig(OUT / f"figS5_rolling{tag}.png", dpi=dpi)
    print("saved figS5_rolling.png (+300dpi)")


if __name__ == "__main__":
    main()
