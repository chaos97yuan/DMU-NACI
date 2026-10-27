"""New main Fig. 5 for the v5 restructured draft: simplified difficult-state
reliability figure with three panels:
  (a) conditional PICP by state (NORMAL / HIGH_N / HARD_DF) for A0-A3
  (b) within-HARD_DF PICP vs mean width (A0-A3, CQR)
  (c) within-HARD_DF PICP vs interval score (A0-A3, CQR)
Reads ONLY the frozen stage9_5 source CSVs (Fig9_conditional_per_seed.csv,
Fig9_HARD_DF_tradeoff_source.csv); writes into manuscript/latex_v5/figs.
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
OUT = ROOT / "figures" / "output_v5"
METHODS = ["A0", "A1", "A2", "A3", "CQR"]
COLORS = {"A0": FS.C_A0, "A1": FS.C_A1, "A2": FS.C_A2, "A3": FS.C_A3, "CQR": FS.C_CQR}


def main() -> None:
    per_seed = pd.read_csv(SRC / "Fig9_conditional_per_seed.csv")
    hard = pd.read_csv(SRC / "Fig9_HARD_DF_tradeoff_source.csv")
    med = per_seed.drop(columns="seed").median()

    FS.setup()
    fig, axes = FS.plt.subplots(1, 3, figsize=(7.2, 2.5))

    # (a) conditional PICP by state, A0-A3
    ax = axes[0]
    sts = ["NORMAL", "HIGH_N", "HARD_DF"]
    x = np.arange(len(sts))
    w = 0.2
    for i, m in enumerate(["A0", "A1", "A2", "A3"]):
        vals = [med[f"{m}_PICP_{st}"] for st in sts]
        ax.bar(x + (i - 1.5) * w, vals, width=w * 0.92, color=COLORS[m],
               label=FS.METHOD_SHORT[m])
        for j, st in enumerate(sts):
            FS.seed_dots(ax, x[j] + (i - 1.5) * w, per_seed[f"{m}_PICP_{st}"],
                         COLORS[m], size=6)
    ax.axhline(0.90, color=FS.C_REF, ls="--", lw=0.9)
    ax.set_xticks(x, ["NORMAL", "HIGH_N", "HARD_DF"], fontsize=7)
    ax.set_ylabel("Conditional PICP")
    ax.set_ylim(0.30, 1.02)
    ax.legend(fontsize=5.6, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              columnspacing=0.7, handletextpad=0.3)

    # (b) within-HARD_DF PICP vs mean width
    ax = axes[1]
    for m in METHODS:
        g = hard[hard["method"] == m]
        ax.scatter(g["HARD_width_kw"], g["HARD_PICP"], s=12, color=COLORS[m],
                   edgecolors="white", linewidths=0.4, zorder=4)
        ax.scatter([g["HARD_width_kw"].median()], [g["HARD_PICP"].median()], s=48,
                   color=COLORS[m], marker="*", zorder=5, label=FS.METHOD_SHORT[m])
    ax.axhline(0.90, color=FS.C_REF, ls="--", lw=0.9)
    ax.set_xlabel("Mean width within HARD_DF (kW)", fontsize=7.5)
    ax.set_ylabel("PICP within HARD_DF")
    ax.set_ylim(0.30, 1.0)
    ax.legend(fontsize=5.6, loc="lower right", handletextpad=0.1, borderaxespad=0.2)

    # (c) within-HARD_DF PICP vs interval score
    ax = axes[2]
    for m in METHODS:
        g = hard[hard["method"] == m]
        ax.scatter(g["HARD_IS"], g["HARD_PICP"], s=12, color=COLORS[m],
                   edgecolors="white", linewidths=0.4, zorder=4)
        ax.scatter([g["HARD_IS"].median()], [g["HARD_PICP"].median()], s=48,
                   color=COLORS[m], marker="*", zorder=5)
    ax.axhline(0.90, color=FS.C_REF, ls="--", lw=0.9)
    ax.set_xlabel("Interval score within HARD_DF (kW)", fontsize=7.5)
    ax.set_ylabel("PICP within HARD_DF")
    ax.set_ylim(0.30, 1.0)
    ax.text(0.03, 0.03, "dots = seeds;  stars = medians",
            transform=ax.transAxes, fontsize=5.8, color="#555555")

    for ax, letter, inside in zip(axes, "abc", (False, True, True)):
        if inside:
            ax.text(0.012, 0.985, f"({letter})", transform=ax.transAxes,
                    fontsize=9, fontweight="bold", va="top", ha="left")
        else:
            FS.panel_label(ax, letter, dx=-0.16, dy=1.10)
    fig.tight_layout(w_pad=1.1)
    OUT.mkdir(parents=True, exist_ok=True)
    for dpi, tag in ((600, ""), (300, "_300dpi")):
        fig.savefig(OUT / f"fig5_harddf{tag}.png", dpi=dpi)
    print("saved fig5_harddf.png (+300dpi)")


if __name__ == "__main__":
    main()
