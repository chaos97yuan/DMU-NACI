"""Stage 9.5 Fig.6: residual-provenance pools on the 2025 cross-year span
(Site 18) + monthly reliability.

(a) A2 interval score  (b) error-scale Spearman  (c) |Std(z)-1|  -- variants
    T1p/T2p/T3p/T1in/T3match/V1/V3; bars = median of 5 seeds, dots = seeds
(d) monthly interval score (seed medians) for T1p/T3p/V3
Source: results/rigorous_v2/s7_final_eval/{metrics_per_seed,monthly_cross_year}.csv
(frozen; numbers already in v3 Tables 4-5). No recomputation.
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
S7 = ROOT / "results" / "rigorous_v2" / "s7_final_eval"

VARIANTS = [("T1p", "T1′", FS.C_T1),
            ("T2p", "T2′", FS.C_T2),
            ("T3p", "T3′\n(depl.)", FS.C_T3),
            ("T1in", "T1-in", "#8C564B"),
            ("T3match", "T3-match", "#0072B2"),
            ("V1", "V1", FS.C_V1),
            ("V3", "V3", FS.C_V3)]


def main() -> None:
    m = pd.read_csv(S7 / "metrics_per_seed.csv")
    mo = pd.read_csv(S7 / "monthly_cross_year.csv")
    cross = m[m["span"] == "cross"]

    src = cross[cross["variant"].isin([v for v, _, _ in VARIANTS])][
        ["variant", "seed", "A2_IS", "A2_PICP", "Spearman", "Std_z"]].copy()
    src["abs_Std_z_minus_1"] = (src["Std_z"] - 1).abs()
    src.to_csv(SRC / "Fig6_provenance_pool_source.csv", index=False)
    mo[mo["variant"].isin(["T1p", "T3p", "V3"])].to_csv(
        SRC / "Fig6_monthly_source.csv", index=False)

    FS.setup()
    fig, axes = FS.plt.subplots(2, 2, figsize=(7.2, 5.0))
    panels = [("A2_IS", "Interval score (kW)", "a", (0, 0), False),
              ("Spearman", r"Spearman($\sigma_t$, $|e_t|$)", "b", (0, 1), False),
              ("abs_Std_z_minus_1", r"$|$Std$(z)-1|$", "c", (1, 0), True)]
    for col, ylab, letter, (r, c), zero_line in panels:
        ax = axes[r, c]
        med, names, cols = [], [], []
        for i, (v, lab, colr) in enumerate(VARIANTS):
            g = src[src["variant"] == v]
            ax.bar(i, g[col].median(), color=colr, width=0.62)
            FS.seed_dots(ax, i, g[col], colr)
            med.append(g[col].median())
            names.append(lab)
            cols.append(colr)
        if zero_line:
            pass
        ax.set_xticks(range(len(VARIANTS)), names, fontsize=6.0)
        ax.set_ylabel(ylab)

    # (d) monthly IS
    ax = axes[1, 1]
    mm = mo[mo["variant"].isin(["T1p", "T3p", "V3"])].groupby(["variant", "month"])["IS"].median().unstack(0)
    styles = {"T1p": (FS.C_T1, "o", 0.9), "T3p": (FS.C_T3, "s", 1.2), "V3": (FS.C_V3, "^", 0.9)}
    labels = {"T1p": "T1' (OOF 2024)", "T3p": "T3' (deployed replay pool)", "V3": "V3 (vintage pooled)"}
    for v in ("T1p", "T3p", "V3"):
        colr, mk, lw = styles[v]
        ax.plot(mm.index, mm[v], color=colr, marker=mk, ms=3, lw=lw, label=labels[v])
    ax.set_xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_xlabel("Month (2025)")
    ax.set_ylabel("Monthly interval score (kW)")
    ax.legend(fontsize=6.2, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
               columnspacing=1.0, handletextpad=0.4)

    for ax, letter in zip(axes.ravel(), "abcd"):
        FS.panel_label(ax, letter, dx=-0.15, dy=1.06)
    fig.tight_layout(w_pad=1.3, h_pad=1.2)
    FS.save(fig, "Fig6_provenance_pools_monthly")


if __name__ == "__main__":
    main()
