"""Stage 9.5 Fig.10: cross-system probabilistic validation, 2025 cross-year span.

Site 18 (5.2 kW fixed-tilt): verified frozen replay (this stage).
Site 1B (23.4 kW dual-axis tracker): results/site1b_val/site1b_eval_per_seed.csv
(frozen Phase-8/9 outputs; FREEZE_MANIFEST_SITE1B.json).
Panels: (a) PICP with 0.90 target  (b) NIS = IS / P_rated  (c) RCE1D.
Bars = median of 5 matched seeds; dots = seeds. Methods A0/A2/A3/CQR.
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
S18_P = 5.2
S1B_P = 23.4
METHODS = ["A0", "A2", "A3", "CQR"]
COLORS = {"A0": FS.C_A0, "A2": FS.C_T2, "A3": FS.C_A3, "CQR": FS.C_CQR}
SHORT = {"A0": "A0", "A2": "A2", "A3": "A3\n(proposed)", "CQR": "CQR"}


def main() -> None:
    pm18 = pd.read_csv(SRC / "Fig8_per_seed_metrics.csv")
    b1 = pd.read_csv(ROOT / "results" / "site1b_val" / "site1b_eval_per_seed.csv")
    b1 = b1[b1["span"] == "cross"]

    rows = []
    for m in METHODS:
        g18 = pm18[f"{m}_PICP"], pm18[f"{m}_IS"] / S18_P, pm18[f"{m}_RCE1D"]
        g1b = (b1[b1["method"] == m]["PICP"], b1[b1["method"] == m]["IS"] / S1B_P,
               b1[b1["method"] == m]["RCE1D"])
        for sysname, (picp, nis, rce) in (("Site 18 (5.2 kW, fixed-tilt)", g18),
                                          ("Site 1B (23.4 kW, dual-axis tracker)", g1b)):
            for s, p, n, r in zip([42, 123, 2026, 3407, 8888], picp, nis, rce):
                rows.append({"system": sysname, "method": m, "seed": s,
                             "PICP": p, "NIS": n, "RCE1D": r})
    src = pd.DataFrame(rows)
    src.to_csv(SRC / "Fig10_cross_system_source.csv", index=False)

    FS.setup()
    fig, axes = FS.plt.subplots(1, 3, figsize=(7.2, 2.8))
    panels = [("PICP", "PICP", 0.90, (0.84, 0.97)),
              ("NIS", "NIS (IS / P_rated)", None, (0, None)),
              ("RCE1D", "RCE1D", None, (0, None))]
    x = np.arange(len(METHODS))
    w = 0.36
    for ax, (col, ylab, ref, ylim) in zip(axes, panels):
        for j, sysname in enumerate(src["system"].unique()):
            g = src[src["system"] == sysname]
            meds = [g[g["method"] == m][col].median() for m in METHODS]
            hatch = "//" if j else None
            ax.bar(x + (j - 0.5) * w, meds, width=w * 0.94, color=[COLORS[m] for m in METHODS],
                   hatch=hatch, edgecolor="white", linewidth=0.4,
                   label=sysname if col == "PICP" else None)
            for i, m in enumerate(METHODS):
                FS.seed_dots(ax, x[i] + (j - 0.5) * w, g[g["method"] == m][col],
                             COLORS[m], size=8)
        if ref is not None:
            ax.axhline(ref, color=FS.C_REF, ls="--", lw=0.9)
            ax.text(len(METHODS) - 0.45, ref + 0.002, "0.90 target", fontsize=6.5,
                    ha="right", color="#555555")
        ax.set_xticks(x, [SHORT[m] for m in METHODS], fontsize=7)
        ax.set_ylabel(ylab)
        if ylim[0] is not None:
            ax.set_ylim(bottom=ylim[0], top=ylim[1])
    import matplotlib.patches as mpatches
    h1 = mpatches.Patch(facecolor="#BBBBBB", label="Site 18 (5.2 kW, fixed-tilt)")
    h2 = mpatches.Patch(facecolor="#BBBBBB", hatch="//", edgecolor="white",
                        label="Site 1B (23.4 kW, dual-axis tracker)")
    fig.legend(handles=[h1, h2], fontsize=6.8, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.13))
    for ax, letter in zip(axes, "abc"):
        FS.panel_label(ax, letter, dx=-0.19, dy=1.08)
    fig.tight_layout(w_pad=1.4)
    FS.save(fig, "Fig10_cross_system")

    med = src.groupby(["system", "method"])[["PICP", "NIS", "RCE1D"]].median().round(4)
    print(med.to_string())


if __name__ == "__main__":
    main()
