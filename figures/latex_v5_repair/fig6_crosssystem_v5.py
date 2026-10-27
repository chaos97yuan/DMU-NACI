"""v5 Fig. 6 (cross-system replication) with the CQR+ACI baseline added.

Same style as the frozen stage9_5 Fig.10 script. Data: frozen
Fig10_cross_system_source.csv (A0/A2/A3/CQR) + the new controlled CQR-ACI run
(results/*/cqr_aci_per_seed.csv, five matched seeds, frozen CQR checkpoints).
Writes manuscript/latex_v5/figs/fig6_crosssystem.png only.
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
METHODS = ["A0", "A2", "A3", "CQR", "CQR+ACI"]
COLORS = {"A0": FS.C_A0, "A2": FS.C_T2, "A3": FS.C_A3, "CQR": FS.C_CQR,
          "CQR+ACI": "#0072B2"}
SHORT = {"A0": "A0", "A2": "A2", "A3": "A3\n(proposed)", "CQR": "CQR",
         "CQR+ACI": "CQR\n+ACI"}


def main() -> None:
    src = pd.read_csv(SRC / "Fig10_cross_system_source.csv")
    s18 = pd.read_csv(ROOT / "results" / "rigorous_v2" / "s9_cqr_aci" / "cqr_aci_per_seed.csv")
    s1b = pd.read_csv(ROOT / "results" / "site1b_val" / "cqr_aci" / "cqr_aci_per_seed.csv")
    extra = []
    for sysname, d in (("Site 18 (5.2 kW, fixed-tilt)", s18),
                       ("Site 1B (23.4 kW, dual-axis tracker)", s1b)):
        for _, r in d.iterrows():
            extra.append({"system": sysname, "method": "CQR+ACI", "seed": int(r["seed"]),
                          "PICP": float(r["PICP"]), "NIS": float(r["NIS"]),
                          "RCE1D": float(r["RCE1D"])})
    src = pd.concat([src, pd.DataFrame(extra)], ignore_index=True)

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
            ax.bar(x + (j - 0.5) * w, meds, width=w * 0.94,
                   color=[COLORS[m] for m in METHODS], hatch=hatch,
                   edgecolor="white", linewidth=0.4,
                   label=sysname if col == "PICP" else None)
            for i, m in enumerate(METHODS):
                FS.seed_dots(ax, x[i] + (j - 0.5) * w, g[g["method"] == m][col],
                             COLORS[m], size=8)
        if ref is not None:
            ax.axhline(ref, color=FS.C_REF, ls="--", lw=0.9)
            ax.text(len(METHODS) - 0.45, ref + 0.002, "0.90 target", fontsize=6.5,
                    ha="right", color="#555555")
        ax.set_xticks(x, [SHORT[m] for m in METHODS], fontsize=6.6)
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
        FS.panel_label(ax, letter, dx=-0.17, dy=1.08)
    fig.tight_layout(w_pad=1.4)
    OUT.mkdir(parents=True, exist_ok=True)
    for dpi, tag in ((600, ""), (300, "_300dpi")):
        fig.savefig(OUT / f"fig6_crosssystem{tag}.png", dpi=dpi)
    med = src.groupby(["system", "method"])[["PICP", "NIS", "RCE1D"]].median().round(4)
    print(med.to_string())


if __name__ == "__main__":
    main()
