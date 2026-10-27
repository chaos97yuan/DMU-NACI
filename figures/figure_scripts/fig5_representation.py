"""Stage 9.5 Fig.5: mean-uncertainty representation diagnosis (R0/R1/R2),
Stage-8 confirmatory Stage-1 experiment (the artifact v3 Section 4.1 quotes).

(a) cross-year Std(z) with the Std(z)=1 reference line
(b) cross-year Gaussian NLL  -- bars = median of 5 seeds, dots = seeds
Different quantities on separate axes (no mixed units).
Source: results/stage8_final_architecture/05_stage1_confirmatory/metrics_per_seed.csv
(frozen; medians 4.9575 / 1.5840 / 4.2519 are exactly the v3 text values).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # release: repo-relative
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "figures" / "figure_scripts"))
import figstyle as FS  # noqa: E402

SRC = ROOT / "figures" / "figure_source_data"
S1C = ROOT / "results" / "stage8_final_architecture" / "05_stage1_confirmatory"

REPS = [("R0_hms", "R0  mean embedding (H_ms)", FS.C_T3),
        ("R1_raw360", "R1  raw window (360-dim)", FS.C_T2),
        ("R2_hms_dynamics", "R2  embedding + dynamics", "#8C564B")]


def main() -> None:
    d = pd.read_csv(S1C / "metrics_per_seed.csv")
    cross = d[d["span"] == "cross"]
    src = cross[cross["mode"].isin([m for m, _, _ in REPS])][["mode", "seed", "Std_z", "NLL"]]
    src.to_csv(SRC / "Fig5_representation_source.csv", index=False)

    FS.setup()
    fig, axes = FS.plt.subplots(1, 2, figsize=(7.0, 2.7))
    specs = [("Std_z", "Cross-year Std(z)", "a", 1.0, "Std(z) = 1"),
             ("NLL", "Cross-year Gaussian NLL (nats)", "b", None, None)]
    for ax, (col, ylab, letter, ref, reflab) in zip(axes, specs):
        for i, (m, lab, colr) in enumerate(REPS):
            g = src[src["mode"] == m]
            ax.bar(i, g[col].median(), color=colr, width=0.55)
            FS.seed_dots(ax, i, g[col], colr)
        ax.set_xticks(range(len(REPS)), [lab for _, lab, _ in REPS], fontsize=6.8)
        ax.set_ylabel(ylab)
        if ref is not None:
            ax.axhline(ref, color=FS.C_REF, ls="--", lw=0.9)
            ax.text(len(REPS) - 0.45, ref, f"  {reflab}", fontsize=6.5,
                    color="#555555", va="bottom", ha="right")
        ax.margins(x=0.06)
        FS.panel_label(ax, letter, dx=-0.17, dy=1.07)
    fig.tight_layout(w_pad=1.6)
    FS.save(fig, "Fig5_representation_diagnosis")


if __name__ == "__main__":
    main()
