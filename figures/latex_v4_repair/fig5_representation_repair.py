"""Layout repair of Fig.5 (representation diagnosis): two-line x tick labels so
they stop overprinting; reference label moved above the dashed line.
Data unchanged: frozen stage-8 confirmatory metrics_per_seed.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # release: repo-relative
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as FS  # noqa: E402

S1C = ROOT / "results" / "stage8_final_architecture" / "05_stage1_confirmatory"

REPS = [("R0_hms", "R0\nmean embedding", FS.C_T3),
        ("R1_raw360", "R1\nraw window", FS.C_T2),
        ("R2_hms_dynamics", "R2\nemb. + dynamics", "#8C564B")]


def main() -> None:
    d = pd.read_csv(S1C / "metrics_per_seed.csv")
    src = d[d["span"] == "cross"]
    src = src[src["mode"].isin([m for m, _, _ in REPS])][["mode", "seed", "Std_z", "NLL"]]

    FS.setup()
    fig, axes = FS.plt.subplots(1, 2, figsize=(7.0, 2.9))
    specs = [("Std_z", "Cross-year Std(z)", "a", 1.0, "Std(z) = 1"),
             ("NLL", "Cross-year Gaussian NLL (nats)", "b", None, None)]
    for ax, (col, ylab, letter, ref, reflab) in zip(axes, specs):
        for i, (m, lab, colr) in enumerate(REPS):
            g = src[src["mode"] == m]
            ax.bar(i, g[col].median(), color=colr, width=0.55)
            FS.seed_dots(ax, i, g[col], colr)
        ax.set_xticks(range(len(REPS)), [lab for _, lab, _ in REPS], fontsize=6.8)
        ax.tick_params(axis="x", pad=2)
        ax.set_ylabel(ylab)
        if ref is not None:
            ax.axhline(ref, color=FS.C_REF, ls="--", lw=0.9)
            ax.text(0.5, ref + 0.15, reflab, fontsize=6.5,
                    color="#555555", va="bottom", ha="center")
        ax.margins(x=0.06)
        FS.panel_label(ax, letter, dx=-0.17, dy=1.07)
    fig.tight_layout(w_pad=1.6)
    FS.save(fig, "Fig5_representation_diagnosis")


if __name__ == "__main__":
    main()
