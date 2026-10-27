"""Layout repair of Fig.9 (conditional reliability): panel-(a) legend moved
above the axes (it used to sit on the bars) and panel-(d) annotation lifted
clear of the first bar. Data pipeline identical to the frozen Stage-9.6 script
(corrected frozen state masks; frozen replay parquets)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # release: repo-relative
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
import figstyle as FS  # noqa: E402
from experiments.rigorous_v2 import common_v2 as C  # noqa: E402

SRC = ROOT / "figures" / "figure_source_data"
METHODS = ["A0", "A1", "A2", "A3", "CQR"]
COLORS = {"A0": FS.C_A0, "A1": FS.C_A1, "A2": FS.C_A2, "A3": FS.C_A3, "CQR": FS.C_CQR}


def main() -> None:
    y25 = C.load_year_windows(2025)
    ng = np.asarray(y25["N_group"]).astype(str)
    hd = np.asarray(y25["hard_DF"]).astype(bool)
    # FROZEN state convention: NORMAL = ~HARD_DF & N_group != HIGH_N; HIGH_N = N_group == HIGH_N
    states = {"NORMAL": ~hd & (ng != "HIGH_N"), "HIGH_N": ng == "HIGH_N", "HARD_DF": hd}
    dl = y25["x_raw"][:, -1, 1] > C.DAYLIGHT_GHI_THRESHOLD

    rows, hard_rows = [], []
    for s in C.SEEDS:
        g = pd.read_parquet(SRC / f"intervals_cross_seed{s}.parquet")
        y = g["y_kw"].to_numpy(float)
        r = {"seed": s}
        for m in METHODS:
            lo, up = g[f"{m}_lo"].to_numpy(float), g[f"{m}_up"].to_numpy(float)
            cov = (y >= lo) & (y <= up)
            for st, mask in list(states.items()) + [("daylight", dl)]:
                r[f"{m}_PICP_{st}"] = float(cov[mask].mean())
            h = states["HARD_DF"]
            rows_h = {"seed": s, "method": m,
                      "HARD_PICP": float(cov[h].mean()),
                      "HARD_width_kw": float(np.maximum(up[h] - lo[h], 0).mean()),
                      "HARD_IS": float(C.interval_score(y[h], lo[h], up[h]).mean())}
            hard_rows.append(rows_h)
        rows.append(r)
    per_seed = pd.DataFrame(rows)
    hard = pd.DataFrame(hard_rows)
    med = per_seed.drop(columns="seed").median()

    FS.setup()
    fig, axes = FS.plt.subplots(2, 2, figsize=(7.2, 5.0))

    # (a) conditional PICP for A0-A3 — legend above the axes
    ax = axes[0, 0]
    sts = ["NORMAL", "HIGH_N", "HARD_DF", "daylight"]
    x = np.arange(len(sts))
    w = 0.2
    for i, m in enumerate(["A0", "A1", "A2", "A3"]):
        vals = [med[f"{m}_PICP_{st}"] for st in sts]
        ax.bar(x + (i - 1.5) * w, vals, width=w * 0.92, color=COLORS[m], label=FS.METHOD_SHORT[m])
        for j, st in enumerate(sts):
            FS.seed_dots(ax, x[j] + (i - 1.5) * w, per_seed[f"{m}_PICP_{st}"], COLORS[m], size=7)
    ax.axhline(0.90, color=FS.C_REF, ls="--", lw=0.9)
    ax.set_xticks(x, ["NORMAL", "HIGH_N", "HARD_DF", "Daylight"], fontsize=7)
    ax.set_ylabel("Conditional PICP")
    ax.set_ylim(0.30, 1.02)
    ax.legend(fontsize=5.9, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              columnspacing=0.7, handletextpad=0.3)

    # (b) HARD_DF PICP vs width
    ax = axes[0, 1]
    for m in METHODS:
        g = hard[hard["method"] == m]
        ax.scatter(g["HARD_width_kw"], g["HARD_PICP"], s=14, color=COLORS[m],
                   edgecolors="white", linewidths=0.5, zorder=4)
        ax.scatter([g["HARD_width_kw"].median()], [g["HARD_PICP"].median()], s=52,
                   color=COLORS[m], marker="*", zorder=5, label=FS.METHOD_SHORT[m])
    ax.axhline(0.90, color=FS.C_REF, ls="--", lw=0.9)
    ax.set_xlabel("Mean width within HARD_DF (kW)")
    ax.set_ylabel("PICP within HARD_DF")
    ax.set_ylim(0.30, 1.0)
    ax.legend(fontsize=6.0, loc="lower right", handletextpad=0.1, borderaxespad=0.2)

    # (c) HARD_DF PICP vs IS
    ax = axes[1, 0]
    for m in METHODS:
        g = hard[hard["method"] == m]
        ax.scatter(g["HARD_IS"], g["HARD_PICP"], s=14, color=COLORS[m],
                   edgecolors="white", linewidths=0.5, zorder=4)
        ax.scatter([g["HARD_IS"].median()], [g["HARD_PICP"].median()], s=52,
                   color=COLORS[m], marker="*", zorder=5)
    ax.axhline(0.90, color=FS.C_REF, ls="--", lw=0.9)
    ax.set_xlabel("Interval score within HARD_DF (kW)")
    ax.set_ylabel("PICP within HARD_DF")
    ax.set_ylim(0.30, 1.0)
    ax.text(0.02, 0.03, "dots = seeds;  stars = 5-seed median",
            transform=ax.transAxes, fontsize=6.0, color="#555555")

    # (d) A3 seed stability — annotation lifted above the bars
    ax = axes[1, 1]
    pm = pd.read_csv(SRC / "Fig8_per_seed_metrics.csv")
    mets = [("PICP", "PICP", 1.0), ("IS", "IS (kW)", None), ("HARD_PICP", "HARD_DF PICP", 0.90)]
    xpos = np.arange(len(mets))
    for i, (col, lab, ref) in enumerate(mets):
        series = (hard.loc[hard["method"] == "A3", "HARD_PICP"].to_numpy()
                  if col == "HARD_PICP" else pm[f"A3_{col}"].to_numpy())
        ax.bar(i, float(np.median(series)), color=FS.C_A3, width=0.5)
        FS.seed_dots(ax, i, series, FS.C_A3)
    ax.set_xticks(xpos, [lab for _, lab, _ in mets], fontsize=7)
    ax.set_ylabel("Proposed method (A3) value")
    ax.set_ylim(0, 1.32)
    ax.text(0.03, 0.97, "bars = 5-seed median\ndots = individual seeds\n(seeds: 42, 123, 2026,\n3407, 8888)",
            transform=ax.transAxes, fontsize=6.3, va="top")

    for ax, letter in zip(axes.ravel(), "abcd"):
        FS.panel_label(ax, letter, dx=-0.15, dy=1.06)
    fig.tight_layout(w_pad=1.4, h_pad=1.2)
    FS.save(fig, "Fig9_conditional_HARD_DF_robustness")


if __name__ == "__main__":
    main()
