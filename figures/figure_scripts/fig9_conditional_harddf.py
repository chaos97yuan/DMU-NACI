"""Stage 9.5 Fig.9: conditional reliability, HARD_DF trade-off, seed robustness
(Site 18, 2025 cross-year span).

(a) PICP by state (NORMAL / HIGH_N / HARD_DF / daylight) for A0/A1/A2/A3
    (v3 Table 5 values; frozen replay <1e-6)
(b) HARD_DF reliability-sharpness plane: PICP vs mean width (kW),
    A0/A1/A2/A3/CQR, seed dots + method medians
(c) HARD_DF PICP vs interval score within HARD_DF (kW) -- the A3-vs-CQR
    trade-off view (coverage higher, within-HARD_DF IS slightly worse)
(d) proposed-method (A3) seed stability: per-seed PICP / IS / HARD_DF PICP
    against the 5-seed medians (descriptive n=5)
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
from experiments.rigorous_v2 import common_v2 as C  # noqa: E402

SRC = ROOT / "figures" / "figure_source_data"
METHODS = ["A0", "A1", "A2", "A3", "CQR"]
COLORS = {"A0": FS.C_A0, "A1": FS.C_A1, "A2": FS.C_A2, "A3": FS.C_A3, "CQR": FS.C_CQR}


def main() -> None:
    y25 = C.load_year_windows(2025)
    ng = np.asarray(y25["N_group"]).astype(str)
    hd = np.asarray(y25["hard_DF"]).astype(bool)
    # FROZEN state convention (stage8 14_state_definition_final_audit.md /
    # diagnose_hetero_representation.state_split): NORMAL = ~HARD_DF & N_group != HIGH_N;
    # HIGH_N = N_group == HIGH_N. (common_v2.state_split's ng=="N" variant is a
    # transcription bug fixed in Stage 9.6; HARD_DF/daylight are unaffected.)
    states = {"NORMAL": ~hd & (ng != "HIGH_N"), "HIGH_N": ng == "HIGH_N", "HARD_DF": hd}
    dl = y25["x_raw"][:, -1, 1] > C.DAYLIGHT_GHI_THRESHOLD

    rows, hard_rows = [], []
    for s in C.SEEDS:
        g = pd.read_parquet(SRC / f"intervals_cross_seed{s}.parquet")
        y = g["y_kw"].to_numpy(float)
        o = pd.DatetimeIndex(g["origin"])
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
            r.update({f"{m}_HARD_PICP": rows_h["HARD_PICP"],
                      f"{m}_HARD_IS": rows_h["HARD_IS"]})
        rows.append(r)
    per_seed = pd.DataFrame(rows)
    hard = pd.DataFrame(hard_rows)
    per_seed.to_csv(SRC / "Fig9_conditional_per_seed.csv", index=False)
    hard.to_csv(SRC / "Fig9_HARD_DF_tradeoff_source.csv", index=False)
    med = per_seed.drop(columns="seed").median()

    FS.setup()
    fig, axes = FS.plt.subplots(2, 2, figsize=(7.2, 5.0))

    # (a) conditional PICP for A0-A3
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
    ax.legend(fontsize=5.9, ncol=4, loc="lower center", columnspacing=0.7,
              handletextpad=0.3)

    # (b) HARD_DF PICP vs width
    ax = axes[0, 1]
    for m in METHODS:
        g = hard[hard["method"] == m]
        ax.scatter(g["HARD_width_kw"], g["HARD_PICP"], s=14, color=COLORS[m],
                   edgecolors="white", linewidths=0.5, zorder=4)
        ax.scatter([g["HARD_width_kw"].median()], [g["HARD_PICP"].median()], s=52,
                   color=COLORS[m], marker="*", zorder=5,
                   label=FS.METHOD_SHORT[m])
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

    # (d) A3 seed stability
    ax = axes[1, 1]
    pm = pd.read_csv(SRC / "Fig8_per_seed_metrics.csv")
    mets = [("PICP", "PICP", 1.0), ("IS", "IS (kW)", None), ("HARD_PICP", "HARD_DF PICP", 0.90)]
    xpos = np.arange(len(mets))
    for i, (col, lab, ref) in enumerate(mets):
        series = per_seed["A3_HARD_PICP"] if col == "HARD_PICP" else pm[f"A3_{col}"]
        ax.bar(i, series.median(), color=FS.C_A3, width=0.5)
        FS.seed_dots(ax, i, series, FS.C_A3)
    ax.set_xticks(xpos, [lab for _, lab, _ in mets], fontsize=7)
    ax.set_ylabel("Proposed method (A3) value")
    ax.text(0.03, 0.95, "bars = 5-seed median\ndots = individual seeds\n(seeds: 42, 123, 2026,\n3407, 8888)",
            transform=ax.transAxes, fontsize=6.3, va="top")

    for ax, letter in zip(axes.ravel(), "abcd"):
        FS.panel_label(ax, letter, dx=-0.15, dy=1.06)
    fig.tight_layout(w_pad=1.4, h_pad=1.2)
    FS.save(fig, "Fig9_conditional_HARD_DF_robustness")

    print("A3 HARD PICP median:", per_seed["A3_HARD_PICP"].median().round(4),
          "| A0:", per_seed["A0_HARD_PICP"].median().round(4),
          "| daylight A3:", per_seed["A3_PICP_daylight"].median().round(4),
          "| HIGH_N A3:", per_seed["A3_PICP_HIGH_N"].median().round(4))


if __name__ == "__main__":
    main()
