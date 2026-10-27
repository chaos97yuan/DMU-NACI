"""Stage 9.5 Fig.8: calibration ablation (A0/A1/A2/A3) + dynamic rolling coverage
on the 2025 cross-year span (Site 18).

(a) PICP (0.90 target line)  (b) interval score  (c) PINAW   -- bars = median
of 5 matched seeds, dots = individual seeds (frozen replay, verified <1e-6)
(d) rolling 1-day PICP        (e) rolling 3-day PICP          -- seed-median
daily coverage; A1/A3/CQR in main, A2 thin gray (full-A2 version supplementary)
(f) RCE1D / RCE3D grouped bars for A0/A1/A2/A3/CQR
No new metric definitions: PICP/PINAW/IS/RCE1D/RCE3D exactly as
experiments/rigorous_v2/common_v2.py::interval_metrics.
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
ROLL_MAIN = ["A1", "A3", "CQR"]


def load_all() -> dict[int, pd.DataFrame]:
    return {s: pd.read_parquet(SRC / f"intervals_cross_seed{s}.parquet") for s in C.SEEDS}


def per_seed_metrics(iv: dict) -> pd.DataFrame:
    rows = []
    for s, g in iv.items():
        o = pd.DatetimeIndex(g["origin"])
        r = {"seed": s}
        for m in METHODS:
            im = C.interval_metrics(g["y_kw"].to_numpy(float),
                                    g[f"{m}_lo"].to_numpy(float),
                                    g[f"{m}_up"].to_numpy(float), o)
            r.update({f"{m}_{k}": im[k] for k in ("PICP", "PINAW", "IS", "RCE1D", "RCE3D")})
        rows.append(r)
    return pd.DataFrame(rows)


def rolling_daily(iv: dict) -> pd.DataFrame:
    """Seed-median daily coverage + 3-day rolling mean, per method."""
    frames = {m: [] for m in METHODS}
    for s, g in iv.items():
        o = pd.DatetimeIndex(g["origin"])
        for m in METHODS:
            cov = ((g["y_kw"] >= g[f"{m}_lo"]) & (g["y_kw"] <= g[f"{m}_up"])).astype(float)
            d = pd.Series(cov.to_numpy(), index=o.normalize()).groupby(level=0).mean()
            frames[m].append(d)
    out = {}
    for m in METHODS:
        df = pd.concat(frames[m], axis=1)
        med = df.median(axis=1)
        out[m] = pd.DataFrame({"day": med.index, f"picp1d": med.to_numpy(),
                               "picp3d": med.rolling(3, min_periods=1).mean().to_numpy()})
    return out


def main() -> None:
    iv = load_all()
    pm = per_seed_metrics(iv)
    pm.to_csv(SRC / "Fig8_per_seed_metrics.csv", index=False)
    rd = rolling_daily(iv)
    roll = rd["A1"][["day"]].copy()
    for m in METHODS:
        roll[f"{m}_picp1d"] = rd[m]["picp1d"].to_numpy()
        roll[f"{m}_picp3d"] = rd[m]["picp3d"].to_numpy()
    roll.to_csv(SRC / "Fig8_rolling_coverage_source.csv", index=False)

    med = pm.drop(columns="seed").median()

    FS.setup()
    fig, axes = FS.plt.subplots(2, 3, figsize=(7.2, 4.6))
    colors = {"A0": FS.C_A0, "A1": FS.C_A1, "A2": FS.C_A2, "A3": FS.C_A3, "CQR": FS.C_CQR}
    short = {"A0": "A0", "A1": "A1", "A2": "A2", "A3": "A3\n(proposed)", "CQR": "CQR"}

    # (a) PICP
    ax = axes[0, 0]
    vals = [med[f"{m}_PICP"] for m in METHODS]
    ax.bar(range(5), vals, color=[colors[m] for m in METHODS], width=0.62)
    for i, m in enumerate(METHODS):
        FS.seed_dots(ax, i, pm[f"{m}_PICP"], colors[m])
    ax.axhline(0.90, color=FS.C_REF, ls="--", lw=0.8)
    ax.text(4.45, 0.902, "0.90 target", fontsize=6.5, color="#555555", va="bottom", ha="right")
    ax.set_xticks(range(5), [short[m] for m in METHODS])
    ax.set_ylabel("PICP")
    ax.set_ylim(0.84, 0.97)

    # (b) IS
    ax = axes[0, 1]
    vals = [med[f"{m}_IS"] for m in METHODS]
    ax.bar(range(5), vals, color=[colors[m] for m in METHODS], width=0.62)
    for i, m in enumerate(METHODS):
        FS.seed_dots(ax, i, pm[f"{m}_IS"], colors[m])
    ax.set_xticks(range(5), [short[m] for m in METHODS])
    ax.set_ylabel("Interval score (kW)")

    # (c) PINAW
    ax = axes[0, 2]
    vals = [med[f"{m}_PINAW"] for m in METHODS]
    ax.bar(range(5), vals, color=[colors[m] for m in METHODS], width=0.62)
    for i, m in enumerate(METHODS):
        FS.seed_dots(ax, i, pm[f"{m}_PINAW"], colors[m])
    ax.set_xticks(range(5), [short[m] for m in METHODS])
    ax.set_ylabel("PINAW")

    # (d)/(e) rolling coverage
    for k, (key, ttl) in enumerate((("picp1d", "1-day rolling PICP"),
                                    ("picp3d", "3-day rolling PICP"))):
        ax = axes[1, k]
        ax.axhline(0.90, color=FS.C_REF, ls="--", lw=0.9, zorder=1)
        for m in METHODS:
            lw, a, z = (0.7, 0.5, 2) if m == "A2" else (1.0, 1.0, 3)
            if m == "A2":
                continue  # A2 shown in the supplementary twin; keep main readable
            ax.plot(rd[m]["day"], rd[m][key], color=colors[m], lw=lw, alpha=a, zorder=z,
                    label=FS.METHOD_SHORT[m])
        ax.set_ylim(0.70, 1.00)
        ax.set_ylabel(ttl)
        import matplotlib.dates as mdates
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        if k == 0:
            ax.legend(loc="lower left", ncol=1, fontsize=6.5)

    # (f) RCE1D/RCE3D
    ax = axes[1, 2]
    x = np.arange(5)
    w = 0.38
    ax.bar(x - w / 2, [med[f"{m}_RCE1D"] for m in METHODS], width=w,
           color=[colors[m] for m in METHODS], alpha=0.95, label="RCE1D")
    ax.bar(x + w / 2, [med[f"{m}_RCE3D"] for m in METHODS], width=w,
           color=[colors[m] for m in METHODS], alpha=0.45, label="RCE3D")
    for i, m in enumerate(METHODS):
        FS.seed_dots(ax, x[i] - w / 2, pm[f"{m}_RCE1D"], colors[m], size=9)
    ax.set_xticks(x, [short[m] for m in METHODS])
    ax.set_ylabel("Rolling coverage error")
    ax.legend(fontsize=6.2, loc="upper right", bbox_to_anchor=(1.02, 1.34), ncol=1)

    for ax, letter in zip(axes.ravel(), "abcdef"):
        FS.panel_label(ax, letter, dx=-0.2, dy=1.07)
    fig.tight_layout(w_pad=1.4, h_pad=1.1)
    FS.save(fig, "Fig8_ablation_dynamic_coverage")

    # supplementary twin with A2 included in rolling panels
    fig, axes = FS.plt.subplots(1, 2, figsize=(7.2, 2.5))
    for k, key in enumerate(("picp1d", "picp3d")):
        ax = axes[k]
        ax.axhline(0.90, color=FS.C_REF, ls="--", lw=0.9, zorder=1)
        for m in METHODS:
            ax.plot(rd[m]["day"], rd[m][key], color=colors[m], lw=1.0, zorder=3,
                    label=FS.METHOD_SHORT[m])
        ax.set_ylim(0.70, 1.00)
        ax.set_ylabel(f"{key[1:-2].replace('picp', '')}-day rolling PICP".capitalize())
        import matplotlib.dates as mdates
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.legend(loc="lower left", fontsize=6.5)
        FS.panel_label(ax, "ab"[k], dx=-0.18, dy=1.08)
    fig.tight_layout()
    FS.save_supp(fig, "FigS_rolling_coverage_with_A2")


if __name__ == "__main__":
    main()
