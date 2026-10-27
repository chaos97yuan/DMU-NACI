"""Layout repair of Fig.8 (ablation + dynamic coverage): month ticks every 2
months (12 labels overprinted in the frozen render), rolling-panel legend
moved above the axes so series no longer cross it. Data pipeline identical to
the frozen script (frozen replay parquets + common_v2.interval_metrics)."""

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


def rolling_daily(iv: dict) -> dict:
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
        out[m] = pd.DataFrame({"day": med.index, "picp1d": med.to_numpy(),
                               "picp3d": med.rolling(3, min_periods=1).mean().to_numpy()})
    return out


def main() -> None:
    iv = load_all()
    pm = per_seed_metrics(iv)
    rd = rolling_daily(iv)

    med = pm.drop(columns="seed").median()

    FS.setup()
    fig, axes = FS.plt.subplots(2, 3, figsize=(7.2, 4.6))
    colors = {"A0": FS.C_A0, "A1": FS.C_A1, "A2": FS.C_A2, "A3": FS.C_A3, "CQR": FS.C_CQR}
    short = {"A0": "A0", "A1": "A1", "A2": "A2", "A3": "A3\n(proposed)", "CQR": "CQR"}

    for k, col in enumerate(("PICP", "IS", "PINAW")):
        ax = axes[0, k]
        vals = [med[f"{m}_{col}"] for m in METHODS]
        ax.bar(range(5), vals, color=[colors[m] for m in METHODS], width=0.62)
        for i, m in enumerate(METHODS):
            FS.seed_dots(ax, i, pm[f"{m}_{col}"], colors[m])
        if col == "PICP":
            ax.axhline(0.90, color=FS.C_REF, ls="--", lw=0.8)
            ax.text(4.45, 0.902, "0.90 target", fontsize=6.5, color="#555555", va="bottom", ha="right")
            ax.set_ylim(0.84, 0.97)
        ax.set_xticks(range(5), [short[m] for m in METHODS])
        ax.set_ylabel({"PICP": "PICP", "IS": "Interval score (kW)", "PINAW": "PINAW"}[col])

    # (d)/(e) rolling coverage — ticks every 2 months, legend above the axes
    import matplotlib.dates as mdates
    for k, (key, ttl) in enumerate((("picp1d", "1-day rolling PICP"),
                                    ("picp3d", "3-day rolling PICP"))):
        ax = axes[1, k]
        ax.axhline(0.90, color=FS.C_REF, ls="--", lw=0.9, zorder=1)
        for m in METHODS:
            if m == "A2":
                continue  # A2 shown in the supplementary twin
            ax.plot(rd[m]["day"], rd[m][key], color=colors[m], lw=1.0, zorder=3,
                    label=FS.METHOD_SHORT[m])
        ax.set_ylim(0.70, 1.00)
        ax.set_ylabel(ttl)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.tick_params(axis="x", labelsize=6.8)
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=4, fontsize=6.0,
                  columnspacing=0.9, handletextpad=0.35)

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

    # supplementary twin with A2 included
    fig, axes = FS.plt.subplots(1, 2, figsize=(7.2, 2.7))
    for k, key in enumerate(("picp1d", "picp3d")):
        ax = axes[k]
        ax.axhline(0.90, color=FS.C_REF, ls="--", lw=0.9, zorder=1)
        for m in METHODS:
            ax.plot(rd[m]["day"], rd[m][key], color=colors[m], lw=1.0, zorder=3,
                    label=FS.METHOD_SHORT[m])
        ax.set_ylim(0.70, 1.00)
        ax.set_ylabel(f"{'1' if key == 'picp1d' else '3'}-day rolling PICP")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.tick_params(axis="x", labelsize=6.8)
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.04), ncol=5, fontsize=6.0,
                  columnspacing=0.8, handletextpad=0.3)
        FS.panel_label(ax, "ab"[k], dx=-0.16, dy=1.16)
    fig.tight_layout()
    FS.save_supp(fig, "FigS_rolling_coverage_with_A2")


if __name__ == "__main__":
    main()
