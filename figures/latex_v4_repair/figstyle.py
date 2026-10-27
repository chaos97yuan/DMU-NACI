"""Stage 9.5 shared publication style for all v4 figures.

Rules (Stage-9.5 brief, section 15): white background, English labels, >=300 dpi
(saved at 600), unified fonts, subplot letters (a)/(b)/(c), color-blind-safe
palette distinguishable in grayscale, no 3D/gradients/shadows/radar/rainbow.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parents[1] / "output_v4"
SRC_DIR = Path(__file__).resolve().parents[1] / "figure_source_data"
SUP_DIR = Path(__file__).resolve().parents[1] / "supplementary"

# Okabe-Ito color-blind-safe palette (+ near-black); all distinguishable in grayscale
C_OBS = "#000000"      # observed power
C_MU = "#000000"       # point forecast (line style separates it from y)
C_A0 = "#E69F00"       # orange
C_A1 = "#56B4E9"       # sky blue
C_A2 = "#009E73"       # bluish green
C_A3 = "#D55E00"       # vermillion  (proposed method)
C_CQR = "#CC79A7"      # reddish purple
C_T1 = "#E69F00"
C_T2 = "#56B4E9"
C_T3 = "#D55E00"
C_TIN = "#009E73"
C_V1 = "#999999"
C_V3 = "#0072B2"
C_REF = "#888888"      # reference lines

METHOD_LABELS = {
    "A0": "A0  absolute fixed conformal",
    "A1": "A1  absolute ACI",
    "A2": "A2  normalized fixed conformal",
    "A3": "A3  normalized delayed ACI (DMU-NACI)",
    "CQR": "CQR",
}
METHOD_SHORT = {"A0": "A0", "A1": "A1", "A2": "A2", "A3": "A3 (proposed)", "CQR": "CQR"}

DPI = 600


def setup(width_in: float = 7.0, height_in: float | None = None) -> plt.Figure:
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": DPI,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "axes.grid": False, "grid.linewidth": 0.4, "grid.alpha": 0.35,
        "axes.facecolor": "white", "figure.facecolor": "white",
        "savefig.facecolor": "white", "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig = plt.figure(figsize=(width_in, height_in) if height_in else (width_in, 4.2))
    return fig


def panel_label(ax, letter: str, dx: float = -0.14, dy: float = 1.04):
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")


def save(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for dpi, tag in ((600, ""), (300, "_300dpi")):
        fig.savefig(FIG_DIR / f"{name}{tag}.png", dpi=dpi)
    plt.close(fig)
    print(f"saved figures/{name}.png (+300dpi)")


def save_supp(fig, name: str):
    SUP_DIR.mkdir(parents=True, exist_ok=True)
    for dpi, tag in ((600, ""), (300, "_300dpi")):
        fig.savefig(SUP_DIR / f"{name}{tag}.png", dpi=dpi)
    plt.close(fig)
    print(f"saved supplementary/{name}.png")


def seed_dots(ax, x: float, vals, color: str, jitter: float = 0.0, size: int = 12):
    """Overlay individual seed points on a bar (bar = median)."""
    import numpy as np
    vals = np.asarray(vals, float)
    ax.scatter([x + jitter] * len(vals), vals, s=size, color="white",
               edgecolors=color, linewidths=0.7, zorder=5)
