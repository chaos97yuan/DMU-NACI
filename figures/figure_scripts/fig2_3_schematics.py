"""Stage 9.5 Fig.2 (DMU-NACI framework) and Fig.3 (multi-scale TCN mean branch).

Schematics drawn with matplotlib patches; structure verified against
src/models/multiscale_tcn.py (36,705 params; branches 5/15/60-min, dilations
(1,2)/(1,2)/(1,1), fusion Linear(96->64)+ReLU+Dropout, head Linear(64->1)).
Solid arrows = deployment-time forward path; dashed = training-only signals.
The delayed-feedback loop is drawn explicitly (issue at t, feedback matures at
t+15 min) so no reader can mistake it for use of the current target.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]  # release: repo-relative
sys.path.insert(0, str(ROOT / "figures" / "figure_scripts"))
import figstyle as FS  # noqa: E402

INK = "#222222"
FILL_MU = "#FDE9D9"
FILL_SG = "#DDEEF7"
FILL_CAL = "#E6F2E6"
FILL_IO = "#FFFFFF"
EDGE = {"mu": "#C77C3B", "sg": "#4A90C4", "cal": "#4C9A63"}


def box(ax, x, y, w, h, text, fc=FILL_IO, ec=INK, fs=7.4, lw=1.0, style="round,pad=0.02",
        dashed=False, weight="normal", tc=INK):
    p = FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw,
                       mutation_scale=1, linestyle="--" if dashed else "-")
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight=weight, linespacing=1.25)
    return (x + w / 2, y, x + w / 2, y + h, x, y + h / 2, x + w, y + h / 2)


def arrow(ax, p0, p1, color=INK, lw=1.1, dashed=False, rad=0.0, ms=7, label=None,
          lab_dx=0, lab_dy=0, fs=6.6):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                        linestyle=(0, (4, 2.2)) if dashed else "solid",
                        connectionstyle=f"arc3,rad={rad}", zorder=3)
    ax.add_patch(a)
    if label:
        ax.text((p0[0] + p1[0]) / 2 + lab_dx, (p0[1] + p1[1]) / 2 + lab_dy, label,
                fontsize=fs, color=color, ha="center", va="center")


def fig2_framework():
    FS.setup()
    fig, ax = FS.plt.subplots(figsize=(7.4, 4.5))
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 62)
    ax.axis("off")

    box(ax, 1, 26, 13, 11, "Input window\n$X_t\\in\\mathbb{R}^{72\\times5}$\n(6 h, 5 channels)",
        fc="#F2F2F2", fs=7.0, weight="bold")

    # mean branch (top)
    box(ax, 19, 40, 17, 12, "Multi-scale TCN\n(mean branch, Fig. 3)", fc=FILL_MU,
        ec=EDGE["mu"], weight="bold")
    box(ax, 41, 40, 13.5, 12, "$\\mu_t^{\\mathrm{raw}}$", fc=FILL_MU, ec=EDGE["mu"])
    box(ax, 58.5, 40, 15.5, 12, "physical projection\nclip($\\cdot$, 0, $P_{\\mathrm{rated}}$)\n$\\to\\ \\mu_t$", fc=FILL_MU, ec=EDGE["mu"], fs=7.0)
    arrow(ax, (14, 35), (19, 45), color=EDGE["mu"], rad=-0.2)
    arrow(ax, (36, 46), (41, 46), color=EDGE["mu"])
    arrow(ax, (54.5, 46), (58.5, 46), color=EDGE["mu"])

    # scale branch (bottom)
    box(ax, 19, 10, 17, 12, "GRU encoder\n(hidden 32, softplus)", fc=FILL_SG, ec=EDGE["sg"],
        weight="bold")
    box(ax, 41, 10, 13.5, 12, "$\\sigma_t$\n(kW)", fc=FILL_SG, ec=EDGE["sg"])
    arrow(ax, (14, 29), (19, 17), color=EDGE["sg"], rad=0.2)
    arrow(ax, (36, 16), (41, 16), color=EDGE["sg"])

    # calibration layer
    box(ax, 58.5, 22, 34, 12, "Delayed ACI calibration layer\n"
        "$\\hat q$ from calibration span;  $\\alpha_t\\!\\leftarrow\\!\\mathrm{clip}\\left(\\alpha_{t-1}+\\gamma(\\alpha-\\mathrm{err}_{t-3})\\right)$",
        fc=FILL_CAL, ec=EDGE["cal"], fs=6.8, weight="bold")
    arrow(ax, (70, 40), (70, 34), color=EDGE["mu"])
    arrow(ax, (54.5, 14), (58.5, 24.5), color=EDGE["sg"], rad=-0.15)

    # output interval
    box(ax, 74, 4, 24, 12, "$C_t=[\\,\\mu_t-q_t\\sigma_t,\\ \\mu_t+q_t\\sigma_t\\,]$\n"
        "90% prediction interval", fc="#FFF6E5", ec=EDGE["mu"], weight="bold", fs=7.2)
    arrow(ax, (75, 22), (82, 16), color=EDGE["cal"])

    # delayed feedback loop (routed outside the interval box, right of it)
    ax.plot([88, 88, 103, 103, 93.5], [4, 1.6, 1.6, 29.5, 29.5], color="#777777", lw=1.0)
    arrow(ax, (93.5, 29.5), (92.6, 29.0), color="#777777", lw=1.0, ms=6)
    ax.plot([92.6, 92.6], [29.0, 27.6], color="#777777", lw=1.0)
    arrow(ax, (92.6, 27.6), (92.6, 25.8), color="#777777", lw=1.0, ms=6)
    ax.text(101.4, 15.5, "delayed feedback\n(3 steps = 15 min):\nerror of the forecast\nissued at $t$ updates\n$\\alpha$ only for issues\n$\\geq t+15$ min",
            rotation=90, fontsize=5.8, color="#555555", ha="center", va="center")

    # training-only residual path (dashed)
    box(ax, 24, 26, 21, 8.5, "$e_t = y_t-\\mu_t^{\\mathrm{phys}}$  (residual pools,\nFig. 4 provenance)", fc="#FCFCFC", dashed=True, fs=6.6)
    arrow(ax, (27, 40), (29, 34.5), color="#888888", dashed=True)
    arrow(ax, (33, 26), (27.5, 22), color="#888888", dashed=True)
    ax.text(46.2, 26.0, "training only\n(Gaussian NLL on $e_t/\\hat\\sigma_y$)", fontsize=6.0,
            color="#666666", ha="left", va="center")

    ax.text(1, 59.5, "Deployment-time forward path: solid arrows. Training-only signals: dashed.",
            fontsize=7.0, color=INK)
    ax.text(1, 56.2, "Branches share no parameters; the scale encoder consumes the raw window, not the mean embedding.",
            fontsize=6.6, color="#555555")
    FS.save(fig, "Fig2_DMU_NACI_framework")


def fig3_tcn():
    FS.setup()
    fig, ax = FS.plt.subplots(figsize=(7.4, 3.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")

    box(ax, 0.5, 22, 11, 12, "input\n$[B,72,5]$\n(power, GHI, DHI,\nT, RH)", fc="#F2F2F2", fs=6.6, weight="bold")
    box(ax, 13.5, 22, 7.5, 12, "transpose\n$[B,5,72]$", fs=6.8)
    arrow(ax, (11.5, 28), (13.5, 28))

    branches = [
        (40.5, "5-min branch", "72 steps", "causal TCN\n2 layers x 32 ch\ndilations (1, 2)", FILL_MU),
        (25.0, "15-min branch", "AvgPool(3) $\\to$ 24 steps", "causal TCN\n2 layers x 32 ch\ndilations (1, 2)", FILL_MU),
        (9.5, "60-min branch", "AvgPool(12) $\\to$ 6 steps", "causal TCN\n2 layers x 32 ch\ndilations (1, 1)", FILL_MU),
    ]
    for y, name, pool, txt, fc in branches:
        box(ax, 23.5, y, 12, 8.5, pool + (" (input)" if "AvgPool" not in pool else ""), fs=6.0, fc="#FAFAFA")
        arrow(ax, (21, 28), (23.5, y + 4.25), rad=0.0)
        box(ax, 38, y, 15.5, 8.5, txt, fc=fc, ec=EDGE["mu"], fs=6.4)
        arrow(ax, (35.5, y + 4.25), (38, y + 4.25))
        ax.text(45.7, y + 10.6, name + f"  $\\to h\\in\\mathbb{{R}}^{{32}}$ (last step)", fontsize=6.6,
                ha="center", color=EDGE["mu"])

    box(ax, 56.5, 22, 10, 12, "concat\n$\\mathbb{R}^{96}$", fs=7.2, fc=FILL_MU, ec=EDGE["mu"])
    for y, yy in zip([40.5, 25.0, 9.5], [29.5, 28.0, 26.5]):
        arrow(ax, (53.5, y + 4.25), (56.5, yy))
    box(ax, 69.5, 22, 14, 12, "Linear 96$\\to$64\nReLU, Dropout 0.1", fs=6.8, fc=FILL_MU, ec=EDGE["mu"])
    arrow(ax, (66.5, 28), (69.5, 28))
    box(ax, 86, 31.5, 13, 10, "mean head\nLinear 64$\\to$1", fs=6.8, fc=FILL_MU, ec=EDGE["mu"])
    arrow(ax, (83.5, 29.5), (86, 35), rad=-0.1)
    box(ax, 86, 14.5, 13, 10, "$\\mu^{\\mathrm{raw}}\\!\\to\\!$ clip(0, 5.2)\n$\\mu^{\\mathrm{phys}}$ (kW)", fs=6.8, fc="#FFF6E5", ec=EDGE["mu"], weight="bold")
    arrow(ax, (92.5, 31.5), (92.5, 24.5))
    ax.text(50, 3.2, "36,705 parameters; trained with AdamW (lr $10^{-3}$, wd $10^{-4}$, batch 512), "
            "early stopping on RMSE (kW) inside the development span only.", fontsize=6.4,
            ha="center", color="#555555")
    FS.save(fig, "Fig3_multiscale_TCN")


if __name__ == "__main__":
    fig2_framework()
    fig3_tcn()
