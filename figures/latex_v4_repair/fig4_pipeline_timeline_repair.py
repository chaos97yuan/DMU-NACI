"""Layout repair of Fig.4 (data provenance / pipeline timeline).

Content identical to the frozen stage9_5 script except layout fixes:
- title moved clear of the year labels;
- T1'/T3' box texts shortened to fit; controls moved to a dedicated dashed box
  (also corrects the T1-in count: 61,208 per the frozen protocol, the frozen
  PNG had a stray "61,211");
- q-hat / gamma annotation boxes wrapped to three lines;
- pink evaluation box widened and its text wrapped so nothing overprints.
"""

from __future__ import annotations

import sys
from pathlib import Path

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]  # release: repo-relative
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as FS  # noqa: E402

INK = "#222222"
C_DEV = "#FDE9D9"
C_SEL = "#DDEEF7"
C_CAL = "#E6F2E6"
C_SAME = "#F2F2F2"
C_CROSS = "#F9D6D5"
C_ARCH = "#EFE6F5"


def tbox(ax, x0, x1, y, h, label, fc, ec, fs=6.6, tc=INK, weight="normal", sub=None):
    ax.add_patch(Rectangle((x0, y), x1 - x0, h, fc=fc, ec=ec, lw=0.8))
    ax.text((x0 + x1) / 2, y + h / 2 + (2.6 if sub else 0), label, ha="center", va="center",
            fontsize=fs, color=tc, fontweight=weight)
    if sub:
        ax.text((x0 + x1) / 2, y + h / 2 - 3.2, sub, ha="center", va="center", fontsize=5.0,
                color="#555555")


def box(ax, x, y, w, h, text, fc="#FFFFFF", ec=INK, fs=6.6, dashed=False, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec=ec,
                                lw=0.9, linestyle="--" if dashed else "-"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, fontweight=weight, linespacing=1.3)


def arrow(ax, p0, p1, color=INK, lw=0.9, dashed=False, rad=0.0, ms=6):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, color=color,
                                 lw=lw, linestyle=(0, (3, 2)) if dashed else "solid",
                                 connectionstyle=f"arc3,rad={rad}", zorder=4))


def main() -> None:
    FS.setup()
    fig, ax = FS.plt.subplots(figsize=(7.4, 5.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ---------------- top lane: year timeline ----------------
    ax.text(0.5, 99, "Temporal roles (frozen boundaries, forecast origins; gradient/early-stop rows by target)",
            fontsize=7.0, fontweight="bold")
    ty, th = 84, 8
    xs = {2021: (2, 18), 2022: (18, 34), 2023: (34, 50), 2024: (50, 82), 2025: (82, 99)}
    for yr, (x0, x1) in xs.items():
        ax.text((x0 + x1) / 2, ty + th + 2.2, str(yr), ha="center", fontsize=7.0,
                fontweight="bold")
    tbox(ax, 2, 18, ty, th, "vintage-model training", C_ARCH, "#9B7EBD", fs=6.0)
    tbox(ax, 18, 34, ty, th, "replay archive E22", C_ARCH, "#9B7EBD", fs=6.0,
         sub="HARD_DF thresholds frozen here")
    tbox(ax, 34, 50, ty, th, "replay archive E23", C_ARCH, "#9B7EBD", fs=6.0)
    b = 50
    for x1, lab, fc, ec, sub in (
            (60.2, "MEAN_GRAD", C_DEV, "#C77C3B", "gradient rows"),
            (60.8, "ES\ntail", C_DEV, "#C77C3B", None),
            (70.0, "SELECT", C_SEL, "#4A90C4", "window + $\\gamma$"),
            (74.0, "CALIB", C_CAL, "#4C9A63", "$\\hat q$ only"),
            (82.0, "SAME", C_SAME, "#888888", "report only")):
        tbox(ax, b, x1, ty, th, lab, fc, ec, fs=5.3, sub=sub)
        b = x1
    ax.text(58, ty - 4.8, "2024 role widths enlarged for legibility (not to scale)",
            fontsize=5.6, color="#555555", ha="center", style="italic")
    tbox(ax, 82, 99, ty, th, "CROSS: 2025 final test", C_CROSS, "#C0504D", fs=5.5,
         weight="bold", sub="never used offline")
    ax.plot([81.7, 81.7], [ty - 1.5, ty + th + 1.5], color="#C0504D", lw=1.0, ls="--")
    ax.text(90.5, ty - 4.8, "freeze manifest written before\nany 2025 target is read",
            fontsize=5.6, color="#C0504D", ha="center")

    ax.text(0.5, 70.5, "2025 excluded from: training, scaling, HPO, early stopping, encoder and window selection, "
            r"$\gamma$ selection, $\hat q$.  Online ACI uses only feedback matured $\geq$ 15 min.",
            fontsize=6.2, color="#8B2B28")

    # ---------------- bottom lane: pool construction ----------------
    ax.text(0.5, 63, "Residual pools for the scale encoder (provenance labels in parentheses)",
            fontsize=7.2, fontweight="bold")

    box(ax, 0.2, 46, 16, 9, "E24_OOF\n4 rolling folds F1-F4\nMay-Aug 2024\n(OOF, 28,418)", fc="#FFF3E8", fs=5.6)
    box(ax, 0.2, 33, 16, 9, "E23_replay\nfrozen mean on 2023\n(REPLAY_OOS, 99,371)", fc="#F3EDF9", fs=5.3)
    box(ax, 0.2, 20, 16, 9, "E22_replay\nfrozen mean on 2022\n(REPLAY_OOS, 102,570)", fc="#F3EDF9", fs=5.3)
    box(ax, 0.2, 4.5, 16, 10.5, "V1 / V3\nvintage models\nM22/M23/M24, next year\n(VINTAGE_OOS)", fc="#F7F7F7", fs=5.2, dashed=True)

    box(ax, 19, 46, 11.5, 9, "T1'\n= E24_OOF\n(OOF, 28,418)", fc="#FFF3E8", fs=6.0)
    box(ax, 19, 33, 11.5, 9, "T2'\n= T1' + E23", fc="#FFF3E8", fs=6.2)
    box(ax, 19, 20, 11.5, 9, "T3' = T2' + E22\n(deployed,\n230,359)", fc="#FFE4C9", fs=6.0, weight="bold")
    box(ax, 18.5, 4.5, 14, 10.5, "controls (not deployed):\nT1-in fitted: 61,208\nT3-match: 28,418", fc="#FFFFFF", fs=5.0, dashed=True)
    for y0 in (50.5, 37.5, 24.5):
        arrow(ax, (16.4, y0), (19, y0))
    arrow(ax, (16.4, 9.5), (18.5, 20), dashed=True)
    ax.text(40, 6.0, "vintage pools do NOT\nenter T1'-T3'", fontsize=5.4, color="#777777",
            ha="center")

    box(ax, 36, 20, 11.5, 20, "GRU $\\sigma$\ntraining\n(Gaussian NLL,\n5 matched seeds)", fc="#DDEEF7", fs=5.8)
    arrow(ax, (30.5, 24.5), (36, 23))
    arrow(ax, (30.5, 37.5), (36, 31))
    arrow(ax, (30.5, 50.5), (36, 38))

    box(ax, 51, 26, 12, 9, "$\\hat q$: normalized\nconformal quantile\n(CALIB span)", fc="#E6F2E6", fs=5.4)
    box(ax, 51, 14, 12, 9, "$\\gamma$: ACI step size\n(SELECT span,\nper seed, frozen)", fc="#DDEEF7", fs=5.4)
    box(ax, 51, 38, 12, 9, "delayed ACI\nfeedback delay\n3 steps = 15 min", fc="#E6F2E6", fs=5.8)

    box(ax, 66.5, 20, 12.5, 22, "2025 prequential\nevaluation\nSAME-year 2024:\nreport-only\ninterval + rolling\ncoverage\n+ conditional states", fc="#F9D6D5", fs=5.4)
    for y in (31, 42.5, 18.5):
        arrow(ax, (63, y), (66.5, 29 + (y - 31) * 0.15))
    arrow(ax, (47.5, 30), (51, 30.5))
    arrow(ax, (47.5, 26), (51, 18.5), rad=0.1)
    arrow(ax, (47.5, 34), (51, 42.5), rad=-0.1)

    box(ax, 81, 26, 18.5, 10, "forecast horizon: target =\norigin + 3 steps (15 min)\nACI feedback delay: 3 steps\n(different concepts)", fc="#FFFFFF", fs=5.4, dashed=True)
    arrow(ax, (79, 31), (81, 31))

    box(ax, 81, 12, 18.5, 10, "pre-registered pool rule\n(SELECT span): feasibility\n$|\\mathrm{Std}(z)-1|\\leq0.35$,\nmin median NLL $\\to$ T3'", fc="#FFFFFF", fs=5.4)
    arrow(ax, (66.5, 24), (81.5, 17), rad=-0.12, dashed=True)

    FS.save(fig, "Fig4_data_pipeline_timeline")


if __name__ == "__main__":
    main()
