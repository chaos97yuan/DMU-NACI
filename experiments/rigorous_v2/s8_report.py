"""rigorous_v2 STEP 8: consolidated report generator.

Reads every stage's artifacts and produces the final audit tables:
  - data-flow audit (E22/E23/E24 provenance)
  - experiment matrix metrics (selection / same-year / cross-year / daylight)
  - paired bootstrap summary
  - case decision (A-E) against the pre-registered rules
Outputs -> results/rigorous_v2/REPORT.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.rigorous_v2 import common_v2 as C  # noqa: E402

R = C.V2


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    return f"{v:.{nd}f}"


def main() -> None:
    decision = json.loads((R / "s6_gru_select" / "selection_decision.json").read_text())
    metrics = pd.read_csv(R / "s7_final_eval" / "metrics_per_seed.csv")
    monthly = pd.read_csv(R / "s7_final_eval" / "monthly_cross_year.csv")
    boot = pd.read_csv(R / "s7_final_eval" / "bootstrap_paired.csv")
    pools = pd.read_csv(R / "s6_gru_select" / "pool_manifest.csv")
    mean_metrics = pd.read_csv(R / "s2_final_mean" / "mean_metrics.csv")
    replay_manifest = json.loads((R / "s4_replay" / "residual_manifest.json").read_text())
    oof_manifest = json.loads((R / "s3_e24_oof" / "residual_manifest.json").read_text())
    vintage_manifest = json.loads((R / "s5_vintage" / "residual_manifest.json").read_text())
    gamma = json.loads((R / "s6_gru_select" / "gamma_selection.json").read_text())

    L: list[str] = []
    A = L.append
    A("# rigorous_v2 -- provenance-controlled residual-window experiment\n")
    A(f"Winner (pre-registered SELECT rule): **{decision['winner']}**\n")

    A("## 1. Data-flow audit\n")
    A("| Pool | Generator | Training data | Forecast span | Provenance | n |")
    A("|---|---|---|---|---|---|")
    folds = {f["fold"]: f for f in oof_manifest["folds"]}
    for f in oof_manifest["fold_audit"]:
        fo = folds[f["fold"]]["fc"]
        A(f"| E24_OOF {f['fold']} | per-fold TCN refit | 2024-01-01..{fo[0]} | "
          f"{fo[0]}..{fo[1]} | OOF | {f['n_fc']} |")
    for y in replay_manifest["per_year"]:
        A(f"| E{str(y['year'])[2:]}_replay | frozen final mean (2024-trained) | 2024 grad+ES | "
          f"{y['year']} full year | REPLAY_OOS | {y['n']} |")
    for v in vintage_manifest["vintages"]:
        A(f"| E{str(v['fc_year'])[2:]}_vintage | vintage {v['tag']} | "
          f"{v['train_years'][0]}-{v['train_years'][-1]} | {v['fc_year']} full year | "
          f"VINTAGE_OOS | {v['n_fc']} |")
    A("")

    A("## 2. Mean branch (retrained, internal early stop)\n")
    med = mean_metrics.groupby("span")[["RMSE", "MAE"]].median()
    A("| Span | RMSE (kW) | MAE (kW) |")
    A("|---|---|---|")
    for s in ("grad", "es", "select", "calib", "same"):
        if s in med.index:
            A(f"| {s} | {med.loc[s, 'RMSE']:.4f} | {med.loc[s, 'MAE']:.4f} |")
    A("")

    A("## 3. Pool composition\n")
    A(pools[["variant", "n", "daylight_share", "abs_e_mean", "abs_e_median", "abs_e_q95"]]
      .to_markdown(index=False))
    A("")

    A("## 4. Experiment matrix (median over 5 seeds)\n")
    for span in ("select", "same", "cross"):
        sub = metrics[metrics["span"] == span]
        piv = sub.groupby("variant").agg(
            Std_z=("Std_z", "median"), NLL=("NLL", "median"),
            Spearman=("Spearman", "median"), PICP=("A2_PICP", "median"),
            PINAW=("A2_PINAW", "median"), IS=("A2_IS", "median"),
            RCE1D=("A2_RCE1D", "median"), HARDDF=("A2_PICP_HARD_DF", "median"),
            daylight_PICP=("A2_PICP_daylight", "median")).reset_index()
        A(f"### span = {span}\n")
        A(piv.to_markdown(index=False, floatfmt=".3f"))
        A("")
    A("### Daylight-only (cross year)\n")
    dl = metrics[metrics["span"] == "cross"].groupby("variant").agg(
        daylight_IS=("A2_IS_daylight", "median"), daylight_PICP=("A2_PICP_daylight", "median"))
    A(dl.to_markdown(floatfmt=".3f"))
    A("")

    A("## 5. A0-A3 ablation (winner pipeline)\n")
    ab = metrics[metrics["variant"].str.contains(":") & (metrics["span"] == "cross")]
    if len(ab):
        piv = ab.groupby("variant").agg(PICP=("A3_PICP", "median"), IS=("A3_IS", "median"),
                                        RCE1D=("A3_RCE1D", "median"),
                                        HARDDF=("A3_PICP_HARD_DF", "median")).reset_index()
        cols = {}
        for v in ab["variant"].unique():
            row = ab[ab["variant"] == v].median(numeric_only=True)
            cols[v] = {k: row.get(f"{v.split(':')[1]}_{k}") for k in ("PICP", "PINAW", "IS",
                                                                       "RCE1D", "PICP_HARD_DF")}
        A(pd.DataFrame(cols).T.to_markdown(floatfmt=".3f"))
    A("")

    A("## 6. Monthly / seasonal (cross year, median)\n")
    mm = monthly.groupby(["variant", "month"])[["Std_z", "Spearman", "PICP", "IS"]].median().reset_index()
    A(mm.to_markdown(index=False, floatfmt=".3f"))
    A("")

    A("## 7. Paired day-block bootstrap (2025, 2000 resamples)\n")
    A(boot.to_markdown(index=False, floatfmt=".4f"))
    A("")

    A("## 8. Gamma (frozen from SELECT)\n")
    A("| seed | gamma_A3 | gamma_A1 |")
    A("|---|---|---|")
    for s, g in gamma.items():
        A(f"| {s} | {g['gamma_A3']} | {g['gamma_A1']} |")
    A("")

    # ---- pre-registered case decision ----
    def medv(variant, span, col):
        sub = metrics[(metrics["variant"] == variant) & (metrics["span"] == span)]
        return float(sub[col].median()) if len(sub) else float("nan")

    def boot_ci(pair, metric):
        sub = boot[(boot["pair"] == pair) & (boot["metric"] == metric)]
        if not len(sub):
            return None
        return float(sub["ci_lower"].median()), float(sub["ci_upper"].median())

    dz = lambda v, s: abs(medv(v, s, "Std_z") - 1.0)
    A("## 9. Pre-registered case assessment\n")
    t3_beats_t1 = dz("T3p", "cross") < dz("T1p", "cross")
    match_beats_t1 = dz("T3match", "cross") < dz("T1p", "cross")
    t1p_beats_t1in = dz("T1p", "cross") < dz("T1in", "cross")
    t3p_close_t1p = abs(dz("T3p", "cross") - dz("T1p", "cross")) < 0.05
    v3_beats_v1 = dz("V3", "cross") < dz("V1", "cross")
    spearman_stable = (metrics[metrics["span"] == "cross"].groupby("variant")["Spearman"]
                       .median().dropna())
    A(f"- T3p better than T1p on cross |Std_z-1| ({dz('T3p','cross'):.3f} vs "
      f"{dz('T1p','cross'):.3f}): **{t3_beats_t1}**")
    A(f"- T3match better than T1p: **{match_beats_t1}** "
      f"(T3match {dz('T3match','cross'):.3f})")
    A(f"- T1p better than T1in (provenance effect): **{t1p_beats_t1in}** "
      f"(T1in {dz('T1in','cross'):.3f})")
    A(f"- V3 better than V1 (vintage chain): **{v3_beats_v1}** "
      f"(V1 {dz('V1','cross'):.3f}, V3 {dz('V3','cross'):.3f})")
    A(f"- Spearman range across variants (cross): "
      f"{spearman_stable.min():.3f}..{spearman_stable.max():.3f}")
    for pair in ("T3p-vs-T1p", "T3match-vs-T1p", "T1p-vs-T1in", "V3-vs-V1"):
        for m in ("abs_Std_z_minus_1", "IS", "PICP", "Spearman"):
            ci = boot_ci(pair, m)
            if ci:
                A(f"- bootstrap {pair} {m}: median CI [{ci[0]:.4f}, {ci[1]:.4f}] "
                  f"{'(significant)' if ci[0] * ci[1] > 0 else '(not significant)'}")
    A("")
    A("Interpretation keys: Case A = T3'>T1' + T3match>T1'; Case B = T3'>T1' but "
      "T3match~=T1'; Case C = provenance dominates; Case D = replay/vintage "
      "disagree; Case E = ranking stable while Std(z) varies.")
    (R / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print("REPORT.md written")


if __name__ == "__main__":
    main()
