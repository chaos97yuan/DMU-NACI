"""Replay the s7 evaluation for ONE seed and compare against the archived
evidence (results/rigorous_v2/s7_final_eval/metrics_per_seed.csv).

This is a verification helper, not a new experiment: it repeats the exact s7
steps for a single seed (mean-TCN inference -> CALIB conformal quantiles ->
SAME/CROSS metrics for the winner's A0/A1/A2/A3 arms) using the archived
checkpoints, then asserts agreement with the frozen five-seed evidence file.
It exists because a full s7 run (5 seeds x 11 GRU variants) takes over an hour
on a single GPU, whereas the one-seed replay runs in a few minutes.

Usage:
    python scripts/replay_s7_seed.py [--seed 42]

Requires the DKASC-derived preopen artifacts (data/processed/external_*)
and outputs/predictions/stage37r_*) plus the archived checkpoints; see
data/README.md. Writes nothing except its console report (pass with
--write-parquet off by default).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.rigorous_v2 import common_v2 as C  # noqa: E402

CK = ROOT / "checkpoints" / "site18"
EVIDENCE = ROOT / "results" / "rigorous_v2" / "s7_final_eval" / "metrics_per_seed.csv"


def metrics_row(y, mu, sg, lo, up, origins, states, dl, prefix, tag):
    r = {"variant": prefix, "span": tag}
    r.update({f"{prefix}_{k}": v for k, v in C.interval_metrics(y, lo, up, origins).items()})
    for st, m in states.items():
        r[f"{prefix}_PICP_{st}"] = float(((y[m] >= lo[m]) & (y[m] <= up[m])).mean())
    r[f"{prefix}_PICP_daylight"] = float(((y[dl] >= lo[dl]) & (y[dl] <= up[dl])).mean())
    r[f"{prefix}_IS_daylight"] = float(C.interval_score(y[dl], lo[dl], up[dl]).mean())
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    seed = args.seed

    sc_raw = json.loads((CK / "site18_scaler.json").read_text())
    sc = {"mean": np.array(sc_raw["mean"], float), "std": np.array(sc_raw["std"], float),
          "target_mean": float(sc_raw["target_mean"]), "target_std": float(sc_raw["target_std"])}
    decision = json.loads((ROOT / "results" / "rigorous_v2" / "s6_gru_select" /
                           "selection_decision.json").read_text())
    gammas = json.loads((ROOT / "results" / "rigorous_v2" / "s6_gru_select" /
                         "gamma_selection.json").read_text())
    winner = decision["winner"]
    print(f"winner pool: {winner}; seed {seed}; device {C.DEVICE}")

    y2024 = C.load_year_windows(2024)
    y2025 = C.load_year_windows(2025)
    calib = C.sub(y2024, C.span_mask(y2024["origins"], *C.CALIB))
    same = C.sub(y2024, C.span_mask(y2024["origins"], *C.SAME))
    spans = {"same": same, "cross": y2025}

    mean_ck = torch.load(CK / f"final_mean_seed{seed}.pt", weights_only=True,
                         map_location=C.DEVICE)
    model = C.MultiScaleTCN(dropout=0.1).to(C.DEVICE)
    model.load_state_dict(mean_ck["state_dict"])
    model.eval()
    gru_ck = torch.load(CK / f"gru_{winner}_seed{seed}.pt", weights_only=True,
                        map_location=C.DEVICE)
    gru = C.ScaleModel("GRU").to(C.DEVICE)
    gru.load_state_dict(gru_ck["state_dict"])
    gru.eval()

    mu_cal = C.mu_phys(C.tcn_mu(model, calib["x_raw"], sc))
    sg_cal = C.sigma_kw(gru, calib["x_raw"], sc)
    cal_abs_sorted = np.sort(np.abs(calib["y_kw"] - mu_cal))
    cal_norm_sorted = np.sort(np.abs(calib["y_kw"] - mu_cal) / (sg_cal + C.EPS))
    q_abs = C.conformal_q(cal_abs_sorted)
    g1 = gammas[str(seed)]["gamma_A1"]
    g3 = gammas[str(seed)]["gamma_A3"]

    ev = pd.read_csv(EVIDENCE)
    ev = ev[ev["seed"] == seed]
    failures = []
    for tag, p in spans.items():
        mu_p = C.mu_phys(C.tcn_mu(model, p["x_raw"], sc))
        sg = C.sigma_kw(gru, p["x_raw"], sc)
        states = C.state_split(p["N_group"], p["hard_DF"]) if "N_group" in p else {}
        dl = p["x_raw"][:, -1, 1] > C.DAYLIGHT_GHI_THRESHOLD
        y = p["y_kw"]
        arms = {
            "A2": (mu_p - float(C.conformal_q(cal_norm_sorted)) * sg,
                   mu_p + float(C.conformal_q(cal_norm_sorted)) * sg),
            "A0": (mu_p - q_abs, mu_p + q_abs),
            "A1": C.run_aci(y, mu_p, np.ones(len(sg)), cal_abs_sorted, p["origins"], g1)[:2],
            "A3": C.run_aci(y, mu_p, sg, cal_norm_sorted, p["origins"], g3)[:2],
        }
        for arm, (lo, up) in arms.items():
            row = metrics_row(y, mu_p, sg, lo, up, p["origins"], states, dl, arm, tag)
            prefix = f"{winner}:{arm}" if arm in ("A0", "A1", "A3") else winner
            ref = ev[(ev["variant"] == prefix) & (ev["span"] == tag)]
            if ref.empty:
                failures.append(f"no evidence row for {prefix}/{tag}")
                continue
            ref = ref.iloc[0]
            cols = [c for c in row if c in ref.index and isinstance(row[c], float)]
            worst, worst_col = 0.0, None
            for c in cols:
                d = abs(float(row[c]) - float(ref[c]))
                if d > worst:
                    worst, worst_col = d, c
            status = "OK" if worst <= 1e-6 else ("CLOSE" if worst <= 1e-4 else "FAIL")
            print(f"  {tag:5s} {arm}: max |Δ| = {worst:.2e} ({worst_col}) [{status}]")
            if worst > 1e-4:
                failures.append(f"{prefix}/{tag}: {worst_col} differs by {worst:.3e}")
    del model, gru
    torch.cuda.empty_cache()

    print()
    if failures:
        print("REPLAY FAILED:")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print(f"REPLAY OK: seed {seed} reproduces the archived s7 metrics "
          f"(max |Δ| <= 1e-4 on every reported column).")


if __name__ == "__main__":
    main()
