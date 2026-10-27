"""rigorous_v2 STEP 7: calibration q (CALIB span only) -> same-year report ->
2025 final prequential evaluation. Runs ONLY after FREEZE_MANIFEST.json exists.

Discipline:
  - CALIB span (2024-09-16..10-02) computes conformal quantiles and the ACI initial
    score grids ONLY (never trains, never selects).
  - SAME span (2024-10-02..12-31) is report-only.
  - CROSS (2025) is report-only + delayed online ACI (15-min matured feedback);
    gamma is frozen from s6 (SELECT span). No offline selection anywhere here.

Every variant is evaluated for reporting; the deployment pipeline is the s6 winner.

Outputs -> results/rigorous_v2/s7_final_eval/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.rigorous_v2 import common_v2 as C  # noqa: E402

OUT = C.V2 / "s7_final_eval"
S2, S6 = C.V2 / "s2_final_mean", C.V2 / "s6_gru_select"
REPORT_VARIANTS = ["T1p", "T2p", "T3p", "T1in", "T3match", "V1", "V2", "V3"]
EQUPD_VARIANTS = ["T1p_equpd", "T3p_equpd", "T3match_equpd"]


def load_gru(variant: str, seed: int, sc: dict):
    ck = torch.load(S6 / "checkpoints" / f"gru_{variant}_seed{seed}.pt", weights_only=False)
    assert ck["scaler_fp"] == C.scaler_fingerprint(sc), "scaler drift vs s6 checkpoint"
    model = C.ScaleModel("GRU").to(C.DEVICE)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model


def row_for(y, mu, sg, lo, up, origins, states, dl, prefix, variant, seed, tag):
    r = {"variant": variant, "seed": seed, "span": tag}
    if prefix == "A2":
        r.update(C.scale_metrics(y, mu, sg))
    r.update({f"{prefix}_{k}": v for k, v in
              C.interval_metrics(y, lo, up, origins).items()})
    for st, m in states.items():
        r[f"{prefix}_PICP_{st}"] = float(((y[m] >= lo[m]) & (y[m] <= up[m])).mean())
    r[f"{prefix}_PICP_daylight"] = float(((y[dl] >= lo[dl]) & (y[dl] <= up[dl])).mean())
    r[f"{prefix}_IS_daylight"] = float(C.interval_score(y[dl], lo[dl], up[dl]).mean())
    return r


def main() -> None:
    assert (C.V2 / "FREEZE_MANIFEST.json").exists(), "run s6 first (freeze manifest)"
    import json
    from scipy.stats import spearmanr
    decision = json.loads((S6 / "selection_decision.json").read_text())
    gammas = json.loads((S6 / "gamma_selection.json").read_text())
    winner = decision["winner"]
    sc_raw = json.loads((S2 / "scaler.json").read_text())
    sc = {"mean": np.array(sc_raw["mean"], float), "std": np.array(sc_raw["std"], float),
          "target_mean": float(sc_raw["target_mean"]), "target_std": float(sc_raw["target_std"])}
    OUT.mkdir(parents=True, exist_ok=True)

    y2024 = C.load_year_windows(2024)
    y2025 = C.load_year_windows(2025)  # opened HERE for the first time, report-only
    spans = {"calib": C.sub(y2024, C.span_mask(y2024["origins"], *C.CALIB)),
             "same": C.sub(y2024, C.span_mask(y2024["origins"], *C.SAME)),
             "cross": y2025}
    for tag in ("calib", "same"):
        C.assert_no_2025_in(spans[tag]["origins"], spans[tag]["targets"], tag)
    assert not set(spans["calib"]["origins"]) & set(spans["same"]["origins"])

    mu_c = {}
    for seed in C.SEEDS:
        model = C.MultiScaleTCN(dropout=0.1).to(C.DEVICE)
        model.load_state_dict(torch.load(S2 / f"final_mean_seed{seed}.pt",
                                         weights_only=False)["state_dict"])
        model.eval()
        for tag, p in spans.items():
            mu_c[f"{tag}|{seed}"] = C.tcn_mu(model, p["x_raw"], sc).astype(np.float32)
        del model
    np.savez_compressed(OUT / "mu_spans.npz", **mu_c)

    all_rows, store, month_rows = [], {}, []
    calib = spans["calib"]
    for seed in C.SEEDS:
        mu_cal = C.mu_phys(mu_c[f"calib|{seed}"].astype(np.float64))
        cal_abs_sorted = np.sort(np.abs(calib["y_kw"] - mu_cal))       # A0/A1 initial grid
        q_abs = C.conformal_q(cal_abs_sorted)
        for variant in REPORT_VARIANTS + EQUPD_VARIANTS:
            model = load_gru(variant, seed, sc)
            sg_cal = C.sigma_kw(model, calib["x_raw"], sc)
            cal_norm_sorted = np.sort(np.abs(calib["y_kw"] - mu_cal) / (sg_cal + C.EPS))
            q_norm = C.conformal_q(cal_norm_sorted)
            store[(variant, seed)] = {"cal_norm_sorted": cal_norm_sorted,
                                      "cal_abs_sorted": cal_abs_sorted}
            for tag in ("same", "cross"):
                p = spans[tag]
                mu_p = C.mu_phys(mu_c[f"{tag}|{seed}"].astype(np.float64))
                sg = C.sigma_kw(model, p["x_raw"], sc)
                store[(variant, seed)][f"sigma_{tag}"] = sg
                states = C.state_split(p["N_group"], p["hard_DF"]) if "N_group" in p else {}
                dl = p["x_raw"][:, -1, 1] > C.DAYLIGHT_GHI_THRESHOLD
                lo, up = mu_p - q_norm * sg, mu_p + q_norm * sg
                all_rows.append(row_for(p["y_kw"], mu_p, sg, lo, up, p["origins"], states, dl,
                                        "A2", variant, seed, tag))
                if variant == winner:
                    g1 = gammas[str(seed)]["gamma_A1"]
                    g3 = gammas[str(seed)]["gamma_A3"]
                    variants_ab = {
                        "A0": (mu_p - q_abs, mu_p + q_abs),
                        "A1": C.run_aci(p["y_kw"], mu_p, np.ones(len(sg)), cal_abs_sorted,
                                        p["origins"], g1)[:2],
                        "A3": C.run_aci(p["y_kw"], mu_p, sg, cal_norm_sorted,
                                        p["origins"], g3)[:2],
                    }
                    for nm, (l, u) in variants_ab.items():
                        all_rows.append(row_for(p["y_kw"], mu_p, sg, l, u, p["origins"], states,
                                                dl, nm, f"{winner}:{nm}", seed, tag))
                # monthly decomposition (cross, main variants)
                if tag == "cross" and variant in ("T1p", "T3p", "T3match", "T1in", "V1", "V3"):
                    o = pd.DatetimeIndex(p["origins"])
                    z = (p["y_kw"] - mu_p) / (sg + C.EPS)
                    for month in range(1, 13):
                        m = np.asarray(o.month == month)
                        if m.sum() == 0:
                            continue
                        month_rows.append({
                            "variant": variant, "seed": seed, "month": month,
                            "n": int(m.sum()), "Std_z": float(np.std(z[m])),
                            "Spearman": float(spearmanr(sg[m], np.abs(p["y_kw"][m] - mu_p[m])).statistic),
                            "PICP": float(((p["y_kw"][m] >= lo[m]) & (p["y_kw"][m] <= up[m])).mean()),
                            "IS": float(C.interval_score(p["y_kw"][m], lo[m], up[m]).mean()),
                            "season": str(C.seasons_southern(np.array([month]))[0])})
            del model
            torch.cuda.empty_cache()
        print(f"seed {seed} done")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "metrics_per_seed.csv", index=False)
    pd.DataFrame(month_rows).to_csv(OUT / "monthly_cross_year.csv", index=False)
    key = [c for c in ("NLL", "Std_z", "Mean_z", "Median_abs_z", "Q95_abs_z", "Spearman",
                       "Median_sigma_kw", "A2_PICP", "A2_PINAW", "A2_IS", "A2_RCE1D",
                       "A2_RCE3D", "A2_LongestMissStreak", "A2_PICP_NORMAL",
                       "A2_PICP_HIGH_N", "A2_PICP_HARD_DF", "A2_PICP_daylight",
                       "A2_IS_daylight", "A0_PICP", "A0_IS", "A1_PICP", "A1_PINAW", "A1_IS",
                       "A3_PICP", "A3_PINAW", "A3_IS", "A3_RCE1D", "A3_PICP_HARD_DF",
                       "A3_PICP_daylight") if c in df.columns]
    df.groupby(["variant", "span"])[key].median().to_csv(OUT / "table_median.csv")

    # ---- paired day-block bootstrap on 2025 (report-only) ----
    p = spans["cross"]
    y, o = p["y_kw"], pd.DatetimeIndex(p["origins"])
    boot = []
    for (a, b) in [("T1p", "T3p"), ("T1p", "T3match"), ("T1in", "T1p"), ("V1", "V3")]:
        for seed in C.SEEDS:
            mu_p = C.mu_phys(mu_c[f"cross|{seed}"].astype(np.float64))
            sa = store[(a, seed)]["sigma_cross"]
            sb = store[(b, seed)]["sigma_cross"]
            qa = float(C.conformal_q(store[(a, seed)]["cal_norm_sorted"]))
            qb = float(C.conformal_q(store[(b, seed)]["cal_norm_sorted"]))
            arrs = [y, mu_p]

            def make(metric, q):
                if metric == "IS":
                    return lambda s, yy, mm: float(C.interval_score(
                        yy, mm - q * s, mm + q * s).mean())
                if metric == "PICP":
                    return lambda s, yy, mm: float(((yy >= mm - q * s) &
                                                     (yy <= mm + q * s)).mean())
                if metric == "abs_Std_z_minus_1":
                    return lambda s, yy, mm: abs(float(np.std((yy - mm) / (s + C.EPS))) - 1.0)
                return lambda s, yy, mm: float(spearmanr(s, np.abs(yy - mm)).statistic)

            for metric in ("IS", "PICP", "abs_Std_z_minus_1", "Spearman"):
                r = C.paired_day_bootstrap(make(metric, qa), make(metric, qb), o,
                                           [sa] + arrs, [sb] + arrs, seed=seed)
                boot.append({"pair": f"{b}-vs-{a}", "seed": seed, "metric": metric, **r})
    pd.DataFrame(boot).to_csv(OUT / "bootstrap_paired.csv", index=False)

    for seed in C.SEEDS:
        mu_p = C.mu_phys(mu_c[f"cross|{seed}"].astype(np.float64))
        sg = store[(winner, seed)]["sigma_cross"]
        pd.DataFrame({"origin": o, "y_kw": y, "mu_phys": mu_p, "sigma": sg}).to_parquet(
            OUT / f"per_sample_cross_{winner}_seed{seed}.parquet", index=False)

    C.wjson(OUT / "run_manifest.json", {
        "winner": winner, "gamma": gammas,
        "calib_span": list(C.CALIB), "n_calib": int(len(calib["y_kw"])),
        "n_same": int(len(spans["same"]["y_kw"])), "n_cross": int(len(spans["cross"]["y_kw"])),
        "bootstrap_pairs": ["T3p-vs-T1p", "T3match-vs-T1p", "T1p-vs-T1in", "V3-vs-V1"],
        "note": "2025 opened here for the first time; report-only + delayed online ACI; "
                "q from CALIB span only; gamma frozen from s6 SELECT",
    })
    print("s7 done")


if __name__ == "__main__":
    main()
