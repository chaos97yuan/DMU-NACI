"""rigorous_v2 STEP 5: forward-vintage residuals (provenance sensitivity arm).

M22: train 2021          -> forecast 2022 full year -> E22_vintage
M23: train 2021+2022     -> forecast 2023 full year -> E23_vintage
M24: train 2021+2022+2023-> forecast 2024 full year -> E24_vintage

For every forecast sample: max(training target) < min(forecast target).
Each vintage fits its own scaler on its gradient rows and early-stops on a tail
slice of its LAST training year. Stage-6 locked config; nothing re-tuned.

These residuals emulate the OPERATIONAL error distribution of the methodology
across years (model vintages differ), in contrast to the replay arm which holds
the deployed model fixed. Purpose: test whether the pooling conclusion is
provenance-dependent (old project chain found T1 > T3 under forward vintages).

Outputs -> results/rigorous_v2/s5_vintage/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.rigorous_v2 import common_v2 as C  # noqa: E402

OUT = C.V2 / "s5_vintage"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    C.assert_role_boundaries_disjoint()
    cache: dict[int, dict] = {}
    audit = []
    for spec in C.VINTAGE:
        tag, years, fc_year, es_start = spec["tag"], spec["years"], spec["fc_year"], spec["es_start"]
        train_parts, es_parts = [], []
        for y in years:
            if y not in cache:
                cache[y] = C.load_year_windows(y)
            d = cache[y]
            m_grad = C.target_mask(d["targets"], f"{y}-01-01 00:00",
                                   es_start if y == years[-1] else f"{y + 1}-01-01 00:00")
            train_parts.append(C.sub(d, m_grad))
            if y == years[-1]:
                m_es = C.target_mask(d["targets"], es_start, f"{y + 1}-01-01 00:00")
                es_parts.append(C.sub(d, m_es))
        grad = {"x_raw": np.concatenate([p["x_raw"] for p in train_parts]),
                "y_kw": np.concatenate([p["y_kw"] for p in train_parts])}
        grad["targets"] = pd.DatetimeIndex(np.concatenate(
            [np.asarray(p["targets"]) for p in train_parts]))
        es = {"x_raw": np.concatenate([p["x_raw"] for p in es_parts]),
              "y_kw": np.concatenate([p["y_kw"] for p in es_parts])}
        es["targets"] = pd.DatetimeIndex(np.concatenate(
            [np.asarray(p["targets"]) for p in es_parts]))
        if fc_year not in cache:
            cache[fc_year] = C.load_year_windows(fc_year)
        fc = cache[fc_year]
        C.assert_training_precedes(np.concatenate([grad["targets"], es["targets"]]),
                                   fc["targets"], f"vintage {tag}")
        assert (pd.DatetimeIndex(fc["origins"]).year == fc_year).all()

        for seed in C.SEEDS:
            ck = OUT / f"{tag}_seed{seed}.pt"
            sc = C.fit_scaler(grad["x_raw"])
            if ck.exists():
                model = C.MultiScaleTCN(dropout=0.1).to(C.DEVICE)
                model.load_state_dict(torch.load(ck, weights_only=True)["state_dict"])
                model.eval()
            else:
                model, info = C.train_tcn(grad, es, sc, seed)
                torch.save({"state_dict": model.state_dict(), "tag": tag, "seed": seed,
                            "info": info, "scaler_fp": C.scaler_fingerprint(sc),
                            "train_years": years}, ck)
            mu = C.tcn_mu(model, fc["x_raw"], sc).astype(np.float64)
            mp = C.mu_phys(mu)
            e = fc["y_kw"] - mp
            df = pd.DataFrame({
                "timestamp": pd.DatetimeIndex(fc["targets"]),
                "origin": pd.DatetimeIndex(fc["origins"]),
                "target": fc["y_kw"], "mu_raw": mu, "mu_phys": mp, "residual": e,
                "abs_residual": np.abs(e), "seed": seed, "fold_id": tag,
                "train_start": f"{years[0]}-01-01", "train_end": f"{years[-1] + 1}-01-01",
                "forecast_start": f"{fc_year}-01-01", "forecast_end": f"{fc_year + 1}-01-01",
                "provenance": C.PROV_VINTAGE, "origin_GHI": fc["origin_GHI"],
            })
            C.assert_provenance(df["provenance"].to_numpy())
            df.to_parquet(OUT / f"E{str(fc_year)[2:]}_vintage_seed{seed}.parquet", index=False)
            del model
            torch.cuda.empty_cache()

        o = pd.DatetimeIndex(fc["origins"])
        audit.append({"tag": tag, "train_years": years, "fc_year": fc_year,
                      "n_grad": int(len(grad["y_kw"])), "n_es": int(len(es["y_kw"])),
                      "n_fc": int(len(fc["y_kw"])),
                      "months": sorted(set(int(m) for m in o.month.unique())),
                      "RMSE": float(np.sqrt(np.mean(e ** 2))), "MAE": float(np.mean(np.abs(e))),
                      "residual_std": float(np.std(e)),
                      "abs_q95": float(np.quantile(np.abs(e), 0.95))})
        print(audit[-1])
    C.wjson(OUT / "residual_manifest.json", {
        "provenance": C.PROV_VINTAGE,
        "definition": "operational-style forward vintages: each year forecast by a model "
                      "trained only on preceding years",
        "vintages": audit,
        "assert": "max(train target) < min(forecast target) per vintage/seed",
        "checkpoint_sha": {p.name: C.sha256_file(p) for p in sorted(OUT.glob("M*_seed*.pt"))},
    })
    print("s5 done")


if __name__ == "__main__":
    main()
