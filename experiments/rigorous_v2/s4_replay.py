"""rigorous_v2 STEP 4: E22_replay / E23_replay -- the frozen final mean (s2)
replayed on archived FULL-YEAR 2022 and 2023 inputs.

Provenance: REPLAY_OOS. These are out-of-sample errors OF THE DEPLOYED MODEL
(it never trained on 2022/2023) replayed retrospectively at deployment time.
They are NOT the operational errors committed by earlier model versions -- the
paper must describe them exactly this way.

Outputs -> results/rigorous_v2/s4_replay/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.rigorous_v2 import common_v2 as C  # noqa: E402

OUT = C.V2 / "s4_replay"
S2 = C.V2 / "s2_final_mean"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    import json
    sc_raw = json.loads((S2 / "scaler.json").read_text())
    sc = {"mean": np.array(sc_raw["mean"], dtype=float), "std": np.array(sc_raw["std"], dtype=float),
          "target_mean": float(sc_raw["target_mean"]), "target_std": float(sc_raw["target_std"])}
    fp = C.scaler_fingerprint(sc)
    audit = []
    for year in (2022, 2023):
        d = C.load_year_windows(year)
        # the replayed model trained ONLY on 2024 rows -> disjoint by year; assert it
        for seed in C.SEEDS:
            ck = torch.load(S2 / f"final_mean_seed{seed}.pt", weights_only=True)
            assert ck["scaler_fp"] == fp, "scaler mismatch vs s2 checkpoint"
            model = C.MultiScaleTCN(dropout=0.1).to(C.DEVICE)
            model.load_state_dict(ck["state_dict"])
            model.eval()
            mu = C.tcn_mu(model, d["x_raw"], sc).astype(np.float64)
            mp = C.mu_phys(mu)
            e = d["y_kw"] - mp
            df = pd.DataFrame({
                "timestamp": pd.DatetimeIndex(d["targets"]),
                "origin": pd.DatetimeIndex(d["origins"]),
                "target": d["y_kw"], "mu_raw": mu, "mu_phys": mp, "residual": e,
                "abs_residual": np.abs(e), "seed": seed, "fold_id": "replay",
                "train_start": "2024-01-01", "train_end": "2024-08-08",
                "forecast_start": f"{year}-01-01", "forecast_end": f"{year + 1}-01-01",
                "provenance": C.PROV_REPLAY, "origin_GHI": d["origin_GHI"],
            })
            C.assert_provenance(df["provenance"].to_numpy())
            assert (pd.DatetimeIndex(df["origin"]).year == year).all()
            df.to_parquet(OUT / f"E{str(year)[2:]}_replay_seed{seed}.parquet", index=False)
            del model
        o = pd.DatetimeIndex(d["origins"])
        audit.append({
            "year": year, "n": int(len(d["y_kw"])),
            "months": sorted(set(int(m) for m in o.month.unique())),
            "first_origin": str(o.min()), "last_origin": str(o.max()),
            "daylight_share": float((d["origin_GHI"] > C.DAYLIGHT_GHI_THRESHOLD).mean()),
            "RMSE": float(np.sqrt(np.mean(e ** 2))), "MAE": float(np.mean(np.abs(e))),
            "residual_std": float(np.std(e)),
            "abs_q50": float(np.quantile(np.abs(e), 0.50)),
            "abs_q95": float(np.quantile(np.abs(e), 0.95)),
            "abs_q99": float(np.quantile(np.abs(e), 0.99)),
        })
        print(audit[-1])
    C.wjson(OUT / "residual_manifest.json", {
        "provenance": C.PROV_REPLAY,
        "definition": "residuals of the frozen deployed forecaster replayed on archived "
                      "inputs (out-of-sample for the model; retrospective by construction; "
                      "executable at deployment date 2024-10-01)",
        "model": "s2 final mean (trained on 2024 gradient rows <2024-07-17 + ES tail "
                 "<2024-08-08), matched seeds",
        "scaler": "common deployment scaler (s2), identical for both years",
        "per_year": audit,
        "assert": "replay years disjoint from model training year (2024) by construction",
    })
    print("s4 done")


if __name__ == "__main__":
    main()
