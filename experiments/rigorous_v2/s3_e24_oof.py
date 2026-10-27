"""rigorous_v2 STEP 3: E24_OOF -- rolling-origin expanding-window OOS residuals
inside the 2024 development window.

For every retained sample t:  max(training target) < target_timestamp(t).
Each fold refits the TCN (Stage-6 locked config), fits its scaler on its own
gradient rows, and early-stops on a tail slice of its OWN training window.
The 2024-08-08+ validation span is never touched here.

Folds (frozen):
  F1 train<05-01 (ES from 04-10) -> forecast 05-01..06-01
  F2 train<06-01 (ES from 05-11) -> forecast 06-01..07-01
  F3 train<07-01 (ES from 06-10) -> forecast 07-01..08-01
  F4 train<08-01 (ES from 07-11) -> forecast 08-01..08-08

Outputs -> results/rigorous_v2/s3_e24_oof/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.rigorous_v2 import common_v2 as C  # noqa: E402

OUT = C.V2 / "s3_e24_oof"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    C.assert_role_boundaries_disjoint()
    y2024 = C.load_year_windows(2024)
    rows_all = {s: [] for s in C.SEEDS}
    audit = []

    for f in C.OOF_FOLDS:
        m_grad = C.target_mask(y2024["targets"], "2024-01-01 00:00", f["es_start"])
        m_es = C.target_mask(y2024["targets"], f["es_start"], f["train_end"])
        m_fc = C.span_mask(y2024["origins"], *f["fc"])
        grad, es, fc = C.sub(y2024, m_grad), C.sub(y2024, m_es), C.sub(y2024, m_fc)
        C.assert_training_precedes(np.concatenate([grad["targets"], es["targets"]]),
                                   fc["targets"], f"fold {f['fold']}")
        C.assert_no_2025_in(fc["origins"], fc["targets"], f"fold {f['fold']} fc")
        assert len(grad["y_kw"]) > 1000 and len(fc["y_kw"]) > 1000, f"fold {f['fold']} too small"

        for seed in C.SEEDS:
            ck = OUT / f"oof_{f['fold']}_seed{seed}.pt"
            sc = C.fit_scaler(grad["x_raw"])
            if ck.exists():
                model = C.MultiScaleTCN(dropout=0.1).to(C.DEVICE)
                model.load_state_dict(torch.load(ck, weights_only=True)["state_dict"])
                model.eval()
                info = torch.load(ck, weights_only=True)["info"]
            else:
                model, info = C.train_tcn(grad, es, sc, seed)
                torch.save({"state_dict": model.state_dict(), "fold": f["fold"], "seed": seed,
                            "info": info, "scaler_fp": C.scaler_fingerprint(sc),
                            "grad_target_span": ["2024-01-01", f["es_start"]],
                            "es_target_span": [f["es_start"], f["train_end"]]}, ck)
            mu = C.tcn_mu(model, fc["x_raw"], sc).astype(np.float64)
            mp = C.mu_phys(mu)
            e = fc["y_kw"] - mp
            rows_all[seed].append(pd.DataFrame({
                "timestamp": pd.DatetimeIndex(fc["targets"]),
                "origin": pd.DatetimeIndex(fc["origins"]),
                "target": fc["y_kw"], "mu_raw": mu, "mu_phys": mp, "residual": e,
                "abs_residual": np.abs(e), "seed": seed, "fold_id": f["fold"],
                "train_start": "2024-01-01", "train_end": f["train_end"],
                "forecast_start": f["fc"][0], "forecast_end": f["fc"][1],
                "provenance": C.PROV_OOF, "origin_GHI": fc["origin_GHI"],
            }))
            del model
            torch.cuda.empty_cache()

        e_fold = fc["y_kw"] - mp  # last-seed residuals for the audit only
        audit.append({"fold": f["fold"], "n_grad": int(m_grad.sum()), "n_es": int(m_es.sum()),
                      "n_fc": int(m_fc.sum()), "fc_first": str(fc["origins"].min()),
                      "fc_last": str(fc["origins"].max()),
                      "residual_mean": float(e_fold.mean()), "residual_std": float(e_fold.std()),
                      "abs_q95": float(np.quantile(np.abs(e_fold), 0.95)),
                      "RMSE": float(np.sqrt(np.mean(e_fold ** 2))),
                      "MAE": float(np.mean(np.abs(e_fold)))})
        print(audit[-1])

    for seed in C.SEEDS:
        df = pd.concat(rows_all[seed], ignore_index=True)
        C.assert_provenance(df["provenance"].to_numpy())
        C.assert_no_2025_in(df["origin"], df["timestamp"], f"E24_OOF seed {seed}")
        o = pd.DatetimeIndex(df["origin"])
        assert (o >= pd.Timestamp("2024-05-01")).all() and (o < pd.Timestamp("2024-08-08")).all()
        df.to_parquet(OUT / f"E24_OOF_seed{seed}.parquet", index=False)

    C.wjson(OUT / "residual_manifest.json", {
        "provenance": C.PROV_OOF,
        "folds": [{k: v for k, v in f.items() if k != "fc"} | {"fc": list(f["fc"])}
                  for f in C.OOF_FOLDS],
        "fold_audit": audit,
        "n_per_seed": int(len(df)),
        "months": sorted(set(int(m) for m in pd.DatetimeIndex(
            pd.concat(rows_all[C.SEEDS[0]])["origin"]).month.unique())),
        "assert": "max(grad+es target) < min(fold forecast target), every fold/seed",
        "checkpoint_sha": {p.name: C.sha256_file(p) for p in sorted(OUT.glob("oof_*.pt"))},
    })
    print("s3 done")


if __name__ == "__main__":
    main()
