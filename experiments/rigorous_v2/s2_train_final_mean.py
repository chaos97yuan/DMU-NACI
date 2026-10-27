"""rigorous_v2 STEP 2: retrain the final mean TCN with an INTERNAL early-stop tail.

Why: the legacy Stage-6 mean early-stopped on 2024-08-08..10-01, the same span later
used for selection and conformal calibration. The v2 protocol keeps every mean-model
decision inside 2024-01-01..08-07:

  gradient rows      target in [2024-01-01, 2024-07-17)
  early-stop tail    target in [2024-07-17, 2024-08-08)
  scaler             fit on gradient rows ONLY
  frozen date        2024-08-08 (nothing after 08-07 23:55 target is ever consumed)

Architecture/hyperparameters are the Stage-6 locked values (dropout 0.1, lr 1e-3,
AdamW wd 1e-4, batch 512, patience 5, max 100 epochs); nothing is re-tuned here.

Outputs -> results/rigorous_v2/s2_final_mean/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.rigorous_v2 import common_v2 as C  # noqa: E402

OUT = C.V2 / "s2_final_mean"
SPANS = {"grad": C.MEAN_GRAD, "es": C.MEAN_ES, "select": C.SELECT, "calib": C.CALIB,
         "same": C.SAME}


def rmse_mae(y, p) -> dict:
    e = y - p
    return {"RMSE": float(np.sqrt(np.mean(e ** 2))), "MAE": float(np.mean(np.abs(e)))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    C.assert_role_boundaries_disjoint()
    y2024 = C.load_year_windows(2024)

    parts = {}
    for tag, (lo, hi) in SPANS.items():
        # mean-model rows are defined on TARGETS (training consumes y);
        # evaluation spans are defined on ORIGINS (issue-time membership)
        if tag in ("grad", "es"):
            m = C.target_mask(y2024["targets"], lo, hi)
        else:
            m = C.span_mask(y2024["origins"], lo, hi)
        parts[tag] = C.sub(y2024, m)
        print(f"{tag}: {m.sum()} rows "
              f"[{parts[tag]['targets'].min()} .. {parts[tag]['targets'].max()}]")

    # causal contract of the final mean
    all_mean_targets = pd.DatetimeIndex(
        np.concatenate([np.asarray(parts["grad"]["targets"]),
                        np.asarray(parts["es"]["targets"])]))
    C.assert_training_precedes(all_mean_targets, parts["select"]["origins"],
                               "final mean vs SELECT")
    for tag in ("grad", "es"):
        C.assert_no_2025_in(parts[tag]["origins"], parts[tag]["targets"], f"mean {tag}")
    assert (pd.DatetimeIndex(parts["es"]["targets"]).max()
            < pd.Timestamp(C.SELECT[0])), "mean ES tail reaches into SELECT"

    sc = C.fit_scaler(parts["grad"]["x_raw"])
    C.wjson(OUT / "scaler.json", {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                  for k, v in sc.items()})
    print("common scaler fingerprint:", C.scaler_fingerprint(sc))

    mu_store, metric_rows = {}, []
    for seed in C.SEEDS:
        ck = OUT / f"final_mean_seed{seed}.pt"
        if ck.exists():
            info = torch.load(ck, weights_only=False)["info"]
            model = C.MultiScaleTCN(dropout=0.1).to(C.DEVICE)
            model.load_state_dict(torch.load(ck, weights_only=False)["state_dict"])
            model.eval()
        else:
            model, info = C.train_tcn(parts["grad"], parts["es"], sc, seed)
            torch.save({"state_dict": model.state_dict(), "cfg": {"dropout": 0.1, "lr": 1e-3},
                        "seed": seed, "scaler_fp": C.scaler_fingerprint(sc), "info": info}, ck)
        for tag in ("grad", "es", "select", "calib", "same"):
            mu = C.tcn_mu(model, parts[tag]["x_raw"], sc)
            mu_store[f"{tag}|{seed}"] = mu.astype(np.float32)
            mp = C.mu_phys(mu.astype(np.float64))
            mm = rmse_mae(parts[tag]["y_kw"], mp)
            metric_rows.append({"seed": seed, "span": tag, **mm,
                                "n": int(len(mu)), "best_epoch": info["best_epoch"]})
            print(f"seed {seed} {tag}: RMSE {mm['RMSE']:.4f} MAE {mm['MAE']:.4f}")
        del model
        torch.cuda.empty_cache()

    np.savez_compressed(OUT / "final_mean_cache.npz", **mu_store)
    pd.DataFrame(metric_rows).to_csv(OUT / "mean_metrics.csv", index=False)
    C.wjson(OUT / "split_manifest.json", {
        "gradients": {"target_span": list(C.MEAN_GRAD), "n": int(len(parts["grad"]["y_kw"]))},
        "early_stop_tail": {"target_span": list(C.MEAN_ES), "n": int(len(parts["es"]["y_kw"]))},
        "scaler_fit_on": "gradient rows only (targets < 2024-07-17)",
        "scaler_fingerprint": C.scaler_fingerprint(sc),
        "frozen_date": "2024-08-08",
        "asserts": ["max(grad+es target) < min(SELECT origin)", "no 2025 anywhere in s2",
                    "role spans disjoint"],
        "checkpoint_sha": {f"seed{s}": C.sha256_file(OUT / f"final_mean_seed{s}.pt")
                           for s in C.SEEDS},
        "stage6_locked_cfg": {"dropout": 0.1, "lr": 1e-3, "wd": 1e-4, "batch": 512,
                              "patience": 5, "max_epochs": 100, "rng": 58003},
        "es_fix": "ES RMSE now computed in kW (legacy stage6 helper compared "
                  "standardized predictions to kW targets)",
    })
    print("s2 done")


if __name__ == "__main__":
    main()
