"""Site 1B Phase 5+6+7: GRU scale encoder -> SELECT gamma -> CALIB q -> FREEZE.

Phase 5  T3' pool = E24_OOF + E23_replay + E22_replay (final Site 18 config; no
         window re-selection); common deployment scaler (Phase 3 mean scaler);
         GRU(5->32->softplus), Gaussian heteroscedastic NLL, 3x50 epochs, rng
         58003, 5 matched seeds -- rigorous_v2 protocol verbatim.
Phase 6  SELECT (2024-08-08..09-15 23:40): gamma_A1/gamma_A3 per seed only.
         CALIB (2024-09-16..10-01 23:40): initial conformal q only.
Phase 7  FREEZE_MANIFEST_SITE1B.json (hashes of data/code/models/scalers/gammas).
2025 is never loaded here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.site1b_validation import common_1b as C  # noqa: E402

OUT = C.OUT


def main() -> None:
    V2 = C.V2
    sc_raw = json.loads((OUT / "site1b_mean_scaler.json").read_text())
    sc = {"mean": np.array(sc_raw["mean"], float), "std": np.array(sc_raw["std"], float),
          "target_mean": float(sc_raw["target_mean"]), "target_std": float(sc_raw["target_std"])}
    fp = V2.scaler_fingerprint(sc)

    g24 = C.load_year_grid(2024)
    w24 = C.windows(g24)
    sel = {k: v[C.span_on_origins(w24, *V2.SELECT)] for k, v in w24.items()}
    cal = {k: v[C.span_on_origins(w24, *V2.CALIB)] for k, v in w24.items()}
    V2.assert_no_2025_in(sel["origins"], sel["targets"], "SELECT")
    V2.assert_no_2025_in(cal["origins"], cal["targets"], "CALIB")
    assert max(sel["origins"]) < min(cal["origins"]), "SELECT/CALIB overlap"

    mu_cache = np.load(OUT / "site1b_mean_cache.npz")
    pools = {}
    for seed in C.SEEDS:
        parts = [pd.read_csv(OUT / f) for f in
                 ("E24_OOF_site1b.csv", "E23_replay_site1b.csv", "E22_replay_site1b.csv")]
        parts = [d[d["seed"] == seed] for d in parts]
        gy22 = C.load_year_grid(2022)
        gy23 = C.load_year_grid(2023)
        w22, w23 = C.windows(gy22), C.windows(gy23)

        def x_by_origins(w, origins):
            pos = pd.Series(np.arange(len(w["origins"])), index=pd.DatetimeIndex(w["origins"]))
            idx = pos.reindex(pd.DatetimeIndex(origins)).to_numpy(dtype=float)
            assert not np.isnan(idx).any()
            return w["x_raw"][idx.astype(int)]

        pools[seed] = {
            "x_raw": np.concatenate([
                x_by_origins(w24, parts[0]["origin"]),
                x_by_origins(w23, parts[1]["origin"]),
                x_by_origins(w22, parts[2]["origin"])]).astype(np.float32),
            "e_kw": np.concatenate([p["residual"].to_numpy(float) for p in parts]),
            "prov": np.concatenate([p["provenance"].to_numpy() for p in parts]),
            "origins": pd.DatetimeIndex(pd.concat([pd.Series(pd.DatetimeIndex(p["origin"]))
                                                   for p in parts]).to_numpy()),
        }
        V2.assert_no_2025_in(pools[seed]["origins"],
                             pools[seed]["origins"] + pd.Timedelta(minutes=15), "T3' pool")
        n_pool = sum(len(d) for d in parts)
        assert len(pools[seed]["e_kw"]) == n_pool == len(pools[seed]["x_raw"]), \
            f"pool misaligned: x {len(pools[seed]['x_raw'])} vs e {len(pools[seed]['e_kw'])} vs csv {n_pool}"
        assert np.isfinite(pools[seed]["x_raw"]).all() and np.isfinite(pools[seed]["e_kw"]).all()
        assert n_pool > 150_000

    rows, sigma_diag = [], {}
    (OUT / "checkpoints").mkdir(exist_ok=True)
    for seed in C.SEEDS:
        p = pools[seed]
        xs = V2.apply_scaler(p["x_raw"], sc)
        estd = (p["e_kw"] / sc["target_std"]).astype(np.float32)
        ck = OUT / "checkpoints" / f"site1b_gru_T3p_seed{seed}.pt"
        if ck.exists():
            model = C.V2.ScaleModel("GRU").to(C.V2.DEVICE)
            model.load_state_dict(torch.load(ck, weights_only=True)["state_dict"])
            model.eval()
        else:
            model, steps, secs = V2.train_gru(xs, estd, seed)
            with torch.no_grad():  # guard against silent numeric blow-up
                probe = model.noise_variance(torch.as_tensor(
                    xs[:4096], dtype=torch.float32, device=C.V2.DEVICE))
                assert torch.isfinite(probe).all(), f"GRU seed {seed} produced NaN"
            torch.save({"state_dict": model.state_dict(), "seed": seed,
                        "scaler_fp": fp, "n_train": len(estd)}, ck)
        mu_sel = C.mu_phys(mu_cache[f"select|{seed}"].astype(np.float64))
        sg_sel = V2.sigma_kw(model, sel["x_raw"], sc)
        r = {"seed": seed, "span": "select", **V2.scale_metrics(sel["y_kw"], mu_sel, sg_sel)}
        rows.append(r)
        # E22 vs E23 sigma support diagnostic (Phase 5 requirement)
        w22d, w23d = C.windows(C.load_year_grid(2022)), C.windows(C.load_year_grid(2023))
        s22 = V2.sigma_kw(model, w22d["x_raw"], sc)
        s23 = V2.sigma_kw(model, w23d["x_raw"], sc)
        sigma_diag[seed] = {
            "sigma_2022_median": float(np.median(s22)), "sigma_2023_median": float(np.median(s23)),
            "sigma_2022_q90": float(np.quantile(s22, .9)), "sigma_2023_q90": float(np.quantile(s23, .9)),
        }
        print({k: round(v, 4) for k, v in r.items() if isinstance(v, float)})
        del model
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(OUT / "site1b_gru_select_metrics.csv", index=False)

    # ---- Phase 6: gamma on SELECT; q on CALIB ----
    gammas = {}
    for seed in C.SEEDS:
        model = C.V2.ScaleModel("GRU").to(C.V2.DEVICE)
        model.load_state_dict(torch.load(OUT / "checkpoints" / f"site1b_gru_T3p_seed{seed}.pt",
                                         weights_only=True)["state_dict"])
        model.eval()
        mu_sel = C.mu_phys(mu_cache[f"select|{seed}"].astype(np.float64))
        sg_sel = V2.sigma_kw(model, sel["x_raw"], sc)
        g3, _ = V2.select_gamma(sel["y_kw"], mu_sel, sg_sel, sel["origins"], use_sigma=True)
        g1, _ = V2.select_gamma(sel["y_kw"], mu_sel, np.ones(len(sg_sel)), sel["origins"],
                                use_sigma=False)
        mu_cal = C.mu_phys(mu_cache[f"calib|{seed}"].astype(np.float64))
        sg_cal = V2.sigma_kw(model, cal["x_raw"], sc)
        q_abs = V2.conformal_q(np.sort(np.abs(cal["y_kw"] - mu_cal)))
        q_norm = V2.conformal_q(np.sort(np.abs(cal["y_kw"] - mu_cal) / (sg_cal + C.EPS)))
        gammas[str(seed)] = {"gamma_A3": g3, "gamma_A1": g1, "q_abs": q_abs, "q_norm": q_norm}
        del model
    (OUT / "site1b_gamma_q.json").write_text(json.dumps(gammas, indent=2), encoding="utf-8")
    print(json.dumps(gammas, indent=1))

    # ---- Phase 7: freeze ----
    (OUT / "FREEZE_MANIFEST_SITE1B.json").write_text(json.dumps({
        "site": C.SITE_ID, "p_rated_kw": C.P_RATED_1B,
        "created": pd.Timestamp.now().isoformat(),
        "protocol": "cross-system validation under shared meteorological environment; "
                    "T3' = E24_OOF + E23_replay + E22_replay (Site 18 final config)",
        "clean_dataset_sha16": C.sha16(C.CLEAN_CSV),
        "mean_ck": {p.name: C.sha16(p) for p in sorted(OUT.glob("site1b_mean_checkpoint_*.pt"))},
        "oof_ck": {p.name: C.sha16(p) for p in sorted(OUT.glob("site1b_oof_*.pt"))},
        "gru_ck": {p.name: C.sha16(p) for p in sorted((OUT / "checkpoints").glob("*.pt"))},
        "scaler_fp": fp,
        "gamma_q": gammas,
        "seeds": C.SEEDS,
        "metrics": ["PICP", "PINAW", "IS", "NIS(=/23.4)", "RCE1D", "RCE3D",
                    "LongestMissStreak", "RMSE/nRMSE", "MAE/nMAE"],
        "bootstrap": "day-block paired, 2000 resamples, A3 vs A0/A2/CQR on IS/NIS/RCE1D",
        "cross_definition": "2025 calendar year, available valid windows (coverage reported)",
        "asserts": ["select_end < calib_start", "calib_end < same_start", "same_end < cross_start",
                    "no 2025 in training/scaler/selection/calibration",
                    "train_end < oof forecast start (every fold)",
                    "capacity == 23.4 kW", "input_dim == 5", "history == 72", "horizon == 3"],
    }, indent=2, default=str), encoding="utf-8")
    print("phase 5+6+7 done; FREEZE_MANIFEST_SITE1B.json written")


if __name__ == "__main__":
    main()
