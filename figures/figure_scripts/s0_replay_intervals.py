"""Stage 9.5 STEP 0: replay the frozen A0/A1/A2/A3/CQR intervals on the 2025
cross-year span from FROZEN components only (no training, no selection).

Everything reproduces results/rigorous_v2/s7_final_eval exactly:
  y / mu_phys / sigma(T3p)  <- per_sample_cross_T3p_seed{S}.parquet (s7 output)
  q_abs, cal_abs_sorted     <- CALIB-span replay of frozen s2 mean + s6 GRU(T3p)
  q_norm, cal_norm_sorted   <- ditto, normalized scores
  gamma_A1, gamma_A3        <- s6_gru_select/gamma_selection.json (frozen)
  A0 = mu +/- q_abs;  A2 = mu +/- q_norm*sigma
  A1 = run_aci(abs);  A3 = run_aci(normalized)
  CQR lo/up <- frozen site18_cqr_seed{S}.pt checkpoints (p8b recipe)

Guard: replayed PICP/PINAW/IS/RCE/streaks must match the frozen CSVs to <1e-6;
any mismatch aborts (this script may not create new numbers).
Outputs -> figure_source_data/intervals_cross_seed{S}.parquet (per-sample
origins/y/mu/sigma + lo/up per method) and replay_verification.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]  # release: repo-relative
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "figures" / "figure_scripts"))

from experiments.rigorous_v2 import common_v2 as C  # noqa: E402
from src.models.multiscale_tcn import MultiScaleTCN  # noqa: E402

S2 = C.V2 / "s2_final_mean"
S6 = C.V2 / "s6_gru_select"
S7 = C.V2 / "s7_final_eval"
CQR_DIR = S7 / "site18_cqr"
OUT = ROOT / "figures" / "figure_source_data"

TOL = 1e-6


def load_cqr_quantiles(seed: int, sc: dict):
    import importlib.util
    spec = importlib.util.spec_from_file_location("p8b", ROOT / "experiments" / "site1b_validation" / "p8b_site18_cqr.py")
    p8b = importlib.util.module_from_spec(spec)
    sys.modules["p8b_fig"] = p8b
    spec.loader.exec_module(p8b)
    model = p8b.CQRNet().to(C.DEVICE)
    model.load_state_dict(torch.load(CQR_DIR / f"site18_cqr_seed{seed}.pt",
                                     weights_only=True)["state_dict"])
    model.eval()
    return model, p8b.quantiles


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    winner = "T3p"
    decision = json.loads((S6 / "selection_decision.json").read_text())
    assert decision["winner"] == winner
    gammas = json.loads((S6 / "gamma_selection.json").read_text())
    sc_raw = json.loads((S2 / "scaler.json").read_text())
    sc = {"mean": np.array(sc_raw["mean"], float), "std": np.array(sc_raw["std"], float),
          "target_mean": float(sc_raw["target_mean"]), "target_std": float(sc_raw["target_std"])}

    y2024 = C.load_year_windows(2024)
    calib = C.sub(y2024, C.span_mask(y2024["origins"], *C.CALIB))
    y2025 = C.load_year_windows(2025)

    frozen = pd.read_csv(S7 / "metrics_per_seed.csv")
    frozen_full = pd.read_csv(S7 / "site18_a0a1_full.csv")
    frozen_cqr = pd.read_csv(CQR_DIR / "site18_cqr_per_seed.csv")

    verify: dict[str, dict] = {}
    for seed in C.SEEDS:
        ps = pd.read_parquet(S7 / f"per_sample_cross_{winner}_seed{seed}.parquet")
        assert len(ps) == len(y2025["y_kw"])
        assert (pd.DatetimeIndex(ps["origin"]) == pd.DatetimeIndex(y2025["origins"])).all()
        y = ps["y_kw"].to_numpy(float)
        mu = ps["mu_phys"].to_numpy(float)
        sg = ps["sigma"].to_numpy(float)
        o = pd.DatetimeIndex(ps["origin"])

        # --- CALIB replay for the frozen quantile grids (identical to s7) ---
        model_m = C.MultiScaleTCN(dropout=0.1).to(C.DEVICE)
        model_m.load_state_dict(torch.load(S2 / f"final_mean_seed{seed}.pt",
                                           weights_only=True)["state_dict"])
        model_m.eval()
        mu_cal = C.mu_phys(C.tcn_mu(model_m, calib["x_raw"], sc).astype(np.float32).astype(np.float64))
        del model_m
        ck = torch.load(S6 / "checkpoints" / f"gru_{winner}_seed{seed}.pt", weights_only=True)
        assert ck["scaler_fp"] == C.scaler_fingerprint(sc), "scaler drift vs s6 checkpoint"
        model_g = C.ScaleModel("GRU").to(C.DEVICE)
        model_g.load_state_dict(ck["state_dict"])
        model_g.eval()
        sg_cal = C.sigma_kw(model_g, calib["x_raw"], sc)
        del model_g
        torch.cuda.empty_cache()
        cal_abs_sorted = np.sort(np.abs(calib["y_kw"] - mu_cal))
        cal_norm_sorted = np.sort(np.abs(calib["y_kw"] - mu_cal) / (sg_cal + C.EPS))
        q_abs = C.conformal_q(cal_abs_sorted)
        q_norm = C.conformal_q(cal_norm_sorted)

        # --- intervals ---
        lo0, up0 = mu - q_abs, mu + q_abs
        lo2, up2 = mu - q_norm * sg, mu + q_norm * sg
        g1 = gammas[str(seed)]["gamma_A1"]
        g3 = gammas[str(seed)]["gamma_A3"]
        lo1, up1, _, _ = C.run_aci(y, mu, np.ones(len(sg)), cal_abs_sorted, o, g1)
        lo3, up3, _, _ = C.run_aci(y, mu, sg, cal_norm_sorted, o, g3)

        model_q, qfun = load_cqr_quantiles(seed, sc)
        qc = qfun(model_q, y2025["x_raw"], sc)
        qhat = C.conformal_q(np.sort(np.maximum(
            qfun(model_q, calib["x_raw"], sc)[:, 0] - calib["y_kw"],
            calib["y_kw"] - qfun(model_q, calib["x_raw"], sc)[:, 2])))
        del model_q
        torch.cuda.empty_cache()
        loq, upq = qc[:, 0] - qhat, qc[:, 2] + qhat

        cols = {"origin": o, "y_kw": y, "mu_phys": mu, "sigma": sg,
                "A0_lo": lo0, "A0_up": up0, "A1_lo": lo1, "A1_up": up1,
                "A2_lo": lo2, "A2_up": up2, "A3_lo": lo3, "A3_up": up3,
                "CQR_lo": loq, "CQR_up": upq, "qhat_CQR": qhat,
                "q_abs": q_abs, "q_norm": q_norm}
        pd.DataFrame(cols).to_parquet(OUT / f"intervals_cross_seed{seed}.parquet", index=False)

        # --- verification against frozen CSVs ---
        def mrow(variant: str) -> pd.Series:
            r = frozen[(frozen["variant"] == variant) & (frozen["seed"] == seed)
                       & (frozen["span"] == "cross")]
            assert len(r) == 1, (variant, seed)
            return r.iloc[0]

        v: dict[str, float] = {}
        for nm, lo, up, variant, pfx in (
                ("A0", lo0, up0, "T3p:A0", "A0"), ("A1", lo1, up1, "T3p:A1", "A1"),
                ("A2", lo2, up2, "T3p", "A2"), ("A3", lo3, up3, "T3p:A3", "A3")):
            im = C.interval_metrics(y, lo, up, o)
            fr = mrow(variant)
            for k in ("PICP", "PINAW", "IS", "RCE1D", "RCE3D", "LongestMissStreak"):
                fk = f"{pfx}_{k}"
                if fk in fr and not pd.isna(fr[fk]):
                    v[f"{nm}_{k}_diff"] = abs(im[k] - float(fr[fk]))
            if nm in ("A0", "A1", "A3"):
                ff = frozen_full[(frozen_full["method"] == nm) & (frozen_full["seed"] == seed)]
                assert len(ff) == 1
                for k in ("PICP", "PINAW", "IS", "RCE1D", "RCE3D"):
                    v[f"{nm}_{k}_diff_vs_full"] = abs(im[k] - float(ff.iloc[0][k]))
        # conditional (A3) + A2 scale cols
        states = C.state_split(y2025["N_group"], y2025["hard_DF"])
        dl = y2025["x_raw"][:, -1, 1] > C.DAYLIGHT_GHI_THRESHOLD
        fr3 = mrow("T3p:A3")
        v["A3_PICP_HARD_DF_diff"] = abs(float(((y[states['HARD_DF']] >= lo3[states['HARD_DF']]) &
                                               (y[states['HARD_DF']] <= up3[states['HARD_DF']])).mean())
                                        - float(fr3["A3_PICP_HARD_DF"]))
        v["A3_PICP_daylight_diff"] = abs(float(((y[dl] >= lo3[dl]) & (y[dl] <= up3[dl])).mean())
                                         - float(fr3["A3_PICP_daylight"]))
        # CQR
        covq = (y >= loq) & (y <= upq)
        d1 = pd.Series(covq.astype(float)).groupby(o.normalize()).mean()
        fc = frozen_cqr[(frozen_cqr["seed"] == seed) & (frozen_cqr["span"] == "cross")].iloc[0]
        v["CQR_PICP_diff"] = abs(float(covq.mean()) - float(fc["PICP"]))
        v["CQR_PINAW_diff"] = abs(float(np.maximum(upq - loq, 0).mean() / C.P_RATED) - float(fc["PINAW"]))
        v["CQR_IS_diff"] = abs(float(C.interval_score(y, loq, upq).mean()) - float(fc["IS"]))
        v["CQR_RCE1D_diff"] = abs(float(np.abs(d1 - 0.9).mean()) - float(fc["RCE1D"]))
        v["CQR_qhat_diff"] = abs(float(qhat) - float(fc["qhat"]))
        verify[str(seed)] = {k: float(x) for k, x in v.items()}
        worst = max(verify[str(seed)].items(), key=lambda kv: kv[1])
        print(f"seed {seed}: worst diff {worst[0]} = {worst[1]:.3e}")
        assert worst[1] < TOL, f"seed {seed}: replay mismatch {worst}"

    (OUT / "replay_verification.json").write_text(json.dumps({
        "tolerance": TOL, "passed": True, "per_seed_max_diff": verify,
        "note": "replayed from frozen checkpoints + s7 parquet + s6 gamma; "
                "matches metrics_per_seed.csv / site18_a0a1_full.csv / "
                "site18_cqr_per_seed.csv to <1e-6"}, indent=2) + "\n")
    print("REPLAY VERIFIED <1e-6 for all seeds")


if __name__ == "__main__":
    main()
