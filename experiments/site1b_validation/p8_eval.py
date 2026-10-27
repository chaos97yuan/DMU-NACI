"""Site 1B Phase 8+9: open 2025 (only after freeze) and run A0/A1/A2/A3/CQR
with day-block paired bootstrap. Report-only; no selection anywhere.

Methods (all 5 matched seeds, gamma/q frozen from p567):
  A0  absolute fixed conformal        [q_abs from CALIB]
  A1  absolute delayed ACI            [gamma_A1 from SELECT]
  A2  normalized fixed conformal      [q_norm from CALIB]
  A3  normalized delayed ACI = DMU-NACI
  CQR canonical conformalized quantile regression (0.05/0.5/0.95 pinball TCN,
      same trunk/optimizer/ES protocol, conformalized on CALIB)
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
    freeze = OUT / "FREEZE_MANIFEST_SITE1B.json"
    assert freeze.exists(), "run p567 first (freeze manifest missing)"
    gq = json.loads((OUT / "site1b_gamma_q.json").read_text())
    sc_raw = json.loads((OUT / "site1b_mean_scaler.json").read_text())
    sc = {"mean": np.array(sc_raw["mean"], float), "std": np.array(sc_raw["std"], float),
          "target_mean": float(sc_raw["target_mean"]), "target_std": float(sc_raw["target_std"])}

    C.open_2025()
    g24 = C.load_year_grid(2024)
    w24 = C.windows(g24)
    same = {k: v[C.span_on_origins(w24, *V2.SAME)] for k, v in w24.items()}
    cal = {k: v[C.span_on_origins(w24, *V2.CALIB)] for k, v in w24.items()}
    g25 = C.load_year_grid(2025)
    w25 = C.windows(g25)
    exp25 = len(pd.date_range("2025-01-01", "2025-12-31 23:55", freq="5min")) - C.HISTORY + 1
    print(f"2025 valid windows {len(w25['y_kw'])} / {exp25} = {len(w25['y_kw'])/exp25*100:.2f}%")

    mu_cache = np.load(OUT / "site1b_mean_cache.npz")
    rows, boot = [], []
    day_code = None
    store = {}   # (method, seed, span) -> (lo, up)

    for seed in C.SEEDS:
        g = gq[str(seed)]
        model = C.V2.MultiScaleTCN(dropout=0.1).to(C.V2.DEVICE)
        model.load_state_dict(torch.load(OUT / f"site1b_mean_checkpoint_seed{seed}.pt",
                                         weights_only=True)["state_dict"])
        model.eval()
        gru = C.V2.ScaleModel("GRU").to(C.V2.DEVICE)
        gru.load_state_dict(torch.load(OUT / "checkpoints" / f"site1b_gru_T3p_seed{seed}.pt",
                                       weights_only=True)["state_dict"])
        gru.eval()
        # CQR model (trained once per seed on the same grad rows / ES protocol)
        cqr_ck = OUT / f"site1b_cqr_seed{seed}.pt"
        m_grad = C.span_on_targets(w24, *V2.MEAN_GRAD)
        m_es = C.span_on_targets(w24, *V2.MEAN_ES)
        grad_rows = {k: v[m_grad] for k, v in w24.items()}
        es_rows = {k: v[m_es] for k, v in w24.items()}
        if cqr_ck.exists():
            # Release fix (see CHANGELOG.md): the checkpoint stores the CQRNet
            # state_dict (trunk.* / head.* keys, exactly what common_1b
            # .train_tcn_cqr saves), so it must be loaded into the same class.
            # The original code constructed a bare MultiScaleTCN here; that path
            # never executed in the original project (checkpoints were absent on
            # the archived run), and it crashes on any re-run. No behaviour
            # change for the archived results.
            class _CQRNet(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.trunk = C.V2.MultiScaleTCN(dropout=0.1)
                    self.head = torch.nn.Linear(64, 3)

                def forward(self, x):
                    return self.head(self.trunk.forward_features(x))

            cqrm = _CQRNet().to(C.V2.DEVICE)
            cqrm.load_state_dict(torch.load(cqr_ck, weights_only=True)["state_dict"])
            cqrm.eval()
        else:
            cqrm, info = C.train_tcn_cqr(grad_rows, es_rows, sc, seed)
            torch.save({"state_dict": cqrm.state_dict(), "seed": seed, "info": info}, cqr_ck)
        qhat = C.cqr_qhat(cal["y_kw"], C.cqr_quantiles(cqrm, cal["x_raw"], sc))

        for span, w in (("same", same), ("cross", w25)):
            mu = C.mu_phys(V2.tcn_mu(model, w["x_raw"], sc))
            sg = V2.sigma_kw(gru, w["x_raw"], sc)
            y, o = w["y_kw"], pd.DatetimeIndex(w["origins"])
            # initial score grids MUST come from CALIB (frozen); eval spans only
            # provide the online ACI feedback stream:
            mu_cal = C.mu_phys(mu_cache[f"calib|{seed}"].astype(np.float64))
            sg_cal = V2.sigma_kw(gru, cal["x_raw"], sc)
            cal_abs_sorted = np.sort(np.abs(cal["y_kw"] - mu_cal))
            cal_norm_sorted = np.sort(np.abs(cal["y_kw"] - mu_cal) / (sg_cal + C.EPS))
            q_abs, q_norm = g["q_abs"], g["q_norm"]
            assert abs(q_norm - V2.conformal_q(cal_norm_sorted)) < 1e-9
            methods = {
                "A0": (mu - q_abs, mu + q_abs),
                "A2": (mu - q_norm * sg, mu + q_norm * sg),
            }
            lo1, up1, _, _ = V2.run_aci(y, mu, np.ones(len(sg)), cal_abs_sorted, o, g["gamma_A1"])
            methods["A1"] = (lo1, up1)
            lo3, up3, _, _ = V2.run_aci(y, mu, sg, cal_norm_sorted, o, g["gamma_A3"])
            methods["A3"] = (lo3, up3)
            qc = C.cqr_quantiles(cqrm, w["x_raw"], sc)
            methods["CQR"] = (qc[:, 0] - qhat, qc[:, 2] + qhat)
            for name, (lo, up) in methods.items():
                r = {"method": name, "seed": seed, "span": span,
                     **C.interval_metrics_1b(y, lo, up, o)}
                if name in ("A0", "A1", "A3"):
                    e = y - mu
                    r["RMSE"] = float(np.sqrt(np.mean(e ** 2)))
                    r["MAE"] = float(np.mean(np.abs(e)))
                    r["nRMSE"] = r["RMSE"] / C.P_RATED_1B
                    r["nMAE"] = r["MAE"] / C.P_RATED_1B
                dl = w["origin_GHI"] > V2.DAYLIGHT_GHI_THRESHOLD
                cov = (y >= lo) & (y <= up)
                r["PICP_daylight"] = float(cov[dl].mean())
                isv = (up - lo) + (2 / V2.ALPHA) * np.maximum(lo - y, 0) \
                    + (2 / V2.ALPHA) * np.maximum(y - up, 0)
                r["IS_daylight"] = float(isv[dl].mean())
                rows.append(r)
                store[(name, seed, span)] = (lo, up)
            print(f"seed {seed} {span}: " + " | ".join(
                f"{n} PICP {next(r['PICP'] for r in rows if r['method']==n and r['seed']==seed and r['span']==span):.3f}"
                for n in ("A0","A2","A3","CQR")))
        del model, gru, cqrm
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "site1b_eval_per_seed.csv", index=False)
    med = df.groupby(["method", "span"])[["PICP", "PINAW", "IS", "NIS", "RCE1D", "RCE3D",
                                           "LongestMissStreak", "PICP_daylight"]].median()
    med.to_csv(OUT / "site1b_eval_median.csv")
    print(med.round(4).to_string())

    # ---- Phase 9: paired day-block bootstrap on CROSS ----
    w = w25
    y = w["y_kw"]
    o = pd.DatetimeIndex(w["origins"])
    uniq, code = np.unique(o.normalize(), return_inverse=True)
    for seed in C.SEEDS:
        for ref in ("A0", "A2", "CQR"):
            for metric in ("IS", "NIS", "RCE1D"):
                def mk(metric=metric):
                    if metric == "IS":
                        f = lambda lo, up, yy, cc: float(((up - lo) + (2/V2.ALPHA)*np.maximum(lo-yy,0)
                                                          + (2/V2.ALPHA)*np.maximum(yy-up,0)).mean())
                    elif metric == "NIS":
                        f = lambda lo, up, yy, cc: float((((up - lo) + (2/V2.ALPHA)*np.maximum(lo-yy,0)
                                                          + (2/V2.ALPHA)*np.maximum(yy-up,0)).mean())
                                                         / C.P_RATED_1B)
                    else:
                        def f(lo, up, yy, cc):
                            cov = ((yy >= lo) & (yy <= up)).astype(float)
                            d1 = pd.Series(cov).groupby(cc).mean()
                            return float(np.abs(d1 - 0.9).mean())
                    return f
                la, ua = store[("A3", seed, "cross")]
                lb, ub = store[(ref, seed, "cross")]
                # diff = metric_b - metric_a = A3 - ref
                res = V2.paired_day_bootstrap(mk(), mk(), o, [lb, ub, y, code],
                                              [la, ua, y, code], seed=seed)
                boot.append({"pair": f"A3-vs-{ref}", "seed": seed, "metric": metric, **res})
    bd = pd.DataFrame(boot)
    bd.to_csv(OUT / "site1b_bootstrap.csv", index=False)
    sm = bd.groupby(["pair", "metric"])[["mean_diff", "ci_lower", "ci_upper"]].median()
    print(sm.round(4).to_string())
    for pair in bd["pair"].unique():
        for metric in ("IS", "NIS", "RCE1D"):
            sub = bd[(bd["pair"] == pair) & (bd["metric"] == metric)]
            sig = int(((sub["ci_lower"] > 0) & (sub["ci_upper"] > 0) |
                       (sub["ci_lower"] < 0) & (sub["ci_upper"] < 0)).sum())
            print(f"{pair} {metric}: {sig}/5 seeds individually significant")

    (OUT / "phase89_manifest.json").write_text(json.dumps({
        "n_cross_windows": int(len(w25["y_kw"])),
        "cross_expected_windows": int(exp25),
        "cross_coverage_pct": round(len(w25["y_kw"]) / exp25 * 100, 2),
    }, indent=2), encoding="utf-8")
    print("phase 8+9 done")


if __name__ == "__main__":
    main()
