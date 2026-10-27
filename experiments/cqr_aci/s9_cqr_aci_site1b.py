"""Site 1B -- CQR + delayed ACI baseline, strictly controlled (Site 18 analog).

Gate order:
  1. REPRODUCE the frozen fixed-CQR metrics from the frozen checkpoints
     (gate: must match results/site1b_val/site1b_eval_per_seed.csv, span=cross).
  2. Only then run CQR-ACI: frozen CQR quantile model + frozen CALIB score set;
     delayed ACI adjusts only the quantile level; same 3-step matured queue,
     alpha bounds and gamma grid/selection criterion as A1/A3.
Artifacts -> results/site1b_val/cqr_aci/
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]  # release: repo-relative (was absolute path)
sys.path.insert(0, str(ROOT))
from experiments.site1b_validation import common_1b as C  # noqa: E402

V2 = C.V2
OUT = ROOT / "results" / "site1b_val" / "cqr_aci"
CQR_CK = ROOT / "results" / "site1b_val" / "site1b_cqr_seed{seed}.pt"
P1B = C.P_RATED_1B

from src.models.multiscale_tcn import MultiScaleTCN  # noqa: E402


class CQRNet(torch.nn.Module):
    """Same wrapper as common_1b.train_tcn_cqr (trunk + 3-quantile head)."""

    def __init__(self):
        super().__init__()
        self.trunk = MultiScaleTCN(dropout=0.1)
        self.head = torch.nn.Linear(64, 3)

    def forward(self, x):
        return self.head(self.trunk.forward_features(x))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_aci_cqr(q05, q95, y, cal_scores_sorted, origins, gamma, alpha0: float = V2.ALPHA):
    """Delayed ACI on the CQR quantile pair; identical queue/bounds to V2.run_aci."""
    import math
    n = len(y)
    k_fn = lambda a: min(max(int(math.ceil((len(cal_scores_sorted) + 1) * (1 - a))), 1),
                         len(cal_scores_sorted))
    lower = np.empty(n)
    upper = np.empty(n)
    alphas = np.empty(n)
    qs = np.empty(n)
    alpha = alpha0
    queue: list[tuple[pd.Timestamp, float]] = []
    qi = 0
    last_update = pd.Timestamp("1900-01-01")
    o_all = pd.DatetimeIndex(origins)
    for i in range(n):
        o = o_all[i]
        while qi < len(queue) and queue[qi][0] <= o:
            alpha = min(max(alpha + gamma * (V2.ALPHA - queue[qi][1]),
                            V2.ALPHA_MIN), V2.ALPHA_MAX)
            last_update = queue[qi][0]
            qi += 1
        assert last_update <= o, "ACI update from the future detected"
        q = cal_scores_sorted[k_fn(alpha) - 1]
        lower[i], upper[i] = q05[i] - q, q95[i] + q
        alphas[i], qs[i] = alpha, q
        e = 0.0 if (y[i] >= lower[i] and y[i] <= upper[i]) else 1.0
        queue.append((o + pd.Timedelta(minutes=V2.ACI_DELAY_MIN), e))
    return lower, upper, alphas, qs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    freeze = json.loads((C.OUT / "FREEZE_MANIFEST_SITE1B.json").read_text())
    gq = json.loads((C.OUT / "site1b_gamma_q.json").read_text())
    sc_raw = json.loads((C.OUT / "site1b_mean_scaler.json").read_text())
    sc = {"mean": np.array(sc_raw["mean"], float), "std": np.array(sc_raw["std"], float),
          "target_mean": float(sc_raw["target_mean"]), "target_std": float(sc_raw["target_std"])}

    C.open_2025()
    g24 = C.load_year_grid(2024)
    w24 = C.windows(g24)
    cal = {k: v[C.span_on_origins(w24, *V2.CALIB)] for k, v in w24.items()}
    sel = {k: v[C.span_on_origins(w24, *V2.SELECT)] for k, v in w24.items()}
    same = {k: v[C.span_on_origins(w24, *V2.SAME)] for k, v in w24.items()}
    g25 = C.load_year_grid(2025)
    w25 = C.windows(g25)
    y25, o25 = w25["y_kw"], pd.DatetimeIndex(w25["origins"])
    mu_cache = np.load(C.OUT / "site1b_mean_cache.npz")

    frozen = pd.read_csv(C.OUT / "site1b_eval_per_seed.csv")
    repro_diffs = []
    rows, boot, store = [], [], {}
    gamma_sel = {}
    for seed in C.SEEDS:
        g = gq[str(seed)]
        cqrm = CQRNet().to(V2.DEVICE)
        cqrm.load_state_dict(torch.load(C.OUT / f"site1b_cqr_seed{seed}.pt",
                                        weights_only=True)["state_dict"])
        cqrm.eval()

        q_cal = C.cqr_quantiles(cqrm, cal["x_raw"], sc)
        cal_scores = np.maximum(q_cal[:, 0] - cal["y_kw"], cal["y_kw"] - q_cal[:, 2])
        cal_sorted = np.sort(cal_scores)
        qhat = C.cqr_qhat(cal["y_kw"], q_cal)

        q_sel = C.cqr_quantiles(cqrm, sel["x_raw"], sc)
        q_same = C.cqr_quantiles(cqrm, same["x_raw"], sc)
        q_cross = C.cqr_quantiles(cqrm, w25["x_raw"], sc)

        # ---------- STEP 1: reproduce frozen fixed CQR ----------
        repro = {}
        for tag, part, q in (("same", same, q_same), ("cross", w25, q_cross)):
            lo, up = q[:, 0] - qhat, q[:, 2] + qhat
            r = C.interval_metrics_1b(part["y_kw"], lo, up,
                                      pd.DatetimeIndex(part["origins"]))
            repro[tag] = r | {"qhat": float(qhat)}
        for tag in ("same", "cross"):
            ref = frozen[(frozen["method"] == "CQR") & (frozen["seed"] == seed)
                         & (frozen["span"] == tag)].iloc[0]
            for k in ("PICP", "PINAW", "IS", "NIS", "RCE1D", "RCE3D", "LongestMissStreak"):
                repro_diffs.append(abs(float(repro[tag][k]) - float(ref[k])))
        max_diff = max(repro_diffs)
        print(f"seed {seed}: fixed-CQR reproduction max|diff| = {max_diff:.3e}")
        assert max_diff < 1e-6, f"CQR reproduction FAILED for seed {seed} (max diff {max_diff})"

        # ---------- STEP 2: gamma selection on SELECT (same criterion as A1/A3) ----------
        cands = []
        for gamma in V2.GAMMA_GRID:
            lo, up, _, _ = run_aci_cqr(q_sel[:, 0], q_sel[:, 2], sel["y_kw"],
                                       cal_sorted, sel["origins"], gamma)
            cov = float(((sel["y_kw"] >= lo) & (sel["y_kw"] <= up)).mean())
            isv = float(V2.interval_score(sel["y_kw"], lo, up).mean())
            cands.append((abs(cov - 0.90), isv, gamma))
        feas = [c for c in cands if c[0] <= 0.02]
        best = (min(feas, key=lambda c: (c[1], c[2])) if feas
                else min(cands, key=lambda c: (c[0], c[2])))
        gamma_sel[seed] = best[2]
        print(f"seed {seed}: selected gamma(CQR-ACI) = {best[2]}")

        # ---------- STEP 3: CROSS with delayed ACI ----------
        lo, up, alphas, qs_t = run_aci_cqr(q_cross[:, 0], q_cross[:, 2], y25,
                                           cal_sorted, o25, gamma_sel[seed])
        r = C.interval_metrics_1b(y25, lo, up, o25)
        rows.append({"method": "CQR-ACI", "seed": seed, "span": "cross", **r})
        store[("CQR-ACI", seed)] = (lo, up)
        det = pd.DataFrame({
            "target_time": o25 + pd.Timedelta(minutes=15),
            "y_true": y25, "q05_raw": q_cross[:, 0], "q50_raw": q_cross[:, 1],
            "q95_raw": q_cross[:, 2], "alpha_t": alphas, "q_t_cqr": qs_t,
            "lower": lo, "upper": up,
            "covered": ((y25 >= lo) & (y25 <= up)).astype(int),
            "gamma": gamma_sel[seed]})
        det.to_csv(OUT / f"cqr_aci_intervals_seed{seed}.csv", index=False)
        pd.DataFrame({"cqr_score_calib": cal_scores}).to_csv(
            OUT / f"cqr_calib_scores_seed{seed}.csv", index=False)

        # ---------- A3 (frozen protocol) and fixed CQR intervals ----------
        model = V2.MultiScaleTCN(dropout=0.1).to(V2.DEVICE)
        model.load_state_dict(torch.load(C.OUT / f"site1b_mean_checkpoint_seed{seed}.pt",
                                         weights_only=True)["state_dict"])
        model.eval()
        mu = C.mu_phys(V2.tcn_mu(model, w25["x_raw"], sc))
        gru = V2.ScaleModel("GRU").to(V2.DEVICE)
        gru.load_state_dict(torch.load(C.OUT / "checkpoints" / f"site1b_gru_T3p_seed{seed}.pt",
                                       weights_only=True)["state_dict"])
        gru.eval()
        sg = V2.sigma_kw(gru, w25["x_raw"], sc)
        mu_cal = C.mu_phys(mu_cache[f"calib|{seed}"].astype(np.float64))
        sg_cal = V2.sigma_kw(gru, cal["x_raw"], sc)
        cal_norm_sorted = np.sort(np.abs(cal["y_kw"] - mu_cal) / (sg_cal + C.EPS))
        la3, ua3, _, _ = V2.run_aci(y25, mu, sg, cal_norm_sorted, o25, g["gamma_A3"])
        a3_chk = C.interval_metrics_1b(y25, la3, ua3, o25)
        ref = frozen[(frozen["method"] == "A3") & (frozen["seed"] == seed)
                     & (frozen["span"] == "cross")].iloc[0]
        for k in ("PICP", "PINAW", "IS", "NIS", "RCE1D", "RCE3D"):
            assert abs(float(a3_chk[k]) - float(ref[k])) < 1e-9, (k, a3_chk[k], float(ref[k]))
        store[("A3", seed)] = (la3, ua3)
        store[("CQR", seed)] = (q_cross[:, 0] - qhat, q_cross[:, 2] + qhat)
        del cqrm, gru
        torch.cuda.empty_cache()

    max_repro = max(repro_diffs)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "cqr_aci_per_seed.csv", index=False)

    # ---------- bootstrap: A3 - CQR-ACI, CQR-ACI - CQR ----------
    def mk_is():
        def f(lo, up, yy, cc):
            return float(V2.interval_score(yy, lo, up).mean())
        return f

    def mk_nis():
        def f(lo, up, yy, cc):
            return float(V2.interval_score(yy, lo, up).mean()) / P1B
        return f

    def mk_rce1d():
        def f(lo, up, yy, cc):
            cov = ((yy >= lo) & (yy <= up)).astype(float)
            d1 = pd.Series(cov).groupby(cc).mean()
            return float(np.abs(d1 - 0.9).mean())
        return f

    uniq, code = np.unique(o25.normalize(), return_inverse=True)
    for seed in C.SEEDS:
        for (name_b, name_a) in (("A3", "CQR-ACI"), ("CQR-ACI", "CQR")):
            lb, ub = store[(name_b, seed)]
            la, ua = store[(name_a, seed)]
            for metric, mk in (("IS", mk_is), ("NIS", mk_nis), ("RCE1D", mk_rce1d)):
                res = V2.paired_day_bootstrap(mk(), mk(), o25,
                                              [la, ua, y25, code],
                                              [lb, ub, y25, code], seed=seed)
                boot.append({"pair": f"{name_b}-vs-{name_a}", "seed": seed,
                             "metric": metric, **res})
    bd = pd.DataFrame(boot)
    bd.to_csv(OUT / "cqr_aci_bootstrap.csv", index=False)

    summary = {}
    for pair in bd["pair"].unique():
        for metric in ("IS", "NIS", "RCE1D"):
            sub = bd[(bd["pair"] == pair) & (bd["metric"] == metric)]
            sig = int(((sub["ci_lower"] > 0) & (sub["ci_upper"] > 0) |
                       (sub["ci_lower"] < 0) & (sub["ci_upper"] < 0)).sum())
            summary[f"{pair}|{metric}"] = {
                "seed_median_diff": float(sub["median_diff"].median()),
                "ci_lower": float(sub["ci_lower"].median()),
                "ci_upper": float(sub["ci_upper"].median()),
                "seeds_ci_excl_0": sig,
            }
    (OUT / "metrics.json").write_text(json.dumps({
        "cqr_aci_per_seed": df.to_dict(orient="records"),
        "bootstrap_summary": summary,
        "fixed_cqr_reproduction_max_abs_diff": float(max_repro),
        "gamma_selection": {str(k): v for k, v in gamma_sel.items()},
    }, indent=2))

    manifest = {
        "cqr_checkpoints": {Path(f"site1b_cqr_seed{s}.pt").name:
                            sha256_file(C.OUT / f"site1b_cqr_seed{s}.pt") for s in C.SEEDS},
        "code_hash": sha256_file(Path(__file__)),
        "seeds": list(C.SEEDS),
        "selected_gamma_cqr_aci": {str(k): v for k, v in gamma_sel.items()},
        "gamma_grid": list(V2.GAMMA_GRID),
        "gamma_selection_criterion": "|PICP-0.90|<=0.02 feasible on SELECT, then min IS, then min gamma (same as A1/A3)",
        "calib_period": list(V2.CALIB),
        "n_calibration_scores": int(len(cal["y_kw"])),
        "feedback_delay_steps": 3,
        "feedback_delay_min": V2.ACI_DELAY_MIN,
        "quantile_rule": "empirical (n+1)-quantile, ceil((n+1)(1-alpha)), clipped to [1,n]; negative q allowed",
        "projection": "none (inherits final CQR convention)",
        "alpha_bounds": [V2.ALPHA_MIN, V2.ALPHA_MAX],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(df.groupby("method")[["PICP", "PINAW", "IS", "NIS", "RCE1D", "RCE3D"]].median().round(4).to_string())
    print(json.dumps(summary, indent=2))
    print("fixed-CQR reproduction max abs diff:", max_repro)


if __name__ == "__main__":
    main()
