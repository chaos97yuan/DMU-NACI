"""rigorous_v2 STEP 9: CQR + delayed ACI baseline (Site 18), strictly controlled.

Gate order (per protocol):
  1. REPRODUCE the frozen fixed-CQR metrics exactly from the frozen checkpoints
     (gate: must match results/rigorous_v2/s7_final_eval/site18_cqr_per_seed.csv).
  2. Only then run CQR-ACI: the CQR quantile model and its CALIB score
     distribution stay frozen; delayed ACI adjusts ONLY the quantile level
     (1 - alpha_t) of that fixed score set, with the same 3-step matured
     feedback queue, alpha bounds and gamma grid/selection criterion as A1/A3.

No retraining, no new gamma grid, no 2025 usage outside matured feedback.
Artifacts -> results/rigorous_v2/s9_cqr_aci/
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]  # release: repo-relative (was absolute path)
sys.path.insert(0, str(ROOT))
from experiments.rigorous_v2 import common_v2 as C  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "p8b", ROOT / "experiments" / "site1b_validation" / "p8b_site18_cqr.py")
p8b = importlib.util.module_from_spec(spec)
sys.modules["p8b"] = p8b
spec.loader.exec_module(p8b)

OUT = ROOT / "results" / "rigorous_v2" / "s9_cqr_aci"
CQR_DIR = ROOT / "results" / "rigorous_v2" / "s7_final_eval" / "site18_cqr"
S2 = ROOT / "results" / "rigorous_v2" / "s2_final_mean"
GRU_DIR = ROOT / "results" / "rigorous_v2" / "s6_gru_select" / "checkpoints"
P18 = 5.2
ALPHA_MIN, ALPHA_MAX = C.ALPHA_MIN, C.ALPHA_MAX


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_aci_cqr(q05, q95, y, cal_scores_sorted, origins, gamma, alpha0: float = C.ALPHA):
    """Delayed ACI on the CQR quantile pair: identical queue/bounds/semantics to
    common_v2.run_aci; only the interval construction differs (asymmetric CQR
    shift instead of mu +/- q*sigma). The CALIB score set is frozen."""
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
            alpha = min(max(alpha + gamma * (C.ALPHA - queue[qi][1]), ALPHA_MIN), ALPHA_MAX)
            last_update = queue[qi][0]
            qi += 1
        assert last_update <= o, "ACI update from the future detected"
        q = cal_scores_sorted[k_fn(alpha) - 1]
        lower[i], upper[i] = q05[i] - q, q95[i] + q
        alphas[i], qs[i] = alpha, q
        e = 0.0 if (y[i] >= lower[i] and y[i] <= upper[i]) else 1.0
        queue.append((o + pd.Timedelta(minutes=C.ACI_DELAY_MIN), e))
    return lower, upper, alphas, qs


def metrics_full(y, lo, up, origins, dl, highn, harddf):
    cov = (y >= lo) & (y <= up)
    o = pd.DatetimeIndex(origins)
    day = o.normalize()
    d1 = pd.Series(cov.astype(float)).groupby(day).mean()
    roll3 = d1.rolling(3, min_periods=1).mean()
    miss = ~cov
    streak = best = 0
    for m in miss:
        streak = streak + 1 if m else 0
        best = max(best, streak)
    isv = C.interval_score(y, lo, up)
    return {
        "PICP": float(cov.mean()),
        "PINAW": float(np.maximum(up - lo, 0).mean() / P18),
        "IS": float(isv.mean()),
        "NIS": float(isv.mean() / P18),
        "RCE1D": float(np.abs(d1 - (1 - C.ALPHA)).mean()),
        "RCE3D": float(np.abs(roll3 - (1 - C.ALPHA)).mean()),
        "LongestMissStreak": int(best),
        "PICP_daylight": float(cov[dl].mean()),
        "PICP_HIGH_N": float(cov[highn].mean()),
        "PICP_HARD_DF": float(cov[harddf].mean()),
        "HARD_DF_width_kW": float(np.maximum(up - lo, 0)[harddf].mean()),
        "HARD_DF_IS_kW": float(isv[harddf].mean()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sc_raw = json.loads((S2 / "scaler.json").read_text())
    sc = {"mean": np.array(sc_raw["mean"], float), "std": np.array(sc_raw["std"], float),
          "target_mean": float(sc_raw["target_mean"]), "target_std": float(sc_raw["target_std"])}
    y2024 = C.load_year_windows(2024)
    y2025 = C.load_year_windows(2025)
    y = x25 = None  # clarity only

    cal = {k: v[C.span_mask(y2024["origins"], *C.CALIB)] for k, v in y2024.items()
           if k in ("x_raw", "y_kw", "origins", "targets")}
    sel = {k: v[C.span_mask(y2024["origins"], *C.SELECT)] for k, v in y2024.items()
           if k in ("x_raw", "y_kw", "origins", "targets")}
    same = {k: v[C.span_mask(y2024["origins"], *C.SAME)] for k, v in y2024.items()
            if k in ("x_raw", "y_kw", "origins", "targets")}
    assert len(cal["y_kw"]) == 4601, f"CALIB score count {len(cal['y_kw'])} != 4601"

    o25 = pd.DatetimeIndex(y2025["origins"])
    dl25 = y2025["x_raw"][:, -1, 1] > C.DAYLIGHT_GHI_THRESHOLD
    ng25 = np.asarray(y2025["N_group"]).astype(str)
    hd25 = np.asarray(y2025["hard_DF"]).astype(bool)
    highn25 = ng25 == "HIGH_N"

    frozen = pd.read_csv(CQR_DIR / "site18_cqr_per_seed.csv")

    repro_diffs = []
    rows, boot, store, a3_all = [], [], {}, []
    gamma_sel = {}
    for seed in C.SEEDS:
        ck = CQR_DIR / f"site18_cqr_seed{seed}.pt"
        model = p8b.CQRNet().to(C.DEVICE)
        model.load_state_dict(torch.load(ck, weights_only=True)["state_dict"])
        model.eval()

        q_cal = p8b.quantiles(model, cal["x_raw"], sc)
        cal_scores = np.maximum(q_cal[:, 0] - cal["y_kw"], cal["y_kw"] - q_cal[:, 2])
        cal_sorted = np.sort(cal_scores)
        qhat = C.conformal_q(cal_sorted)

        q_sel = p8b.quantiles(model, sel["x_raw"], sc)
        q_same = p8b.quantiles(model, same["x_raw"], sc)
        q_cross = p8b.quantiles(model, y2025["x_raw"], sc)

        # ---------- STEP 1: reproduce frozen fixed CQR ----------
        repro = {}
        for tag, part, q in (("same", same, q_same), ("cross", y2025, q_cross)):
            lo, up = q[:, 0] - qhat, q[:, 2] + qhat
            cov = (part["y_kw"] >= lo) & (part["y_kw"] <= up)
            o = pd.DatetimeIndex(part["origins"])
            d1 = pd.Series(cov.astype(float)).groupby(o.normalize()).mean()
            isv = (up - lo) + 20 * np.maximum(lo - part["y_kw"], 0) + 20 * np.maximum(part["y_kw"] - up, 0)
            repro[tag] = {"PICP": float(cov.mean()),
                          "PINAW": float(np.maximum(up - lo, 0).mean() / P18),
                          "IS": float(isv.mean()), "NIS": float(isv.mean() / P18),
                          "RCE1D": float(np.abs(d1 - 0.9).mean()), "qhat": float(qhat)}
        for tag in ("same", "cross"):
            ref = frozen[(frozen["seed"] == seed) & (frozen["span"] == tag)].iloc[0]
            for k in ("PICP", "PINAW", "IS", "NIS", "RCE1D", "qhat"):
                repro_diffs.append(abs(repro[tag][k] - float(ref[k])))
        max_diff = max(repro_diffs)
        print(f"seed {seed}: fixed-CQR reproduction max|diff| = {max_diff:.3e}")
        assert max_diff < 1e-6, f"CQR reproduction FAILED for seed {seed} (max diff {max_diff})"

        # ---------- STEP 2: gamma selection on SELECT (same criterion as A1/A3) ----------
        cands = []
        for gamma in C.GAMMA_GRID:
            lo, up, _, _ = run_aci_cqr(q_sel[:, 0], q_sel[:, 2], sel["y_kw"],
                                       cal_sorted, sel["origins"], gamma)
            cov = float(((sel["y_kw"] >= lo) & (sel["y_kw"] <= up)).mean())
            isv = float(C.interval_score(sel["y_kw"], lo, up).mean())
            cands.append((abs(cov - 0.90), isv, gamma))
        feas = [c for c in cands if c[0] <= 0.02]
        best = (min(feas, key=lambda c: (c[1], c[2])) if feas
                else min(cands, key=lambda c: (c[0], c[2])))
        gamma_sel[seed] = best[2]
        print(f"seed {seed}: selected gamma(CQR-ACI) = {best[2]}")

        # ---------- STEP 3: CROSS with delayed ACI ----------
        lo, up, alphas, qs_t = run_aci_cqr(q_cross[:, 0], q_cross[:, 2], y2025["y_kw"],
                                           cal_sorted, y2025["origins"], gamma_sel[seed])
        m = metrics_full(y2025["y_kw"], lo, up, y2025["origins"], dl25, highn25, hd25)
        rows.append({"method": "CQR-ACI", "seed": seed, "span": "cross", **m})
        store[("CQR-ACI", seed)] = (lo, up)
        det = pd.DataFrame({
            "target_time": o25 + pd.Timedelta(minutes=15),
            "y_true": y2025["y_kw"], "q05_raw": q_cross[:, 0], "q50_raw": q_cross[:, 1],
            "q95_raw": q_cross[:, 2], "alpha_t": alphas, "q_t_cqr": qs_t,
            "lower": lo, "upper": up,
            "covered": ((y2025["y_kw"] >= lo) & (y2025["y_kw"] <= up)).astype(int),
            "gamma": gamma_sel[seed]})
        det.to_csv(OUT / f"cqr_aci_intervals_seed{seed}.csv", index=False)
        pd.DataFrame({"cqr_score_calib": cal_scores}).to_csv(
            OUT / f"cqr_calib_scores_seed{seed}.csv", index=False)

        # ---------- A3 per-sample intervals: frozen replay-verified arrays ----------
        rep = pd.read_parquet(ROOT / "results" / "stage9_5_v4_visual_completion"
                              / "figure_source_data" / f"intervals_cross_seed{seed}.parquet")
        assert np.allclose(rep["y_kw"].to_numpy(float), y2025["y_kw"], atol=1e-9)
        la3 = rep["A3_lo"].to_numpy(float)
        ua3 = rep["A3_up"].to_numpy(float)
        a3_chk = C.interval_metrics(y2025["y_kw"], la3, ua3, y2025["origins"])
        a3_all.append(a3_chk)
        store[("A3", seed)] = (la3, ua3)
        store[("CQR", seed)] = (q_cross[:, 0] - qhat, q_cross[:, 2] + qhat)
        del model
        torch.cuda.empty_cache()

    # A3 sanity: 5-seed medians of the frozen replay intervals must equal the frozen dictionary
    a3_med = pd.DataFrame(a3_all).median()
    assert abs(a3_med["PICP"] - 0.899775) < 5e-4 and abs(a3_med["IS"] - 0.654303) < 5e-4, a3_med
    print("A3 replay-interval medians match frozen dictionary: "
          f"PICP {a3_med['PICP']:.6f}, IS {a3_med['IS']:.6f}")

    max_repro = max(repro_diffs)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "cqr_aci_per_seed.csv", index=False)

    # ---------- bootstrap: A3 - CQR-ACI, CQR-ACI - CQR ----------
    def mk_is():
        def f(lo, up, yy, cc):
            return float(C.interval_score(yy, lo, up).mean())
        return f

    def mk_rce1d():
        def f(lo, up, yy, cc):
            cov = ((yy >= lo) & (yy <= up)).astype(float)
            d1 = pd.Series(cov).groupby(cc).mean()
            return float(np.abs(d1 - 0.9).mean())
        return f

    o25 = pd.DatetimeIndex(y2025["origins"])
    uniq, code = np.unique(o25.normalize(), return_inverse=True)
    for seed in C.SEEDS:
        for (name_b, name_a) in (("A3", "CQR-ACI"), ("CQR-ACI", "CQR")):
            lb, ub = store[(name_b, seed)]
            la, ua = store[(name_a, seed)]
            for metric, mk in (("IS", mk_is), ("RCE1D", mk_rce1d)):
                res = C.paired_day_bootstrap(mk(), mk(), o25,
                                             [la, ua, y2025["y_kw"], code],
                                             [lb, ub, y2025["y_kw"], code], seed=seed)
                boot.append({"pair": f"{name_b}-vs-{name_a}", "seed": seed,
                             "metric": metric, **res})
    bd = pd.DataFrame(boot)
    bd.to_csv(OUT / "cqr_aci_bootstrap.csv", index=False)

    summary = {}
    for pair in bd["pair"].unique():
        for metric in ("IS", "RCE1D"):
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
        "cqr_checkpoints": {p.name: sha256_file(p) for p in sorted(CQR_DIR.glob("*.pt"))},
        "code_hash": sha256_file(Path(__file__)),
        "seeds": list(C.SEEDS),
        "selected_gamma_cqr_aci": {str(k): v for k, v in gamma_sel.items()},
        "gamma_grid": list(C.GAMMA_GRID),
        "gamma_selection_criterion": "|PICP-0.90|<=0.02 feasible on SELECT, then min IS, then min gamma (same as A1/A3)",
        "calib_period": list(C.CALIB),
        "n_calibration_scores": 4601,
        "feedback_delay_steps": 3,
        "feedback_delay_min": C.ACI_DELAY_MIN,
        "quantile_rule": "empirical (n+1)-quantile, ceil((n+1)(1-alpha)), clipped to [1,n]; negative q allowed",
        "projection": "none (inherits final CQR convention)",
        "alpha_bounds": [C.ALPHA_MIN, C.ALPHA_MAX],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(df.groupby("method")[["PICP", "PINAW", "IS", "NIS", "RCE1D", "RCE3D",
                                "PICP_HARD_DF"]].median().round(4).to_string())
    print(json.dumps(summary, indent=2))
    print("fixed-CQR reproduction max abs diff:", max_repro)


if __name__ == "__main__":
    main()
