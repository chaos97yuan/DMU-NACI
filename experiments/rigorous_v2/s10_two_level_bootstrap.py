"""rigorous_v2 STEP 10: two-level (seed x day-block) bootstrap for the cross-seed
median paired differences quoted in the manuscript.

Why this exists: the frozen per-seed day-block bootstrap
(common_v2.paired_day_bootstrap) was summarised in the manuscript by the median
across seeds of the per-seed CI BOUNDS. That interval contains no seed-level
uncertainty: whenever per-seed CIs do not cross, it numerically equals the
median seed's own CI (verified: Site 1B A3-CQR IS -> seed 3407's CI; Site 18
A3-CQR IS -> seed 2026's CI). It must therefore not be read as a 95% CI for the
cross-seed median difference. This script recomputes every affected quoted CI
with a two-level bootstrap:

    outer level: resample the five seeds with replacement;
    inner level: within each picked seed, one day-block resample of the 2025
                 cross span (same picked days for both arms => paired);
    statistic :  median over the five picked seeds of the paired difference.

Reported point estimates are UNCHANGED (median of within-seed paired
differences); only the interval definition changes. Per-seed day-block CIs and
the k/5 seed-exclusion counts stay descriptive as before.

Per-sample intervals are deterministically replayed from frozen artifacts and
asserted against the frozen per-seed metrics (mean IS, 1e-9 for the exact-replay
arms, 1e-6 for checkpoint-replay arms) before any resampling:
  Site 18: A0/A2/A3 for the winner (s7 per-sample parquet + frozen CALIB scores
           + frozen gamma); T1p/T3match/T1in/V1/V3 A2-style arms (frozen GRU
           checkpoints); CQR and CQR-ACI (frozen s9 interval CSVs + qhat);
  Site 1B: A3 (frozen mean/GRU checkpoints + mu cache + frozen gamma),
           CQR and CQR-ACI (frozen s9 interval CSVs + qhat from the frozen
           CALIB score CSVs).

Outputs -> results/rigorous_v2/s10_two_level_bootstrap/
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.rigorous_v2 import common_v2 as C  # noqa: E402
from experiments.site1b_validation import common_1b as C1B  # noqa: E402

S2 = C.V2 / "s2_final_mean"
S6 = C.V2 / "s6_gru_select"
S7 = C.V2 / "s7_final_eval"
S9_18 = C.V2 / "s9_cqr_aci"
S9_1B = C1B.OUT / "cqr_aci"
OUT = C.V2 / "s10_two_level_bootstrap"

B_OUTER = 2000          # outer draws (matches the paper's quoted resample count)
RNG_SEED = 2026         # documented draw seed for the two-level bootstrap
POOL_ARMS = ["T1p", "T3match", "T1in", "V1", "V3"]    # s7 pool-pair arms (+ winner T3p)
METHOD_PAIRS = [("A3", "A0"), ("A3", "A2"), ("A3", "CQR"), ("A3", "CQR-ACI")]
POOL_PAIRS = [("A2", "T1p"), ("T3match", "T1p"), ("T1p", "T1in"), ("V3", "V1")]  # (b, a); "A2" = winner T3p under its A2-style fixed arm
METHOD_METRICS = ["IS", "RCE1D"]
POOL_METRICS = ["IS", "PICP", "abs_Std_z_minus_1", "Spearman"]


def load_s7_module():
    """Import s7_final_eval without running it (guarded main) to reuse load_gru."""
    return C.load_module("s7_mod_final_eval", "experiments/rigorous_v2/s7_final_eval.py")


def empty_cache() -> None:
    if C.DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def torch_load(path: Path):
    return torch.load(path, weights_only=True)


def is_values(y: np.ndarray, lo: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Per-row interval-score contribution (alpha = 0.10 frozen)."""
    alpha = C.ALPHA
    return ((up - lo) + (2.0 / alpha) * np.maximum(lo - y, 0.0)
            + (2.0 / alpha) * np.maximum(y - up, 0.0))


class DayStats:
    """Sufficient statistics of one arm on one seed for the resampled metrics.

    cnt is the per-day row count (identical across arms and seeds). IS/PICP and
    |Std(z)-1| follow from day sums; RCE1D needs per-day coverage means;
    Spearman needs row access (sig / abse), so only arms compared on Spearman
    carry sig.
    """

    def __init__(self, y, mu, dcode, n_days, cnt, lo=None, up=None, sig=None):
        self.cnt = cnt
        if lo is not None:
            self.isv_sum = np.bincount(dcode, weights=is_values(y, lo, up), minlength=n_days)
            cov = ((y >= lo) & (y <= up)).astype(float)
            self.cov_sum = np.bincount(dcode, weights=cov, minlength=n_days)
            self.cov_day = self.cov_sum / np.maximum(self.cnt, 1.0)
        if sig is not None:
            z = (y - mu) / (sig + C.EPS)
            self.z_sum = np.bincount(dcode, weights=z, minlength=n_days)
            self.z2_sum = np.bincount(dcode, weights=z * z, minlength=n_days)
            self.sig = sig
            self.abse = np.abs(y - mu)


def stat_full(stats: DayStats, metric: str) -> float:
    """Full-sample statistic (matching the frozen metric definitions)."""
    n = stats.cnt.sum()
    if metric == "IS":
        return float(stats.isv_sum.sum() / n)
    if metric == "PICP":
        return float(stats.cov_sum.sum() / n)
    if metric == "abs_Std_z_minus_1":
        m = stats.z_sum.sum() / n
        var = max(stats.z2_sum.sum() / n - m * m, 0.0)
        return float(abs(np.sqrt(var) - 1.0))
    if metric == "RCE1D":
        return float(np.abs(stats.cov_day - (1 - C.ALPHA)).mean())
    if metric == "Spearman":
        r1 = rankdata(stats.sig)
        r2 = rankdata(stats.abse)
        return float(np.corrcoef(r1, r2)[0, 1])
    raise KeyError(metric)


def _gather_sums(table: np.ndarray, sp: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """table (S,D); sp (B,S); dy (B,S,D) -> (B,S) per-pick day-sum gathers."""
    B, S = sp.shape
    out = np.empty((B, S))
    for j in range(S):
        rows = table[sp[:, j]]                       # (B, D)
        out[:, j] = np.take_along_axis(rows, dy[:, j, :], axis=1).sum(axis=1)
    return out


def two_level_boot(statsA, statsB, metric, dcode, n_days, day_rows, cnt_day,
                   b_outer=B_OUTER, rng_seed=RNG_SEED):
    """Draws (B,) of median_j[ statB_j(resample) - statA_j(resample) ].

    statsA / statsB: list over seeds (identical order) of DayStats. The same
    picked days are applied to both arms (paired), matching the frozen inner
    day-block scheme.
    """
    rng = np.random.default_rng(rng_seed)
    S = len(statsA)
    sp = rng.integers(0, S, size=(b_outer, S))
    dy = rng.integers(0, n_days, size=(b_outer, S, n_days))
    cnt_table = np.stack(cnt_day)

    if metric in ("IS", "PICP", "abs_Std_z_minus_1"):
        if metric == "IS":
            tA = np.stack([s.isv_sum for s in statsA])
            tB = np.stack([s.isv_sum for s in statsB])
            den = _gather_sums(cnt_table, sp, dy)
            per = (tA, tB)
        elif metric == "PICP":
            tA = np.stack([s.cov_sum for s in statsA])
            tB = np.stack([s.cov_sum for s in statsB])
            den = _gather_sums(cnt_table, sp, dy)
            per = (tA, tB)
        else:
            zsA = np.stack([s.z_sum for s in statsA]); z2A = np.stack([s.z2_sum for s in statsA])
            zsB = np.stack([s.z_sum for s in statsB]); z2B = np.stack([s.z2_sum for s in statsB])
            den = _gather_sums(cnt_table, sp, dy)

            def abstd(zs, z2):
                sz = _gather_sums(zs, sp, dy)
                sz2 = _gather_sums(z2, sp, dy)
                var = np.maximum(sz2 / den - (sz / den) ** 2, 0.0)
                return np.abs(np.sqrt(var) - 1.0)
            return np.median(abstd(zsB, z2B) - abstd(zsA, z2A), axis=1)
        gA = _gather_sums(per[0], sp, dy)
        gB = _gather_sums(per[1], sp, dy)
        return np.median(gB / den - gA / den, axis=1)

    diff = np.empty((b_outer, S))
    if metric == "RCE1D":
        covA = np.stack([s.cov_day for s in statsA])
        covB = np.stack([s.cov_day for s in statsB])
        devA = np.abs(covA - (1 - C.ALPHA))
        devB = np.abs(covB - (1 - C.ALPHA))
        for b in range(b_outer):
            for j in range(S):
                u = np.unique(dy[b, j])
                diff[b, j] = devB[sp[b, j], u].mean() - devA[sp[b, j], u].mean()
        return np.median(diff, axis=1)
    if metric == "Spearman":
        t0 = time.time()
        for b in range(b_outer):
            for j in range(S):
                s = sp[b, j]
                rows = np.concatenate([day_rows[s][d] for d in dy[b, j]])
                r1 = rankdata(statsA[s].sig[rows])
                r2 = rankdata(statsA[s].abse[rows])
                sa = np.corrcoef(r1, r2)[0, 1]
                r1 = rankdata(statsB[s].sig[rows])
                r2 = rankdata(statsB[s].abse[rows])
                sb = np.corrcoef(r1, r2)[0, 1]
                diff[b, j] = sb - sa
            if (b + 1) % 400 == 0:
                print(f"    {metric} draw {b + 1}/{b_outer} ({time.time() - t0:.0f}s)",
                      flush=True)
        return np.median(diff, axis=1)
    raise KeyError(metric)


def _day_grid(o: pd.DatetimeIndex):
    day = o.normalize()
    uniq, dcode = np.unique(day, return_inverse=True)
    n_days = len(uniq)
    cnt = np.bincount(dcode, minlength=n_days).astype(float)
    lo_rows = {d: np.flatnonzero(dcode == d) for d in range(n_days)}
    day_rows = [lo_rows for _ in C.SEEDS]     # identical day grid across seeds
    cnt_day = [cnt for _ in C.SEEDS]
    return dcode, n_days, day_rows, cnt_day


def build_site18():
    s7 = load_s7_module()
    sc_raw = json.loads((S2 / "scaler.json").read_text())
    sc = {"mean": np.array(sc_raw["mean"], float), "std": np.array(sc_raw["std"], float),
          "target_mean": float(sc_raw["target_mean"]), "target_std": float(sc_raw["target_std"])}
    gammas = json.loads((S6 / "gamma_selection.json").read_text())
    metrics = pd.read_csv(S7 / "metrics_per_seed.csv")
    cqr_ref = pd.read_csv(S7 / "site18_cqr" / "site18_cqr_per_seed.csv")
    cqraci_ref = pd.read_csv(S9_18 / "cqr_aci_per_seed.csv")

    y24 = C.load_year_windows(2024)
    calib = C.sub(y24, C.span_mask(y24["origins"], *C.CALIB))
    y25 = C.load_year_windows(2025)
    y, o = y25["y_kw"], pd.DatetimeIndex(y25["origins"])
    mu_npz = np.load(S7 / "mu_spans.npz")
    dcode, n_days, day_rows, cnt_day = _day_grid(o)

    arms = {name: {} for name in ["A0", "A2", "A3", "CQR", "CQR-ACI"] + POOL_ARMS}
    for seed in C.SEEDS:
        mu_cal = C.mu_phys(mu_npz[f"calib|{seed}"].astype(np.float64))
        mu_cr = C.mu_phys(mu_npz[f"cross|{seed}"].astype(np.float64))
        pq = pd.read_parquet(S7 / f"per_sample_cross_T3p_seed{seed}.parquet")
        assert len(pq) == len(y) and (pd.DatetimeIndex(pq["origin"]) == o).all()
        assert np.allclose(pq["y_kw"].to_numpy(float), y, atol=1e-8)
        sg_t3p = pq["sigma"].to_numpy(float)

        gru_t3p = s7.load_gru("T3p", seed, sc)
        assert np.allclose(C.sigma_kw(gru_t3p, y25["x_raw"], sc), sg_t3p, atol=1e-6), \
            f"T3p cross sigma replay mismatch (seed {seed})"
        sg_cal_t3p = C.sigma_kw(gru_t3p, calib["x_raw"], sc)
        del gru_t3p
        empty_cache()

        cal_abs = np.sort(np.abs(calib["y_kw"] - mu_cal))
        q_abs = C.conformal_q(cal_abs)
        caln_t3p = np.sort(np.abs(calib["y_kw"] - mu_cal) / (sg_cal_t3p + C.EPS))
        q_t3p = C.conformal_q(caln_t3p)

        a = {}
        a["A0"] = {"lo": mu_cr - q_abs, "up": mu_cr + q_abs, "sig": None}
        a["A2"] = {"lo": mu_cr - q_t3p * sg_t3p, "up": mu_cr + q_t3p * sg_t3p, "sig": sg_t3p}
        la3, ua3, _, _ = C.run_aci(y, mu_cr, sg_t3p, caln_t3p, o, gammas[str(seed)]["gamma_A3"])
        a["A3"] = {"lo": la3, "up": ua3, "sig": sg_t3p}

        for v in POOL_ARMS:
            model = s7.load_gru(v, seed, sc)
            sg_cal = C.sigma_kw(model, calib["x_raw"], sc)
            sg_cr = C.sigma_kw(model, y25["x_raw"], sc)
            del model
            empty_cache()
            q_v = C.conformal_q(np.sort(np.abs(calib["y_kw"] - mu_cal) / (sg_cal + C.EPS)))
            a[v] = {"lo": mu_cr - q_v * sg_cr, "up": mu_cr + q_v * sg_cr, "sig": sg_cr}
            ref = metrics[(metrics["variant"] == v) & (metrics["seed"] == seed)
                          & (metrics["span"] == "cross")]["A2_IS"].iloc[0]
            mine = float(C.interval_score(y, a[v]["lo"], a[v]["up"]).mean())
            assert abs(mine - float(ref)) < 1e-6, (v, seed, mine, ref)

        cqr_det = pd.read_csv(S9_18 / f"cqr_aci_intervals_seed{seed}.csv")
        assert len(cqr_det) == len(y) and np.allclose(cqr_det["y_true"].to_numpy(float), y)
        assert (pd.DatetimeIndex(cqr_det["target_time"]) == o + pd.Timedelta(minutes=15)).all()
        qhat = float(cqr_ref[(cqr_ref["seed"] == seed)
                             & (cqr_ref["span"] == "cross")]["qhat"].iloc[0])
        a["CQR"] = {"lo": cqr_det["q05_raw"].to_numpy(float) - qhat,
                    "up": cqr_det["q95_raw"].to_numpy(float) + qhat, "sig": None}
        a["CQR-ACI"] = {"lo": cqr_det["lower"].to_numpy(float),
                        "up": cqr_det["upper"].to_numpy(float), "sig": None}

        for var, col, arm, tol in (("T3p:A0", "A0_IS", "A0", 1e-9),
                                   ("T3p", "A2_IS", "A2", 1e-9),
                                   ("T3p:A3", "A3_IS", "A3", 1e-9)):
            ref = metrics[(metrics["variant"] == var) & (metrics["seed"] == seed)
                          & (metrics["span"] == "cross")][col].iloc[0]
            mine = float(C.interval_score(y, a[arm]["lo"], a[arm]["up"]).mean())
            assert abs(mine - float(ref)) < tol, (arm, seed, mine, ref)
        ref = cqr_ref[(cqr_ref["seed"] == seed) & (cqr_ref["span"] == "cross")]["IS"].iloc[0]
        assert abs(float(C.interval_score(y, a["CQR"]["lo"], a["CQR"]["up"]).mean())
                   - float(ref)) < 1e-6, ("CQR", seed)
        ref = cqraci_ref[(cqraci_ref["method"] == "CQR-ACI") & (cqraci_ref["seed"] == seed)
                         & (cqraci_ref["span"] == "cross")]["IS"].iloc[0]
        assert abs(float(C.interval_score(y, a["CQR-ACI"]["lo"], a["CQR-ACI"]["up"]).mean())
                   - float(ref)) < 1e-6, ("CQR-ACI", seed)

        for name, d in a.items():
            arms[name][seed] = DayStats(y, mu_cr, dcode, n_days, cnt_day[0],
                                        lo=d["lo"], up=d["up"], sig=d["sig"])
        print(f"[site18] seed {seed}: arms replayed + frozen-metric assertions passed", flush=True)
    return {"arms": arms, "dcode": dcode, "n_days": n_days,
            "day_rows": day_rows, "cnt_day": cnt_day}


def build_site1b():
    sc_raw = json.loads((C1B.OUT / "site1b_mean_scaler.json").read_text())
    sc = {"mean": np.array(sc_raw["mean"], float), "std": np.array(sc_raw["std"], float),
          "target_mean": float(sc_raw["target_mean"]), "target_std": float(sc_raw["target_std"])}
    gq = json.loads((C1B.OUT / "site1b_gamma_q.json").read_text())
    frozen = pd.read_csv(C1B.OUT / "site1b_eval_per_seed.csv")
    cqraci_ref = pd.read_csv(S9_1B / "cqr_aci_per_seed.csv")

    C1B.open_2025()
    g24 = C1B.load_year_grid(2024)
    w24 = C1B.windows(g24)
    cal = {k: v[C1B.span_on_origins(w24, *C.CALIB)] for k, v in w24.items()}
    g25 = C1B.load_year_grid(2025)
    w25 = C1B.windows(g25)
    y, o = w25["y_kw"], pd.DatetimeIndex(w25["origins"])
    mu_cache = np.load(C1B.OUT / "site1b_mean_cache.npz")
    dcode, n_days, day_rows, cnt_day = _day_grid(o)

    arms = {"A3": {}, "A2": {}, "A0": {}, "CQR": {}, "CQR-ACI": {}}
    for seed in C.SEEDS:
        model = C.MultiScaleTCN(dropout=0.1).to(C.DEVICE)
        model.load_state_dict(torch_load(C1B.OUT /
                                         f"site1b_mean_checkpoint_seed{seed}.pt")["state_dict"])
        model.eval()
        mu = C1B.mu_phys(C.tcn_mu(model, w25["x_raw"], sc))
        del model
        empty_cache()
        gru = C.ScaleModel("GRU").to(C.DEVICE)
        gru.load_state_dict(torch_load(C1B.OUT / "checkpoints" /
                                       f"site1b_gru_T3p_seed{seed}.pt")["state_dict"])
        gru.eval()
        sg = C.sigma_kw(gru, w25["x_raw"], sc)
        mu_cal = C1B.mu_phys(mu_cache[f"calib|{seed}"].astype(np.float64))
        sg_cal = C.sigma_kw(gru, cal["x_raw"], sc)
        del gru
        empty_cache()
        caln = np.sort(np.abs(cal["y_kw"] - mu_cal) / (sg_cal + C.EPS))
        la3, ua3, _, _ = C.run_aci(y, mu, sg, caln, o, gq[str(seed)]["gamma_A3"])
        ref = frozen[(frozen["method"] == "A3") & (frozen["seed"] == seed)
                     & (frozen["span"] == "cross")]["IS"].iloc[0]
        assert abs(float(C.interval_score(y, la3, ua3).mean()) - float(ref)) < 1e-9, ("A3", seed)

        # fixed A2 (normalized) and A0 (absolute) arms from the same frozen CALIB scores
        q_norm = C.conformal_q(caln)
        q_abs = C.conformal_q(np.sort(np.abs(cal["y_kw"] - mu_cal)))
        lo_a2, up_a2 = mu - q_norm * sg, mu + q_norm * sg
        lo_a0, up_a0 = mu - q_abs, mu + q_abs
        for meth, l_, u_ in (("A2", lo_a2, up_a2), ("A0", lo_a0, up_a0)):
            ref = frozen[(frozen["method"] == meth) & (frozen["seed"] == seed)
                         & (frozen["span"] == "cross")]["IS"].iloc[0]
            assert abs(float(C.interval_score(y, l_, u_).mean()) - float(ref)) < 1e-6, \
                (meth, seed)

        det = pd.read_csv(S9_1B / f"cqr_aci_intervals_seed{seed}.csv")
        assert len(det) == len(y) and np.allclose(det["y_true"].to_numpy(float), y)
        assert (pd.DatetimeIndex(det["target_time"]) == o + pd.Timedelta(minutes=15)).all()
        cal_scores = pd.read_csv(S9_1B / f"cqr_calib_scores_seed{seed}.csv")["cqr_score_calib"]
        qhat = C.conformal_q(np.sort(cal_scores.to_numpy(float)))
        lo_cqr = det["q05_raw"].to_numpy(float) - qhat
        up_cqr = det["q95_raw"].to_numpy(float) + qhat
        ref = frozen[(frozen["method"] == "CQR") & (frozen["seed"] == seed)
                     & (frozen["span"] == "cross")]["IS"].iloc[0]
        assert abs(float(C.interval_score(y, lo_cqr, up_cqr).mean()) - float(ref)) < 1e-6, \
            ("CQR", seed)
        lo_a = det["lower"].to_numpy(float)
        up_a = det["upper"].to_numpy(float)
        ref = cqraci_ref[(cqraci_ref["method"] == "CQR-ACI") & (cqraci_ref["seed"] == seed)
                         & (cqraci_ref["span"] == "cross")]["IS"].iloc[0]
        assert abs(float(C.interval_score(y, lo_a, up_a).mean()) - float(ref)) < 1e-6, \
            ("CQR-ACI", seed)

        arms["A3"][seed] = DayStats(y, mu, dcode, n_days, cnt_day[0], lo=la3, up=ua3)
        arms["A2"][seed] = DayStats(y, mu, dcode, n_days, cnt_day[0], lo=lo_a2, up=up_a2)
        arms["A0"][seed] = DayStats(y, mu, dcode, n_days, cnt_day[0], lo=lo_a0, up=up_a0)
        arms["CQR"][seed] = DayStats(y, mu, dcode, n_days, cnt_day[0], lo=lo_cqr, up=up_cqr)
        arms["CQR-ACI"][seed] = DayStats(y, mu, dcode, n_days, cnt_day[0], lo=lo_a, up=up_a)
        print(f"[site1b] seed {seed}: A0/A2/A3/CQR/CQR-ACI replayed + assertions passed", flush=True)
    return {"arms": arms, "dcode": dcode, "n_days": n_days,
            "day_rows": day_rows, "cnt_day": cnt_day}


def run_site(data, pairs, metrics, site, rows):
    for (b_name, a_name) in pairs:
        for metric in metrics:
            statsA = [data["arms"][a_name][s] for s in C.SEEDS]
            statsB = [data["arms"][b_name][s] for s in C.SEEDS]
            point = float(np.median([stat_full(statsB[j], metric) - stat_full(statsA[j], metric)
                                     for j in range(len(C.SEEDS))]))
            t0 = time.time()
            draws = two_level_boot(statsA, statsB, metric, data["dcode"], data["n_days"],
                                   data["day_rows"], data["cnt_day"])
            lo, hi = np.quantile(draws, [0.025, 0.975])
            rows.append({"site": site, "pair": f"{b_name}-vs-{a_name}", "metric": metric,
                         "point_median_seed_diff": point,
                         "ci_lower_two_level": float(lo), "ci_upper_two_level": float(hi),
                         "draw_median": float(np.median(draws)), "B": B_OUTER,
                         "seconds": round(time.time() - t0, 1)})
            print(f"[{site}] {b_name}-vs-{a_name} {metric}: point {point:+.4f} "
                  f"two-level 95% CI [{lo:+.4f}, {hi:+.4f}] ({rows[-1]['seconds']}s)", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    data18 = build_site18()
    run_site(data18, METHOD_PAIRS, METHOD_METRICS, "site18", rows)
    run_site(data18, POOL_PAIRS, POOL_METRICS, "site18", rows)
    print(f"site18 done in {(time.time() - t0) / 60:.1f} min", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "two_level_bootstrap.csv", index=False)

    t1 = time.time()
    data1b = build_site1b()
    run_site(data1b, [("A3", "CQR"), ("A3", "CQR-ACI"), ("A3", "A0"), ("A3", "A2")],
             METHOD_METRICS, "site1b", rows)
    print(f"site1b done in {(time.time() - t1) / 60:.1f} min", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "two_level_bootstrap.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps({
        "B_outer": B_OUTER, "rng_seed": RNG_SEED,
        "outer": "resample 5 seeds with replacement",
        "inner": "one day-block resample of the 2025 cross span per picked seed "
                 "(same picked days for both arms)",
        "statistic": "median across the five picked seeds of the paired difference",
        "point_estimate": "median across seeds of full-sample paired differences (unchanged)",
        "arms_replay_validated": "mean IS vs frozen per-seed CSVs (1e-9 exact-replay arms, "
                                 "1e-6 checkpoint-replay arms)",
        "code_hash": C.sha256_file(Path(__file__)),
    }, indent=2))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
