"""rigorous_v2 STEP 6: provenance-controlled pools -> GRU scale encoders ->
PRE-REGISTERED window selection -> freeze manifest.

Variants (all trained with the SAME GRU/optimizer/loss/seed protocol and the SAME
common deployment scaler; pools differ ONLY in residual provenance/composition):

  T1p    E24_OOF                                   (single-year OOS baseline) [eligible]
  T2p    E24_OOF + E23_replay                      [eligible]
  T3p    E24_OOF + E23_replay + E22_replay         [eligible]
  T1in   final-mean fitted residual (grad+ES rows) (IN_SAMPLE provenance control)
  T3match T3p subsampled to |T1p|, stratified by month-of-year x daylight (control)
  V1/V2/V3  vintage pools (provenance sensitivity arm; NOT deployment-eligible)
  *_equpd    same-update sensitivity (fixed optimizer updates = T1p's budget)

PRE-REGISTERED selection rule (SELECT span 2024-08-08..09-16 only; 2025 never
loaded in this script):
  feasible: median |Std_z - 1| <= 0.35 ; winner: argmin median NLL ; ties -> smaller
  window. Eligible set = {T1p, T2p, T3p} (deployment-faithful replay chain).

gamma for the winner (A3) and for absolute ACI (A1) selected per seed on the SAME
SELECT span with the Stage-7 rule.

Outputs -> results/rigorous_v2/s6_gru_select/  and  FREEZE_MANIFEST.json
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.rigorous_v2 import common_v2 as C  # noqa: E402

OUT = C.V2 / "s6_gru_select"
S2, S3, S4, S5 = (C.V2 / d for d in
                  ("s2_final_mean", "s3_e24_oof", "s4_replay", "s5_vintage"))
ELIGIBLE = ["T1p", "T2p", "T3p"]
ORDER = {"T1p": 0, "T2p": 1, "T3p": 2}



def pool_x_by_origins(year_part: dict, origins: pd.DatetimeIndex) -> np.ndarray:
    pos = pd.Series(np.arange(len(year_part["origins"])), index=year_part["origins"])
    idx = pos.reindex(pd.DatetimeIndex(origins)).to_numpy(dtype=float)
    assert not np.isnan(idx).any(), "pool origin missing from year windows"
    return year_part["x_raw"][idx.astype(int)]


def build_pools(seed: int, y2024: dict, y2022: dict, y2023: dict, mu_cache) -> dict:
    oof = pd.read_parquet(S3 / f"E24_OOF_seed{seed}.parquet")
    e23 = pd.read_parquet(S4 / f"E23_replay_seed{seed}.parquet")
    e22 = pd.read_parquet(S4 / f"E22_replay_seed{seed}.parquet")
    e24v = pd.read_parquet(S5 / f"E24_vintage_seed{seed}.parquet")
    e23v = pd.read_parquet(S5 / f"E23_vintage_seed{seed}.parquet")
    e22v = pd.read_parquet(S5 / f"E22_vintage_seed{seed}.parquet")
    # V-int arms: the 2024 vintage pool is trimmed to origins < 2024-08-08 so that
    # NO evaluation row (SELECT/CALIB/SAME) appears in any training pool. The
    # 2022/2023 vintage pools are already disjoint from all evaluation spans.
    cut = pd.Timestamp(C.MEAN_ES[1])
    e24v = e24v[pd.to_datetime(e24v["origin"]) < cut].reset_index(drop=True)

    def pack(dfs, xs):
        return {"x_raw": np.concatenate(xs).astype(np.float32),
                "e_kw": np.concatenate([df["residual"].to_numpy(float) for df in dfs]),
                "origins": pd.DatetimeIndex(pd.concat([df["origin"] for df in dfs])
                                            .to_numpy()),
                "prov": np.concatenate([df["provenance"].to_numpy() for df in dfs])}

    x_oof = pool_x_by_origins(y2024, pd.DatetimeIndex(oof["origin"]))
    x_e22 = pool_x_by_origins(y2022, pd.DatetimeIndex(e22["origin"]))
    x_e23 = pool_x_by_origins(y2023, pd.DatetimeIndex(e23["origin"]))
    x_e22v = pool_x_by_origins(y2022, pd.DatetimeIndex(e22v["origin"]))
    x_e23v = pool_x_by_origins(y2023, pd.DatetimeIndex(e23v["origin"]))
    x_e24v = pool_x_by_origins(y2024, pd.DatetimeIndex(e24v["origin"]))

    # T1in: fitted residuals of the final mean on its own grad+ES rows
    m_g = C.target_mask(y2024["targets"], *C.MEAN_GRAD)
    m_e = C.target_mask(y2024["targets"], *C.MEAN_ES)
    x_in = np.concatenate([y2024["x_raw"][m_g], y2024["x_raw"][m_e]])
    y_in = np.concatenate([y2024["y_kw"][m_g], y2024["y_kw"][m_e]])
    mu_in = np.concatenate([C.mu_phys(mu_cache[f"grad|{seed}"].astype(np.float64)),
                            C.mu_phys(mu_cache[f"es|{seed}"].astype(np.float64))])
    t1in = {"x_raw": x_in.astype(np.float32), "e_kw": y_in - mu_in,
            "origins": pd.DatetimeIndex(pd.DatetimeIndex(y2024["origins"])[m_g]
                                        .append(pd.DatetimeIndex(y2024["origins"])[m_e])),
            "prov": np.full(len(y_in), C.PROV_INSAMPLE)}

    pools = {
        "T1p": pack([oof], [x_oof]),
        "T2p": pack([oof, e23], [x_oof, x_e23]),
        "T3p": pack([oof, e23, e22], [x_oof, x_e23, x_e22]),
        "T1in": t1in,
        "V1": pack([e24v], [x_e24v]),
        "V2": pack([e23v, e24v], [x_e23v, x_e24v]),
        "V3": pack([e22v, e23v, e24v], [x_e22v, x_e23v, x_e24v]),
    }
    for p in pools.values():
        C.assert_provenance(p["prov"])
        C.assert_no_2025_in(p["origins"], p["origins"] + C.HORIZON_TD, "pool")
    return pools


def stratified_match(pool: dict, ref: dict, seed: int) -> np.ndarray:
    """Subsample pool to len(ref) matching month-of-year x daylight strata of ref.
    Daylight is the frozen rule: GHI at origin (input window last step, channel 1)
    > 20 W/m2. Strata absent in the pool are filled from neighbouring months with
    the same daylight state."""
    rng = np.random.default_rng(seed)
    pm = pd.DatetimeIndex(pool["origins"]).month.to_numpy()
    rm = pd.DatetimeIndex(ref["origins"]).month.to_numpy()
    pool_dl = pool["x_raw"][:, -1, 1] > C.DAYLIGHT_GHI_THRESHOLD
    ref_dl = ref["x_raw"][:, -1, 1] > C.DAYLIGHT_GHI_THRESHOLD
    pstrat = pm.astype(str) + "_" + pool_dl.astype(int).astype(str)
    rstrat = rm.astype(str) + "_" + ref_dl.astype(int).astype(str)
    pick = np.zeros(len(pstrat), dtype=bool)
    for s in np.unique(rstrat):
        need = int((rstrat == s).sum())
        month, dl = s.split("_")
        cand = np.flatnonzero(pstrat == s)
        if len(cand) == 0:
            dl_flag = pool_dl if dl == "1" else ~pool_dl
            cand = np.flatnonzero(dl_flag & (np.abs(pm - int(month)) <= 1))
            assert len(cand) > 0, f"stratum {s} missing in pool"
        take = rng.choice(cand, size=min(need, len(cand)), replace=False)
        pick[take] = True
    deficit = len(ref["e_kw"]) - int(pick.sum())
    if deficit > 0:
        rest = np.flatnonzero(~pick)
        pick[rng.choice(rest, size=deficit, replace=False)] = True
    return pick


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    C.assert_role_boundaries_disjoint()
    import json
    sc = json.loads((S2 / "scaler.json").read_text())
    sc = {"mean": np.array(sc["mean"], float), "std": np.array(sc["std"], float),
          "target_mean": float(sc["target_mean"]), "target_std": float(sc["target_std"])}
    fp = C.scaler_fingerprint(sc)

    y2024 = C.load_year_windows(2024)
    y2022 = C.load_year_windows(2022)
    y2023 = C.load_year_windows(2023)
    mu_cache = np.load(S2 / "final_mean_cache.npz")
    m_sel = C.span_mask(y2024["origins"], *C.SELECT)
    sel = C.sub(y2024, m_sel)
    C.assert_no_2025_in(sel["origins"], sel["targets"], "SELECT")

    (OUT / "checkpoints").mkdir(exist_ok=True)
    pool_stats, rows = [], []
    sel_sigma = {}          # (variant, seed) -> sigma on SELECT
    equpd_budget = None
    for seed in C.SEEDS:
        pools = build_pools(seed, y2024, y2022, y2023, mu_cache)
        # stratified T3match (draw tied to this seed)
        pick = stratified_match(pools["T3p"], pools["T1p"], seed)
        pools["T3match"] = {"x_raw": pools["T3p"]["x_raw"][pick],
                            "e_kw": pools["T3p"]["e_kw"][pick],
                            "origins": pd.DatetimeIndex(pools["T3p"]["origins"])[pick],
                            "prov": pools["T3p"]["prov"][pick]}
        if equpd_budget is None:
            equpd_budget = 3 * 50 * int(np.ceil(len(pools["T1p"]["e_kw"]) / 512))
            for name, p in pools.items():
                o = pd.DatetimeIndex(p["origins"])
                pool_stats.append({
                    "variant": name, "seed": seed, "n": len(p["e_kw"]),
                    "prov_mix": {k: int(v) for k, v in
                                 zip(*np.unique(p["prov"], return_counts=True))},
                    "months": sorted(set(int(m) for m in o.month.unique())),
                    "daylight_share": float((p["x_raw"][:, -1, 1] > C.DAYLIGHT_GHI_THRESHOLD).mean()),
                    "abs_e_mean": float(np.mean(np.abs(p["e_kw"]))),
                    "abs_e_median": float(np.median(np.abs(p["e_kw"]))),
                    "abs_e_q95": float(np.quantile(np.abs(p["e_kw"]), 0.95)),
                })
        budgets = {v: (equpd_budget if v in ("T1p", "T3p", "T3match") else None)
                   for v in pools}
        for suffix in ("", "_equpd"):
            for name, p in pools.items():
                if suffix == "_equpd" and name not in ("T1p", "T3p", "T3match"):
                    continue
                vname = name + suffix
                updates = budgets[name] if suffix else None
                xs = C.apply_scaler(p["x_raw"], sc)
                estd = (p["e_kw"] / sc["target_std"]).astype(np.float32)
                ck = OUT / "checkpoints" / f"gru_{vname}_seed{seed}.pt"
                if ck.exists() and "V" not in vname:
                    model = C.ScaleModel("GRU").to(C.DEVICE)
                    sd = torch.load(ck, weights_only=False)
                    assert sd["n_train"] == len(estd), f"{vname}: stale checkpoint n_train"
                    model.load_state_dict(sd["state_dict"])
                    model.eval()
                    steps, secs = sd["updates"], 0.0
                else:
                    model, steps, secs = C.train_gru(xs, estd, seed, total_updates=updates)
                    torch.save({"state_dict": model.state_dict(), "variant": vname, "seed": seed,
                                "scaler_fp": fp, "n_train": len(estd), "updates": steps}, ck)
                sg = C.sigma_kw(model, sel["x_raw"], sc)
                sel_sigma[(vname, seed)] = sg
                mu_p = C.mu_phys(mu_cache[f"select|{seed}"].astype(np.float64))
                r = {"variant": vname, "seed": seed, "span": "select",
                     "n_train": len(estd), "updates": steps, "train_s": secs,
                     **C.scale_metrics(sel["y_kw"], mu_p, sg)}
                rows.append(r)
                print(r["variant"], seed, "NLL", round(r["NLL"], 3), "Std_z", round(r["Std_z"], 3),
                      "Spearman", round(r["Spearman"], 3), "updates", steps)
                del model
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "metrics_per_seed.csv", index=False)
    med = df.groupby("variant")[["NLL", "Std_z", "Mean_z", "Median_abs_z", "Q95_abs_z",
                                 "Spearman", "Pearson", "Median_sigma_kw"]].median()
    med.to_csv(OUT / "table_median.csv")
    pd.DataFrame(pool_stats).to_csv(OUT / "pool_manifest.csv", index=False)

    # ---- pre-registered decision (SELECT, main protocol, eligible set) ----
    main_df = df[~df["variant"].str.endswith("_equpd")]
    med_nll = main_df[main_df["variant"].isin(ELIGIBLE)].groupby("variant")["NLL"].median()
    dz = main_df[main_df["variant"].isin(ELIGIBLE)].groupby("variant")["Std_z"].apply(
        lambda s: abs(s.median() - 1.0))
    feasible = [v for v in ELIGIBLE if dz[v] <= 0.35] or list(ELIGIBLE)
    winner = min(feasible, key=lambda v: (med_nll[v], ORDER[v]))
    decision = {"rule": "feasible median |Std_z-1|<=0.35 on SELECT; argmin median NLL; "
                        "ties -> smaller window; eligible={T1p,T2p,T3p}; scaler=common",
                "eligible": ELIGIBLE,
                "median_NLL": med_nll.to_dict(), "abs_Std_z_minus_1": dz.to_dict(),
                "feasible": feasible, "winner": winner,
                "note": "SELECT span only; 2025 never loaded in s6"}
    C.wjson(OUT / "selection_decision.json", decision)
    print(json.dumps(decision, indent=2))

    # ---- gamma for winner (A3) and absolute ACI (A1), per seed, SELECT span ----
    gammas = {}
    for seed in C.SEEDS:
        mu_p = C.mu_phys(mu_cache[f"select|{seed}"].astype(np.float64))
        sg = sel_sigma[(winner, seed)]
        g3, det3 = C.select_gamma(sel["y_kw"], mu_p, sg, sel["origins"], use_sigma=True)
        g1, det1 = C.select_gamma(sel["y_kw"], mu_p, np.ones(len(sg)), sel["origins"], use_sigma=False)
        gammas[seed] = {"gamma_A3": g3, "gamma_A1": g1, "detail_A3": det3, "detail_A1": det1}
    C.wjson(OUT / "gamma_selection.json", gammas)

    # ---- freeze manifest ----
    C.wjson(C.V2 / "FREEZE_MANIFEST.json", {
        "created": pd.Timestamp.now().isoformat(),
        "role_boundaries": {"MEAN_GRAD": list(C.MEAN_GRAD), "MEAN_ES": list(C.MEAN_ES),
                            "SELECT": list(C.SELECT), "CALIB": list(C.CALIB),
                            "SAME": list(C.SAME), "CROSS": list(C.CROSS)},
        "scaler_fingerprint": fp,
        "selection_decision": decision,
        "gamma_selection": {str(k): {"gamma_A3": v["gamma_A3"], "gamma_A1": v["gamma_A1"]}
                            for k, v in gammas.items()},
        "winner": winner,
        "pool_manifest_rows": len(pool_stats),
        "gru_checkpoints": {p.name: C.sha256_file(p) for p in
                            sorted((OUT / "checkpoints").glob("gru_*.pt"))},
        "mean_checkpoints": {p.name: C.sha256_file(p) for p in sorted(S2.glob("final_mean_*.pt"))},
        "code_hashes": {p.name: C.sha256_file(p) for p in
                        sorted((C.PROJECT_ROOT / "experiments" / "rigorous_v2").glob("*.py"))},
        "asserts": ["role spans disjoint", "no 2025 in any pool/SELECT artifact",
                    "provenance labels within allowed set",
                    "scaler identical across variants (single fingerprint)"],
        "daylight_rule": f"origin GHI > {C.DAYLIGHT_GHI_THRESHOLD} W/m2 (frozen)",
    })
    print("s6 done; FREEZE_MANIFEST.json written")


if __name__ == "__main__":
    main()
