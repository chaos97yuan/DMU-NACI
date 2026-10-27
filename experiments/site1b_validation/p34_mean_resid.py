"""Site 1B Phase 3+4: frozen-architecture mean training + residual library.

Phase 3  mean TCN (rigorous_v2 architecture & training loop verbatim):
         grad rows target in [2024-01-01, 2024-07-17); ES tail [07-17, 08-08);
         scaler fit on grad rows only; 5 matched seeds; P_rated = 23.4 kW.
Phase 4  E24_OOF (rigorous_v2 fold boundaries, per-fold scaler + internal tail ES)
         + E23_replay / E22_replay (frozen final mean replayed on full 2023/2022).
2025 is never loaded here (enforced by the loader guard).
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


def rmse_mae(y, p):
    e = np.asarray(y) - np.asarray(p)
    return {"RMSE": float(np.sqrt(np.mean(e ** 2))), "MAE": float(np.mean(np.abs(e))),
            "nRMSE": float(np.sqrt(np.mean(e ** 2)) / C.P_RATED_1B),
            "nMAE": float(np.mean(np.abs(e)) / C.P_RATED_1B)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    V2 = C.V2
    V2.assert_role_boundaries_disjoint()
    g24 = C.load_year_grid(2024)
    w24 = C.windows(g24)

    # ---- Phase 3: development rows by TARGET span (training consumes y) ----
    m_grad = C.target_span_mask(g24, *V2.MEAN_GRAD)
    m_es = C.target_span_mask(g24, *V2.MEAN_ES)
    wm_grad = C.span_on_targets(w24, *V2.MEAN_GRAD)
    wm_es = C.span_on_targets(w24, *V2.MEAN_ES)
    # rows whose 6-h history is entirely inside the year and target in span
    grad_rows = {k: v[wm_grad] for k, v in w24.items()}
    es_rows = {k: v[wm_es] for k, v in w24.items()}
    # note: origin-span membership implies target in span (origin < target);
    # equivalence asserted below against the grid-level masks
    tg = pd.DatetimeIndex(grad_rows["targets"])
    assert tg.min() >= pd.Timestamp(V2.MEAN_GRAD[0]) and tg.max() < pd.Timestamp(V2.MEAN_GRAD[1])
    te = pd.DatetimeIndex(es_rows["targets"])
    assert te.min() >= pd.Timestamp(V2.MEAN_ES[0]) and te.max() < pd.Timestamp(V2.MEAN_ES[1])
    assert int(m_grad.sum()) >= len(grad_rows["y_kw"]) and int(m_es.sum()) >= len(es_rows["y_kw"])
    print(f"grad rows {len(grad_rows['y_kw'])}, es rows {len(es_rows['y_kw'])}")

    sc = V2.fit_scaler(grad_rows["x_raw"])
    (OUT / "site1b_mean_scaler.json").write_text(json.dumps(
        {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in sc.items()}), encoding="utf-8")
    fp = V2.scaler_fingerprint(sc)

    span_eval = {"select": V2.SELECT, "calib": V2.CALIB, "same": V2.SAME}
    eval_w = {t: {k: v[C.span_on_origins(w24, *s)] for k, v in w24.items()}
              for t, s in span_eval.items()}
    mu_store, point_rows = {}, []
    for seed in C.SEEDS:
        ck = OUT / f"site1b_mean_checkpoint_seed{seed}.pt"
        if ck.exists():
            model = C.V2.MultiScaleTCN(dropout=0.1).to(C.V2.DEVICE)
            model.load_state_dict(torch.load(ck, weights_only=False)["state_dict"])
            model.eval()
            info = torch.load(ck, weights_only=False)["info"]
        else:
            model, info = V2.train_tcn(grad_rows, es_rows, sc, seed)
            torch.save({"state_dict": model.state_dict(), "site": "1B", "seed": seed,
                        "scaler_fp": fp, "info": info, "p_rated_kw": C.P_RATED_1B}, ck)
        for tag, rows in [("grad", grad_rows), ("es", es_rows), *eval_w.items()]:
            mu = V2.tcn_mu(model, rows["x_raw"], sc)
            mu_store[f"{tag}|{seed}"] = mu.astype(np.float32)
            if tag in ("grad", "es", "same"):
                m = rmse_mae(rows["y_kw"], C.mu_phys(mu.astype(np.float64)))
                point_rows.append({"seed": seed, "span": tag, **m,
                                   "n": int(len(mu)), "best_epoch": info["best_epoch"]})
                print(f"seed {seed} {tag}: RMSE {m['RMSE']:.3f} nRMSE {m['nRMSE']:.4f}")
        del model
        torch.cuda.empty_cache()
    np.savez_compressed(OUT / "site1b_mean_cache.npz", **mu_store)
    pd.DataFrame(point_rows).to_csv(OUT / "site1b_mean_point_metrics.csv", index=False)
    med = pd.DataFrame(point_rows).groupby("span")[["RMSE", "MAE", "nRMSE", "nMAE"]].median()
    print(med.round(4).to_string())

    # ---- Phase 4a: E24_OOF ----
    oof_frames = {s: [] for s in C.SEEDS}
    fold_audit = []
    for f in V2.OOF_FOLDS:
        m_g = C.target_span_mask(g24, "2024-01-01 00:00", f["es_start"])
        m_e = C.target_span_mask(g24, f["es_start"], f["train_end"])
        f_g = {k: v[C.span_on_targets(w24, "2024-01-01 00:00", f["es_start"])]
               for k, v in w24.items()}
        f_e = {k: v[C.span_on_targets(w24, f["es_start"], f["train_end"])]
               for k, v in w24.items()}
        f_fc = {k: v[C.span_on_origins(w24, *f["fc"])] for k, v in w24.items()}
        V2.assert_training_precedes(
            np.concatenate([np.asarray(f_g["targets"]), np.asarray(f_e["targets"])]),
            f_fc["targets"], f"1B fold {f['fold']}")
        V2.assert_no_2025_in(f_fc["origins"], f_fc["targets"], f"1B fold {f['fold']}")
        for seed in C.SEEDS:
            ck = OUT / f"site1b_oof_{f['fold']}_seed{seed}.pt"
            fsc = V2.fit_scaler(f_g["x_raw"])
            if ck.exists():
                model = C.V2.MultiScaleTCN(dropout=0.1).to(C.V2.DEVICE)
                model.load_state_dict(torch.load(ck, weights_only=False)["state_dict"])
                model.eval()
            else:
                model, info = V2.train_tcn(f_g, f_e, fsc, seed)
                torch.save({"state_dict": model.state_dict(), "fold": f["fold"],
                            "seed": seed, "info": info}, ck)
            mu = V2.tcn_mu(model, f_fc["x_raw"], fsc).astype(np.float64)
            mp = C.mu_phys(mu)
            oof_frames[seed].append(pd.DataFrame({
                "timestamp": pd.DatetimeIndex(f_fc["targets"]),
                "origin": pd.DatetimeIndex(f_fc["origins"]),
                "target": f_fc["y_kw"], "mu_raw": mu, "mu_phys": mp,
                "residual": f_fc["y_kw"] - mp, "seed": seed, "fold": f["fold"],
                "provenance": "OOF", "origin_GHI": f_fc["origin_GHI"]}))
            del model
            torch.cuda.empty_cache()
        e = np.asarray(f_fc["y_kw"]) - mp
        fold_audit.append({"fold": f["fold"], "n_fc": int(len(e)),
                           "RMSE": float(np.sqrt(np.mean(e ** 2))),
                           "MAE": float(np.mean(np.abs(e))),
                           "abs_q95": float(np.quantile(np.abs(e), 0.95))})
        print("OOF", fold_audit[-1])
    oof_all = pd.concat([pd.concat(oof_frames[s], ignore_index=True) for s in C.SEEDS],
                        ignore_index=True)
    oof_all.to_csv(OUT / "E24_OOF_site1b.csv", index=False)

    # ---- Phase 4b: E22/E23 replay with the frozen final mean ----
    replay_audit = {}
    for year in (2022, 2023):
        gy = C.load_year_grid(year)
        wy = C.windows(gy)
        frames = []
        for seed in C.SEEDS:
            model = C.V2.MultiScaleTCN(dropout=0.1).to(C.V2.DEVICE)
            model.load_state_dict(torch.load(OUT / f"site1b_mean_checkpoint_seed{seed}.pt",
                                             weights_only=False)["state_dict"])
            model.eval()
            mu = V2.tcn_mu(model, wy["x_raw"], sc).astype(np.float64)
            mp = C.mu_phys(mu)
            frames.append(pd.DataFrame({
                "timestamp": pd.DatetimeIndex(wy["targets"]),
                "origin": pd.DatetimeIndex(wy["origins"]),
                "target": wy["y_kw"], "mu_raw": mu, "mu_phys": mp,
                "residual": wy["y_kw"] - mp, "seed": seed, "fold": "replay",
                "provenance": "REPLAY_OOS", "origin_GHI": wy["origin_GHI"]}))
            del model
        dfy = pd.concat(frames, ignore_index=True)
        dfy.to_csv(OUT / f"E{str(year)[2:]}_replay_site1b.csv", index=False)
        e = dfy["residual"].to_numpy()
        o = pd.DatetimeIndex(dfy["origin"])
        replay_audit[year] = {
            "n": int(len(dfy)), "mean_e": float(e.mean()), "std_e": float(e.std()),
            "MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "abs_q50": float(np.quantile(np.abs(e), .5)),
            "abs_q90": float(np.quantile(np.abs(e), .9)),
            "abs_q95": float(np.quantile(np.abs(e), .95)),
            "day_share": float((dfy["origin_GHI"] > V2.DAYLIGHT_GHI_THRESHOLD).mean()),
            "monthly_n": {int(m): int(c) for m, c in
                          pd.Series(1, index=o.month).groupby(level=0).count().items()},
        }
        print(year, replay_audit[year])

    (OUT / "phase34_manifest.json").write_text(json.dumps({
        "p_rated_kw": C.P_RATED_1B, "scaler_fp": fp,
        "grad_rows": int(len(grad_rows["y_kw"])), "es_rows": int(len(es_rows["y_kw"])),
        "fold_audit": fold_audit, "replay_audit": replay_audit,
        "mean_ck": {p.name: C.sha16(p) for p in sorted(OUT.glob("site1b_mean_checkpoint_*.pt"))},
        "oof_ck": {p.name: C.sha16(p) for p in sorted(OUT.glob("site1b_oof_*.pt"))},
        "note_2023": "2023 replay retains the sustained reduced-power operational state; "
                     "no capacity renormalization applied",
    }, indent=2, default=str), encoding="utf-8")
    print("phase 3+4 done")


if __name__ == "__main__":
    main()
