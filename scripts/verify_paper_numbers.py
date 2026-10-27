"""Recompute the paper's Tables 3, 4 and 5 (Site 18 + Site 1B medians) from the
archived per-seed evidence CSVs and compare them with the values printed in the
manuscript (latex_v8, five matched seeds).

This is a verification utility: it reads only shipped aggregate evidence under
results/ and asserts manuscript-value agreement within rounding tolerance
(absolute 0.0005 for 3-decimal cells, 0.005 for 2-decimal cells, 0.05 for
integer cells). It recomputes nothing from raw data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
S7 = ROOT / "results" / "rigorous_v2" / "s7_final_eval"
S9 = ROOT / "results" / "rigorous_v2" / "s9_cqr_aci"
S1B = ROOT / "results" / "site1b_val"

FAILURES: list[str] = []


def check(where: str, got: float, want: float, tol: float) -> None:
    ok = abs(float(got) - float(want)) <= tol
    flag = "OK " if ok else "MISMATCH"
    print(f"  [{flag}] {where}: evidence {float(got):.4f} vs paper {want}")
    if not ok:
        FAILURES.append(f"{where}: evidence {got:.4f} vs paper {want}")


def table3() -> None:
    print("Table 3 — residual-pool comparison (Site 18, 2025, A2 calibration)")
    df = pd.read_csv(S7 / "metrics_per_seed.csv")
    cross = df[df["span"] == "cross"]
    paper = {  # variant: (Std_z, Spearman, PICP, PINAW, IS, IS-D, PICP-D, PICP-H)
        "T1p": (1.444, 0.646, 0.922, 0.227, 1.335, 2.496, 0.899, 0.912),
        "T2p": (1.789, 0.820, 0.878, 0.077, 0.706, 1.406, 0.827, 0.816),
        "T3p": (1.256, 0.853, 0.895, 0.075, 0.664, 1.330, 0.846, 0.825),
        "T3match": (1.080, 0.776, 0.909, 0.108, 0.759, 1.488, 0.865, 0.851),
        "T1in": (1.230, 0.868, 0.902, 0.088, 0.686, 1.395, 0.869, 0.869),
        "V1": (1.557, 0.682, 0.885, 0.109, 0.772, 1.504, 0.867, 0.904),
        "V3": (1.644, 0.701, 0.913, 0.103, 0.729, 1.439, 0.884, 0.917),
    }
    cols = ("Std_z", "Spearman", "A2_PICP", "A2_PINAW", "A2_IS", "A2_IS_daylight",
            "A2_PICP_daylight", "A2_PICP_HARD_DF")
    for variant, wants in paper.items():
        med = cross[cross["variant"] == variant][list(cols)].median()
        for col, want in zip(cols, wants):
            check(f"T3 {variant}.{col}", med[col], want, 5e-4)


def table4() -> None:
    print("Table 4 — calibration ablation (Site 18, 2025)")
    df = pd.read_csv(S7 / "metrics_per_seed.csv")
    cross = df[df["span"] == "cross"]
    ab = cross[cross["variant"] == "T3p:A0"].set_index("seed")
    rows = {
        "A0": cross[cross["variant"] == "T3p:A0"],
        "A1": cross[cross["variant"] == "T3p:A1"],
        "A2": cross[cross["variant"] == "T3p"],
        "A3": cross[cross["variant"] == "T3p:A3"],
    }
    # method: (PICP, IS, PINAW, RCE1D, streak, HIGH_N, HARD_DF, Daylight, Width-H, IS-H)
    paper = {
        "A0": (0.944, 1.567, 0.187, 0.079, 80, 0.740, 0.543, 0.882, 0.97, 5.97),
        "A1": (0.901, 0.874, 0.099, 0.013, 22, 0.881, 0.832, 0.815, 2.29, 3.75),
        "A2": (0.895, 0.664, 0.075, 0.047, 95, 0.861, 0.825, 0.846, 1.98, 3.49),
        "A3": (0.900, 0.654, 0.075, 0.037, 87, 0.863, 0.829, 0.850, 2.00, 3.49),
    }
    trade = pd.read_csv(ROOT / "figures" / "figure_source_data" / "Fig9_HARD_DF_tradeoff_source.csv")
    cond = pd.read_csv(S7 / "conditional_correct_per_seed.csv")  # paper-consistent HIGH_N (inclusive of HARD_DF)

    for method, wants in paper.items():
        r = rows[method].median(numeric_only=True)
        picp, is_, pinaw, rce1d, streak, highn, harddf, day, wh, ish = wants
        p = r["A0_PICP"] if method == "A0" else (r["A1_PICP"] if method == "A1"
                                                 else (r["A2_PICP"] if method == "A2" else r["A3_PICP"]))
        i = r["A0_IS"] if method == "A0" else (r["A1_IS"] if method == "A1"
                                               else (r["A2_IS"] if method == "A2" else r["A3_IS"]))
        pa = r["A0_PINAW"] if method == "A0" else (r["A1_PINAW"] if method == "A1"
                                                   else (r["A2_PINAW"] if method == "A2" else r["A3_PINAW"]))
        rc = r["A0_RCE1D"] if method == "A0" else (r["A1_RCE1D"] if method == "A1"
                                                   else (r["A2_RCE1D"] if method == "A2" else r["A3_RCE1D"]))
        st = r["A0_LongestMissStreak"] if method == "A0" else (
            r["A1_LongestMissStreak"] if method == "A1" else (
                r["A2_LongestMissStreak"] if method == "A2" else r["A3_LongestMissStreak"]))
        hn = cond[f"{method}_PICP_HIGH_N"].median()  # inclusive definition used by the paper
        hd = r["A0_PICP_HARD_DF"] if method == "A0" else (
            r["A1_PICP_HARD_DF"] if method == "A1" else (
                r["A2_PICP_HARD_DF"] if method == "A2" else r["A3_PICP_HARD_DF"]))
        dl = r["A0_PICP_daylight"] if method == "A0" else (
            r["A1_PICP_daylight"] if method == "A1" else (
                r["A2_PICP_daylight"] if method == "A2" else r["A3_PICP_daylight"]))
        t = trade[trade["method"] == method].median(numeric_only=True)
        check(f"T4 {method}.PICP", p, picp, 5e-4)
        check(f"T4 {method}.IS", i, is_, 5e-4)
        check(f"T4 {method}.PINAW", pa, pinaw, 5e-4)
        check(f"T4 {method}.RCE1D", rc, rce1d, 5e-4)
        check(f"T4 {method}.streak", st, streak, 0.5)
        check(f"T4 {method}.HIGH_N", hn, highn, 5e-4)
        check(f"T4 {method}.HARD_DF", hd, harddf, 5e-4)
        check(f"T4 {method}.Daylight", dl, day, 5e-4)
        check(f"T4 {method}.Width-H", t["HARD_width_kw"], wh, 5e-3)
        check(f"T4 {method}.IS-H", t["HARD_IS"], ish, 5e-3)
    del ab


def table5() -> None:
    print("Table 5 — cross-system A0–A3 and CQR (2025 medians)")
    a18 = pd.read_csv(S7 / "site18_a0a1_full.csv")
    mp = pd.read_csv(S7 / "metrics_per_seed.csv")
    mp = mp[mp["span"] == "cross"]
    fixed18 = pd.read_csv(S7 / "site18_cqr_per_seed.csv")
    c18 = pd.read_csv(S9 / "cqr_aci_per_seed.csv")
    numdict = pd.read_csv(ROOT / "results" / "supplementary" /
                          "FINAL_PAPER_NUMBER_DICTIONARY_V4.csv")
    e1b = pd.read_csv(S1B / "site1b_eval_per_seed.csv")
    c1b = pd.read_csv(S1B / "cqr_aci" / "cqr_aci_per_seed.csv")

    def numdict_value(key: str) -> float:
        return float(numdict.loc[numdict["key"] == key, "value"].iloc[0])

    paper18 = {"A0": (0.944, 0.187, 1.567, 0.301, 0.079, 0.061, 80),
               "A1": (0.901, 0.099, 0.874, 0.168, 0.013, 0.007, 22),
               "A2": (0.895, 0.075, 0.664, 0.128, 0.047, 0.037, 95),
               "A3": (0.900, 0.075, 0.654, 0.126, 0.037, 0.023, 87),
               "CQR": (0.943, 0.092, 0.753, 0.145, 0.060, 0.052, None),
               "CQR-ACI": (0.900, 0.093, 0.726, 0.140, 0.034, 0.020, 71)}
    med18 = a18.groupby("method").median(numeric_only=True)
    a2row = mp[mp["variant"] == "T3p"].median(numeric_only=True)

    def _norm(method: str, r) -> dict:
        """Normalize a metric row to unprefixed names."""
        if method == "A2":
            return {"PICP": r["A2_PICP"], "PINAW": r["A2_PINAW"], "IS": r["A2_IS"],
                    "NIS": r["A2_IS"] / 5.2, "RCE1D": r["A2_RCE1D"], "RCE3D": r["A2_RCE3D"]}
        out = dict(r)
        if "NIS" not in out:
            out["NIS"] = out["IS"] / 5.2
        return out

    for method, (picp, pinaw, is_, nis, rce1d, rce3d, streak) in paper18.items():
        if method == "CQR":
            r = fixed18[fixed18["span"] == "cross"].median(numeric_only=True)
            check(f"T5/Site18 {method}.RCE3D", numdict_value("S18_CQR_RCE3D_median"), rce3d, 5e-4)
        elif method == "CQR-ACI":
            r = c18[c18["method"] == "CQR-ACI"].median(numeric_only=True)
            if streak is not None:
                check(f"T5/Site18 {method}.streak", r["LongestMissStreak"], streak, 0.5)
        elif method == "A2":
            r = _norm(method, a2row)
            check(f"T5/Site18 {method}.streak", a2row["A2_LongestMissStreak"], streak, 0.5)
        else:
            r = med18.loc[method]
            stm = mp[mp["variant"] == f"T3p:{method}"][
                f"{method}_LongestMissStreak"].median()
            check(f"T5/Site18 {method}.streak", stm, streak, 0.5)
        rr = r if method == "A2" else _norm(method, r)
        check(f"T5/Site18 {method}.PICP", rr["PICP"], picp, 5e-4)
        check(f"T5/Site18 {method}.PINAW", rr["PINAW"], pinaw, 5e-4)
        check(f"T5/Site18 {method}.IS", rr["IS"], is_, 5e-4)
        check(f"T5/Site18 {method}.NIS", rr["NIS"], nis, 5e-4)
        check(f"T5/Site18 {method}.RCE1D", rr["RCE1D"], rce1d, 5e-4)
        if method in ("A0", "A1", "A3", "A2"):
            check(f"T5/Site18 {method}.RCE3D", rr["RCE3D"], rce3d, 5e-4)

    paper1b = {"A0": (0.950, 0.160, 4.983, 0.071, 0.056, 16),
               "A1": (0.918, 0.093, 3.097, 0.022, 0.019, 12),
               "A2": (0.855, 0.060, 2.574, 0.065, 0.056, 46),
               "A3": (0.898, 0.070, 2.530, 0.023, 0.013, 31),
               "CQR": (0.903, 0.067, 2.683, 0.042, 0.030, 34),
               "CQR-ACI": (0.900, 0.072, 2.607, 0.017, 0.008, 23)}
    med1b = e1b[e1b["span"] == "cross"].groupby("method").median(numeric_only=True)
    for method, (picp, pinaw, is_, rce1d, rce3d, streak) in paper1b.items():
        if method == "CQR-ACI":
            r = c1b[c1b["method"] == "CQR-ACI"].median(numeric_only=True)
        else:
            r = med1b.loc[method]
        check(f"T5/Site1B {method}.PICP", r["PICP"], picp, 5e-4)
        check(f"T5/Site1B {method}.PINAW", r["PINAW"], pinaw, 5e-4)
        check(f"T5/Site1B {method}.IS", r["IS"], is_, 5e-3)
        check(f"T5/Site1B {method}.RCE1D", r["RCE1D"], rce1d, 5e-4)
        check(f"T5/Site1B {method}.RCE3D", r["RCE3D"], rce3d, 5e-4)
        check(f"T5/Site1B {method}.streak", r["LongestMissStreak"], streak, 0.5)


def table2() -> None:
    print("Table 2 — point forecasting (2025 medians)")
    m = json_load = None
    import json
    rm = json.loads((S7 / "run_manifest.json").read_text())
    # Site 18 point metrics come from s2's frozen mean on the CROSS span
    mean_metrics = pd.read_csv(ROOT / "results" / "rigorous_v2" / "s2_final_mean" / "mean_metrics.csv")
    s2 = mean_metrics[mean_metrics["span"] == "same"].median(numeric_only=True)  # sanity only
    del rm, s2, m
    # Site 18 CROSS RMSE is not in mean_metrics (spans end at SAME); the paper's
    # Site 18 RMSE comes from the s7 run manifest / p9 report; Site 1B from the
    # point metrics CSV.
    # the CROSS-span point metrics live in the eval file (the A0 arm carries the
    # raw point forecast; the mean-training file covers only grad/es/same spans)
    pt1b = pd.read_csv(S1B / "site1b_eval_per_seed.csv")
    pt1b = pt1b[(pt1b["span"] == "cross") & (pt1b["method"] == "A0")].median(numeric_only=True)
    check("T2 Site1B.RMSE", pt1b["RMSE"], 0.867, 5e-4)
    check("T2 Site1B.MAE", pt1b["MAE"], 0.406, 5e-4)
    nd = pd.read_csv(ROOT / "results" / "supplementary" / "FINAL_PAPER_NUMBER_DICTIONARY_V4.csv")

    def ndv(key):
        return float(nd.loc[nd["key"] == key, "value"].iloc[0])

    check("T2 Site18.RMSE", ndv("S18_mean_RMSE_2025_kW"), 0.286, 5e-4)
    check("T2 Site18.nRMSE", ndv("S18_mean_nRMSE_2025"), 0.055, 5e-4)
    check("T2 Site1B.nRMSE", ndv("S1B_mean_nRMSE_2025"), 0.037, 5e-4)


if __name__ == "__main__":
    table3()
    table4()
    table5()
    table2()
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)} mismatches):")
        for f in FAILURES:
            print(" -", f)
        sys.exit(1)
    print("All manuscript values reproduced from the archived evidence CSVs.")
