"""Stage 9.5 Fig.7 case-day selection: PRE-REGISTERED, performance-blind.

The three display days for the prediction-interval case study are chosen from
raw 2025 series and the FROZEN HARD_DF labels ONLY. No interval, score,
coverage, IS, or any method output is accessed here; the plotting script runs
afterwards and only reads the chosen dates from
figure_source_data/Fig7_case_days.csv.

Rule:
  universe  = all 2025 calendar days (2025_core.csv + window origins with the
              frozen HARD_DF labels that produced the paper's Table 5)
  eligible  = (i) >=95% of the day's 5-min core samples finite in power AND GHI;
              (ii) >=240 of 288 possible forecast origins present.
  features per day (raw series only):
    c_d = median(GHI daytime)/max(GHI daytime)           clearness (smooth bell)
    v_d = mean(|dGHI| at daytime steps)/max(GHI daytime)  flicker
    e_d = daily PV energy (kWh)
    h_d = HARD_DF origin share of the day (frozen labels)
  Case 1 clear/stable      : argmax  z(c) - z(v)
  Case 2 cloud-variable    : argmax  z(v) - z(c)  among days with e_d >= median(e)
  Case 3 ramp / HARD_DF    : argmax  h_d (tie-break: mean daytime |dP|)
  distinctness: if one day wins two cases, assign it to the case where its
  standardized margin is larger and use the runner-up day for the other case.
  display seed = 42 (first element of the frozen seed vector [42,123,2026,3407,8888]).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # release: repo-relative
sys.path.insert(0, str(ROOT))
SRC = ROOT / "figures" / "figure_source_data"
STAGE = SRC.parent
DAYLIGHT = 20.0


def main() -> None:
    core = pd.read_csv(ROOT / "data" / "processed" / "external_2025_preopen" / "2025_core.csv")
    core["timestamp"] = pd.to_datetime(core["timestamp"])
    core["day"] = core["timestamp"].dt.normalize()

    iv = pd.read_parquet(SRC / "intervals_cross_seed42.parquet")  # origins only; no lo/up used
    from experiments.rigorous_v2 import common_v2 as C
    y25 = C.load_year_windows(2025)
    o = pd.DatetimeIndex(iv["origin"])
    assert (o == pd.DatetimeIndex(y25["origins"])).all()
    hd = np.asarray(y25["hard_DF"]).astype(bool)
    org = pd.DataFrame({"origin": o, "day": o.normalize(), "hard_df": hd})

    rows = []
    for d, g in core.groupby("day"):
        p = g["Active_Power"].to_numpy(float)
        ghi = g["Global_Horizontal_Radiation"].to_numpy(float)
        m = ghi > DAYLIGHT
        both = m[:-1] & m[1:]
        finite_share = float((np.isfinite(p) & np.isfinite(ghi)).mean())
        og = org[org["day"] == d]
        c_d = (float(np.nanmedian(ghi[m]) / np.nanmax(ghi[m]))
               if m.sum() > 10 and np.nanmax(ghi[m]) > 50 else np.nan)
        v_d = (float(np.nanmean(np.abs(np.diff(ghi))[both]) / np.nanmax(ghi[m]))
               if m.sum() > 10 and np.nanmax(ghi[m]) > 50 and both.sum() > 5 else np.nan)
        rows.append({
            "day": d.strftime("%Y-%m-%d"), "finite_share": finite_share,
            "n_origins": int(len(og)), "hard_df_share": float(og["hard_df"].mean()) if len(og) else np.nan,
            "clearness": c_d, "flicker": v_d,
            "energy_kwh": float(np.nansum(np.clip(p, 0, None)) * 5.0 / 60.0),
            "mean_abs_dP_daytime": float(np.nanmean(np.abs(np.diff(p))[both])) if both.sum() > 5 else np.nan,
        })
    df = pd.DataFrame(rows)
    elig = df[(df["finite_share"] >= 0.95) & (df["n_origins"] >= 240) &
              np.isfinite(df["clearness"]) & np.isfinite(df["flicker"])].copy()
    for f in ("clearness", "flicker"):
        elig[f"z_{f}"] = (elig[f] - elig[f].mean()) / elig[f].std()
    elig["s_clear"] = elig["z_clearness"] - elig["z_flicker"]
    elig["s_var"] = elig["z_flicker"] - elig["z_clearness"]
    e_med = elig["energy_kwh"].median()

    cand_clear = elig.sort_values("s_clear", ascending=False)
    case1 = cand_clear.iloc[0]
    var_pool = elig[elig["energy_kwh"] >= e_med].sort_values("s_var", ascending=False)
    case2 = var_pool.iloc[0]
    cand_hard = elig.sort_values(["hard_df_share", "mean_abs_dP_daytime"], ascending=False)
    case3 = cand_hard.iloc[0]

    # distinctness rule
    picks: dict[str, pd.Series] = {}
    margins = {"case1": ("s_clear", case1), "case2": ("s_var", case2), "case3": ("hard_df_share", case3)}
    used: set[str] = set()
    order = sorted(margins.items(), key=lambda kv: -float(kv[1][1][kv[1][0]]))
    for name, (col, row) in order:
        rank_col = {"case1": "s_clear", "case2": "s_var", "case3": "hard_df_share"}[name]
        pool = cand_clear if name == "case1" else (var_pool if name == "case2" else cand_hard)
        for _, r in pool.iterrows():
            if r["day"] not in used:
                picks[name] = r
                used.add(r["day"])
                break
    assert len({picks[k]["day"] for k in picks}) == 3

    out = pd.DataFrame([
        {"case": "C1_clear_stable", "day": picks["case1"]["day"],
         "rule": "argmax z(clearness) - z(flicker)",
         "clearness": picks["case1"]["clearness"], "flicker": picks["case1"]["flicker"],
         "hard_df_share": picks["case1"]["hard_df_share"], "energy_kwh": picks["case1"]["energy_kwh"]},
        {"case": "C2_cloud_variable", "day": picks["case2"]["day"],
         "rule": "argmax z(flicker) - z(clearness) among energy >= median",
         "clearness": picks["case2"]["clearness"], "flicker": picks["case2"]["flicker"],
         "hard_df_share": picks["case2"]["hard_df_share"], "energy_kwh": picks["case2"]["energy_kwh"]},
        {"case": "C3_ramp_harddf", "day": picks["case3"]["day"],
         "rule": "argmax frozen HARD_DF origin share (tie: mean daytime |dP|)",
         "clearness": picks["case3"]["clearness"], "flicker": picks["case3"]["flicker"],
         "hard_df_share": picks["case3"]["hard_df_share"], "energy_kwh": picks["case3"]["energy_kwh"]},
    ])
    out["display_seed"] = 42
    out.to_csv(SRC / "Fig7_case_days.csv", index=False)
    df.to_csv(SRC / "Fig7_day_features_all2025.csv", index=False)
    elig.to_csv(SRC / "Fig7_day_features_eligible.csv", index=False)
    print(out[["case", "day", "clearness", "flicker", "hard_df_share", "energy_kwh"]].to_string(index=False))

    top = lambda pool, col: pool.head(6)[["day", col, "clearness", "flicker", "hard_df_share"]].round(3).to_string(index=False)
    md = f"""# 02_INTERVAL_CASE_SELECTION.md

Purpose: choose the three case days of the prediction-interval figure (Fig. 7)
by rules that cannot see any method output. Selection was computed and written
to `figure_source_data/Fig7_case_days.csv` BEFORE any interval was plotted.

## Pre-registered rule

- Universe: all 2025 calendar days. Raw inputs: the cleaned 5-min core
  (`2025_core.csv`) and the frozen HARD_DF origin labels (the same labels that
  produced Table 5; thresholds frozen on 2022 training data; not redefined).
- Eligibility: >=95% of the day's 5-min samples finite in power AND GHI, and
  >=240 of 288 possible forecast origins present ({len(elig)} eligible days).
- Daily features (raw series only):
  - clearness c_d = median(daytime GHI) / max(daytime GHI), daytime = GHI > 20 W/m2
  - flicker  v_d = mean(|Delta GHI| over daytime successive steps) / max(daytime GHI)
  - energy   e_d = daily PV energy (kWh)
  - HARD_DF share h_d = frozen HARD_DF origins / day origins
- Case assignment (z = standardized over eligible days):
  - C1 clear/stable: argmax z(c) - z(v)
  - C2 cloud-variable: argmax z(v) - z(c) among days with e_d >= median(e)
  - C3 ramp/HARD_DF-rich: argmax h_d, tie-break mean daytime |dP|
- Distinctness: if one day wins two cases it is assigned to the case with its
  larger standardized margin; the other case takes its runner-up.
- Display seed: 42 (first element of the frozen seed vector), fixed in advance;
  no seed selection by performance.

## Chosen days

```
{out[['case', 'day', 'rule']].to_string(index=False)}
```

Feature values of the chosen days:

```
{out[['case', 'day', 'clearness', 'flicker', 'hard_df_share', 'energy_kwh']].round(3).to_string(index=False)}
```

Top-ranked days per criterion (for transparency):

C1 candidates (by z(c)-z(v)):
```
{top(cand_clear, 's_clear')}
```

C2 candidates (by z(v)-z(c), energy >= median):
```
{top(var_pool, 's_var')}
```

C3 candidates (by HARD_DF share):
```
{top(cand_hard, 'hard_df_share')}
```

## Independence statement

The selection code reads only `2025_core.csv` (power/GHI), window origin
timestamps, and the frozen HARD_DF boolean labels. It does not open any
`intervals_cross_seed*.parquet` lo/up column, conformal quantile, score, or
metric file; those are consumed only by the plotting script
(`fig7_interval_examples.py`), which reads `Fig7_case_days.csv` as its only
day input. HARD_DF is defined from input-window and target dynamics by
thresholds frozen before any 2025 evaluation (see
`results/stage8_final_architecture/14_state_definition_final_audit.md`), so its
density is a property of the data, not of any compared method.
"""
    (STAGE / "02_INTERVAL_CASE_SELECTION.md").write_text(md, encoding="utf-8")
    print("wrote 02_INTERVAL_CASE_SELECTION.md")


if __name__ == "__main__":
    main()
