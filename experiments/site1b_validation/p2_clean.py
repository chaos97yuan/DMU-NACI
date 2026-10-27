"""Site 1B Phase 2: clean five-variable 5-min dataset (no interpolation)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.site1b_validation import common_1b as C  # noqa: E402


def main() -> None:
    C.OUT.mkdir(parents=True, exist_ok=True)
    raw_rows = 0
    for y in (2022, 2023, 2024, 2025):
        n = sum(1 for _ in open(C.PROJECT_ROOT / f"data/raw/Alice_Springs_{y}.csv")) - 1
        raw_rows += n
    clean = C.build_clean()
    counts = clean.attrs.get("counts", {})
    per_year, win_rows = [], []
    ren = {"active_power_kw": "power", "ghi_wm2": "GHI", "dhi_wm2": "DHI",
           "temperature_c": "T", "relative_humidity_pct": "RH"}
    for y in (2022, 2023, 2024, 2025):
        # coverage statistics are data audit only (no model decision touches 2025
        # before the freeze); computed straight from the clean frame
        g = (clean[clean["timestamp"].dt.year == y].set_index("timestamp")
             .rename(columns=ren)[C.FEAT_ORDER])
        m = (g.notna().all(axis=1)).to_numpy()
        exp = len(g)
        w = C.windows(g)
        per_year.append({"year": y, "grid_rows": int(exp), "all5_valid": int(m.sum()),
                         "all5_pct": round(float(m.mean()) * 100, 2)})
        win_rows.append({"year": y, "valid_windows": len(w["y_kw"])})
        print(per_year[-1], win_rows[-1])
    manifest = {
        "site": C.SITE_ID, "p_rated_kw": C.P_RATED_1B,
        "input_dim": C.INPUT_DIM, "sampling_min": C.SAMPLING_MIN,
        "history_steps": C.HISTORY, "horizon_steps": C.HORIZON,
        "raw_rows_scanned": int(raw_rows), "clean_rows": int(len(clean)),
        "cleaning_counts": counts,
        "rules": {"sentinel >90000 / <-1e8 / inf -> NaN": True,
                  "power > 25 kW -> NaN (auto-removes 2025-02-04 16:30 spike)": True,
                  "night tare (small negative power) kept": True,
                  "radiation [-5,0) -> 0; < -5 -> NaN; > 1500 -> NaN": True,
                  "T outside [-50,60] -> NaN; RH outside [0,100] -> NaN": True,
                  "no interpolation, no backfill; incomplete windows dropped": True},
        "per_year": per_year, "valid_windows": win_rows,
        "dataset_sha16": C.sha16(C.CLEAN_CSV),
        "columns": list(pd.read_csv(C.CLEAN_CSV, nrows=0).columns),
    }
    (C.OUT / "phase2_clean_manifest.json").write_text(json.dumps(manifest, indent=2),
                                                      encoding="utf-8")
    print("phase2 done:", C.CLEAN_CSV)


if __name__ == "__main__":
    main()
