"""Read-only source-endpoint audit; never evaluate forecast performance."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.schema import HISTORY_STEPS, HORIZON_STEPS, SAMPLING_MINUTES

FLAGS = {
    "Active_Power": "power_imputed",
    "Global_Horizontal_Radiation": "ghi_imputed",
    "Diffuse_Horizontal_Radiation": "dhi_imputed",
    "Weather_Temperature_Celsius": "temperature_imputed",
    "Weather_Relative_Humidity": "humidity_imputed",
}
LINEAGE_COLUMNS = ["affected_feature", "missing_timestamp", "left_endpoint_timestamp",
                   "right_endpoint_timestamp", "interpolation_method", "segment_id"]


def sha256(path: Path) -> str:
    """Hash an artifact without modifying it."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str) + "\n")


def is_future_endpoint_dependent(forecast_origin: pd.Timestamp,
                                 history_window_rows: pd.DatetimeIndex,
                                 interpolation_lineage: pd.DataFrame) -> bool:
    """An in-history filled cell is unsafe iff its source right endpoint is > origin.

    Causal forward fills have no right endpoint. Equality at the origin is safe.
    The same predicate is used for every year and split.
    """
    if interpolation_lineage.empty:
        return False
    rows = interpolation_lineage
    return bool((rows.missing_timestamp.isin(history_window_rows)
                 & (pd.to_datetime(rows.right_endpoint_timestamp) > forecast_origin)).any())


def reconstruct_lineage(flags: pd.DataFrame, core: pd.DataFrame | None = None) -> pd.DataFrame:
    """Reconstruct frozen two-sided linear runs from persisted per-feature imputation flags.

    These flags record actual successful fills after QC, not ideal raw missingness.
    Unfilled long gaps remain unmarked. With core supplied, verify every filled
    value against the two saved endpoints without replacing any values.
    """
    ts = pd.DatetimeIndex(pd.to_datetime(flags.timestamp))
    if ts.hasnans or not ts.is_unique or not (ts[1:] - ts[:-1] ==
            pd.Timedelta(minutes=SAMPLING_MINUTES)).all():
        raise ValueError("Lineage requires an aligned complete sampling grid")
    if core is not None and not pd.DatetimeIndex(pd.to_datetime(core.timestamp)).equals(ts):
        raise ValueError("Core and flags not aligned")
    records = []
    for feature, col in FLAGS.items():
        marked = flags[col].to_numpy()
        if not np.isin(marked, [0, 1]).all():
            raise ValueError(f"Invalid imputation flags: {col}")
        transitions = np.diff(np.r_[0, marked, 0].astype(int))
        for start, end in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)):
            if start == 0 or end == len(ts):
                raise ValueError(f"Unresolved endpoint beyond available split: {feature}")
            if core is not None:
                v = core[feature].to_numpy(dtype=float)
                expected = v[start-1] + (v[end]-v[start-1]) * np.arange(1, end-start+1)/(end-start+1)
                if not np.allclose(v[start:end], expected, rtol=1e-10, atol=1e-9):
                    raise ValueError(f"Saved interpolation not reproduced: {feature} {ts[start]}")
            for row in range(start, end):
                records.append(dict(zip(LINEAGE_COLUMNS, [feature, ts[row], ts[start-1], ts[end],
                    "time_linear_two_sided", int(flags.segment_id.iloc[row])])) )
    return pd.DataFrame(records, columns=LINEAGE_COLUMNS)


def audit_candidates(flags: pd.DataFrame, lineage: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Enumerate ALL original candidate origins and retain separate QC/causality decisions."""
    ts = pd.DatetimeIndex(pd.to_datetime(flags.timestamp))
    origin = np.arange(HISTORY_STEPS-1, len(ts)-HORIZON_STEPS)
    start = origin-HISTORY_STEPS+1
    seg = flags.segment_id.to_numpy()
    missing = flags.row_has_core_missing.to_numpy()
    breaks = np.r_[0, np.cumsum(seg[1:] != seg[:-1])]
    bad = np.r_[0, np.cumsum((seg <= 0) | (missing != 0))]
    eligible = (breaks[origin] == breaks[start]) & ((bad[origin+1]-bad[start]) == 0)
    summary = pd.DataFrame({"sample_index": np.arange(len(origin)),
        "forecast_origin_timestamp": ts[origin], "history_end_index": origin,
        "segment_id": seg[origin], "history_qc_eligible": eligible,
        "future_endpoint_dependent": False})
    details = []
    for j, row in lineage.iterrows():
        left = max(HISTORY_STEPS-1, ts.get_loc(row.missing_timestamp))
        right = min(len(ts)-HORIZON_STEPS, ts.get_loc(row.right_endpoint_timestamp))
        for k in range(left, right):
            history = ts[max(0, k-HISTORY_STEPS+1):k+1]
            if not is_future_endpoint_dependent(ts[k], history, lineage.loc[[j]]):
                continue
            idx = k-HISTORY_STEPS+1
            summary.loc[idx, "future_endpoint_dependent"] = True
            details.append({**row.to_dict(), "sample_index": idx,
                "forecast_origin_timestamp": ts[k], "origin_segment_id": int(seg[k]),
                "right_endpoint_minus_origin_minutes": (row.right_endpoint_timestamp-ts[k]).total_seconds()/60,
                "future_endpoint_dependent": True,
                "history_qc_eligible": bool(eligible[idx]),
                "exclusion_reason": "future_interpolation_endpoint"})
    summary["retained"] = summary.history_qc_eligible & ~summary.future_endpoint_dependent
    columns = LINEAGE_COLUMNS + ["sample_index", "forecast_origin_timestamp", "origin_segment_id",
        "right_endpoint_minus_origin_minutes", "future_endpoint_dependent", "history_qc_eligible", "exclusion_reason"]
    return summary, pd.DataFrame(details, columns=columns)


def rule_definition() -> dict:
    source = inspect.getsource(is_future_endpoint_dependent)
    return {"definition": "in-history filled cell with source right endpoint strictly after forecast origin",
            "helper_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "history_steps": HISTORY_STEPS, "sampling_minutes": SAMPLING_MINUTES}


def severity(summary: pd.DataFrame, resolved: bool = True) -> tuple[str, str]:
    """Conservative screening only: any used dependency blocks Phase D."""
    if not resolved:
        return "LEVEL_4_UNRESOLVED", "HOLD"
    if summary.UsedDependent.sum() == 0:
        return ("LEVEL_1_NEGLIGIBLE_UNUSED_ONLY" if summary.RawDependent.sum() else
                "LEVEL_0_CAUSAL_SAFE"), "PASS"
    if (summary.UsedRate >= .001).any():
        return "LEVEL_3_MATERIAL_USED_DEPENDENCY", "FAIL"
    return "LEVEL_2_MINOR_USED_DEPENDENCY", "HOLD"
