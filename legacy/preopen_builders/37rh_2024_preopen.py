#!/usr/bin/env python3
"""Stage37R-H: 2024 preopen causal-safe build and external lock.

This script builds the 2024 target-blind external universe and frozen
preopen predictions. It audits raw timestamps, rebuilds causal-safe windows,
exports H_ms with the frozen causal-clean TCN, computes frozen D/F/N routing
signals, and applies frozen 2022 Calibration q values.

It intentionally does not read 2024 future target Active_Power, compute any
performance metric, train TCN/BSCN, recalibrate q, or touch 2025 data beyond
confirming that a raw file exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import importlib.util
import json
import logging
import math
import os
import pickle
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data.schema import (
    COLUMN_MAP,
    FEATURE_COLUMNS,
    HISTORY_STEPS,
    HORIZON_STEPS,
    NUMERIC_COLUMNS,
    OUTPUT_COLUMNS,
    REQUIRED_RAW_COLUMNS,
    SAMPLING_MINUTES,
    SITE_NAME,
)
from src.eval.compact_final_reliability import CONFORMAL_EPSILON_KW
from src.eval.historical_interpolation_causality_audit import audit_candidates, reconstruct_lineage
from src.features.dynamic_index import DynamicIndexTransformer
from src.features.historical_familiarity import (
    CONTEXT_COLUMNS,
    DAYLIGHT_GHI_THRESHOLD,
    FamiliarityThresholds,
    build_neighbor_index,
    build_query_context,
    compute_distance_statistics,
    distance_to_familiarity,
    query_neighbors,
    transform_context,
    validate_memory_context,
)
from src.features.scaler import FeatureStandardizer
from src.models.bscn import BayesianStochasticConfigurationNetwork


STAGE = "Stage37R-H"
YEAR = 2024
EXPECTED_RAW_ROWS_REFERENCE = 105386
FULL_YEAR_REFERENCE_ROWS = 366 * 24 * 12
EXPECTED_INTERVAL_MINUTES = SAMPLING_MINUTES
EXPECTED_TCN_SHA = "548434f86cc7c6d9d95c70aca9483ea4d8b8a8b6138a1edefb4eabf4b235f762"
EXPECTED_Q_SHA = "0f4a4a096427915f9e50927a478eda905b53fbd39afb5fca79826516c039750b"
EXPECTED_2023_FINAL_SHA = "9cce649759c883004503fca355e32ee1fb975bd1c890b2a847026fd79817ed54"
EXPECTED_MASTER_SHA = "eeab54c009cd4d9d13e202119919ac01fc28135c18bab478835882743f81028a"
EXPECTED_H_PROTOCOL_SHA = "43f9d6c383fe17b4eddc49daab253623ef88c6adf5510caac19c83a3116253fd"
EXPECTED_I_PROTOCOL_SHA = "c9ebdd0d3106b9ce5a0dd0ebbb43c7bd5945454b73664940133ae0c8dad444fa"
SEEDS = (42, 123, 2026, 3407, 8888)
COMPACT_NODES = {42: 11, 123: 12, 2026: 18, 3407: 12, 8888: 36}
ALPHAS = (0.10, 0.05)
GROUPS = ("LOW_N", "MID_N", "HIGH_N")
METHODS = ("M0", "MG", "RMG", "MP")
LAMBDA_FINAL = 0.1
FAMILIARITY_K = 20
HMS_BATCH_SIZE = 512

PROTOCOL_DIR = PROJECT_ROOT / "outputs" / "protocols" / "postexternal_extensions"
POST_G_DIR = PROJECT_ROOT / "outputs" / "metrics" / "stage37r_postexternal_lock"
STAGE_E_DIR = PROJECT_ROOT / "outputs" / "metrics" / "stage37r_preexternal_freeze"
CAUSAL_ROOT = PROJECT_ROOT / "data" / "processed" / "causal_clean_2022"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "external_2024_preopen"
FEATURE_DIR = PROJECT_ROOT / "outputs" / "features" / "stage37r_2024_preopen"
PRED_DIR = PROJECT_ROOT / "outputs" / "predictions" / "stage37r_2024_preopen"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics" / "stage37r_2024_preopen"
LOG_PATH = PROJECT_ROOT / "logs" / "37rh_2024_preopen.log"

H_PROTOCOL = PROTOCOL_DIR / "stage37r_h_2024_preopen_protocol.json"
I_PROTOCOL = PROTOCOL_DIR / "stage37r_i_2024_external_protocol.json"
FINAL_2023_LOCK = POST_G_DIR / "FINAL_2023_EXTERNAL_LOCK.json"
MASTER_LOCK = POST_G_DIR / "POSTEXTERNAL_EXTENSION_MASTER_LOCK.json"
G_DECISION = POST_G_DIR / "stage37r_g_decision.json"
OFFICIAL_Q = STAGE_E_DIR / "stage37r_official_calibration_quantiles.json"
TCN_CKPT = PROJECT_ROOT / "outputs" / "checkpoints" / "causal_clean_2022" / "multiscale_tcn_causal_clean_best_cuda.pt"
INPUT_SCALER = PROJECT_ROOT / "outputs" / "checkpoints" / "causal_clean_2022" / "input_scaler_causal_clean.json"
HMS_SCALER = CAUSAL_ROOT / "h_ms" / "h_ms_scaler_causal_clean_cuda.pkl"
BSCN_CONFIG = PROJECT_ROOT / "outputs" / "checkpoints" / "bscn_baseline_config.json"
BSCN_DIR = PROJECT_ROOT / "outputs" / "checkpoints" / "causal_clean_2022" / "bscn"
DYNAMIC_CONFIG = CAUSAL_ROOT / "dynamic_index_config.json"
MEMORY_2021 = PROJECT_ROOT / "data" / "processed" / "history" / "site18_2021_memory.csv"
CONTEXT_SCALER = PROJECT_ROOT / "data" / "processed" / "history" / "historical_context_scaler.pkl"
FAMILIARITY_THRESHOLDS = CAUSAL_ROOT / "historical_familiarity_thresholds.json"
TRAIN_THRESHOLDS = CAUSAL_ROOT / "train_derived_thresholds.json"


class Stage37RHError(RuntimeError):
    """Raised when a Stage37R-H preopen invariant fails."""


def setup_logging() -> logging.Logger:
    """Configure console and file logging for this standalone stage."""
    for path in (PROCESSED_DIR, FEATURE_DIR, PRED_DIR, METRICS_DIR, LOG_PATH.parent):
        path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage37r_h_2024_preopen")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def sha256(path: Path) -> str:
    """Return a streaming SHA256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    """Convert pandas/numpy objects to JSON-safe primitives."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat(sep=" ")
    if isinstance(value, Path):
        return str(value.relative_to(PROJECT_ROOT)) if value.is_absolute() and value.is_relative_to(PROJECT_ROOT) else str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_sha_sidecar(path: Path, sidecar: Path | None = None) -> str:
    """Write a SHA256 sidecar and return the digest."""
    digest = sha256(path)
    out = sidecar if sidecar is not None else path.with_suffix(path.suffix + ".sha256")
    out.write_text(digest + "\n", encoding="utf-8")
    return digest


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file with a clear error message."""
    if not path.is_file():
        raise Stage37RHError(f"Missing required JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_protocol_integrity() -> dict[str, Any]:
    """Verify Stage37R-G locks and frozen 2024 protocol SHAs."""
    decision = load_json(G_DECISION).get("decision")
    checks = {
        "stage37r_g_decision": decision == "POSTEXTERNAL_EXTENSION_READY",
        "2023_final_lock": sha256(FINAL_2023_LOCK) == EXPECTED_2023_FINAL_SHA,
        "extension_master_lock": sha256(MASTER_LOCK) == EXPECTED_MASTER_SHA,
        "2024_preopen_protocol": sha256(H_PROTOCOL) == EXPECTED_H_PROTOCOL_SHA,
        "2024_future_external_protocol": sha256(I_PROTOCOL) == EXPECTED_I_PROTOCOL_SHA,
    }
    if not all(checks.values()):
        raise Stage37RHError(f"Protocol integrity check failed: {checks}")
    h_protocol = load_json(H_PROTOCOL)
    raw_avail = h_protocol.get("raw_availability", {})
    boundary_ok = (
        h_protocol.get("stage") == STAGE
        and raw_avail.get("year") == YEAR
        and raw_avail.get("first_timestamp") == "2024-01-01 00:00:00"
        and raw_avail.get("last_timestamp") == "2024-12-31 23:55:00"
        and h_protocol.get("future_target_performance_opened") is False
    )
    if not boundary_ok:
        raise Stage37RHError("YEAR_BOUNDARY_PROTOCOL_AMBIGUOUS")
    return {
        "checks": checks,
        "annual_boundary_rule": "interpreted from frozen Stage37R-H raw_availability: timestamps in [2024-01-01 00:00:00, 2025-01-01 00:00:00)",
        "h_protocol": h_protocol,
        "i_protocol_sha256": sha256(I_PROTOCOL),
    }


def identify_2024_raw(protocol: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Identify the unique 2024 Site18 raw source without reading 2025 content."""
    expected = protocol["raw_availability"]
    raw_dir = PROJECT_ROOT / "data" / "raw"
    candidates = [p for p in raw_dir.glob("*.csv") if "2024" in p.name and not p.name.endswith(":Zone.Identifier")]
    rows = []
    for path in candidates:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        schema_ok = all(col in header for col in REQUIRED_RAW_COLUMNS)
        if not schema_ok:
            rows.append({"path": path, "schema_ok": False})
            continue
        ts = pd.to_datetime(pd.read_csv(path, usecols=["timestamp"])["timestamp"], errors="coerce")
        rows.append(
            {
                "path": path,
                "schema_ok": True,
                "sha256": sha256(path),
                "rows": int(len(ts)),
                "first_timestamp": str(ts.min()),
                "last_timestamp": str(ts.max()),
                "site_token_ok": "Alice_Springs" in path.name,
                "reference_rows_ok": int(len(ts)) == EXPECTED_RAW_ROWS_REFERENCE,
            }
        )
    matches = [
        row for row in rows
        if row.get("schema_ok")
        and row.get("sha256") == expected["sha256"]
        and row.get("rows") == expected["row_count"]
        and row.get("first_timestamp") == expected["first_timestamp"]
        and row.get("last_timestamp") == expected["last_timestamp"]
        and row.get("site_token_ok")
    ]
    if len(matches) != 1:
        raise Stage37RHError(f"2024_RAW_SOURCE_AMBIGUOUS: candidates={json_safe(rows)}")
    raw_path = Path(matches[0]["path"])
    exists_2025 = any(p.name.endswith("2025.csv") for p in raw_dir.glob("*.csv"))
    audit = {
        "candidate_count": len(candidates),
        "candidates": rows,
        "selected_path": raw_path,
        "stage37r_g_reference_rows": EXPECTED_RAW_ROWS_REFERENCE,
        "raw_2025_exists_checked_by_filename_only": bool(exists_2025),
        "raw_2025_content_read": False,
        "PASS": True,
    }
    return raw_path, audit


def timestamp_gap_table(missing: pd.DatetimeIndex) -> pd.DataFrame:
    """Summarize missing 5-minute timestamp runs in the full-year grid."""
    if len(missing) == 0:
        return pd.DataFrame(columns=["gap_id", "start_timestamp", "end_timestamp", "missing_points", "duration_minutes"])
    groups = []
    start = missing[0]
    prev = missing[0]
    for ts in missing[1:]:
        if ts - prev == pd.Timedelta(minutes=SAMPLING_MINUTES):
            prev = ts
            continue
        groups.append((start, prev))
        start = prev = ts
    groups.append((start, prev))
    rows = []
    for i, (a, b) in enumerate(groups, start=1):
        n = int((b - a) / pd.Timedelta(minutes=SAMPLING_MINUTES)) + 1
        rows.append({"gap_id": i, "start_timestamp": a, "end_timestamp": b, "missing_points": n, "duration_minutes": n * SAMPLING_MINUTES})
    return pd.DataFrame(rows)


def raw_data_audit(raw_path: Path) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    """Audit raw 2024 timestamps and calendar coverage before cleaning."""
    raw = pd.read_csv(raw_path, usecols=REQUIRED_RAW_COLUMNS)
    header = pd.read_csv(raw_path, nrows=0).columns.tolist()
    ts = pd.to_datetime(raw["timestamp"], errors="coerce")
    sorted_ts = ts.dropna().sort_values().reset_index(drop=True)
    diffs = sorted_ts.diff().dropna()
    interval_counts = diffs.value_counts().sort_index()
    five_min = pd.Timedelta(minutes=SAMPLING_MINUTES)
    timeline = pd.date_range(f"{YEAR}-01-01 00:00:00", f"{YEAR}-12-31 23:55:00", freq=f"{SAMPLING_MINUTES}min")
    present = pd.DatetimeIndex(sorted_ts.drop_duplicates())
    missing = timeline.difference(present)
    extra_before = present[present < timeline[0]]
    extra_after = present[present > timeline[-1]]
    gaps = timestamp_gap_table(missing)
    gaps.to_csv(METRICS_DIR / "2024_timestamp_gap_audit.csv", index=False)
    raw_audit = {
        "raw_file_path": raw_path,
        "file_sha256": sha256(raw_path),
        "row_count": int(len(raw)),
        "column_count": int(len(header)),
        "schema": "DKASC Site18 mapped by src.data.schema.COLUMN_MAP",
        "site_id": SITE_NAME,
        "required_schema_present": all(col in header for col in REQUIRED_RAW_COLUMNS),
        "first_timestamp": str(sorted_ts.min()),
        "last_timestamp": str(sorted_ts.max()),
        "unique_timestamps": int(ts.nunique(dropna=True)),
        "duplicate_timestamps": int(ts.duplicated(keep=False).sum()),
        "timestamp_nat_count": int(ts.isna().sum()),
        "time_order_violations": int((ts.diff().dropna() <= pd.Timedelta(0)).sum()),
        "sampling_interval_distribution_minutes": {str(int(k / pd.Timedelta(minutes=1))): int(v) for k, v in interval_counts.items()},
        "five_min_interval_ratio": float((diffs == five_min).mean()) if len(diffs) else math.nan,
        "missing_timestamps": int(len(missing)),
        "gap_segments": int(len(gaps)),
        "long_gap_segments": int((gaps["duration_minutes"] >= 30).sum()) if not gaps.empty else 0,
        "maximum_gap_duration_minutes": int(gaps["duration_minutes"].max()) if not gaps.empty else 0,
        "PASS": True,
    }
    calendar_audit = {
        "year": YEAR,
        "leap_year": True,
        "full_year_reference_rows": FULL_YEAR_REFERENCE_ROWS,
        "boundary_rule": "[2024-01-01 00:00:00, 2025-01-01 00:00:00)",
        "expected_first_timestamp": str(timeline[0]),
        "expected_last_timestamp": str(timeline[-1]),
        "actual_first_timestamp": str(sorted_ts.min()),
        "actual_last_timestamp": str(sorted_ts.max()),
        "unique_timestamps_in_boundary": int(len(present.intersection(timeline))),
        "missing_timestamps_in_boundary": int(len(missing)),
        "extra_timestamps_before_boundary": int(len(extra_before)),
        "extra_timestamps_after_boundary": int(len(extra_after)),
        "duplicates": int(ts.duplicated(keep=False).sum()),
        "coverage_complete": bool(len(missing) == 0 and len(extra_before) == 0 and len(extra_after) == 0),
        "PASS": True,
    }
    write_json(METRICS_DIR / "2024_raw_data_audit.json", raw_audit)
    write_json(METRICS_DIR / "2024_calendar_coverage_audit.json", calendar_audit)
    if raw_audit["row_count"] != EXPECTED_RAW_ROWS_REFERENCE:
        raise Stage37RHError(f"2024 raw row count differs from Stage37R-G reference: {raw_audit['row_count']} != {EXPECTED_RAW_ROWS_REFERENCE}")
    return raw_audit, calendar_audit, raw


def load_clean_module() -> Any:
    """Load scripts/02_clean_data.py as a reusable module."""
    path = PROJECT_ROOT / "scripts" / "02_clean_data.py"
    spec = importlib.util.spec_from_file_location("stage37_clean_data", path)
    if spec is None or spec.loader is None:
        raise Stage37RHError(f"Cannot import cleaning module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean_2024_raw(raw: pd.DataFrame, logger: logging.Logger) -> dict[str, Any]:
    """Apply the frozen 2022 cleaning functions to a 2024 full grid."""
    clean_mod = load_clean_module()
    df = raw.rename(columns=COLUMN_MAP).loc[:, OUTPUT_COLUMNS].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    start = pd.Timestamp("2024-01-01 00:00:00")
    end = pd.Timestamp("2024-12-31 23:55:00")
    in_year = (df["timestamp"] >= start) & (df["timestamp"] <= end)
    df = df.loc[in_year].copy()
    if df["timestamp"].duplicated().any():
        raise Stage37RHError("Duplicate timestamps in 2024 raw data; no frozen duplicate resolver.")
    timeline = pd.date_range(start, end, freq=f"{SAMPLING_MINUTES}min")
    reindexed = df.sort_values("timestamp").set_index("timestamp").reindex(timeline)
    reindexed.index.name = "timestamp"
    reindexed = reindexed.reset_index()
    flags = pd.DataFrame({"timestamp": reindexed["timestamp"]})
    flags["timestamp_was_missing"] = flags["timestamp"].isin(timeline.difference(pd.DatetimeIndex(df["timestamp"]))).astype(int)
    flags = clean_mod.initialize_quality_flags(flags)
    before_qc = reindexed.copy()
    cleaned, flags, active_stats = clean_mod.clean_active_power(reindexed, flags)
    cleaned, flags, radiation_stats = clean_mod.clean_radiation_values(cleaned, flags)
    cleaned, flags, humidity_stats = clean_mod.clean_humidity(cleaned, flags, logger)
    cleaned, flags, rainfall_stats = clean_mod.clean_rainfall(cleaned, flags)
    qc_before_imputation = cleaned.copy()
    cleaned, flags, interpolation_stats = clean_mod.apply_variable_specific_interpolation(cleaned, flags)
    cleaned, flags = clean_mod.build_segment_ids(cleaned, flags)
    core = clean_mod.build_core_dataset(cleaned)
    extended = clean_mod.build_extended_dataset(cleaned)
    for name, frame in (("core", core), ("extended", extended), ("flags", flags), ("qc_before_imputation", qc_before_imputation)):
        frame.to_csv(PROCESSED_DIR / f"2024_{name}.csv", index=False)
    audit = {
        "cleaning_source": "scripts/02_clean_data.py frozen functions",
        "full_grid_rows": int(len(core)),
        "timestamp_was_missing": int(flags["timestamp_was_missing"].sum()),
        "core_missing_rows": int(core[[c for c in FEATURE_COLUMNS]].isna().any(axis=1).sum()),
        "segment_count": int((flags["segment_id"] > 0).astype(int).diff().fillna(0).eq(1).sum() + (flags["segment_id"].iloc[0] > 0)),
        "active_stats": active_stats,
        "radiation_stats": radiation_stats,
        "humidity_stats": humidity_stats,
        "rainfall_stats": rainfall_stats,
        "interpolation_stats": interpolation_stats,
        "before_qc_rows": int(len(before_qc)),
        "PASS": True,
    }
    write_json(METRICS_DIR / "2024_cleaning_audit.json", audit)
    return {"core": core, "extended": extended, "flags": flags, "audit": audit}


def build_causal_safe_windows(core: pd.DataFrame, flags: pd.DataFrame) -> dict[str, Any]:
    """Build target-blind 2024 causal-safe windows from cleaned full-grid data."""
    timestamps = pd.to_datetime(core["timestamp"]).reset_index(drop=True)
    if not timestamps.equals(pd.to_datetime(flags["timestamp"]).reset_index(drop=True)):
        raise Stage37RHError("2024 core/flags timestamps are not aligned")
    if not timestamps.diff().dropna().eq(pd.Timedelta(minutes=SAMPLING_MINUTES)).all():
        raise Stage37RHError("2024 cleaned core is not a complete 5-minute grid")
    lineage = reconstruct_lineage(flags, core)
    candidates, details = audit_candidates(flags, lineage)
    details = details.copy()
    if not details.empty:
        details = details.rename(columns={"origin_segment_id": "segment_id"})
        details["right_endpoint_minus_origin_minutes"] = details["right_endpoint_minus_origin_minutes"].astype(float)
    details.to_csv(METRICS_DIR / "2024_future_endpoint_dependency_cells.csv", index=False)
    affected = candidates.loc[candidates["future_endpoint_dependent"]].copy()
    affected["exclusion_reason"] = "future_interpolation_endpoint"
    affected.to_csv(METRICS_DIR / "2024_future_endpoint_excluded_origins.csv", index=False)

    features = core.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    seg = flags["segment_id"].to_numpy(dtype=np.int64)
    missing = flags["row_has_core_missing"].fillna(1).to_numpy(dtype=np.int8)
    imputed_cols = ["power_imputed", "ghi_imputed", "dhi_imputed", "temperature_imputed", "humidity_imputed"]
    imputed_any = flags[imputed_cols].fillna(0).max(axis=1).to_numpy(dtype=np.int8)
    future_dep = candidates["future_endpoint_dependent"].to_numpy(dtype=bool)

    total_candidates = len(candidates)
    X_rows: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []
    reject_gap = reject_missing = reject_future = 0
    horizon_td = pd.Timedelta(minutes=HORIZON_STEPS * SAMPLING_MINUTES)
    year_start = pd.Timestamp("2024-01-01 00:00:00")
    year_end = pd.Timestamp("2025-01-01 00:00:00")

    for start_idx in range(total_candidates):
        history_end = start_idx + HISTORY_STEPS - 1
        target_idx = history_end + HORIZON_STEPS
        origin_ts = timestamps.iloc[history_end]
        target_ts = origin_ts + horizon_td
        if not (year_start <= origin_ts < year_end and year_start <= target_ts < year_end):
            reject_gap += 1
            continue
        hseg = seg[start_idx : history_end + 1]
        if hseg[0] <= 0 or not np.all(hseg == hseg[0]):
            reject_gap += 1
            continue
        if np.any(missing[start_idx : history_end + 1] != 0) or not np.isfinite(features[start_idx : history_end + 1]).all():
            reject_missing += 1
            continue
        if future_dep[start_idx]:
            reject_future += 1
            continue
        X_rows.append(features[start_idx : history_end + 1].astype(np.float32, copy=True))
        meta_rows.append(
            {
                "sample_index": len(meta_rows),
                "candidate_index": start_idx,
                "history_start_index": start_idx,
                "history_end_index": history_end,
                "target_index_bookkeeping_only": target_idx,
                "history_start_timestamp": timestamps.iloc[start_idx],
                "history_end_timestamp": origin_ts,
                "forecast_origin_timestamp": origin_ts,
                "target_timestamp": target_ts,
                "global_segment_id": int(hseg[0]),
                "history_has_any_imputation": int(imputed_any[start_idx : history_end + 1].max()),
                "origin_GHI": float(core["Global_Horizontal_Radiation"].iloc[history_end]),
                "causal_safe": True,
            }
        )
    if not X_rows:
        raise Stage37RHError("No 2024 causal-safe windows survived")
    X = np.stack(X_rows).astype(np.float32, copy=False)
    meta = pd.DataFrame(meta_rows)
    if not np.array_equal(meta["sample_index"].to_numpy(dtype=int), np.arange(len(meta))):
        raise Stage37RHError("Manifest sample_index is not sequential")
    if not pd.to_datetime(meta["forecast_origin_timestamp"]).is_monotonic_increasing:
        raise Stage37RHError("Manifest origins are not sorted")
    if not ((pd.to_datetime(meta["target_timestamp"]) - pd.to_datetime(meta["forecast_origin_timestamp"])) == horizon_td).all():
        raise Stage37RHError("Target timestamp is not origin + 15 minutes")
    retained_candidates = candidates.loc[meta["candidate_index"].to_numpy(dtype=int)]
    remaining = int(retained_candidates["future_endpoint_dependent"].sum())
    if remaining != 0:
        raise Stage37RHError("2024_CAUSAL_BUILD_FAILED: retained future dependency > 0")
    np.save(PROCESSED_DIR / "X_2024_preopen.npy", X)
    meta.to_csv(PROCESSED_DIR / "2024_causal_safe_manifest.csv", index=False)
    meta.to_csv(METRICS_DIR / "2024_causal_safe_manifest.csv", index=False)
    audit = {
        "history_steps": HISTORY_STEPS,
        "horizon_steps": HORIZON_STEPS,
        "sampling_minutes": SAMPLING_MINUTES,
        "candidate_origins": int(total_candidates),
        "retained_count": int(len(meta)),
        "excluded_total": int(total_candidates - len(meta)),
        "excluded_due_gap_or_boundary": int(reject_gap),
        "excluded_due_history_missing_or_nonfinite": int(reject_missing),
        "excluded_due_future_interpolation_endpoint": int(reject_future),
        "future_dependent_cells": int(len(details)),
        "unique_affected_origins": int(affected["sample_index"].nunique()) if not affected.empty else 0,
        "median_right_endpoint_delay_minutes": float(details["right_endpoint_minus_origin_minutes"].median()) if not details.empty else 0.0,
        "max_right_endpoint_delay_minutes": float(details["right_endpoint_minus_origin_minutes"].max()) if not details.empty else 0.0,
        "remaining_future_dependent": remaining,
        "target_finiteness_used_for_filtering": False,
        "future_power_used": False,
        "PASS": True,
    }
    write_json(METRICS_DIR / "2024_causal_integrity_audit.json", audit)
    return {"X": X, "metadata": meta, "lineage": lineage, "candidate_summary": candidates, "dependency_details": details, "audit": audit}


def load_tcn_model() -> Any:
    """Load the frozen causal-clean CUDA TCN checkpoint for eval inference."""
    import torch
    from src.models.multiscale_tcn import MultiScaleTCN

    if sha256(TCN_CKPT) != EXPECTED_TCN_SHA:
        raise Stage37RHError("Frozen TCN SHA mismatch")
    checkpoint = torch.load(TCN_CKPT, map_location="cpu")
    config = checkpoint["model_config"]
    model = MultiScaleTCN(**config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def export_hms(X: np.ndarray) -> dict[str, Any]:
    """Transform raw windows and export frozen TCN H_ms plus BSCN-standardized H_ms."""
    import joblib
    import torch

    scaler = FeatureStandardizer.load(INPUT_SCALER)
    X_scaled = scaler.transform(X).astype(np.float32, copy=False)
    model, checkpoint = load_tcn_model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(X_scaled), HMS_BATCH_SIZE):
            xb = torch.as_tensor(X_scaled[start : start + HMS_BATCH_SIZE], dtype=torch.float32, device=device)
            hb = model.forward_features(xb)
            if hb.ndim != 2 or hb.shape[1] != model.embedding_dim or not torch.isfinite(hb).all():
                raise Stage37RHError(f"Invalid H_ms batch shape/finite: {tuple(hb.shape)}")
            chunks.append(hb.cpu().numpy().astype(np.float32, copy=False))
        h1 = model.forward_features(torch.as_tensor(X_scaled[:32], dtype=torch.float32, device=device))
        h2 = model.forward_features(torch.as_tensor(X_scaled[:32], dtype=torch.float32, device=device))
        deterministic = bool(torch.allclose(h1, h2, rtol=1e-6, atol=1e-6))
    if not deterministic:
        raise Stage37RHError("Frozen TCN eval forward is not deterministic")
    hms = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    with HMS_SCALER.open("rb") as handle:
        hms_scaler = pickle.load(handle)
    hms_std = hms_scaler.transform(hms.astype(np.float64, copy=False)).astype(np.float32, copy=False)
    if hms.shape != (len(X), 64) or hms_std.shape != (len(X), 64):
        raise Stage37RHError(f"H_ms shape mismatch: raw={hms.shape} std={hms_std.shape}")
    if not np.isfinite(hms).all() or not np.isfinite(hms_std).all():
        raise Stage37RHError("H_ms contains NaN/Inf")
    raw_path = FEATURE_DIR / "hms_2024_preopen.npy"
    std_path = PROCESSED_DIR / "hms_external_2024_std.npy"
    np.save(raw_path, hms.astype(np.float32, copy=False))
    np.save(std_path, hms_std.astype(np.float32, copy=False))
    audit = {
        "tcn_checkpoint": TCN_CKPT,
        "tcn_sha256": sha256(TCN_CKPT),
        "input_scaler": INPUT_SCALER,
        "input_scaler_sha256": sha256(INPUT_SCALER),
        "hms_scaler": HMS_SCALER,
        "hms_scaler_sha256": sha256(HMS_SCALER),
        "device": str(device),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_best_val_rmse": float(checkpoint.get("best_val_rmse", math.nan)),
        "shape": list(hms.shape),
        "std_shape": list(hms_std.shape),
        "dtype": str(hms.dtype),
        "finite": True,
        "deterministic_forward": deterministic,
        "training_executed": False,
        "path": raw_path,
        "std_path": std_path,
        "hms_sha256": sha256(raw_path),
        "hms_std_sha256": sha256(std_path),
        "PASS": True,
    }
    write_json(METRICS_DIR / "2024_h_ms_integrity.json", audit)
    return {"hms": hms, "hms_std": hms_std, "audit": audit}


def compute_dynamic_and_familiarity(X: np.ndarray, metadata: pd.DataFrame) -> dict[str, Any]:
    """Compute frozen D, F, N and routing groups without target access."""
    dyn_cfg = load_json(DYNAMIC_CONFIG)
    transformer = DynamicIndexTransformer.from_dict(dyn_cfg)
    dyn = transformer.transform(X, return_components=True)
    D = np.asarray(dyn.dynamic_index, dtype=np.float64)
    if not np.isfinite(D).all() or D.min() < -1e-8 or D.max() > 1 + 1e-8:
        raise Stage37RHError("D outside finite [0,1]")

    import joblib

    memory = pd.read_csv(MEMORY_2021, parse_dates=["timestamp", "future_timestamp"])
    validate_memory_context(memory)
    if set(memory["timestamp"].dt.year.unique().astype(int)) != {2021}:
        raise Stage37RHError("Historical memory is not 2021-only")
    context_scaler = joblib.load(CONTEXT_SCALER)
    query_context = build_query_context(X, metadata, split="external_2024")
    query_scaled = transform_context(query_context, context_scaler)
    memory_scaled = transform_context(memory.loc[:, CONTEXT_COLUMNS], context_scaler)
    is_daylight = (query_context["ghi"].to_numpy(dtype=np.float64) > DAYLIGHT_GHI_THRESHOLD).astype(np.int8)
    day_pos = np.flatnonzero(memory["is_daylight"].to_numpy(dtype=int) == 1)
    night_pos = np.flatnonzero(memory["is_daylight"].to_numpy(dtype=int) == 0)
    memory_ids = memory["memory_id"].to_numpy(dtype=np.int64)
    day_index, day_backend = build_neighbor_index(memory_scaled[day_pos], backend="auto")
    night_index, night_backend = build_neighbor_index(memory_scaled[night_pos], backend="auto")
    if day_backend != night_backend:
        raise Stage37RHError("Familiarity day/night KNN backend mismatch")
    distances = np.empty((len(X), FAMILIARITY_K), dtype=np.float32)
    day_q = np.flatnonzero(is_daylight == 1)
    night_q = np.flatnonzero(is_daylight == 0)
    if len(day_q):
        res = query_neighbors(query_scaled[day_q], day_index, memory_ids, day_pos, k=FAMILIARITY_K, backend_name=day_backend)
        distances[day_q] = res.neighbor_distances
    if len(night_q):
        res = query_neighbors(query_scaled[night_q], night_index, memory_ids, night_pos, k=FAMILIARITY_K, backend_name=night_backend)
        distances[night_q] = res.neighbor_distances
    dist_stats = compute_distance_statistics(distances)
    thr = load_json(FAMILIARITY_THRESHOLDS)
    if "day" in thr:
        thresholds = FamiliarityThresholds(
            day_q05=float(thr["day"]["Q05"]),
            day_q95=float(thr["day"]["Q95"]),
            night_q05=float(thr["night"]["Q05"]),
            night_q95=float(thr["night"]["Q95"]),
            source="2022_train_queries_only",
        )
    else:
        thresholds = FamiliarityThresholds(
            day_q05=float(thr["day_q05"]),
            day_q95=float(thr["day_q95"]),
            night_q05=float(thr["night_q05"]),
            night_q95=float(thr["night_q95"]),
            source=str(thr.get("source", "2022_train_queries_only")),
        )
    F32, _N32 = distance_to_familiarity(dist_stats["hist_mean_distance"], is_daylight, thresholds)
    F = F32.astype(np.float64)
    N = 1.0 - F
    if not np.allclose(F + N, 1.0, rtol=0, atol=1e-12):
        raise Stage37RHError("N=1-F identity failed")
    train_thr = load_json(TRAIN_THRESHOLDS)
    groups = assign_n_groups(N, train_thr)
    hard = (D >= float(train_thr["D_Q75"])) & (F <= float(train_thr["F_Q25"]))
    context = metadata[["sample_index", "forecast_origin_timestamp", "target_timestamp", "global_segment_id"]].copy()
    context["D"] = D
    context["F"] = F
    context["N"] = N
    context["N_group"] = groups
    context["HARD_DF"] = hard
    context["deployment_daylight"] = is_daylight
    context.to_csv(PROCESSED_DIR / "2024_context_preopen.csv", index=False)
    np.save(PROCESSED_DIR / "d_2024_preopen.npy", D.astype(np.float32))
    np.save(PROCESSED_DIR / "f_2024_preopen.npy", F.astype(np.float32))
    np.save(PROCESSED_DIR / "n_2024_preopen.npy", N.astype(np.float32))
    return {
        "D": D,
        "F": F,
        "N": N,
        "groups": groups,
        "hard_df": hard,
        "is_daylight": is_daylight,
        "context": context,
        "audit": {
            "memory_path": MEMORY_2021,
            "memory_years": [2021],
            "memory_rows": int(len(memory)),
            "context_scaler": CONTEXT_SCALER,
            "familiarity_thresholds": FAMILIARITY_THRESHOLDS,
            "N_thresholds": {"N_Q25": float(train_thr["N_Q25"]), "N_Q75": float(train_thr["N_Q75"])},
            "D_F_thresholds": {"D_Q75": float(train_thr["D_Q75"]), "F_Q25": float(train_thr["F_Q25"])},
            "k_neighbors": FAMILIARITY_K,
            "knn_backend": day_backend,
            "N_identity": True,
            "fit_on_2024": False,
            "PASS": True,
        },
    }


def assign_n_groups(N: np.ndarray, thresholds: dict[str, Any]) -> np.ndarray:
    """Assign frozen LOW/MID/HIGH N groups from Stage37R-A thresholds."""
    labels = np.full(len(N), "MID_N", dtype=object)
    labels[N <= float(thresholds["N_Q25"])] = "LOW_N"
    labels[N >= float(thresholds["N_Q75"])] = "HIGH_N"
    return labels


def group_support(context: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute target-blind support counts for N groups and HARD_DF."""
    rows = []
    ts = pd.to_datetime(context["target_timestamp"])
    dates = ts.dt.date
    months = ts.dt.to_period("M").astype(str)
    masks = {g: context["N_group"].to_numpy() == g for g in GROUPS}
    masks["HARD_DF"] = context["HARD_DF"].to_numpy(dtype=bool)
    for group, mask in masks.items():
        rows.append(
            {
                "group": group,
                "samples": int(mask.sum()),
                "fraction": float(mask.mean()) if len(mask) else 0.0,
                "unique_days": int(pd.Series(dates[mask]).nunique()),
                "months": int(pd.Series(months[mask]).nunique()),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(METRICS_DIR / "2024_group_support.csv", index=False)
    return table, {"rows": table.to_dict(orient="records"), "PASS": bool((table["samples"] > 0).all())}


def q_lookup(bundle: dict[str, Any], seed: int, alpha: float) -> dict[str, Any]:
    """Return official q family dictionaries for seed and alpha."""
    return bundle["seeds"][f"seed_{seed}"]["groups"][f"alpha_{alpha:.2f}"]


def predict_kw(model: BayesianStochasticConfigurationNetwork, X: np.ndarray, target_scaler: FeatureStandardizer, target_std: float) -> tuple[np.ndarray, np.ndarray]:
    """Predict BSCN posterior mean and total predictive std in kW."""
    mu_scaled, sigma_scaled = model.predict_distribution(X)
    mu_kw = target_scaler.inverse_transform_target(mu_scaled).astype(np.float64)
    sigma_kw = sigma_scaled.astype(np.float64) * target_std
    return mu_kw, sigma_kw


def assert_preopen_columns(frame: pd.DataFrame, path: Path) -> None:
    """Ensure no target/performance columns are present in preopen predictions."""
    forbidden_tokens = ("y_true", "target_value", "actual_power", "error", "abs_error", "covered", "picp", "interval_score", "residual", "rmse", "mae")
    bad = [col for col in frame.columns if any(tok in col.lower() for tok in forbidden_tokens)]
    if bad:
        raise Stage37RHError(f"Forbidden preopen columns in {path}: {bad}")


def collect_model_hashes() -> dict[str, Any]:
    """Verify frozen TCN and compact BSCN SHA256 digests and node counts."""
    if sha256(TCN_CKPT) != EXPECTED_TCN_SHA:
        raise Stage37RHError("TCN SHA mismatch")
    bscn = {}
    for seed in SEEDS:
        manifest = load_json(BSCN_DIR / f"seed{seed}" / "manifest.json")
        model_path = PROJECT_ROOT / manifest["compact_model"]
        digest = sha256(model_path)
        if digest != manifest["compact_sha256"]:
            raise Stage37RHError(f"seed {seed}: compact BSCN SHA mismatch")
        model = BayesianStochasticConfigurationNetwork.load(model_path)
        if int(model.best_node_count_) != COMPACT_NODES[seed]:
            raise Stage37RHError(f"seed {seed}: node count {model.best_node_count_} != frozen {COMPACT_NODES[seed]}")
        bscn[str(seed)] = {"path": manifest["compact_model"], "sha256": digest, "nodes": int(model.best_node_count_)}
    return {"tcn_cuda": {"path": str(TCN_CKPT.relative_to(PROJECT_ROOT)), "sha256": sha256(TCN_CKPT)}, "compact_bscn": bscn}


def build_preopen_predictions(metadata: pd.DataFrame, context: pd.DataFrame, hms_std: np.ndarray) -> dict[str, Any]:
    """Run frozen compact BSCN inference and write per-seed interval CSVs."""
    q_bundle = load_json(OFFICIAL_Q)
    if sha256(OFFICIAL_Q) != EXPECTED_Q_SHA:
        raise Stage37RHError("Official q SHA mismatch")
    base_config = load_json(BSCN_CONFIG)
    target_scaler_path = Path(base_config["target_scaler_path"])
    if not target_scaler_path.is_absolute():
        target_scaler_path = PROJECT_ROOT / target_scaler_path
    target_scaler = FeatureStandardizer.load(target_scaler_path)
    target_std = float(base_config["target_std_kw"])
    if not np.isclose(target_scaler.std_[target_scaler.target_feature_index], target_std, rtol=0, atol=1e-9):
        raise Stage37RHError("target_std_kw does not match frozen scaler")
    D = context["D"].to_numpy(dtype=np.float64)
    F = context["F"].to_numpy(dtype=np.float64)
    N = context["N"].to_numpy(dtype=np.float64)
    labels = context["N_group"].astype(str).to_numpy()
    hard = context["HARD_DF"].to_numpy(dtype=bool)
    rows = []
    routing_mismatch = {m: 0 for m in ("MG", "RMG", "MP")}
    prediction_hashes = {}
    for seed in SEEDS:
        model_path = PROJECT_ROOT / load_json(BSCN_DIR / f"seed{seed}" / "manifest.json")["compact_model"]
        model = BayesianStochasticConfigurationNetwork.load(model_path)
        mu, sigma = predict_kw(model, hms_std.astype(np.float64, copy=False), target_scaler, target_std)
        if not np.isfinite(mu).all() or not np.isfinite(sigma).all() or (sigma <= 0).any():
            raise Stage37RHError(f"seed {seed}: invalid mu/sigma_B")
        out = pd.DataFrame(
            {
                "sample_index": metadata["sample_index"].to_numpy(dtype=np.int64),
                "forecast_origin_timestamp": metadata["forecast_origin_timestamp"].astype(str),
                "target_timestamp": metadata["target_timestamp"].astype(str),
                "F": F,
                "N": N,
                "D": D,
                "group": labels,
                "HARD_DF": hard,
                "mu": mu,
                "sigma_B": sigma,
            }
        )
        for alpha in ALPHAS:
            suffix = f"{int(round((1.0 - alpha) * 100))}"
            qd = q_lookup(q_bundle, seed, alpha)
            q_values = {
                "M0": np.full(len(out), float(qd["M0"]["GLOBAL"]["q"]), dtype=np.float64),
                "MG": np.zeros(len(out), dtype=np.float64),
                "RMG": np.zeros(len(out), dtype=np.float64),
                "MP": np.zeros(len(out), dtype=np.float64),
            }
            for group in GROUPS:
                sel = labels == group
                q_values["MG"][sel] = float(qd["MG"][group]["q"])
                q_values["RMG"][sel] = float(qd["RMG"][group]["q"])
                q_values["MP"][sel] = float(qd["MP"][group]["q"])
            if np.any(q_values["MG"] > q_values["RMG"] + 1e-12) or np.any(q_values["RMG"] > q_values["MP"] + 1e-12):
                raise Stage37RHError(f"seed {seed}: interval hierarchy violated")
            for method in METHODS:
                half = q_values[method] * (sigma + CONFORMAL_EPSILON_KW)
                out[f"q_{method}_{suffix}"] = q_values[method]
                out[f"halfwidth_{method}_{suffix}"] = half
                out[f"lower_{method}_{suffix}"] = mu - half
                out[f"upper_{method}_{suffix}"] = mu + half
            for group in GROUPS:
                sel = labels == group
                routing_mismatch["MG"] += int(np.sum(q_values["MG"][sel] != float(qd["MG"][group]["q"])))
                routing_mismatch["RMG"] += int(np.sum(q_values["RMG"][sel] != float(qd["RMG"][group]["q"])))
                routing_mismatch["MP"] += int(np.sum(q_values["MP"][sel] != float(qd["MP"][group]["q"])))
        path = PRED_DIR / f"2024_preopen_seed{seed}.csv"
        assert_preopen_columns(out, path)
        out.to_csv(path, index=False)
        prediction_hashes[str(seed)] = {"path": path, "sha256": sha256(path), "rows": int(len(out))}
        rows.append(
            {
                "seed": seed,
                "rows": int(len(out)),
                "nodes": COMPACT_NODES[seed],
                "mu_finite": bool(np.isfinite(mu).all()),
                "sigma_positive": bool((sigma > 0).all()),
                "mu_mean": float(mu.mean()),
                "sigma_mean": float(sigma.mean()),
                "path": path,
                "sha256": prediction_hashes[str(seed)]["sha256"],
            }
        )
    if any(v != 0 for v in routing_mismatch.values()):
        raise Stage37RHError(f"Routing mismatch: {routing_mismatch}")
    audit = {
        "official_q": OFFICIAL_Q,
        "official_q_sha256": sha256(OFFICIAL_Q),
        "rows_per_seed": int(len(metadata)),
        "per_seed": rows,
        "routing_mismatch": routing_mismatch,
        "recalibration_performed": False,
        "bscn_training_performed": False,
        "target_joined": False,
        "PASS": True,
    }
    pd.DataFrame(rows).to_csv(METRICS_DIR / "2024_preopen_prediction_audit.csv", index=False)
    write_json(METRICS_DIR / "2024_preopen_prediction_audit.json", audit)
    return {"audit": audit, "prediction_hashes": prediction_hashes, "routing_mismatch": routing_mismatch}


def allowed_preopen_stats(context: pd.DataFrame, prediction_audit: dict[str, Any]) -> dict[str, Any]:
    """Compute target-blind distribution summaries only."""
    rows = []
    for name in ("D", "F", "N"):
        values = context[name].to_numpy(dtype=np.float64)
        rows.append({"variable": name, "mean": float(values.mean()), "std": float(values.std(ddof=0)), "min": float(values.min()), "median": float(np.median(values)), "max": float(values.max())})
    for seed_row in prediction_audit["per_seed"]:
        frame = pd.read_csv(Path(seed_row["path"]))
        for name in ("mu", "sigma_B", "halfwidth_M0_90", "halfwidth_MG_90", "halfwidth_RMG_90", "halfwidth_MP_90"):
            values = frame[name].to_numpy(dtype=np.float64)
            rows.append({"variable": f"seed{seed_row['seed']}_{name}", "mean": float(values.mean()), "std": float(values.std(ddof=0)), "min": float(values.min()), "median": float(np.median(values)), "max": float(values.max())})
    table = pd.DataFrame(rows)
    table.to_csv(METRICS_DIR / "2024_preopen_allowed_distribution_stats.csv", index=False)
    return {"path": METRICS_DIR / "2024_preopen_allowed_distribution_stats.csv", "rows": int(len(table))}


def artifact_hashes(paths: dict[str, Path]) -> dict[str, Any]:
    """Hash generated artifacts."""
    payload = {name: {"path": path, "sha256": sha256(path)} for name, path in paths.items() if path.is_file()}
    write_json(METRICS_DIR / "2024_preopen_artifact_hashes.json", payload)
    return payload


def write_registry_snapshot() -> dict[str, Any]:
    """Write a Stage37R-H registry snapshot without altering old locks."""
    payload = {
        "stage": STAGE,
        "2023": "OPENED_FINAL_LOCKED",
        "2024": "PREOPEN_LOCKED",
        "2025": "UNOPENED",
        "compact_representation_sensitivity": "REGISTERED",
        "original_registry": str((POST_G_DIR / "extension_experiment_registry.json").relative_to(PROJECT_ROOT)),
        "old_locks_modified": False,
    }
    write_json(METRICS_DIR / "extension_registry_stage37r_h_snapshot.json", payload)
    return payload


def write_lock_and_summary(
    raw_audit: dict[str, Any],
    calendar_audit: dict[str, Any],
    source_audit: dict[str, Any],
    causal_audit: dict[str, Any],
    group_table: pd.DataFrame,
    hms_audit: dict[str, Any],
    model_hashes: dict[str, Any],
    pred: dict[str, Any],
    protocol: dict[str, Any],
    registry: dict[str, Any],
    allowed_stats: dict[str, Any],
) -> dict[str, Any]:
    """Write final preopen lock, summary, decision and report."""
    lock_payload = {
        "stage": STAGE,
        "year": YEAR,
        "status": "PREOPEN_LOCKED",
        "TARGET_PERFORMANCE_OPENED": False,
        "2024_TARGET_PERFORMANCE_OPENED": False,
        "2025_TARGET_PERFORMANCE_OPENED": False,
        "raw_file_path": raw_audit["raw_file_path"],
        "raw_sha256": raw_audit["file_sha256"],
        "raw_rows": raw_audit["row_count"],
        "timestamp_range": [raw_audit["first_timestamp"], raw_audit["last_timestamp"]],
        "candidate_origins": causal_audit["candidate_origins"],
        "causal_exclusions": causal_audit["excluded_due_future_interpolation_endpoint"],
        "retained_count": causal_audit["retained_count"],
        "remaining_dependency": causal_audit["remaining_future_dependent"],
        "group_counts": {row["group"]: int(row["samples"]) for row in group_table.to_dict(orient="records")},
        "H_ms_shape": hms_audit["shape"],
        "H_ms_sha256": hms_audit["hms_sha256"],
        "model_hashes": model_hashes,
        "prediction_hashes": pred["prediction_hashes"],
        "official_q_sha256": EXPECTED_Q_SHA,
        "preopen_protocol_sha256": EXPECTED_H_PROTOCOL_SHA,
        "future_external_protocol_sha256": EXPECTED_I_PROTOCOL_SHA,
        "method_hierarchy": {"BASELINE": "M0", "PRIMARY": "MG", "SECONDARY": {"method": "RMG", "lambda": LAMBDA_FINAL}, "CONTROL": "MP"},
        "routing_mismatch": pred["routing_mismatch"],
        "target_finiteness_used_for_filtering": False,
        "performance_metrics_computed": False,
        "registry_snapshot": registry,
    }
    lock_path = METRICS_DIR / "2024_EXTERNAL_PREOPEN_LOCK.json"
    write_json(lock_path, lock_payload)
    lock_sha = write_sha_sidecar(lock_path, METRICS_DIR / "2024_EXTERNAL_PREOPEN_LOCK.sha256")
    summary = {
        "decision": "2024_PREOPEN_READY",
        "protocol_integrity": protocol["checks"],
        "source_audit": source_audit,
        "raw_audit": raw_audit,
        "calendar_audit": calendar_audit,
        "causal_audit": causal_audit,
        "group_support": group_table.to_dict(orient="records"),
        "hms_integrity": hms_audit,
        "model_hashes": model_hashes,
        "prediction_audit": pred["audit"],
        "allowed_preopen_stats": allowed_stats,
        "lock_sha256": lock_sha,
        "2024_TARGET_PERFORMANCE_OPENED": False,
        "2025_TARGET_PERFORMANCE_OPENED": False,
    }
    write_json(METRICS_DIR / "stage37r_h_summary.json", summary)
    write_json(METRICS_DIR / "stage37r_h_decision.json", {"decision": "2024_PREOPEN_READY"})
    lines = [
        "Stage37R-H 2024 Preopen Causal-Safe Build",
        "",
        "PROTOCOL INTEGRITY",
        f"2023 Final Lock: {protocol['checks']['2023_final_lock']}",
        f"Master Lock: {protocol['checks']['extension_master_lock']}",
        f"2024 Preopen Protocol: {protocol['checks']['2024_preopen_protocol']}",
        f"2024 Future External Protocol: {protocol['checks']['2024_future_external_protocol']}",
        "",
        "RAW AUDIT",
        f"file: {raw_audit['raw_file_path']}",
        f"rows: {raw_audit['row_count']}",
        f"unique timestamps: {raw_audit['unique_timestamps']}",
        f"first/last: {raw_audit['first_timestamp']} / {raw_audit['last_timestamp']}",
        f"duplicates: {raw_audit['duplicate_timestamps']}",
        f"gap segments: {raw_audit['gap_segments']}",
        f"max gap minutes: {raw_audit['maximum_gap_duration_minutes']}",
        f"full-year reference: {FULL_YEAR_REFERENCE_ROWS}",
        "",
        "CAUSAL AUDIT",
        f"candidate origins: {causal_audit['candidate_origins']}",
        f"future-dependent cells: {causal_audit['future_dependent_cells']}",
        f"affected origins: {causal_audit['unique_affected_origins']}",
        f"median delay minutes: {causal_audit['median_right_endpoint_delay_minutes']}",
        f"max delay minutes: {causal_audit['max_right_endpoint_delay_minutes']}",
        f"retained: {causal_audit['retained_count']}",
        f"remaining dependency: {causal_audit['remaining_future_dependent']}",
        "",
        "GROUP SUPPORT",
        group_table.to_string(index=False),
        "",
        "FROZEN INFERENCE",
        f"H_ms shape: {hms_audit['shape']}",
        f"prediction rows per seed: {pred['audit']['rows_per_seed']}",
        f"routing mismatch: {pred['routing_mismatch']}",
        "",
        "PREOPEN LOCK",
        f"SHA256: {lock_sha}",
        "status: PREOPEN_LOCKED",
        "2024_TARGET_PERFORMANCE_OPENED=False",
        "2025_TARGET_PERFORMANCE_OPENED=False",
        "",
        "DECISION",
        "2024_PREOPEN_READY",
    ]
    (METRICS_DIR / "stage37r_h_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    """Run Stage37R-H end to end."""
    logger = setup_logging()
    start = time.perf_counter()
    logger.info("Start Stage37R-H 2024 preopen build.")
    protocol = verify_protocol_integrity()
    raw_path, source_audit = identify_2024_raw(protocol["h_protocol"])
    logger.info("Selected raw source: %s", raw_path)
    raw_audit, calendar_audit, raw = raw_data_audit(raw_path)
    cleaned = clean_2024_raw(raw, logger)
    windows = build_causal_safe_windows(cleaned["core"], cleaned["flags"])
    logger.info("Retained causal-safe windows: %d", len(windows["metadata"]))
    hms = export_hms(windows["X"])
    dynamics = compute_dynamic_and_familiarity(windows["X"], windows["metadata"])
    group_table, _group_audit = group_support(dynamics["context"])
    model_hashes = collect_model_hashes()
    pred = build_preopen_predictions(windows["metadata"], dynamics["context"], hms["hms_std"])
    allowed_stats = allowed_preopen_stats(dynamics["context"], pred["audit"])
    registry = write_registry_snapshot()
    paths = {
        "manifest": PROCESSED_DIR / "2024_causal_safe_manifest.csv",
        "X": PROCESSED_DIR / "X_2024_preopen.npy",
        "H_ms": FEATURE_DIR / "hms_2024_preopen.npy",
        "H_ms_std": PROCESSED_DIR / "hms_external_2024_std.npy",
        "context": PROCESSED_DIR / "2024_context_preopen.csv",
    }
    paths.update({f"prediction_seed{seed}": PRED_DIR / f"2024_preopen_seed{seed}.csv" for seed in SEEDS})
    hashes = artifact_hashes(paths)
    summary = write_lock_and_summary(
        raw_audit,
        calendar_audit,
        source_audit,
        windows["audit"],
        group_table,
        hms["audit"],
        model_hashes,
        pred,
        protocol,
        registry,
        allowed_stats,
    )
    summary["artifact_hashes"] = hashes
    summary["total_seconds"] = float(time.perf_counter() - start)
    write_json(METRICS_DIR / "stage37r_h_summary.json", summary)
    logger.info("Stage37R-H completed: 2024_PREOPEN_READY in %.2fs", summary["total_seconds"])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        write_json(METRICS_DIR / "stage37r_h_decision.json", {"decision": "2024_PREOPEN_FAILED", "error": str(exc)})
        raise
