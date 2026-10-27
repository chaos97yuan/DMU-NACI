"""Historical familiarity utilities for 2022 PV forecasting states.

This module builds the Stage16 Historical Context baseline:

    C_t -> nearest 2021 historical contexts -> distance -> Novelty N_t -> Familiarity F_t

The context uses only information available at the forecast origin ``t`` or
earlier. Historical future outcomes are intentionally excluded from the distance
space and are aggregated only after neighbors are selected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from scipy.spatial import cKDTree

    SCIPY_CKDTREE_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when scipy is absent.
    cKDTree = None
    SCIPY_CKDTREE_AVAILABLE = False

try:
    from sklearn.neighbors import NearestNeighbors

    SKLEARN_NEIGHBORS_AVAILABLE = True
except Exception:  # pragma: no cover - sklearn is already required for StandardScaler.
    NearestNeighbors = None
    SKLEARN_NEIGHBORS_AVAILABLE = False

from src.data.schema import (
    FEATURE_COLUMNS,
    FEATURE_INDEX,
    HISTORY_STEPS,
    HORIZON_STEPS,
    SAMPLING_MINUTES,
)
from src.features.dynamic_index import COMPONENT_NAMES, compute_raw_components


POWER_NORM_RATED_CAPACITY_KW = 5.236
DAYLIGHT_GHI_THRESHOLD = 20.0
DEFAULT_K = 20
FAMILIARITY_EPS = 1e-8

CONTEXT_COLUMNS = [
    "sin_doy",
    "cos_doy",
    "sin_tod",
    "cos_tod",
    "power_norm",
    "ghi",
    "temperature",
    "relative_humidity",
    "rG",
    "rP",
    "vG",
    "slope_gap",
]
DISTANCE_EXCLUDED_COLUMNS = [
    "D_t",
    "future_power_15m",
    "delta_power_15m",
    "future_timestamp",
    "target_active_power",
    "target_GHI",
    "target_DHI",
    "memory_id",
    "segment_id",
]


@dataclass(frozen=True)
class NeighborQueryResult:
    """Nearest-neighbor result for one query split."""

    neighbor_memory_ids: np.ndarray
    neighbor_distances: np.ndarray
    neighbor_positions: np.ndarray
    backend: str


@dataclass(frozen=True)
class FamiliarityThresholds:
    """Day/night distance thresholds fitted only from 2022 Train queries."""

    day_q05: float
    day_q95: float
    night_q05: float
    night_q95: float
    source: str = "2022_train_queries_only"


def resolve_context_columns() -> list[str]:
    """Return the fixed 12 context columns used by Stage16 KNN retrieval."""
    return list(CONTEXT_COLUMNS)


def _required_feature_index(column: str) -> int:
    if column not in FEATURE_INDEX:
        raise ValueError(f"Required feature {column!r} is absent from FEATURE_INDEX.")
    idx = int(FEATURE_INDEX[column])
    if FEATURE_COLUMNS[idx] != column:
        raise ValueError(f"FEATURE_INDEX is inconsistent for {column!r}.")
    return idx


def build_cyclical_time_features(timestamps: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    """Build Stage15-compatible DOY/TOD cyclic features.

    The timestamp represents forecast origin ``t``. Target timestamps are not
    used in the context.
    """
    ts = pd.to_datetime(pd.Series(timestamps), errors="coerce")
    if ts.isna().any():
        raise ValueError("Cannot build cyclical features from NaT timestamps.")
    year_start = pd.to_datetime(ts.dt.year.astype(str) + "-01-01")
    next_year_start = pd.to_datetime((ts.dt.year + 1).astype(str) + "-01-01")
    year_seconds = (next_year_start - year_start).dt.total_seconds()
    seconds_since_year_start = (ts - year_start).dt.total_seconds()
    doy_position = seconds_since_year_start / year_seconds
    seconds_since_midnight = (
        ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second + ts.dt.microsecond / 1_000_000
    )
    tod_position = seconds_since_midnight / 86400.0
    return pd.DataFrame(
        {
            "sin_doy": np.sin(2.0 * np.pi * doy_position),
            "cos_doy": np.cos(2.0 * np.pi * doy_position),
            "sin_tod": np.sin(2.0 * np.pi * tod_position),
            "cos_tod": np.cos(2.0 * np.pi * tod_position),
        }
    )


def validate_memory_context(memory: pd.DataFrame) -> None:
    """Validate 2021 memory context and historical outcome columns."""
    required = CONTEXT_COLUMNS + [
        "memory_id",
        "timestamp",
        "is_daylight",
        "future_power_15m",
        "delta_power_15m",
    ]
    missing = [column for column in required if column not in memory.columns]
    if missing:
        raise ValueError(f"Historical memory is missing required columns: {missing}")
    timestamps = pd.to_datetime(memory["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Historical memory timestamp contains NaT.")
    if not (timestamps.dt.year == 2021).all():
        raise ValueError("Historical memory contains rows outside year 2021.")
    if memory["memory_id"].duplicated().any():
        raise ValueError("Historical memory memory_id contains duplicates.")
    context = memory.loc[:, CONTEXT_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(context).all():
        raise ValueError("2021 memory context contains NaN or Inf.")
    outcomes = memory.loc[:, ["future_power_15m", "delta_power_15m"]].to_numpy(dtype=np.float64)
    if not np.isfinite(outcomes).all():
        raise ValueError("2021 historical outcome columns contain NaN or Inf.")
    daylight_values = set(memory["is_daylight"].astype(int).unique().tolist())
    if not daylight_values.issubset({0, 1}):
        raise ValueError(f"Historical memory is_daylight must be binary, got {daylight_values}.")


def validate_query_inputs(
    split: str,
    X: np.ndarray,
    metadata: pd.DataFrame,
    hms_rows: int | None = None,
) -> None:
    """Validate query windows and metadata alignment without using y target."""
    expected_features = len(FEATURE_COLUMNS)
    if X.ndim != 3:
        raise ValueError(f"{split}: expected X [N,{HISTORY_STEPS},{expected_features}], got {X.shape}.")
    if X.shape[1] != HISTORY_STEPS or X.shape[2] != expected_features:
        raise ValueError(f"{split}: X shape mismatch, got {X.shape}.")
    if not np.isfinite(X).all():
        bad = np.argwhere(~np.isfinite(X))
        first = bad[0].tolist()
        raise ValueError(f"{split}: query X contains NaN/Inf at index {first}.")
    required = ["sample_index", "history_end_timestamp", "target_timestamp"]
    missing = [column for column in required if column not in metadata.columns]
    if missing:
        raise ValueError(f"{split}: metadata missing required columns: {missing}")
    if len(X) != len(metadata):
        raise ValueError(f"{split}: X rows {len(X)} != metadata rows {len(metadata)}.")
    if hms_rows is not None and len(X) != hms_rows:
        raise ValueError(f"{split}: X rows {len(X)} != H_ms rows {hms_rows}.")
    expected_index = np.arange(len(metadata))
    if not np.array_equal(metadata["sample_index"].to_numpy(), expected_index):
        raise ValueError(f"{split}: sample_index must be exactly 0...N-1.")
    origin = pd.to_datetime(metadata["history_end_timestamp"], errors="coerce")
    target = pd.to_datetime(metadata["target_timestamp"], errors="coerce")
    if origin.isna().any() or target.isna().any():
        raise ValueError(f"{split}: history_end_timestamp/target_timestamp contains NaT.")
    expected_delta = pd.Timedelta(minutes=HORIZON_STEPS * SAMPLING_MINUTES)
    if not ((target - origin) == expected_delta).all():
        raise ValueError(f"{split}: target_timestamp - forecast_origin_ts must be exactly {expected_delta}.")


def build_query_context(X: np.ndarray, metadata: pd.DataFrame, split: str = "query") -> pd.DataFrame:
    """Build 2022 query context from raw physical windows and forecast-origin time.

    Uses ``X[:, -1, :]`` for current PV/GHI/temperature/RH and
    ``compute_raw_components(X)`` for the four dynamic components. It never
    reads the future target value or target weather.
    """
    validate_query_inputs(split, X, metadata)
    origin_ts = pd.to_datetime(metadata["history_end_timestamp"], errors="coerce").reset_index(drop=True)
    cyc = build_cyclical_time_features(origin_ts)
    power_idx = _required_feature_index("Active_Power")
    ghi_idx = _required_feature_index("Global_Horizontal_Radiation")
    temp_idx = _required_feature_index("Weather_Temperature_Celsius")
    rh_idx = _required_feature_index("Weather_Relative_Humidity")
    current = X[:, -1, :].astype(np.float64, copy=False)
    raw_components = compute_raw_components(X).astype(np.float64, copy=False)
    if list(COMPONENT_NAMES) != ["ghi_ramp_1h", "pv_ramp_1h", "ghi_std_1h", "ghi_slope_gap_1h_6h"]:
        raise RuntimeError(f"Unexpected dynamic component order: {COMPONENT_NAMES}")
    context = pd.DataFrame(
        {
            "sin_doy": cyc["sin_doy"].to_numpy(dtype=np.float64),
            "cos_doy": cyc["cos_doy"].to_numpy(dtype=np.float64),
            "sin_tod": cyc["sin_tod"].to_numpy(dtype=np.float64),
            "cos_tod": cyc["cos_tod"].to_numpy(dtype=np.float64),
            "power_norm": current[:, power_idx] / POWER_NORM_RATED_CAPACITY_KW,
            "ghi": current[:, ghi_idx],
            "temperature": current[:, temp_idx],
            "relative_humidity": current[:, rh_idx],
            "rG": raw_components[:, 0],
            "rP": raw_components[:, 1],
            "vG": raw_components[:, 2],
            "slope_gap": raw_components[:, 3],
        }
    )
    if not np.isfinite(context.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{split}: query context contains NaN or Inf.")
    return context.loc[:, CONTEXT_COLUMNS]


def fit_historical_context_scaler(memory_context: pd.DataFrame) -> StandardScaler:
    """Fit StandardScaler exclusively on 2021 historical memory contexts."""
    if list(memory_context.columns) != CONTEXT_COLUMNS:
        raise ValueError(f"Context columns mismatch: {list(memory_context.columns)}")
    values = memory_context.to_numpy(dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(CONTEXT_COLUMNS):
        raise ValueError(f"Expected memory context [N,{len(CONTEXT_COLUMNS)}], got {values.shape}.")
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Memory context must be non-empty and finite.")
    scaler = StandardScaler()
    scaler.fit(values)
    if int(scaler.n_samples_seen_) != len(memory_context):
        raise RuntimeError("Scaler n_samples_seen_ does not equal 2021 memory rows.")
    return scaler


def transform_context(context: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """Transform context with an already-fitted 2021 memory scaler."""
    if list(context.columns) != CONTEXT_COLUMNS:
        raise ValueError(f"Context columns mismatch: {list(context.columns)}")
    values = context.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Context contains NaN or Inf.")
    transformed = scaler.transform(values)
    if not np.isfinite(transformed).all():
        raise ValueError("Scaled context contains NaN or Inf.")
    return transformed.astype(np.float64, copy=False)


def build_neighbor_index(memory_scaled: np.ndarray, backend: str = "auto") -> Any:
    """Build an exact Euclidean nearest-neighbor index for one daylight pool."""
    if memory_scaled.ndim != 2 or memory_scaled.shape[1] != len(CONTEXT_COLUMNS):
        raise ValueError(f"memory_scaled must be [N,{len(CONTEXT_COLUMNS)}], got {memory_scaled.shape}.")
    if not np.isfinite(memory_scaled).all():
        raise ValueError("memory_scaled contains NaN or Inf.")
    if backend == "auto":
        backend = "ckdtree" if SCIPY_CKDTREE_AVAILABLE else "sklearn"
    if backend == "ckdtree":
        if not SCIPY_CKDTREE_AVAILABLE:
            raise RuntimeError("scipy.spatial.cKDTree is not available.")
        return cKDTree(memory_scaled), "scipy.cKDTree"
    if backend == "sklearn":
        if not SKLEARN_NEIGHBORS_AVAILABLE:
            raise RuntimeError("sklearn.neighbors.NearestNeighbors is not available.")
        index = NearestNeighbors(metric="euclidean", algorithm="auto")
        index.fit(memory_scaled)
        return index, "sklearn.NearestNeighbors"
    raise ValueError(f"Unknown backend {backend!r}.")


def query_neighbors(
    query_scaled: np.ndarray,
    index: Any,
    pool_memory_ids: np.ndarray,
    pool_positions: np.ndarray,
    k: int = DEFAULT_K,
    backend_name: str = "scipy.cKDTree",
) -> NeighborQueryResult:
    """Query K exact Euclidean neighbors from a prefiltered memory pool."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")
    if len(pool_memory_ids) < k:
        raise ValueError(f"Retrieval pool has {len(pool_memory_ids)} rows, smaller than K={k}.")
    if query_scaled.ndim != 2 or query_scaled.shape[1] != len(CONTEXT_COLUMNS):
        raise ValueError(f"query_scaled must be [N,{len(CONTEXT_COLUMNS)}], got {query_scaled.shape}.")
    if backend_name == "scipy.cKDTree":
        distances, local_indices = index.query(query_scaled, k=k)
    elif backend_name == "sklearn.NearestNeighbors":
        distances, local_indices = index.kneighbors(query_scaled, n_neighbors=k, return_distance=True)
    else:
        raise ValueError(f"Unsupported backend_name {backend_name!r}.")
    if k == 1:
        distances = distances[:, None]
        local_indices = local_indices[:, None]
    memory_ids = pool_memory_ids[local_indices]
    positions = pool_positions[local_indices]
    if not np.isfinite(distances).all():
        raise ValueError("Neighbor distances contain NaN or Inf.")
    return NeighborQueryResult(
        neighbor_memory_ids=memory_ids.astype(np.int64, copy=False),
        neighbor_distances=distances.astype(np.float32, copy=False),
        neighbor_positions=positions.astype(np.int64, copy=False),
        backend=backend_name,
    )


def aggregate_neighbor_outcomes(memory: pd.DataFrame, neighbor_positions: np.ndarray) -> dict[str, np.ndarray]:
    """Aggregate historical outcomes after context-distance neighbors are fixed."""
    if neighbor_positions.ndim != 2:
        raise ValueError(f"neighbor_positions must be 2D [N,K], got {neighbor_positions.shape}.")
    delta = memory["delta_power_15m"].to_numpy(dtype=np.float64)[neighbor_positions]
    future = memory["future_power_15m"].to_numpy(dtype=np.float64)[neighbor_positions]
    return {
        "hist_deltaP_mean": np.mean(delta, axis=1).astype(np.float32),
        "hist_deltaP_std": np.std(delta, axis=1, ddof=0).astype(np.float32),
        "hist_deltaP_median": np.median(delta, axis=1).astype(np.float32),
        "hist_future_power_mean": np.mean(future, axis=1).astype(np.float32),
        "hist_future_power_std": np.std(future, axis=1, ddof=0).astype(np.float32),
    }


def fit_familiarity_thresholds(
    train_mean_distance: np.ndarray,
    train_is_daylight: np.ndarray,
    q_low: float = 0.05,
    q_high: float = 0.95,
) -> FamiliarityThresholds:
    """Fit distance-to-novelty thresholds from 2022 Train queries only."""
    distances = np.asarray(train_mean_distance, dtype=np.float64)
    daylight = np.asarray(train_is_daylight, dtype=int)
    if distances.ndim != 1 or daylight.ndim != 1 or len(distances) != len(daylight):
        raise ValueError("train_mean_distance and train_is_daylight must be aligned 1D arrays.")
    if not np.isfinite(distances).all():
        raise ValueError("Train mean distances contain NaN or Inf.")
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError(f"Invalid quantiles q_low={q_low}, q_high={q_high}.")
    day_dist = distances[daylight == 1]
    night_dist = distances[daylight == 0]
    if len(day_dist) == 0 or len(night_dist) == 0:
        raise ValueError("Train queries must contain both daylight and night samples.")
    day_q05, day_q95 = np.quantile(day_dist, [q_low, q_high])
    night_q05, night_q95 = np.quantile(night_dist, [q_low, q_high])
    if day_q95 <= day_q05 or night_q95 <= night_q05:
        raise ValueError(
            f"Invalid familiarity thresholds: day=({day_q05},{day_q95}), "
            f"night=({night_q05},{night_q95})."
        )
    return FamiliarityThresholds(
        day_q05=float(day_q05),
        day_q95=float(day_q95),
        night_q05=float(night_q05),
        night_q95=float(night_q95),
    )


def distance_to_familiarity(
    mean_distance: np.ndarray,
    is_daylight: np.ndarray,
    thresholds: FamiliarityThresholds,
    eps: float = FAMILIARITY_EPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Map mean historical distance to Novelty N_t and Familiarity F_t."""
    distances = np.asarray(mean_distance, dtype=np.float64)
    daylight = np.asarray(is_daylight, dtype=int)
    if distances.ndim != 1 or daylight.ndim != 1 or len(distances) != len(daylight):
        raise ValueError("mean_distance and is_daylight must be aligned 1D arrays.")
    if not np.isfinite(distances).all():
        raise ValueError("mean_distance contains NaN or Inf.")
    novelty = np.empty_like(distances, dtype=np.float64)
    day_mask = daylight == 1
    night_mask = ~day_mask
    novelty[day_mask] = (distances[day_mask] - thresholds.day_q05) / (
        thresholds.day_q95 - thresholds.day_q05 + eps
    )
    novelty[night_mask] = (distances[night_mask] - thresholds.night_q05) / (
        thresholds.night_q95 - thresholds.night_q05 + eps
    )
    novelty = np.clip(novelty, 0.0, 1.0)
    familiarity = 1.0 - novelty
    validate_familiarity_output(familiarity, novelty)
    return familiarity.astype(np.float32), novelty.astype(np.float32)


def validate_familiarity_output(familiarity: np.ndarray, novelty: np.ndarray) -> None:
    """Validate finite, [0,1], and F_t + N_t = 1."""
    f = np.asarray(familiarity, dtype=np.float64)
    n = np.asarray(novelty, dtype=np.float64)
    if f.shape != n.shape or f.ndim != 1:
        raise ValueError(f"F/N must be aligned 1D arrays, got {f.shape} and {n.shape}.")
    if not np.isfinite(f).all() or not np.isfinite(n).all():
        raise ValueError("F_t or N_t contains NaN/Inf.")
    if f.min() < -1e-8 or f.max() > 1.0 + 1e-8 or n.min() < -1e-8 or n.max() > 1.0 + 1e-8:
        raise ValueError("F_t/N_t must be in [0,1].")
    if not np.allclose(f + n, 1.0, rtol=1e-6, atol=1e-6):
        raise ValueError("F_t + N_t must equal 1.")


def compute_distance_statistics(distances: np.ndarray) -> dict[str, np.ndarray]:
    """Compute per-query distance summaries for K neighbors."""
    d = np.asarray(distances, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError(f"distances must be [N,K], got {d.shape}.")
    return {
        "hist_min_distance": np.min(d, axis=1).astype(np.float32),
        "hist_mean_distance": np.mean(d, axis=1).astype(np.float32),
        "hist_median_distance": np.median(d, axis=1).astype(np.float32),
        "hist_std_distance": np.std(d, axis=1, ddof=0).astype(np.float32),
        "hist_max_distance": np.max(d, axis=1).astype(np.float32),
    }


def validate_neighbor_year(memory: pd.DataFrame, neighbor_positions: np.ndarray) -> None:
    """Fail if any selected neighbor is not from year 2021."""
    timestamps = pd.to_datetime(memory["timestamp"], errors="coerce")
    years = timestamps.dt.year.to_numpy()[neighbor_positions]
    if not np.all(years == 2021):
        raise ValueError("Neighbor set contains non-2021 memory rows.")


def validate_same_daylight_retrieval(
    query_is_daylight: np.ndarray,
    memory: pd.DataFrame,
    neighbor_positions: np.ndarray,
) -> None:
    """Validate every query retrieves only memory rows with the same daylight flag."""
    query_flags = np.asarray(query_is_daylight, dtype=int)[:, None]
    memory_flags = memory["is_daylight"].to_numpy(dtype=int)[neighbor_positions]
    if query_flags.shape[0] != memory_flags.shape[0] or not np.all(memory_flags == query_flags):
        raise ValueError("Same-daylight retrieval check failed.")


def scaler_summary_dict(scaler: StandardScaler, feature_names: list[str]) -> dict[str, Any]:
    """Return JSON-safe StandardScaler diagnostics."""
    return {
        "feature_names": list(feature_names),
        "mean_": [float(v) for v in scaler.mean_],
        "scale_": [float(v) for v in scaler.scale_],
        "var_": [float(v) for v in scaler.var_],
        "n_samples_seen_": int(scaler.n_samples_seen_),
        "fit_source": "2021_memory_only",
    }
