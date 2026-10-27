from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schema import FEATURE_INDEX, HISTORY_STEPS
from src.features.historical_familiarity import (
    CONTEXT_COLUMNS,
    DISTANCE_EXCLUDED_COLUMNS,
    POWER_NORM_RATED_CAPACITY_KW,
    aggregate_neighbor_outcomes,
    build_neighbor_index,
    build_query_context,
    distance_to_familiarity,
    fit_familiarity_thresholds,
    fit_historical_context_scaler,
    query_neighbors,
    resolve_context_columns,
    transform_context,
    validate_neighbor_year,
    validate_query_inputs,
    validate_same_daylight_retrieval,
)


def make_context(values: list[float], n: int = 1) -> pd.DataFrame:
    arr = np.tile(np.asarray(values, dtype=np.float64), (n, 1))
    return pd.DataFrame(arr, columns=CONTEXT_COLUMNS)


def make_memory() -> pd.DataFrame:
    rows = []
    base_values = [
        [0, 1, 0, 1, 0.0, 0.0, 20.0, 40.0, 0.0, 0.00, 0.0, 0.0],
        [0, 1, 0, 1, 0.1, 0.0, 20.0, 40.0, 0.0, 0.00, 0.0, 0.0],
        [0, 1, 0, 1, 0.8, 800.0, 25.0, 30.0, 10.0, 0.10, 5.0, 20.0],
        [0, 1, 0, 1, 0.9, 900.0, 25.0, 30.0, 10.0, 0.10, 5.0, 20.0],
    ]
    daylight = [0, 0, 1, 1]
    for i, values in enumerate(base_values):
        row = dict(zip(CONTEXT_COLUMNS, values, strict=True))
        row.update(
            {
                "memory_id": i,
                "timestamp": pd.Timestamp("2021-01-01") + pd.Timedelta(minutes=5 * i),
                "is_daylight": daylight[i],
                "future_power_15m": float(i + 1),
                "delta_power_15m": float(i) * 0.5,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def make_window(n: int = 2) -> np.ndarray:
    X = np.zeros((n, HISTORY_STEPS, 5), dtype=np.float32)
    steps = np.arange(HISTORY_STEPS, dtype=np.float32)
    for i in range(n):
        X[i, :, FEATURE_INDEX["Active_Power"]] = 1.0 + 0.01 * steps + i
        X[i, :, FEATURE_INDEX["Global_Horizontal_Radiation"]] = 100.0 + steps + 10.0 * i
        X[i, :, FEATURE_INDEX["Diffuse_Horizontal_Radiation"]] = 50.0 + 0.5 * steps
        X[i, :, FEATURE_INDEX["Weather_Temperature_Celsius"]] = 22.0 + i
        X[i, :, FEATURE_INDEX["Weather_Relative_Humidity"]] = 45.0 + i
    return X


def make_metadata(n: int = 2) -> pd.DataFrame:
    origin = pd.date_range("2022-01-01 05:55:00", periods=n, freq="5min")
    return pd.DataFrame(
        {
            "sample_index": np.arange(n),
            "history_end_timestamp": origin,
            "target_timestamp": origin + pd.Timedelta(minutes=15),
        }
    )


def test_context_column_selection_and_exclusions():
    assert resolve_context_columns() == CONTEXT_COLUMNS
    for forbidden in ["future_power_15m", "delta_power_15m", "D_t", "target_GHI"]:
        assert forbidden not in CONTEXT_COLUMNS
        assert forbidden in DISTANCE_EXCLUDED_COLUMNS


def test_target_timestamp_does_not_change_context():
    X = make_window(1)
    metadata_a = make_metadata(1)
    metadata_b = metadata_a.copy()
    metadata_b["target_timestamp"] = metadata_b["target_timestamp"] + pd.Timedelta(days=7)
    with pytest.raises(ValueError):
        build_query_context(X, metadata_b, split="bad_horizon")

    metadata_c = metadata_a.copy()
    metadata_c["target_timestamp"] = metadata_c["history_end_timestamp"] + pd.Timedelta(minutes=15)
    context_a = build_query_context(X, metadata_a, split="a")
    context_c = build_query_context(X, metadata_c, split="c")
    pd.testing.assert_frame_equal(context_a, context_c)


def test_power_norm_uses_5_236_and_query_nan_fails():
    X = make_window(1)
    metadata = make_metadata(1)
    context = build_query_context(X, metadata)
    expected = X[0, -1, FEATURE_INDEX["Active_Power"]] / POWER_NORM_RATED_CAPACITY_KW
    assert np.isclose(context.loc[0, "power_norm"], expected)
    X_bad = X.copy()
    X_bad[0, -1, FEATURE_INDEX["Active_Power"]] = np.nan
    with pytest.raises(ValueError):
        build_query_context(X_bad, metadata)


def test_scaler_only_fit_memory_and_transform_query_without_refit():
    memory = make_memory()
    scaler = fit_historical_context_scaler(memory[CONTEXT_COLUMNS])
    before_mean = scaler.mean_.copy()
    query = make_context([0, 1, 0, 1, 2.0, 1000.0, 30.0, 20.0, 20.0, 0.2, 10.0, 40.0], n=3)
    transformed = transform_context(query, scaler)
    assert transformed.shape == (3, len(CONTEXT_COLUMNS))
    assert np.array_equal(before_mean, scaler.mean_)
    assert int(scaler.n_samples_seen_) == len(memory)


def test_nearest_neighbors_order_and_outcome_aggregation():
    memory = make_memory()
    scaler = fit_historical_context_scaler(memory[CONTEXT_COLUMNS])
    scaled = transform_context(memory[CONTEXT_COLUMNS], scaler)
    day_positions = np.flatnonzero(memory["is_daylight"].to_numpy(dtype=int) == 1)
    index, backend = build_neighbor_index(scaled[day_positions], backend="auto")
    result = query_neighbors(
        scaled[[2]],
        index,
        memory["memory_id"].to_numpy(dtype=np.int64)[day_positions],
        day_positions,
        k=2,
        backend_name=backend,
    )
    assert result.neighbor_memory_ids.tolist() == [[2, 3]]
    assert result.neighbor_distances[0, 0] <= result.neighbor_distances[0, 1]
    outcomes = aggregate_neighbor_outcomes(memory, result.neighbor_positions)
    assert np.isclose(outcomes["hist_deltaP_mean"][0], (1.0 + 1.5) / 2.0)
    assert np.isclose(outcomes["hist_future_power_mean"][0], (3.0 + 4.0) / 2.0)


def test_daylight_and_night_retrieval_are_isolated():
    memory = make_memory()
    scaler = fit_historical_context_scaler(memory[CONTEXT_COLUMNS])
    scaled = transform_context(memory[CONTEXT_COLUMNS], scaler)
    flags = memory["is_daylight"].to_numpy(dtype=int)
    for flag in [0, 1]:
        positions = np.flatnonzero(flags == flag)
        index, backend = build_neighbor_index(scaled[positions])
        result = query_neighbors(
            scaled[positions[:1]],
            index,
            memory["memory_id"].to_numpy(dtype=np.int64)[positions],
            positions,
            k=2,
            backend_name=backend,
        )
        validate_same_daylight_retrieval(np.array([flag]), memory, result.neighbor_positions)


def test_familiarity_threshold_mapping_and_train_only_stability():
    train_dist = np.array([1, 2, 3, 4, 10, 12, 14, 16], dtype=float)
    daylight = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=int)
    thresholds = fit_familiarity_thresholds(train_dist, daylight)
    original = (thresholds.day_q05, thresholds.day_q95, thresholds.night_q05, thresholds.night_q95)
    f, n = distance_to_familiarity(np.array([thresholds.day_q05, thresholds.day_q95]), np.array([1, 1]), thresholds)
    assert np.allclose(f, [1.0, 0.0])
    assert np.allclose(n, [0.0, 1.0])
    f_val, n_val = distance_to_familiarity(np.array([100.0, -100.0]), np.array([0, 0]), thresholds)
    assert np.all((0.0 <= f_val) & (f_val <= 1.0))
    assert np.allclose(f_val + n_val, 1.0)
    assert original == (thresholds.day_q05, thresholds.day_q95, thresholds.night_q05, thresholds.night_q95)


def test_sample_order_and_sample_id_misalignment_fail_loudly():
    X = make_window(3)
    metadata = make_metadata(3)
    validate_query_inputs("ok", X, metadata, hms_rows=3)
    bad = metadata.copy()
    bad.loc[1, "sample_index"] = 99
    with pytest.raises(ValueError):
        validate_query_inputs("bad_sample_index", X, bad, hms_rows=3)


def test_neighbor_year_non_2021_fails():
    memory = make_memory()
    memory.loc[0, "timestamp"] = pd.Timestamp("2022-01-01")
    with pytest.raises(ValueError):
        validate_neighbor_year(memory, np.array([[0, 1]]))


def test_deterministic_neighbor_output():
    memory = make_memory()
    scaler = fit_historical_context_scaler(memory[CONTEXT_COLUMNS])
    scaled = transform_context(memory[CONTEXT_COLUMNS], scaler)
    positions = np.flatnonzero(memory["is_daylight"].to_numpy(dtype=int) == 1)
    index, backend = build_neighbor_index(scaled[positions])
    kwargs = dict(
        query_scaled=scaled[positions],
        index=index,
        pool_memory_ids=memory["memory_id"].to_numpy(dtype=np.int64)[positions],
        pool_positions=positions,
        k=2,
        backend_name=backend,
    )
    first = query_neighbors(**kwargs)
    second = query_neighbors(**kwargs)
    assert np.array_equal(first.neighbor_memory_ids, second.neighbor_memory_ids)
    assert np.allclose(first.neighbor_distances, second.neighbor_distances)
