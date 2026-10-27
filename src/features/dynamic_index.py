"""Continuous historical dynamics index D_t for PV forecasting windows.

本模块实现论文流程中的连续天气/过程动态指数：

    D_t = 0.35 * r_G~ + 0.30 * r_P~ + 0.20 * v_G~ + 0.15 * s_G~

D_t 是 physically interpretable proxy for recent PV/weather dynamics，只使用
forecast origin t 及之前的6小时历史窗口：

    [t-5h55min, ..., t] -> D_t

预测目标是 t+15min，因此本模块绝不使用未来target、target GHI、TCN误差、
H_ms或任何未来天气。鲁棒归一化参数必须只在Train窗口上fit；Validation、
Calibration和Test只能调用transform。

本文件是纯特征工程核心：不读写磁盘、不绘图、不依赖sklearn/pandas/torch。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from src.data.schema import (
        FEATURE_COLUMNS,
        FEATURE_INDEX,
        HISTORY_STEPS,
        SAMPLING_MINUTES,
        SITE_RATED_POWER_KW,
    )
except ModuleNotFoundError:  # 支持 `python src/features/dynamic_index.py` 直接自检。
    import os
    import sys

    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from src.data.schema import (
        FEATURE_COLUMNS,
        FEATURE_INDEX,
        HISTORY_STEPS,
        SAMPLING_MINUTES,
        SITE_RATED_POWER_KW,
    )


ACTIVE_POWER_COLUMN = "Active_Power"
GHI_COLUMN = "Global_Horizontal_Radiation"
SHORT_WINDOW_HOURS = 1
DEFAULT_Q_LOW = 0.05
DEFAULT_Q_HIGH = 0.95
DEFAULT_EPS = 1e-8
NEGATIVE_TOLERANCE = -1e-12

COMPONENT_NAMES = [
    "ghi_ramp_1h",
    "pv_ramp_1h",
    "ghi_std_1h",
    "ghi_slope_gap_1h_6h",
]
DEFAULT_WEIGHTS = np.array([0.35, 0.30, 0.20, 0.15], dtype=np.float64)


@dataclass(frozen=True)
class DynamicIndexResult:
    """Result container for D_t computation.

    Attributes
    ----------
    raw_components:
        原始动态分量，shape [N, 4]，顺序与 COMPONENT_NAMES 一致。
    normalized_components:
        使用Train Q5/Q95归一化到[0,1]后的动态分量，shape [N, 4]。
    dynamic_index:
        连续动态指数 D_t，shape [N]，范围[0,1]。
    """

    raw_components: np.ndarray
    normalized_components: np.ndarray
    dynamic_index: np.ndarray


def get_short_window_steps(sampling_minutes: int = SAMPLING_MINUTES) -> int:
    """根据采样间隔计算最近1小时点数。"""
    if sampling_minutes <= 0:
        raise ValueError(f"sampling_minutes must be positive, got {sampling_minutes}.")
    if 60 % sampling_minutes != 0:
        raise ValueError(
            f"60 minutes must be divisible by sampling_minutes={sampling_minutes} "
            "to define an exact recent-1h window."
        )
    return int(60 // sampling_minutes)


def _required_feature_index(column: str) -> int:
    """从schema.FEATURE_INDEX读取列位置，禁止手写特征索引。"""
    if column not in FEATURE_INDEX:
        raise ValueError(
            f"Required feature {column!r} is not in FEATURE_INDEX. "
            f"Available features: {list(FEATURE_COLUMNS)}"
        )
    index = int(FEATURE_INDEX[column])
    if index < 0 or index >= len(FEATURE_COLUMNS) or FEATURE_COLUMNS[index] != column:
        raise ValueError(f"FEATURE_INDEX is inconsistent for column {column!r}.")
    return index


def validate_window_array(X: np.ndarray) -> np.ndarray:
    """Validate raw physical PV windows with shape [N, HISTORY_STEPS, F].

    错误信息会特别提示不要误把H_ms [N,64]传进来。
    """
    array = np.asarray(X)
    expected_features = len(FEATURE_COLUMNS)
    if array.ndim != 3:
        raise ValueError(
            f"Expected X with shape [N,{HISTORY_STEPS},{expected_features}], got {array.shape}. "
            "Did you accidentally pass H_ms instead of raw PV windows?"
        )
    if array.shape[1] != HISTORY_STEPS:
        raise ValueError(
            f"Expected history dimension {HISTORY_STEPS}, got {array.shape[1]}. "
            "D_t must be computed from the full 6h raw history window."
        )
    if array.shape[2] != expected_features:
        raise ValueError(
            f"Expected feature dimension {expected_features}, got {array.shape[2]}. "
            f"FEATURE_COLUMNS={list(FEATURE_COLUMNS)}"
        )
    if array.shape[0] == 0:
        raise ValueError("X contains zero samples.")
    if HISTORY_STEPS < get_short_window_steps():
        raise ValueError(
            f"HISTORY_STEPS={HISTORY_STEPS} is shorter than recent-1h steps={get_short_window_steps()}."
        )
    if not np.isfinite(array).all():
        raise ValueError("X contains NaN or Inf. DynamicIndexTransformer will not repair data.")
    return array.astype(np.float64, copy=False)


def _linear_slope_batch(values: np.ndarray, sampling_minutes: int = SAMPLING_MINUTES) -> np.ndarray:
    """Vectorized OLS slope for values [N,T] using time in hours.

    时间轴为 0, 1/12, 2/12, ... hours（当前5min采样），因此斜率单位是
    GHI unit / hour，而不是每个采样step的变化量。
    """
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError(f"_linear_slope_batch expects [N,T], got shape {y.shape}.")
    if y.shape[1] < 2:
        raise ValueError("At least two time points are required for OLS slope.")
    if not np.isfinite(y).all():
        raise ValueError("values contains NaN or Inf.")

    dt_hours = sampling_minutes / 60.0
    x = np.arange(y.shape[1], dtype=np.float64) * dt_hours
    x_centered = x - np.mean(x)
    denominator = float(np.sum(x_centered**2))
    if denominator <= 0.0:
        raise ValueError("Invalid slope denominator.")
    y_centered = y - np.mean(y, axis=1, keepdims=True)
    return (y_centered @ x_centered) / denominator


def compute_raw_components(X: np.ndarray) -> np.ndarray:
    """Compute raw dynamic components from raw physical windows.

    Component order:
    0. ghi_ramp_1h: max adjacent absolute GHI change in recent 1h.
    1. pv_ramp_1h: max adjacent absolute PV change in recent 1h / rated power.
    2. ghi_std_1h: recent 1h GHI standard deviation, ddof=0.
    3. ghi_slope_gap_1h_6h: |recent 1h GHI slope - full 6h GHI slope|.

    所有分量只使用历史窗口内部的Active_Power和GHI，不使用未来target。
    """
    array = validate_window_array(X)
    short_steps = get_short_window_steps()
    power_idx = _required_feature_index(ACTIVE_POWER_COLUMN)
    ghi_idx = _required_feature_index(GHI_COLUMN)

    power_short = array[:, -short_steps:, power_idx]
    ghi_short = array[:, -short_steps:, ghi_idx]
    ghi_full = array[:, :, ghi_idx]

    ghi_ramp = np.max(np.abs(np.diff(ghi_short, axis=1)), axis=1)
    pv_ramp = np.max(np.abs(np.diff(power_short, axis=1)), axis=1) / float(SITE_RATED_POWER_KW)
    ghi_std = np.std(ghi_short, axis=1, ddof=0)
    slope_short = _linear_slope_batch(ghi_short)
    slope_long = _linear_slope_batch(ghi_full)
    slope_gap = np.abs(slope_short - slope_long)

    components = np.column_stack([ghi_ramp, pv_ramp, ghi_std, slope_gap]).astype(np.float64, copy=False)
    return _validate_raw_components(components).astype(np.float32, copy=False)


def compute_raw_components_dict(X: np.ndarray) -> dict[str, np.ndarray]:
    """Return raw components as a dictionary keyed by COMPONENT_NAMES."""
    components = compute_raw_components(X)
    return {name: components[:, idx].copy() for idx, name in enumerate(COMPONENT_NAMES)}


def _validate_raw_components(raw_components: np.ndarray) -> np.ndarray:
    """Validate raw components [N,4], finite and non-negative."""
    components = np.asarray(raw_components, dtype=np.float64)
    if components.ndim != 2 or components.shape[1] != len(COMPONENT_NAMES):
        raise ValueError(
            f"raw_components must have shape [N,{len(COMPONENT_NAMES)}], got {components.shape}."
        )
    if components.shape[0] == 0:
        raise ValueError("raw_components contains zero samples.")
    if not np.isfinite(components).all():
        raise ValueError("raw_components contains NaN or Inf.")
    min_value = float(np.min(components))
    if min_value < NEGATIVE_TOLERANCE:
        raise ValueError(f"raw_components contains negative values below tolerance: min={min_value}.")
    if min_value < 0.0:
        components = np.maximum(components, 0.0)
    return components


def compute_saturation_rates(normalized_components: np.ndarray) -> dict[str, dict[str, float]]:
    """Compute per-component fractions clipped to 0 or 1.

    该函数只做统计，不会修改归一化参数，也不会根据Val/Test重新fit。
    """
    normalized = np.asarray(normalized_components, dtype=np.float64)
    if normalized.ndim != 2 or normalized.shape[1] != len(COMPONENT_NAMES):
        raise ValueError(f"normalized_components must have shape [N,4], got {normalized.shape}.")
    if not np.isfinite(normalized).all():
        raise ValueError("normalized_components contains NaN or Inf.")
    return {
        name: {
            "fraction_zero": float(np.mean(normalized[:, idx] <= 0.0)),
            "fraction_one": float(np.mean(normalized[:, idx] >= 1.0)),
        }
        for idx, name in enumerate(COMPONENT_NAMES)
    }


class DynamicIndexTransformer:
    """Train-only robust transformer for continuous historical dynamics index D_t.

    D_t使用4个历史动态分量：GHI短时爬坡、PV短时爬坡、GHI短时波动、
    GHI短期-长期趋势差。默认权重是第一版物理先验：
    GHI快速变化直接反映云遮挡等短时间太阳资源变化，因此r_G权重最高；
    PV ramp反映系统实际历史输出变化，因此r_P也占较高权重。这些权重不是
    最终实验最优权重，不在本类中搜索或学习。

    鲁棒归一化参数Q5/Q95只能由Train窗口fit。Val/Cal/Test必须只调用transform。
    """

    def __init__(
        self,
        weights: list[float] | tuple[float, ...] | np.ndarray | None = None,
        q_low: float = DEFAULT_Q_LOW,
        q_high: float = DEFAULT_Q_HIGH,
        site_rated_power_kw: float = SITE_RATED_POWER_KW,
        eps: float = DEFAULT_EPS,
    ) -> None:
        self.component_names = list(COMPONENT_NAMES)
        self.weights_ = self._validate_weights(DEFAULT_WEIGHTS if weights is None else weights)
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.site_rated_power_kw = float(site_rated_power_kw)
        self.eps = float(eps)
        self.q_low_values_: np.ndarray | None = None
        self.q_high_values_: np.ndarray | None = None
        self.degenerate_components_: list[str] = []
        self.train_component_summary_: dict[str, dict[str, float]] | None = None
        self.is_fitted_ = False
        self._validate_init_config()

    @staticmethod
    def _validate_weights(weights: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
        """Validate weights length, non-negativity, finite values and sum≈1."""
        array = np.asarray(weights, dtype=np.float64)
        if array.shape != (len(COMPONENT_NAMES),):
            raise ValueError(f"weights must have length {len(COMPONENT_NAMES)}, got shape {array.shape}.")
        if not np.isfinite(array).all():
            raise ValueError("weights must be finite.")
        if np.any(array < 0.0):
            raise ValueError(f"weights must be non-negative, got {array.tolist()}.")
        if abs(float(np.sum(array)) - 1.0) >= 1e-8:
            raise ValueError(f"weights must sum to 1 within 1e-8, got sum={float(np.sum(array))}.")
        return array

    def _validate_init_config(self) -> None:
        """Validate quantile/rated-power/eps configuration."""
        if not 0.0 <= self.q_low < self.q_high <= 1.0:
            raise ValueError(f"Require 0 <= q_low < q_high <= 1, got {self.q_low}, {self.q_high}.")
        if self.site_rated_power_kw <= 0.0:
            raise ValueError(f"site_rated_power_kw must be positive, got {self.site_rated_power_kw}.")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be positive, got {self.eps}.")

    @property
    def short_window_steps(self) -> int:
        """Recent 1h sample count derived from SAMPLING_MINUTES."""
        return get_short_window_steps(SAMPLING_MINUTES)

    def fit(self, X_train: np.ndarray) -> "DynamicIndexTransformer":
        """Fit Train-only Q5/Q95 robust normalization parameters.

        只应对Train窗口调用本函数。它不会也不能读取Val/Cal/Test。
        """
        raw = compute_raw_components(X_train).astype(np.float64)
        q_low_values = np.quantile(raw, self.q_low, axis=0)
        q_high_values = np.quantile(raw, self.q_high, axis=0)
        ranges = q_high_values - q_low_values
        degenerate_mask = ranges < self.eps
        self.degenerate_components_ = [
            self.component_names[idx] for idx in np.flatnonzero(degenerate_mask)
        ]
        self.q_low_values_ = q_low_values.astype(np.float64, copy=False)
        self.q_high_values_ = q_high_values.astype(np.float64, copy=False)
        self.train_component_summary_ = {
            name: {
                "mean": float(np.mean(raw[:, idx])),
                "std": float(np.std(raw[:, idx], ddof=0)),
                "min": float(np.min(raw[:, idx])),
                "max": float(np.max(raw[:, idx])),
                "q_low": float(q_low_values[idx]),
                "q_high": float(q_high_values[idx]),
                "range": float(ranges[idx]),
                "degenerate": bool(degenerate_mask[idx]),
            }
            for idx, name in enumerate(self.component_names)
        }
        self.is_fitted_ = True
        return self

    def _check_fitted(self) -> None:
        """Prevent accidental transform-before-fit leakage patterns."""
        if not self.is_fitted_ or self.q_low_values_ is None or self.q_high_values_ is None:
            raise RuntimeError("DynamicIndexTransformer has not been fitted. Fit on Train only before transform.")

    def transform_components(self, raw_components: np.ndarray) -> np.ndarray:
        """Normalize raw components to [0,1] using Train-fitted Q5/Q95."""
        self._check_fitted()
        raw = _validate_raw_components(raw_components)
        denominator = self.q_high_values_ - self.q_low_values_
        denominator = np.where(denominator < self.eps, self.eps, denominator)
        normalized = (raw - self.q_low_values_) / denominator
        normalized = np.clip(normalized, 0.0, 1.0)
        if not np.isfinite(normalized).all():
            raise ValueError("normalized components contain NaN or Inf.")
        if np.min(normalized) < -1e-7 or np.max(normalized) > 1.0 + 1e-7:
            raise ValueError("normalized components are outside [0,1] beyond tolerance.")
        return normalized.astype(np.float32, copy=False)

    def transform(
        self,
        X: np.ndarray,
        return_components: bool = False,
    ) -> np.ndarray | DynamicIndexResult:
        """Compute D_t for windows using already fitted Train-only parameters.

        默认返回 D [N]。若 return_components=True，返回包含raw、normalized和D的
        DynamicIndexResult。
        """
        raw = compute_raw_components(X)
        normalized = self.transform_components(raw)
        dynamic_index = normalized.astype(np.float64) @ self.weights_
        dynamic_index = np.clip(dynamic_index, 0.0, 1.0)
        if not np.isfinite(dynamic_index).all():
            raise ValueError("D_t contains NaN or Inf.")
        if np.min(dynamic_index) < -1e-7 or np.max(dynamic_index) > 1.0 + 1e-7:
            raise ValueError("D_t is outside [0,1] beyond tolerance.")
        d_float32 = dynamic_index.astype(np.float32, copy=False)
        if return_components:
            return DynamicIndexResult(
                raw_components=raw.astype(np.float32, copy=False),
                normalized_components=normalized.astype(np.float32, copy=False),
                dynamic_index=d_float32,
            )
        return d_float32

    def fit_transform(
        self,
        X_train: np.ndarray,
        return_components: bool = False,
    ) -> np.ndarray | DynamicIndexResult:
        """Fit on Train windows, then transform the same Train windows."""
        self.fit(X_train)
        return self.transform(X_train, return_components=return_components)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable fitted transformer configuration."""
        return {
            "component_names": list(self.component_names),
            "weights": [float(v) for v in self.weights_],
            "q_low": float(self.q_low),
            "q_high": float(self.q_high),
            "q_low_values": None if self.q_low_values_ is None else [float(v) for v in self.q_low_values_],
            "q_high_values": None if self.q_high_values_ is None else [float(v) for v in self.q_high_values_],
            "eps": float(self.eps),
            "site_rated_power_kw": float(self.site_rated_power_kw),
            "sampling_minutes": int(SAMPLING_MINUTES),
            "history_steps": int(HISTORY_STEPS),
            "short_window_steps": int(self.short_window_steps),
            "is_fitted": bool(self.is_fitted_),
            "degenerate_components": list(self.degenerate_components_),
            "train_component_summary": self.train_component_summary_,
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "DynamicIndexTransformer":
        """Restore transformer from JSON-compatible config."""
        component_names = config.get("component_names")
        if component_names != COMPONENT_NAMES:
            raise ValueError(
                f"component_names mismatch: config has {component_names}, expected {COMPONENT_NAMES}."
            )
        if int(config.get("sampling_minutes")) != SAMPLING_MINUTES:
            raise ValueError("sampling_minutes in config does not match current schema.")
        if int(config.get("history_steps")) != HISTORY_STEPS:
            raise ValueError("history_steps in config does not match current schema.")

        transformer = cls(
            weights=config.get("weights"),
            q_low=float(config.get("q_low", DEFAULT_Q_LOW)),
            q_high=float(config.get("q_high", DEFAULT_Q_HIGH)),
            site_rated_power_kw=float(config.get("site_rated_power_kw", SITE_RATED_POWER_KW)),
            eps=float(config.get("eps", DEFAULT_EPS)),
        )
        if bool(config.get("is_fitted", False)):
            q_low_values = np.asarray(config.get("q_low_values"), dtype=np.float64)
            q_high_values = np.asarray(config.get("q_high_values"), dtype=np.float64)
            expected_shape = (len(COMPONENT_NAMES),)
            if q_low_values.shape != expected_shape or q_high_values.shape != expected_shape:
                raise ValueError("q_low_values/q_high_values shapes do not match component count.")
            if not np.isfinite(q_low_values).all() or not np.isfinite(q_high_values).all():
                raise ValueError("q_low_values/q_high_values contain NaN or Inf.")
            transformer.q_low_values_ = q_low_values
            transformer.q_high_values_ = q_high_values
            transformer.degenerate_components_ = list(config.get("degenerate_components", []))
            transformer.train_component_summary_ = config.get("train_component_summary")
            transformer.is_fitted_ = True
        return transformer

    def summary(self) -> dict[str, Any]:
        """Return transformer state without printing."""
        ranges = None
        if self.q_low_values_ is not None and self.q_high_values_ is not None:
            ranges = [float(v) for v in (self.q_high_values_ - self.q_low_values_)]
        return {
            "fitted": bool(self.is_fitted_),
            "component_names": list(self.component_names),
            "weights": [float(v) for v in self.weights_],
            "q_low": float(self.q_low),
            "q_high": float(self.q_high),
            "q_low_values": None if self.q_low_values_ is None else [float(v) for v in self.q_low_values_],
            "q_high_values": None if self.q_high_values_ is None else [float(v) for v in self.q_high_values_],
            "ranges": ranges,
            "history_steps": int(HISTORY_STEPS),
            "short_window_steps": int(self.short_window_steps),
            "sampling_minutes": int(SAMPLING_MINUTES),
            "site_rated_power_kw": float(self.site_rated_power_kw),
            "degenerate_components": list(self.degenerate_components_),
        }


def _empty_synthetic_windows(n_samples: int) -> np.ndarray:
    """Create finite synthetic raw windows [N,72,F] for self-check only."""
    X = np.zeros((n_samples, HISTORY_STEPS, len(FEATURE_COLUMNS)), dtype=np.float64)
    return X


def _set_power_and_ghi(X: np.ndarray, power: np.ndarray, ghi: np.ndarray) -> np.ndarray:
    """Fill Active_Power and GHI columns in synthetic windows."""
    result = X.copy()
    result[:, :, _required_feature_index(ACTIVE_POWER_COLUMN)] = power
    result[:, :, _required_feature_index(GHI_COLUMN)] = ghi
    return result


def _run_self_check() -> None:
    """Synthetic self-check; does not read project data or write files."""
    short_steps = get_short_window_steps()
    assert short_steps == 12

    # Case A: 完全稳定，所有动态分量应为0。
    stable = _empty_synthetic_windows(2)
    stable_power = np.full((2, HISTORY_STEPS), 1.0)
    stable_ghi = np.full((2, HISTORY_STEPS), 500.0)
    stable = _set_power_and_ghi(stable, stable_power, stable_ghi)
    stable_raw = compute_raw_components(stable)
    assert stable_raw.shape == (2, 4)
    assert np.allclose(stable_raw, 0.0, atol=1e-6), stable_raw

    # Case B: 线性GHI每小时增加100，短期和长期斜率一致，s_G≈0。
    t_hours = np.arange(HISTORY_STEPS, dtype=np.float64) * (SAMPLING_MINUTES / 60.0)
    linear_ghi = 50.0 + 100.0 * t_hours
    linear_power = 1.0 + 0.05 * t_hours
    linear = _empty_synthetic_windows(2)
    linear = _set_power_and_ghi(
        linear,
        np.tile(linear_power, (2, 1)),
        np.tile(linear_ghi, (2, 1)),
    )
    slope = _linear_slope_batch(np.tile(linear_ghi, (2, 1)))
    assert np.allclose(slope, 100.0, atol=1e-9), slope
    linear_raw = compute_raw_components(linear)
    assert linear_raw[0, 0] > 0.0
    assert linear_raw[0, 2] > 0.0
    assert abs(float(linear_raw[0, 3])) < 1e-5, linear_raw[0, 3]

    # Case C: PV最近1小时相邻变化0.52kW，r_P应为0.52/5.2=0.1。
    pv_ramp_case = _empty_synthetic_windows(1)
    pv_values = np.zeros((1, HISTORY_STEPS), dtype=np.float64)
    pv_values[:, -short_steps:] = np.arange(short_steps, dtype=np.float64) * 0.52
    ghi_values = np.full((1, HISTORY_STEPS), 100.0)
    pv_ramp_case = _set_power_and_ghi(pv_ramp_case, pv_values, ghi_values)
    pv_ramp_raw = compute_raw_components(pv_ramp_case)
    assert np.allclose(pv_ramp_raw[0, 1], 0.1, atol=1e-6), pv_ramp_raw[0, 1]

    # Case D: 最后1小时突然快速变化，slope gap应明显大于0。
    abrupt = _empty_synthetic_windows(2)
    abrupt_ghi = np.full((2, HISTORY_STEPS), 200.0)
    abrupt_ghi[:, -short_steps:] = 200.0 + np.arange(short_steps) * 60.0
    abrupt_power = np.full((2, HISTORY_STEPS), 1.0)
    abrupt_power[:, -short_steps:] = 1.0 + np.arange(short_steps) * 0.08
    abrupt = _set_power_and_ghi(abrupt, abrupt_power, abrupt_ghi)
    abrupt_raw = compute_raw_components(abrupt)
    assert abrupt_raw[0, 0] > linear_raw[0, 0]
    assert abrupt_raw[0, 2] > linear_raw[0, 2]
    assert abrupt_raw[0, 3] > 100.0, abrupt_raw[0, 3]

    # Fit/transform: 只用Train fit；transform Val不得改变Q5/Q95。
    train = np.concatenate([stable, linear, abrupt], axis=0)
    transformer = DynamicIndexTransformer()
    result = transformer.fit_transform(train, return_components=True)
    assert result.raw_components.shape == (6, 4)
    assert result.normalized_components.shape == (6, 4)
    assert result.dynamic_index.shape == (6,)
    assert np.isfinite(result.dynamic_index).all()
    assert np.min(result.dynamic_index) >= 0.0 and np.max(result.dynamic_index) <= 1.0
    q_low_before = transformer.q_low_values_.copy()
    q_high_before = transformer.q_high_values_.copy()
    val_d = transformer.transform(abrupt)
    assert val_d.shape == (2,)
    assert np.array_equal(transformer.q_low_values_, q_low_before)
    assert np.array_equal(transformer.q_high_values_, q_high_before)

    # Serialization round-trip.
    config = transformer.to_dict()
    restored = DynamicIndexTransformer.from_dict(config)
    original_d = transformer.transform(train)
    restored_d = restored.transform(train)
    assert np.allclose(original_d, restored_d, rtol=1e-7, atol=1e-7)

    saturation = compute_saturation_rates(result.normalized_components)
    assert set(saturation.keys()) == set(COMPONENT_NAMES)

    print("DynamicIndexTransformer self-check passed.")
    print(f"component_names: {COMPONENT_NAMES}")
    print(f"short_window_steps: {short_steps}")
    print("slope time unit: hours")
    print(f"default_weights: {DEFAULT_WEIGHTS.tolist()}")
    print(f"stable raw components first row: {stable_raw[0].tolist()}")
    print(f"known PV ramp r_P: {float(pv_ramp_raw[0, 1]):.8f}")
    print(f"known GHI slope: {float(slope[0]):.8f}")
    print(f"abrupt raw components first row: {abrupt_raw[0].tolist()}")
    print(f"D_t range on synthetic train: [{float(result.dynamic_index.min()):.8f}, {float(result.dynamic_index.max()):.8f}]")
    print("serialization round-trip: passed")
    print("Train-only Q5/Q95 transform immutability: passed")


if __name__ == "__main__":
    _run_self_check()
