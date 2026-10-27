"""Feature standardization utilities for PV forecasting windows.

This module implements transparent NumPy-based z-score standardization. It is
fit by external scripts only, typically using Train data selected upstream. It
does not read project data, clean data, split data, build windows, shuffle,
train models, compute D_t, or depend on sklearn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np

from src.data.schema import FEATURE_COLUMNS, FEATURE_INDEX, HISTORY_STEPS, TARGET_COLUMN


class FeatureStandardizer:
    """光伏特征标准化器，使用训练集统计量执行标准Z-score变换。

    参数只通过 ``fit`` 或 ``load`` 获得。统计量使用 population std
    ``ddof=0``，与常见 StandardScaler 行为一致。目标 Active_Power 使用
    FEATURE_COLUMNS 中同一套 mean/std，不额外拟合第二套 target scaler。
    """

    version = 1
    scaler_type = "standard_zscore"
    ddof = 0

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        eps: float = 1e-8,
    ) -> None:
        self.feature_columns = list(feature_columns) if feature_columns is not None else list(FEATURE_COLUMNS)
        self.eps = float(eps)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.n_samples_: int | None = None
        self.fitted_ = False
        self._validate_feature_order()

    def _validate_feature_order(self) -> None:
        """验证特征顺序与当前schema完全一致。"""
        if self.feature_columns != list(FEATURE_COLUMNS):
            raise ValueError(
                "feature_columns must exactly match src.data.schema.FEATURE_COLUMNS. "
                f"Got {self.feature_columns}, expected {list(FEATURE_COLUMNS)}."
            )
        if TARGET_COLUMN not in self.feature_columns:
            raise ValueError(f"TARGET_COLUMN {TARGET_COLUMN!r} is not in FEATURE_COLUMNS.")

    @property
    def target_feature_index(self) -> int:
        """返回目标 Active_Power 在特征维度中的位置，不手写固定索引。"""
        if TARGET_COLUMN in FEATURE_INDEX:
            index = int(FEATURE_INDEX[TARGET_COLUMN])
            if self.feature_columns[index] != TARGET_COLUMN:
                raise ValueError("FEATURE_INDEX is inconsistent with FEATURE_COLUMNS for target column.")
            return index
        return self.feature_columns.index(TARGET_COLUMN)

    def fit(self, X: np.ndarray) -> "FeatureStandardizer":
        """使用外部传入的二维训练点 ``[N, F]`` 拟合mean/std。

        ``fit`` 只接受二维数组，不接受滑动窗口三维数组。调用方应在外部
        完成Train有效行筛选，不能在这里自动删除NaN或Inf。
        """
        array = np.asarray(X)
        if array.ndim != 2:
            raise ValueError(f"fit(X) requires a 2D array [N, F], got shape {array.shape}.")
        if array.shape[1] != len(self.feature_columns):
            raise ValueError(
                f"fit(X) feature dimension must be {len(self.feature_columns)}, got {array.shape[1]}."
            )
        if len(array) == 0:
            raise ValueError("fit(X) received an empty array.")
        if not np.isfinite(array).all():
            raise ValueError("fit(X) contains NaN or Inf. Scaler will not drop or repair values.")

        mean = np.mean(array, axis=0, dtype=np.float64)
        std = np.std(array, axis=0, ddof=self.ddof, dtype=np.float64)
        low_std = std < self.eps
        if np.any(low_std):
            bad = [self.feature_columns[i] for i in np.flatnonzero(low_std)]
            raise ValueError(f"Feature std is below eps={self.eps}: {bad}. Refusing unsafe scaling.")

        self.mean_ = mean.astype(np.float64, copy=False)
        self.std_ = std.astype(np.float64, copy=False)
        self.n_samples_ = int(array.shape[0])
        self.fitted_ = True
        return self

    def _check_fitted(self) -> None:
        """禁止在未fit或load之前执行transform。"""
        if not self.fitted_ or self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler has not been fitted.")

    def _validate_feature_array(self, X: np.ndarray, method_name: str) -> np.ndarray:
        """验证二维或三维特征数组，沿最后一维执行标准化。"""
        self._check_fitted()
        array = np.asarray(X)
        if array.ndim not in (2, 3):
            raise ValueError(f"{method_name} supports only [N, F] or [N, T, F], got shape {array.shape}.")
        if array.shape[-1] != len(self.feature_columns):
            raise ValueError(
                f"{method_name} last dimension must be {len(self.feature_columns)}, got {array.shape[-1]}."
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{method_name} input contains NaN or Inf. Scaler will not repair values.")
        return array

    def transform(self, X: np.ndarray) -> np.ndarray:
        """标准化二维 ``[N,F]`` 或三维 ``[N,T,F]`` 特征数组，返回float32副本。"""
        array = self._validate_feature_array(X, "transform")
        transformed = (array - self.mean_) / self.std_
        if not np.isfinite(transformed).all():
            raise ValueError("transform output contains NaN or Inf.")
        return transformed.astype(np.float32, copy=False)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """将标准化后的二维或三维特征数组反变换回原始物理尺度。"""
        array = self._validate_feature_array(X_scaled, "inverse_transform")
        restored = array * self.std_ + self.mean_
        if not np.isfinite(restored).all():
            raise ValueError("inverse_transform output contains NaN or Inf.")
        return restored.astype(np.float32, copy=False)

    def _validate_target_array(self, y: np.ndarray, method_name: str) -> np.ndarray:
        """验证目标数组，支持 ``[N]`` 和 ``[N,1]``。"""
        self._check_fitted()
        array = np.asarray(y)
        if array.ndim == 1:
            pass
        elif array.ndim == 2 and array.shape[1] == 1:
            pass
        else:
            raise ValueError(f"{method_name} supports [N] or [N,1], got shape {array.shape}.")
        if len(array) == 0:
            raise ValueError(f"{method_name} received an empty target array.")
        if not np.isfinite(array).all():
            raise ValueError(f"{method_name} input contains NaN or Inf.")
        return array

    def transform_target(self, y: np.ndarray) -> np.ndarray:
        """使用Active_Power对应的mean/std标准化目标y，返回float32。"""
        array = self._validate_target_array(y, "transform_target")
        idx = self.target_feature_index
        transformed = (array - self.mean_[idx]) / self.std_[idx]
        if not np.isfinite(transformed).all():
            raise ValueError("transform_target output contains NaN or Inf.")
        return transformed.astype(np.float32, copy=False)

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """将标准化目标反变换回真实Active_Power尺度，返回float32。"""
        array = self._validate_target_array(y_scaled, "inverse_transform_target")
        idx = self.target_feature_index
        restored = array * self.std_[idx] + self.mean_[idx]
        if not np.isfinite(restored).all():
            raise ValueError("inverse_transform_target output contains NaN or Inf.")
        return restored.astype(np.float32, copy=False)

    def save(self, path: str | Path, fit_scope: str | None = None) -> Path:
        """保存Scaler参数到JSON，便于科研审计和复现实验。"""
        self._check_fitted()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "version": self.version,
            "scaler_type": self.scaler_type,
            "feature_columns": list(self.feature_columns),
            "target_column": TARGET_COLUMN,
            "mean": [float(v) for v in self.mean_],
            "std": [float(v) for v in self.std_],
            "n_samples": int(self.n_samples_),
            "eps": float(self.eps),
            "ddof": int(self.ddof),
        }
        if fit_scope is not None:
            payload["fit_scope"] = str(fit_scope)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "FeatureStandardizer":
        """从JSON恢复Scaler，并验证特征顺序与当前schema完全一致。"""
        input_path = Path(path)
        with input_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        feature_columns = payload.get("feature_columns")
        if feature_columns != list(FEATURE_COLUMNS):
            raise ValueError(
                "Scaler feature_columns differ from current schema.FEATURE_COLUMNS. "
                f"JSON has {feature_columns}, schema has {list(FEATURE_COLUMNS)}. "
                "Refusing to reorder automatically."
            )
        if payload.get("target_column") != TARGET_COLUMN:
            raise ValueError(
                f"Scaler target_column {payload.get('target_column')!r} does not match schema target {TARGET_COLUMN!r}."
            )
        if payload.get("scaler_type") != cls.scaler_type:
            raise ValueError(f"Unsupported scaler_type: {payload.get('scaler_type')!r}.")

        scaler = cls(feature_columns=feature_columns, eps=float(payload.get("eps", 1e-8)))
        scaler.mean_ = np.asarray(payload["mean"], dtype=np.float64)
        scaler.std_ = np.asarray(payload["std"], dtype=np.float64)
        if scaler.mean_.shape != (len(FEATURE_COLUMNS),) or scaler.std_.shape != (len(FEATURE_COLUMNS),):
            raise ValueError("Loaded mean/std shapes do not match current feature count.")
        if np.any(scaler.std_ < scaler.eps):
            raise ValueError("Loaded scaler contains std below eps.")
        if not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.std_).all():
            raise ValueError("Loaded scaler mean/std contain NaN or Inf.")
        scaler.n_samples_ = int(payload["n_samples"])
        scaler.fitted_ = True
        return scaler

    def summary(self) -> dict[str, Any]:
        """返回Scaler状态和目标Active_Power统计摘要。"""
        target_mean = None
        target_std = None
        if self.fitted_ and self.mean_ is not None and self.std_ is not None:
            idx = self.target_feature_index
            target_mean = float(self.mean_[idx])
            target_std = float(self.std_[idx])
        return {
            "fitted": bool(self.fitted_),
            "feature_columns": list(self.feature_columns),
            "mean": None if self.mean_ is None else [float(v) for v in self.mean_],
            "std": None if self.std_ is None else [float(v) for v in self.std_],
            "n_samples": self.n_samples_,
            "target_column": TARGET_COLUMN,
            "target_mean": target_mean,
            "target_std": target_std,
            "history_steps": HISTORY_STEPS,
        }


def _run_self_check() -> None:
    """使用人工小矩阵执行轻量自检，不读取真实项目数据。"""
    X = np.array(
        [
            [0.0, 100.0, 20.0, 25.0, 40.0],
            [1.0, 200.0, 30.0, 26.0, 42.0],
            [2.0, 300.0, 40.0, 27.0, 44.0],
            [3.0, 400.0, 50.0, 28.0, 46.0],
        ],
        dtype=np.float64,
    )
    X_windows = np.stack([X[:3], X[1:4]], axis=0)
    y = X[:, FEATURE_INDEX[TARGET_COLUMN]]

    scaler = FeatureStandardizer().fit(X)
    X_scaled = scaler.transform(X)
    X_restored = scaler.inverse_transform(X_scaled)
    Xw_scaled = scaler.transform(X_windows)
    Xw_restored = scaler.inverse_transform(Xw_scaled)
    y_scaled = scaler.transform_target(y)
    y_restored = scaler.inverse_transform_target(y_scaled)
    y_column = y.reshape(-1, 1)
    y_column_scaled = scaler.transform_target(y_column)
    y_column_restored = scaler.inverse_transform_target(y_column_scaled)

    feature_error = float(np.max(np.abs(X_restored.astype(np.float64) - X)))
    window_error = float(np.max(np.abs(Xw_restored.astype(np.float64) - X_windows)))
    target_error = float(np.max(np.abs(y_restored.astype(np.float64) - y)))
    target_column_error = float(np.max(np.abs(y_column_restored.astype(np.float64) - y_column)))

    assert X_scaled.shape == X.shape
    assert Xw_scaled.shape == X_windows.shape
    assert y_scaled.shape == y.shape
    assert y_column_scaled.shape == y_column.shape
    assert np.allclose(X_restored, X, rtol=1e-6, atol=1e-6)
    assert np.allclose(Xw_restored, X_windows, rtol=1e-6, atol=1e-6)
    assert np.allclose(y_restored, y, rtol=1e-6, atol=1e-6)
    assert np.allclose(y_column_restored, y_column, rtol=1e-6, atol=1e-6)

    temp_path = Path("/tmp/pv_feature_standardizer_self_check.json")
    scaler.save(temp_path, fit_scope="self_check_synthetic_data")
    loaded = FeatureStandardizer.load(temp_path)
    loaded_scaled = loaded.transform(X)
    assert np.allclose(loaded_scaled, X_scaled, rtol=1e-7, atol=1e-7)

    print("FeatureStandardizer self-check passed.")
    print(f"feature_count: {len(FEATURE_COLUMNS)}")
    print(f"target_column: {TARGET_COLUMN}")
    print(f"max feature roundtrip error: {feature_error:.8g}")
    print(f"max window roundtrip error: {window_error:.8g}")
    print(f"max target roundtrip error: {target_error:.8g}")
    print(f"max target [N,1] roundtrip error: {target_column_error:.8g}")
    print(f"save/load check: {temp_path}")


if __name__ == "__main__":
    _run_self_check()
