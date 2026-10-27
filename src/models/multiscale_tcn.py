"""Multi-scale TCN temporal encoder for Site 18 PV forecasting.

Multi-scale TCN 在本项目中作为成熟、轻量的辅助时序编码器，用于从
已经标准化的 6h 历史窗口中提取固定维度 H_ms 表征。它不是当前研究的
核心创新算法；后续创新位于天气动态指数 D_t、Dynamic-aware BSCN 和 DUAC。

输入采用数据层 canonical shape: [B, T, F]，当前为 [B, 72, 5]。
模型内部只做：
1. transpose 到 Conv1d 所需 [B, F, T]；
2. 对同一个 6h 窗口构造 5min / 15min / 60min 三个尺度；
3. 三个经典 causal TCN 分支编码；
4. concat + Linear 融合为 H_ms。

本模块不读取数据、不加载 Scaler、不做标准化、不构造标签、不训练模型。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from src.data.schema import FEATURE_COLUMNS, HISTORY_STEPS
from src.models.tcn_blocks import TemporalConvNet


DEFAULT_BRANCH_CHANNELS = 32
DEFAULT_KERNEL_SIZE = 3
DEFAULT_DROPOUT = 0.1
DEFAULT_EMBEDDING_DIM = 64
FIFTEEN_MIN_POOL = 3
SIXTY_MIN_POOL = 12
DEFAULT_FIVE_MIN_DILATIONS = (1, 2)
DEFAULT_FIFTEEN_MIN_DILATIONS = (1, 2)
DEFAULT_SIXTY_MIN_DILATIONS = (1, 1)


class MultiScaleTCN(nn.Module):
    """轻量 Multi-scale TCN 编码器。

    Parameters
    ----------
    num_features:
        输入特征数，默认使用 schema.FEATURE_COLUMNS 的长度。
    history_steps:
        历史窗口长度，默认使用 schema.HISTORY_STEPS，当前为 72。
    branch_channels:
        每个 TCN 分支每层输出通道数；默认 32，并构造 [32, 32]。
    kernel_size:
        TCN Conv1d kernel size，默认 3。
    dropout:
        TCN 与融合层 dropout，默认 0.1。
    embedding_dim:
        最终 H_ms 维度，默认 64。
    """

    def __init__(
        self,
        num_features: int | None = None,
        history_steps: int = HISTORY_STEPS,
        branch_channels: int = DEFAULT_BRANCH_CHANNELS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        dropout: float = DEFAULT_DROPOUT,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        five_min_dilations: Sequence[int] = DEFAULT_FIVE_MIN_DILATIONS,
        fifteen_min_dilations: Sequence[int] = DEFAULT_FIFTEEN_MIN_DILATIONS,
        sixty_min_dilations: Sequence[int] = DEFAULT_SIXTY_MIN_DILATIONS,
    ) -> None:
        super().__init__()
        resolved_num_features = len(FEATURE_COLUMNS) if num_features is None else num_features
        self._validate_init_args(
            num_features=resolved_num_features,
            history_steps=history_steps,
            branch_channels=branch_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            embedding_dim=embedding_dim,
        )
        self._validate_pooling_lengths(history_steps)

        self.num_features = int(resolved_num_features)
        self.history_steps = int(history_steps)
        self.branch_channels = int(branch_channels)
        self.kernel_size = int(kernel_size)
        self.dropout = float(dropout)
        self.embedding_dim = int(embedding_dim)
        self.feature_columns = list(FEATURE_COLUMNS)

        branch_hidden_channels = [self.branch_channels, self.branch_channels]
        self.tcn_5min = TemporalConvNet(
            num_inputs=self.num_features,
            channels=branch_hidden_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            dilations=five_min_dilations,
        )
        self.tcn_15min = TemporalConvNet(
            num_inputs=self.num_features,
            channels=branch_hidden_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            dilations=fifteen_min_dilations,
        )
        self.tcn_60min = TemporalConvNet(
            num_inputs=self.num_features,
            channels=branch_hidden_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            dilations=sixty_min_dilations,
        )

        self.pool_15min = nn.AvgPool1d(
            kernel_size=FIFTEEN_MIN_POOL,
            stride=FIFTEEN_MIN_POOL,
        )
        self.pool_60min = nn.AvgPool1d(
            kernel_size=SIXTY_MIN_POOL,
            stride=SIXTY_MIN_POOL,
        )

        concat_dim = self.branch_channels * 3
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.prediction_head = nn.Linear(self.embedding_dim, 1)

    @staticmethod
    def _validate_init_args(
        *,
        num_features: int,
        history_steps: int,
        branch_channels: int,
        kernel_size: int,
        dropout: float,
        embedding_dim: int,
    ) -> None:
        """验证初始化参数，尽早暴露配置错误。"""
        if num_features <= 0:
            raise ValueError(f"num_features must be positive, got {num_features}.")
        if history_steps <= 0:
            raise ValueError(f"history_steps must be positive, got {history_steps}.")
        if branch_channels <= 0:
            raise ValueError(f"branch_channels must be positive, got {branch_channels}.")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}.")

    @staticmethod
    def _validate_pooling_lengths(history_steps: int) -> None:
        """确保固定池化尺度不会静默截断历史窗口。"""
        for pool_size in (FIFTEEN_MIN_POOL, SIXTY_MIN_POOL):
            if history_steps % pool_size != 0:
                raise ValueError(
                    f"history_steps={history_steps} must be divisible by pool_size={pool_size}. "
                    "Please update pooling design explicitly instead of truncating windows."
                )

    def _validate_input_shape(self, x: torch.Tensor) -> None:
        """验证输入为 canonical [B, T, F]，不自动猜测或转置错误输入。"""
        if x.ndim != 3:
            raise ValueError(f"MultiScaleTCN expects [B, T, F], got shape {tuple(x.shape)}.")
        if x.size(1) != self.history_steps:
            raise ValueError(
                f"Expected history_steps={self.history_steps} at dim=1, got {x.size(1)}. "
                "Input should be [B, 72, 5], not [B, 5, 72]."
            )
        if x.size(2) != self.num_features:
            raise ValueError(
                f"Expected num_features={self.num_features} at dim=2, got {x.size(2)}."
            )

    def _build_multiscale_inputs(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """从同一个 6h 窗口构造 5min / 15min / 60min 三个尺度输入。

        输入 x: [B, 72, 5]
        输出:
        - x_5min: [B, 5, 72]
        - x_15min: [B, 5, 24]
        - x_60min: [B, 5, 6]
        """
        self._validate_input_shape(x)
        x_5min = x.transpose(1, 2).contiguous()
        x_15min = self.pool_15min(x_5min)
        x_60min = self.pool_60min(x_5min)
        return x_5min, x_15min, x_60min

    def forward_branch_features(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回三个尺度的 branch embedding，均为 [B, branch_channels]。"""
        x_5min, x_15min, x_60min = self._build_multiscale_inputs(x)

        out_5min = self.tcn_5min(x_5min)
        out_15min = self.tcn_15min(x_15min)
        out_60min = self.tcn_60min(x_60min)

        h_5min = out_5min[:, :, -1]
        h_15min = out_15min[:, :, -1]
        h_60min = out_60min[:, :, -1]
        return h_5min, h_15min, h_60min

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """返回 Multi-scale TCN 编码后的 H_ms，shape 为 [B, embedding_dim]。"""
        h_5min, h_15min, h_60min = self.forward_branch_features(x)
        h_cat = torch.cat([h_5min, h_15min, h_60min], dim=1)
        return self.fusion(h_cat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """临时监督预测头，输入 [B,72,5]，输出 y_hat [B]。"""
        h_ms = self.forward_features(x)
        return self.prediction_head(h_ms).squeeze(-1)


def count_trainable_parameters(model: nn.Module) -> int:
    """统计可训练参数量。"""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _has_any_gradient(module: nn.Module) -> bool:
    """检查模块中是否至少有一个可训练参数获得梯度。"""
    return any(
        parameter.requires_grad and parameter.grad is not None
        for parameter in module.parameters()
    )


def _assert_shape(name: str, actual: torch.Size | tuple[int, ...], expected: tuple[int, ...]) -> None:
    """简单 shape assert，错误信息更易读。"""
    actual_tuple = tuple(actual)
    if actual_tuple != expected:
        raise AssertionError(f"{name} shape failed: expected {expected}, got {actual_tuple}.")


def _run_self_check() -> None:
    """轻量自检：多尺度 shape、H_ms、prediction、finite 与梯度路径。"""
    torch.manual_seed(42)
    batch_size = 8
    model = MultiScaleTCN()
    model.train()
    x = torch.randn(batch_size, HISTORY_STEPS, len(FEATURE_COLUMNS))

    x_5min, x_15min, x_60min = model._build_multiscale_inputs(x)
    _assert_shape("5min scale", x_5min.shape, (batch_size, len(FEATURE_COLUMNS), 72))
    _assert_shape("15min scale", x_15min.shape, (batch_size, len(FEATURE_COLUMNS), 24))
    _assert_shape("60min scale", x_60min.shape, (batch_size, len(FEATURE_COLUMNS), 6))

    h_5min, h_15min, h_60min = model.forward_branch_features(x)
    _assert_shape("H_5", h_5min.shape, (batch_size, DEFAULT_BRANCH_CHANNELS))
    _assert_shape("H_15", h_15min.shape, (batch_size, DEFAULT_BRANCH_CHANNELS))
    _assert_shape("H_60", h_60min.shape, (batch_size, DEFAULT_BRANCH_CHANNELS))

    h_cat = torch.cat([h_5min, h_15min, h_60min], dim=1)
    _assert_shape("H_cat", h_cat.shape, (batch_size, DEFAULT_BRANCH_CHANNELS * 3))

    h_ms = model.forward_features(x)
    _assert_shape("H_ms", h_ms.shape, (batch_size, DEFAULT_EMBEDDING_DIM))
    if not torch.isfinite(h_ms).all():
        raise AssertionError("H_ms contains NaN or Inf.")

    y_hat = model(x)
    _assert_shape("prediction", y_hat.shape, (batch_size,))
    if not torch.isfinite(y_hat).all():
        raise AssertionError("prediction contains NaN or Inf.")

    loss = y_hat.mean()
    loss.backward()
    gradient_checks = {
        "5min branch": _has_any_gradient(model.tcn_5min),
        "15min branch": _has_any_gradient(model.tcn_15min),
        "60min branch": _has_any_gradient(model.tcn_60min),
        "fusion": _has_any_gradient(model.fusion),
        "prediction_head": _has_any_gradient(model.prediction_head),
    }
    failed = [name for name, ok in gradient_checks.items() if not ok]
    if failed:
        raise AssertionError(f"Gradient path failed for: {failed}.")

    parameter_count = count_trainable_parameters(model)
    print("MultiScaleTCN self-check passed.")
    print(f"Input shape: {tuple(x.shape)}")
    print(f"5min scale shape: {tuple(x_5min.shape)}")
    print(f"15min scale shape: {tuple(x_15min.shape)}")
    print(f"60min scale shape: {tuple(x_60min.shape)}")
    print(f"H_5 shape: {tuple(h_5min.shape)}")
    print(f"H_15 shape: {tuple(h_15min.shape)}")
    print(f"H_60 shape: {tuple(h_60min.shape)}")
    print(f"H_cat shape: {tuple(h_cat.shape)}")
    print(f"H_ms shape: {tuple(h_ms.shape)}")
    print(f"Prediction shape: {tuple(y_hat.shape)}")
    print(f"Trainable parameters: {parameter_count}")
    print(f"Gradient checks: {gradient_checks}")


if __name__ == "__main__":
    _run_self_check()
