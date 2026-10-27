"""经典小型 Temporal Convolutional Network 基础组件。

本模块只实现 Multi-scale TCN 需要复用的标准 TCN block：
causal Conv1d、dilated convolution、residual connection、ReLU、
Dropout 与 weight normalization。输入和输出均采用 PyTorch Conv1d
格式 [B, C, T]，不在本文件中进行 [B, T, F] 与 [B, F, T] 的转换。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


DEFAULT_KERNEL_SIZE = 3
DEFAULT_DROPOUT = 0.1


class Chomp1d(nn.Module):
    """删除 Conv1d 右侧多出的时间步，保持 causal convolution 的长度一致。

    PyTorch Conv1d 的 padding 是左右两侧都会补零。对于 causal TCN，
    使用 padding=(kernel_size - 1) * dilation 后，需要裁掉右侧多出的
    chomp_size 个时间点，确保输出长度等于输入长度，并避免未来信息泄漏。
    """

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        if chomp_size < 0:
            raise ValueError(f"chomp_size must be non-negative, got {chomp_size}.")
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回裁掉右侧 padding 后的张量。"""
        if self.chomp_size == 0:
            return x.contiguous()
        if x.size(-1) <= self.chomp_size:
            raise ValueError(
                "Chomp size is larger than or equal to the temporal dimension: "
                f"chomp_size={self.chomp_size}, temporal_length={x.size(-1)}."
            )
        return x[:, :, : -self.chomp_size].contiguous()


def _causal_padding(kernel_size: int, dilation: int) -> int:
    """计算 causal Conv1d 需要的显式 padding。"""
    if kernel_size <= 0:
        raise ValueError(f"kernel_size must be positive, got {kernel_size}.")
    if dilation <= 0:
        raise ValueError(f"dilation must be positive, got {dilation}.")
    return (kernel_size - 1) * dilation


def _make_weight_norm_conv1d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    *,
    dilation: int = 1,
    padding: int = 0,
) -> nn.Conv1d:
    """创建带 weight normalization 的 Conv1d。

    当前环境 PyTorch 支持推荐 API：
    torch.nn.utils.parametrizations.weight_norm。先做普通 Conv1d 初始化，
    再应用 parametrization，避免直接操作 parametrized weight 的细节。
    """
    conv = nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
        dilation=dilation,
    )
    _init_conv1d(conv)
    return weight_norm(conv)


def _init_conv1d(conv: nn.Conv1d) -> None:
    """使用稳定、常见的 Kaiming 初始化 Conv1d。"""
    nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


class TemporalBlock(nn.Module):
    """经典 TCN residual block。

    结构：
    Conv1d -> Chomp1d -> ReLU -> Dropout ->
    Conv1d -> Chomp1d -> ReLU -> Dropout ->
    residual add -> ReLU

    输入输出均为 [B, C, T]，时间长度 T 保持不变。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}.")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        padding = _causal_padding(kernel_size, dilation)

        self.conv1 = _make_weight_norm_conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = _make_weight_norm_conv1d(
            out_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )

        if in_channels == out_channels:
            self.downsample: nn.Module | None = None
        else:
            projection = nn.Conv1d(in_channels, out_channels, kernel_size=1)
            _init_conv1d(projection)
            self.downsample = projection

        self.final_relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 residual TCN block 前向传播。"""
        main = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        if main.shape != residual.shape:
            raise RuntimeError(
                "TemporalBlock residual shape mismatch: "
                f"main={tuple(main.shape)}, residual={tuple(residual.shape)}."
            )
        return self.final_relu(main + residual)


class TemporalConvNet(nn.Module):
    """由多个 TemporalBlock 堆叠而成的经典 TCN。

    Parameters
    ----------
    num_inputs:
        输入通道数，即 [B, C, T] 中的 C。
    channels:
        每个 TemporalBlock 的输出通道数列表。
    kernel_size:
        Conv1d kernel size，默认 3。
    dropout:
        Dropout 概率，默认 0.1。
    dilations:
        每个 block 的 dilation。若为 None，默认使用 1, 2, 4, ...
    """

    def __init__(
        self,
        num_inputs: int,
        channels: Sequence[int],
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        dropout: float = DEFAULT_DROPOUT,
        dilations: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if num_inputs <= 0:
            raise ValueError(f"num_inputs must be positive, got {num_inputs}.")
        if len(channels) == 0:
            raise ValueError("channels must contain at least one output channel.")
        if any(channel <= 0 for channel in channels):
            raise ValueError(f"all channels must be positive, got {list(channels)}.")

        if dilations is None:
            resolved_dilations = [2**i for i in range(len(channels))]
        else:
            if len(channels) != len(dilations):
                raise ValueError(
                    "channels and dilations must have the same length: "
                    f"len(channels)={len(channels)}, len(dilations)={len(dilations)}."
                )
            if any(dilation <= 0 for dilation in dilations):
                raise ValueError(f"all dilations must be positive, got {list(dilations)}.")
            resolved_dilations = list(dilations)

        layers: list[nn.Module] = []
        for i, out_channels in enumerate(channels):
            in_channels = num_inputs if i == 0 else channels[i - 1]
            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=resolved_dilations[i],
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)
        self.num_inputs = num_inputs
        self.channels = list(channels)
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.dilations = resolved_dilations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 [B, C_in, T]，输出 [B, C_out, T]。"""
        if x.ndim != 3:
            raise ValueError(f"TemporalConvNet expects [B, C, T], got shape {tuple(x.shape)}.")
        if x.size(1) != self.num_inputs:
            raise ValueError(
                f"Expected input channels={self.num_inputs}, got {x.size(1)}."
            )
        return self.network(x)


def compute_receptive_field(
    kernel_size: int,
    dilations: Sequence[int],
    *,
    conv_layers_per_block: int = 2,
) -> int:
    """计算 TCN 理论感受野点数。

    每个 TemporalBlock 包含两层相同 dilation 的卷积，因此默认每个 block
    对 receptive field 的增量为 2 * (kernel_size - 1) * dilation。
    """
    if conv_layers_per_block <= 0:
        raise ValueError(
            "conv_layers_per_block must be positive, "
            f"got {conv_layers_per_block}."
        )
    if kernel_size <= 0:
        raise ValueError(f"kernel_size must be positive, got {kernel_size}.")
    if any(dilation <= 0 for dilation in dilations):
        raise ValueError(f"all dilations must be positive, got {list(dilations)}.")
    return 1 + sum(
        conv_layers_per_block * (kernel_size - 1) * dilation
        for dilation in dilations
    )


def _assert_shape_test(
    *,
    input_shape: tuple[int, int, int],
    channels: Sequence[int],
    kernel_size: int,
    dilations: Sequence[int],
) -> tuple[int, int, int]:
    """运行一次 shape 自检并返回输出 shape。"""
    model = TemporalConvNet(
        num_inputs=input_shape[1],
        channels=channels,
        kernel_size=kernel_size,
        dropout=DEFAULT_DROPOUT,
        dilations=dilations,
    )
    model.eval()
    x = torch.randn(*input_shape)
    with torch.no_grad():
        out = model(x)
    expected_shape = (input_shape[0], channels[-1], input_shape[2])
    if tuple(out.shape) != expected_shape:
        raise AssertionError(
            f"Shape test failed: expected {expected_shape}, got {tuple(out.shape)}."
        )
    return tuple(out.shape)


def _assert_causality() -> None:
    """验证修改未来输入不会影响过去输出。"""
    torch.manual_seed(42)
    model = TemporalConvNet(
        num_inputs=5,
        channels=[16, 16],
        kernel_size=3,
        dropout=0.0,
        dilations=[1, 2],
    )
    model.eval()

    x1 = torch.randn(4, 5, 72)
    x2 = x1.clone()
    t = 30
    x2[:, :, t + 1 :] = torch.randn_like(x2[:, :, t + 1 :]) * 10.0

    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)

    if not torch.allclose(out1[:, :, : t + 1], out2[:, :, : t + 1], atol=1e-6, rtol=1e-5):
        max_diff = (out1[:, :, : t + 1] - out2[:, :, : t + 1]).abs().max().item()
        raise AssertionError(
            "Causality test failed: future inputs changed past outputs. "
            f"max_diff={max_diff:.6g}"
        )


def _run_self_check() -> None:
    """轻量自检：shape、长度保持、残差兼容与 causality。"""
    torch.manual_seed(7)
    results = [
        _assert_shape_test(
            input_shape=(8, 5, 72),
            channels=[32, 32],
            kernel_size=3,
            dilations=[1, 2],
        ),
        _assert_shape_test(
            input_shape=(8, 5, 24),
            channels=[32, 32],
            kernel_size=3,
            dilations=[1, 2],
        ),
        _assert_shape_test(
            input_shape=(8, 5, 6),
            channels=[32, 32],
            kernel_size=3,
            dilations=[1, 1],
        ),
    ]
    _assert_causality()

    receptive_field = compute_receptive_field(kernel_size=3, dilations=[1, 2])
    print("TemporalConvNet self-check passed.")
    print(f"Test 1 output shape: {results[0]}")
    print(f"Test 2 output shape: {results[1]}")
    print(f"Test 3 output shape: {results[2]}")
    print("Causality test: passed")
    print(f"Receptive field for kernel_size=3, dilations=[1, 2]: {receptive_field} points")


if __name__ == "__main__":
    _run_self_check()
