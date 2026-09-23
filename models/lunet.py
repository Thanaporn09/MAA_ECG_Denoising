from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def same_padding(kernel_size: int, dilation: int) -> int:
    return ((kernel_size - 1) // 2) * dilation


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        use_norm: bool,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            padding=same_padding(kernel_size, dilation),
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=True)
        self.norm = nn.BatchNorm1d(out_channels) if use_norm else nn.Identity()
        self.activation = nn.GELU()

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        signal = self.depthwise(signal)
        signal = self.pointwise(signal)
        signal = self.norm(signal)
        return self.activation(signal)


class GroupConvBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        use_norm: bool,
        max_groups: int = 8,
    ) -> None:
        super().__init__()
        groups = max(
            candidate
            for candidate in range(1, min(max_groups, in_channels) + 1)
            if in_channels % candidate == 0
        )
        self.grouped = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            padding=same_padding(kernel_size, dilation),
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.norm1 = nn.BatchNorm1d(in_channels) if use_norm else nn.Identity()
        self.activation1 = nn.GELU()
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=True)
        self.norm2 = nn.BatchNorm1d(out_channels) if use_norm else nn.Identity()
        self.activation2 = nn.GELU()

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        signal = self.activation1(self.norm1(self.grouped(signal)))
        signal = self.pointwise(signal)
        return self.activation2(self.norm2(signal))


def padded_max_pool1d(signal: torch.Tensor, kernel_size: int) -> torch.Tensor:
    length = signal.shape[-1]
    output_length = math.ceil(length / kernel_size)
    padding = max(0, (output_length - 1) * kernel_size + kernel_size - length)
    if padding:
        left = padding // 2
        signal = F.pad(signal, (left, padding - left))
    return F.max_pool1d(signal, kernel_size=kernel_size, stride=kernel_size)


class _DeterministicLinearInterpolation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, signal, left, right, left_weight, right_weight, backward_matrix):
        ctx.save_for_backward(backward_matrix)
        shape = (1,) * (signal.ndim - 1) + (-1,)
        return (
            signal.index_select(-1, left) * left_weight.view(shape)
            + signal.index_select(-1, right) * right_weight.view(shape)
        )

    @staticmethod
    def backward(ctx, gradient):
        (backward_matrix,) = ctx.saved_tensors
        return torch.matmul(gradient, backward_matrix), None, None, None, None, None


def _deterministic_linear_upsample(
    signal: torch.Tensor, output_length: int
) -> torch.Tensor:
    input_length = int(signal.shape[-1])
    destination = torch.arange(output_length, dtype=torch.float64)
    source = ((destination + 0.5) * input_length / output_length - 0.5).clamp(
        0.0, float(input_length - 1)
    )
    left = source.floor().long()
    right = (left + 1).clamp_max(input_length - 1)
    right_weight = source - left.to(source.dtype)
    left_weight = 1.0 - right_weight
    samples = torch.arange(input_length).unsqueeze(0)
    backward_matrix = (
        (samples == left[:, None]).double() * left_weight[:, None]
        + (samples == right[:, None]).double() * right_weight[:, None]
    )
    return _DeterministicLinearInterpolation.apply(
        signal,
        left.to(signal.device),
        right.to(signal.device),
        left_weight.to(device=signal.device, dtype=signal.dtype),
        right_weight.to(device=signal.device, dtype=signal.dtype),
        backward_matrix.to(device=signal.device, dtype=signal.dtype),
    )


def linear_upsample(signal: torch.Tensor, output_length: int) -> torch.Tensor:
    if signal.shape[-1] == output_length:
        return signal
    if torch.are_deterministic_algorithms_enabled():
        return _deterministic_linear_upsample(signal, output_length)
    return F.interpolate(signal, size=output_length, mode="linear", align_corners=False)


class LUNet1D(nn.Module):
    encoder_attr_names = ("encoder1", "encoder2", "encoder3", "bottleneck")
    decoder_attr_names = ("decoder1", "decoder2", "decoder3", "output_projection")

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        enc_channels: tuple[int, int, int] = (16, 32, 64),
        dec_channels: tuple[int, int, int] = (32, 16, 16),
        encoder_kernels: tuple[int, int, int, int] = (7, 5, 3, 3),
        decoder_kernels: tuple[int, int, int] = (3, 5, 7),
        dilation: int = 2,
        pool_kernels: tuple[int, int, int] = (5, 2, 2),
        use_norm: bool = True,
        use_skip: bool = False,
    ) -> None:
        super().__init__()
        if not (len(enc_channels) == len(dec_channels) == len(pool_kernels) == 3):
            raise ValueError("LUNet requires exactly three encoder/decoder stages")
        if len(encoder_kernels) != 4 or len(decoder_kernels) != 3:
            raise ValueError("LUNet requires four encoder and three decoder kernels")
        if use_skip:
            raise ValueError("Manuscript LUNet requires use_skip=false")
        self.enc_channels = tuple(enc_channels)
        self.dec_channels = tuple(dec_channels)
        self.pool_kernels = tuple(pool_kernels)
        self.use_skip = False
        self.encoder1 = DepthwiseSeparableConv1d(
            in_channels, enc_channels[0], encoder_kernels[0], dilation, use_norm
        )
        self.encoder2 = DepthwiseSeparableConv1d(
            enc_channels[0], enc_channels[1], encoder_kernels[1], dilation, use_norm
        )
        self.encoder3 = DepthwiseSeparableConv1d(
            enc_channels[1], enc_channels[2], encoder_kernels[2], dilation, use_norm
        )
        self.bottleneck = DepthwiseSeparableConv1d(
            enc_channels[2], enc_channels[2], encoder_kernels[3], dilation, use_norm
        )
        self.decoder1 = GroupConvBlock1d(
            enc_channels[2], dec_channels[0], decoder_kernels[0], dilation, use_norm
        )
        self.decoder2 = GroupConvBlock1d(
            dec_channels[0], dec_channels[1], decoder_kernels[1], dilation, use_norm
        )
        self.decoder3 = GroupConvBlock1d(
            dec_channels[1], dec_channels[2], decoder_kernels[2], dilation, use_norm
        )
        self.output_projection = nn.Conv1d(dec_channels[2], out_channels, 1)

    @property
    def bottleneck_channels(self) -> int:
        return int(self.enc_channels[-1])

    def encode_stages(
        self, signal: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[int, int, int],
    ]:
        encoder1 = self.encoder1(signal)
        encoder2 = self.encoder2(padded_max_pool1d(encoder1, self.pool_kernels[0]))
        encoder3 = self.encoder3(padded_max_pool1d(encoder2, self.pool_kernels[1]))
        bottleneck = self.bottleneck(
            padded_max_pool1d(encoder3, self.pool_kernels[2])
        )
        return encoder1, encoder2, encoder3, bottleneck, (
            int(encoder1.shape[-1]),
            int(encoder2.shape[-1]),
            int(encoder3.shape[-1]),
        )

    def encode(self, signal: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        *_, bottleneck, lengths = self.encode_stages(signal)
        return bottleneck, lengths

    def decode(self, memory: torch.Tensor, lengths: tuple[int, int, int]) -> torch.Tensor:
        length1, length2, length3 = (int(value) for value in lengths)
        decoder1 = self.decoder1(linear_upsample(memory, length3))
        decoder2 = self.decoder2(linear_upsample(decoder1, length2))
        decoder3 = self.decoder3(linear_upsample(decoder2, length1))
        return self.output_projection(decoder3)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        bottleneck, lengths = self.encode(signal)
        return self.decode(bottleneck, lengths)

    def shape_trace(self, input_length: int) -> dict[str, tuple[int, int]]:
        if input_length <= 0:
            raise ValueError("input_length must be positive")
        length1 = math.ceil(input_length / self.pool_kernels[0])
        length2 = math.ceil(length1 / self.pool_kernels[1])
        length3 = math.ceil(length2 / self.pool_kernels[2])
        return {
            "input": (1, input_length),
            "encoder1": (self.enc_channels[0], input_length),
            "encoder2": (self.enc_channels[1], length1),
            "encoder3": (self.enc_channels[2], length2),
            "bottleneck": (self.enc_channels[2], length3),
            "decoder1": (self.dec_channels[0], length2),
            "decoder2": (self.dec_channels[1], length1),
            "decoder3": (self.dec_channels[2], input_length),
            "output": (1, input_length),
        }


def build_lunet(config: dict) -> LUNet1D:
    allowed = {
        "in_channels",
        "out_channels",
        "enc_channels",
        "dec_channels",
        "encoder_kernels",
        "decoder_kernels",
        "dilation",
        "pool_kernels",
        "use_norm",
        "use_skip",
    }
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unsupported LUNet fields: {sorted(unknown)}")
    return LUNet1D(**config)
