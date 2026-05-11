from collections.abc import Sequence

import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor):
        x = nn.functional.pad(x, (self.padding, 0), "constant", 0)
        return self.conv(x)


class CausalConvTranspose1d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1
    ):
        super().__init__()
        self.cut_last = kernel_size - stride
        self.conv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        return x[..., : -self.cut_last]


class ResidualUnit(nn.Module):
    def __init__(self, N: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ELU(),
            CausalConv1d(
                kernel_size=7, in_channels=N, out_channels=N, dilation=dilation
            ),
            nn.ELU(),
            CausalConv1d(kernel_size=1, in_channels=N, out_channels=N),
        )

    def forward(self, x: torch.Tensor):
        return x + self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, N: int, S: int):
        super().__init__()
        self.net = nn.Sequential(
            ResidualUnit(N // 2, dilation=1),
            ResidualUnit(N // 2, dilation=3),
            ResidualUnit(N // 2, dilation=9),
            nn.ELU(),
            CausalConv1d(
                kernel_size=2 * S, in_channels=N // 2, out_channels=N, stride=S
            ),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, out_channels: int, hidden_channels: int, strides: Sequence[int]):
        super().__init__()

        C = hidden_channels
        encoder_blocks = []
        for stride in strides:
            C *= 2
            encoder_blocks.append(EncoderBlock(N=C, S=stride))

        self.net = nn.Sequential(
            CausalConv1d(kernel_size=7, in_channels=1, out_channels=C),
            *encoder_blocks,
            nn.ELU(),
            CausalConv1d(kernel_size=3, in_channels=C, out_channels=out_channels)
        )

    def forward(self, x: torch.Tensor):
        return {"embeddings": self.net(x)}


class DecoderBlock(nn.Module):
    def __init__(self, N: int, S: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ELU(),
            CausalConvTranspose1d(
                kernel_size=2 * S, in_channels=2 * N, out_channels=N, stride=S
            ),
            ResidualUnit(N, dilation=1),
            ResidualUnit(N, dilation=3),
            ResidualUnit(N, dilation=9),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, strides: Sequence[int]):
        super().__init__()

        C = hidden_channels * (2 ** len(strides))
        decoder_blocks = []
        for stride in strides:
            C //= 2
            decoder_blocks.append(EncoderBlock(N=C, S=stride))

        self.net = nn.Sequential(
            CausalConv1d(
                kernel_size=7,
                in_channels=in_channels,
                out_channels=hidden_channels * (2 ** len(strides)),
            ),
            *decoder_blocks,
            nn.ELU(),
            CausalConv1d(kernel_size=7, in_channels=C, out_channels=1)
        )

    def forward(self, x: torch.Tensor):
        return {"reconstruction": self.net(x)}
