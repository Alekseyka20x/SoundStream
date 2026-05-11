import torch
import torch.nn as nn


class ResidualUnit(nn.Module):
    def __init__(self, N: int, m: int, s: tuple[int, int], slope: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LeakyReLU(negative_slope=slope),
            nn.Conv2d(
                kernel_size=(3, 3), in_channels=N, out_channels=N, padding=(1, 1)
            ),
            nn.LeakyReLU(negative_slope=slope),
            nn.Conv2d(
                kernel_size=(s[0] + 2, s[1] + 2),
                in_channels=N,
                out_channels=N * m,
                stride=s,
                padding=(s[0] // 2 + 1, s[1] // 2 + 1),
            ),
        )
        self.skip = nn.Conv2d(
            kernel_size=(1, 1), in_channels=N, out_channels=N * m, stride=s
        )

    def forward(self, x: torch.Tensor):
        return self.net(x) + self.skip(x)


class STFTDiscriminator(nn.Module):
    def __init__(
        self, hidden_channels: int, window_length: int, hop_length: int, slope: float
    ):
        super().__init__()
        self.window_length = window_length
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(window_length))

        F = self.window_length // 2 + 1
        C = hidden_channels
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    kernel_size=(7, 7), in_channels=2, out_channels=C, padding=(3, 3)
                ),
                ResidualUnit(N=C, m=2, s=(1, 2), slope=slope),
                ResidualUnit(N=2 * C, m=2, s=(2, 2), slope=slope),
                ResidualUnit(N=4 * C, m=1, s=(1, 2), slope=slope),
                ResidualUnit(N=4 * C, m=2, s=(2, 2), slope=slope),
                ResidualUnit(N=8 * C, m=1, s=(1, 2), slope=slope),
                ResidualUnit(N=8 * C, m=2, s=(2, 2), slope=slope),
            ]
        )
        self.final_layer = nn.Conv2d(
            kernel_size=(1, F // 2**6 + 1), in_channels=16 * C, out_channels=1
        )

    def forward(self, x: torch.Tensor):
        x = torch.stft(
            x.squeeze(1),
            n_fft=self.window_length,
            hop_length=self.hop_length,
            return_complex=True,
            window=self.window,
        )
        # (B, F, T/H, (re, im)) -> (B, 2, T/H, F)
        x = torch.view_as_real(x).permute(0, 3, 2, 1)

        output = {"features": []}
        for layer in self.conv_layers:
            x = layer(x)
            output["features"].append(x)
        x = self.final_layer(x)
        output["logits"] = x.squeeze(-1)
        return {"stft_discriminator": output}
