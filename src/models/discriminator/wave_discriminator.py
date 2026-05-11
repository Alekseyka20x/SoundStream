import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class DiscriminatorBlock(nn.Module):
    def __init__(self, slope: float):
        super().__init__()
        self.conv_layers = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(kernel_size=15, in_channels=1, out_channels=16, padding=7)
                ),
                weight_norm(
                    nn.Conv1d(
                        kernel_size=41,
                        in_channels=16,
                        out_channels=64,
                        groups=4,
                        stride=4,
                        padding=20,
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        kernel_size=41,
                        in_channels=64,
                        out_channels=256,
                        groups=16,
                        stride=4,
                        padding=20,
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        kernel_size=41,
                        in_channels=256,
                        out_channels=1024,
                        groups=64,
                        stride=4,
                        padding=20,
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        kernel_size=41,
                        in_channels=1024,
                        out_channels=1024,
                        groups=256,
                        stride=4,
                        padding=20,
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        kernel_size=5, in_channels=1024, out_channels=1024, padding=2
                    )
                ),
            ]
        )
        self.final_layer = weight_norm(
            nn.Conv1d(kernel_size=3, in_channels=1024, out_channels=1, padding=1)
        )
        self.activation = nn.LeakyReLU(negative_slope=slope)

    def forward(self, x: torch.Tensor):
        output = {"features": []}
        for layer in self.conv_layers:
            x = layer(x)
            x = self.activation(x)
            output["features"].append(x)
        x = self.final_layer(x)
        output["logits"] = x
        return output


class WaveDiscriminator(nn.Module):
    def __init__(self, n_wave_discriminators: int, slope: float):
        super().__init__()
        self.downsample = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)
        self.wave_discriminators = nn.ModuleList(
            [DiscriminatorBlock(slope=slope) for _ in range(n_wave_discriminators)]
        )

    def forward(self, x: torch.Tensor):
        output = {}
        for disc_num, disc in enumerate(self.wave_discriminators):
            output[f"wave_discriminator_{disc_num}"] = disc(x)
            x = self.downsample(x)
        return output
