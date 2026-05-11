import torch
import torch.nn as nn

from .stft_discriminator import STFTDiscriminator
from .wave_discriminator import WaveDiscriminator


class Discriminator(nn.Module):
    def __init__(
        self,
        wave_discriminator: WaveDiscriminator,
        stft_discriminator: STFTDiscriminator,
    ):
        super().__init__()
        self.wave_discriminator = wave_discriminator
        self.stft_discriminator = stft_discriminator

    def forward(self, x: torch.Tensor):
        output = {}
        output.update(self.wave_discriminator(x))
        output.update(self.stft_discriminator(x))
        return output
