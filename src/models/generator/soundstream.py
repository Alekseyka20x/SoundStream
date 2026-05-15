import torch
import torch.nn as nn

from .encoder_decoder import Decoder, Encoder
from .rvq import RVQ


class SoundStream(nn.Module):
    def __init__(self, encoder: Encoder, rvq: RVQ, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.rvq = rvq
        self.decoder = decoder

    def forward(self, x):
        embeddings = self.encoder(x)["embeddings"]
        rvq_output = self.rvq(embeddings)
        quantized_embeddings = rvq_output["quantized_embeddings"]
        reconstruction = self.decoder(quantized_embeddings)["reconstruction"]
        return {
            "embeddings": embeddings,
            "quantized_embeddings": quantized_embeddings,
            "reconstruction": reconstruction,
            "codebook_indices": rvq_output["codebook_indices"],
        }

    @torch.inference_mode()
    def encode(self, x):
        return self.rvq(self.encoder(x))["codebook_indices"]

    @torch.inference_mode()
    def decode(self, x):
        return self.decoder(self.rvq.decode(x))
