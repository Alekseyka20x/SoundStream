from .rvq import RVQ
from .encoder_decoder import Encoder, Decoder

import torch.nn as nn
import torch


class SoundStream(nn.Module):
    def __init__(self, encoder: Encoder, rvq: RVQ, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.rvq = rvq
        self.decoder = decoder
    
    def forward(self, x):
        embeddings = self.encoder(x)["embeddings"]
        quantized_embeddings = self.rvq(embeddings)["quantized_embeddings"]
        reconstruction = self.decoder(quantized_embeddings)["reconstruction"]
        return {
            "embeddings": embeddings,
            "quantized_embeddings": quantized_embeddings,
            "reconstruction": reconstruction
        }

    @torch.inference_mode()
    def encode(self, x):
        return self.rvq(self.encoder(x))["codebook_indices"]
    
    @torch.inference_mode()
    def decode(self, x):
        return self.decoder(self.rvq.decode(x))
