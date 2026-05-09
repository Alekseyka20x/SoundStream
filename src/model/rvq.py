import torch.nn as nn


class RVQ(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return {
            "quantized_embeddings": ...,
            "codebook_indices": ...
        }

    def decode(self, x):
        return ...
