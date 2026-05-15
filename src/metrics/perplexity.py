import torch

from src.metrics.base_metric import BaseMetric


class Perplexity(BaseMetric):
    def __init__(
        self, codebook_size: int, n_quantizers: int, div_length: int, eps: float = 1e-9
    ):
        super().__init__(name="perplexity")
        self.codebook_size = codebook_size
        self.n_quantizers = n_quantizers
        self.div_length = div_length
        self.eps = eps

    def __call__(
        self, codebook_indices: torch.Tensor, pad_lengths: torch.Tensor, **batch
    ):
        B = codebook_indices.shape[0]
        codebook_indices = codebook_indices.reshape(B, self.n_quantizers, -1)
        result = {}

        for quantizer_i in range(self.n_quantizers):
            indices = torch.concatenate(
                [
                    codebook_indices[
                        b, quantizer_i, : pad_lengths[b].item() // self.div_length
                    ]
                    for b in range(B)
                ]
            )
            p = torch.bincount(indices, minlength=self.codebook_size) / indices.shape[0]
            perplexity = torch.exp(-torch.sum(p * torch.log(p + self.eps))).item()
            result[f"perplexity_codebook_{quantizer_i}"] = perplexity

        return result
