import torch
from torch import nn
from typing import Any


class AdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, descriminator_logits: dict[Any, dict[str, torch.Tensor]], **batch):
        K = len(descriminator_logits) - 1
        loss = 0
        for obj in descriminator_logits:
            logits = obj["logits"]
            labels = obj["labels"]
            loss += torch.max(1 - labels * logits).mean(axis=-1)
        loss /= K
        return {
            "loss": loss.mean()
        }


class FeatureLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, **batch):
        ...


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, **batch):
        ...


class CommitmentLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(embeddings: torch.Tensor, quantized_embeddings: torch.Tensor, **batch):
        return {
            "loss": nn.functional.mse_loss(embeddings, quantized_embeddings)
        }
