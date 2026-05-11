from typing import Any, Literal

import torch
from torch import nn


class AdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        discriminator_output: dict[Any, dict[str, Any]],
        label: Literal["real", "fake"],
        **batch
    ):
        K = len(discriminator_output) - 1
        loss = 0
        for output in discriminator_output.values():
            logits: torch.Tensor = output["logits"]
            loss += torch.max(1 - (1 if label == "real" else -1) * logits).mean(axis=-1)
        loss /= K
        return {"loss": loss.mean()}


class FeatureLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        discriminator_output_real: dict[Any, dict[str, Any]],
        discriminator_output_fake: dict[Any, dict[str, Any]],
        **batch
    ):
        K = len(discriminator_output_real) - 1
        loss = 0
        for disc_label in discriminator_output_real.keys():
            features_real = discriminator_output_real[disc_label]["features"].detach()
            features_fake = discriminator_output_fake[disc_label]["features"]
            for feature_real, feature_fake in zip(features_real, features_fake):
                loss += torch.abs(feature_real - feature_fake).mean(axis=-1)
            L = len(features_real)
        loss /= K * L
        return {"loss": loss.mean()}


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, **batch):
        ...


class CommitmentLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, embeddings: torch.Tensor, quantized_embeddings: torch.Tensor, **batch
    ):
        return {"loss": nn.functional.mse_loss(embeddings, quantized_embeddings)}
