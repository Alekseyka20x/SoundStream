from typing import Any

import torch.nn as nn

from .base_losses import (
    AdversarialLoss,
    CommitmentLoss,
    FeatureLoss,
    ReconstructionLoss,
)


class GeneratorLoss(nn.Module):
    def __init__(
        self,
        lambda_adversarial: float,
        lambda_feature: float,
        lambda_reconstruction: float,
        lambda_commitment: float,
    ):
        super().__init__()

        self.lambda_adversarial = lambda_adversarial
        self.lambda_feature = lambda_feature
        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_commitment = lambda_commitment

        self.adversarial_loss = AdversarialLoss()
        self.feature_loss = FeatureLoss()
        self.reconstruction_loss = ReconstructionLoss()
        self.commitment_loss = CommitmentLoss()

    def forward(self, **batch):
        adversarial_loss = self.adversarial_loss(
            batch["discriminator_output_fake"], "real"
        )["loss"]
        feature_loss = self.feature_loss(**batch)["loss"]
        reconstruction_loss = self.reconstruction_loss(**batch)["loss"]
        commitment_loss = self.commitment_loss(**batch)["loss"]
        loss = (
            self.lambda_adversarial * adversarial_loss
            + self.lambda_feature * feature_loss
            + self.lambda_reconstruction * reconstruction_loss
            + self.lambda_commitment * commitment_loss
        )
        return {
            "loss_adversarial": adversarial_loss,
            "loss_feature": feature_loss,
            "loss_reconstruction": reconstruction_loss,
            "loss_commitment": commitment_loss,
            "loss": loss,
        }
