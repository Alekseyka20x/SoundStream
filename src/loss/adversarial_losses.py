from typing import Any, Literal

import torch
import torch.nn as nn


class AdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        discriminator_output: dict[Any, dict[str, Any]],
        label: Literal["real", "fake"],
        **batch
    ):
        K = len(discriminator_output)
        loss = 0
        for output in discriminator_output.values():
            logits: torch.Tensor = output["logits"]
            loss += torch.clamp(
                1 - (1 if label == "real" else -1) * logits, min=0
            ).mean()
        loss /= K
        return {"loss": loss}


class AdversarialDiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.adversarial_loss = AdversarialLoss()

    def forward(self, discriminator_output_real, discriminator_output_fake, **batch):
        loss_real = self.adversarial_loss(discriminator_output_real, "real")["loss"]
        loss_fake = self.adversarial_loss(discriminator_output_fake, "fake")["loss"]
        return {"loss": loss_real + loss_fake}


class AdversarialGeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.adversarial_loss = AdversarialLoss()

    def forward(self, discriminator_output_fake, **batch):
        loss_fake = self.adversarial_loss(discriminator_output_fake, "real")["loss"]
        return {"loss": loss_fake}
