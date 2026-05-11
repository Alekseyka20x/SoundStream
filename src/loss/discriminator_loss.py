import torch.nn as nn

from .base_losses import AdversarialLoss


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.adversarial_loss = AdversarialLoss()

    def forward(self, discriminator_output_real, discriminator_output_fake, **batch):
        loss_real = self.adversarial_loss(discriminator_output_real, "real")
        loss_fake = self.adversarial_loss(discriminator_output_fake, "fake")
        return loss_real + loss_fake
