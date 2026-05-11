from typing import Any

import torch

from src.metrics import BaseMetric
from src.trainer.base_trainer import BaseTrainer
from src.utils.train_utils import TrainableModel


class GanTrainer(BaseTrainer):
    def __init__(self, generator: TrainableModel, discriminator: TrainableModel):
        self.G = generator
        self.D = discriminator

    def process_batch(self, batch: dict[str, Any], metrics: list[BaseMetric]):
        real_data: torch.Tensor = batch["audio"]

        if self.G.model.training:
            G_output = self.G.model(real_data)
            fake_data: torch.Tensor = G_output["reconstruction"]
            batch.update(G_output)

            # Disctiminator update
            D_output_fake_detached = self.D.model(fake_data.detach())
            D_output_real = self.D.model(real_data)
            D_loss = self.D.loss_function(
                discriminator_output_real=D_output_real,
                discriminator_output_fake=D_output_fake_detached,
            )["loss"]

            batch.update(
                {
                    "discriminator_loss": D_loss.item(),
                    "discriminator_output_real": D_output_real,
                }
            )

            self.D.optimizer.zero_grad()
            D_loss.backward()
            self.D.optimizer.step()

            # Generator update
            D_output_fake = self.D.model(fake_data)
            batch.update({"discriminator_output_fake": D_output_fake})
            G_losses = self.G.loss_function(**batch)
            G_loss = G_losses["loss"]
            G_losses.pop("loss")

            self.G.optimizer.zero_grad()
            G_loss.backward()
            self.G.optimizer.step(0)

            batch.update(G_losses)
            batch.update({"generator_loss": G_loss.item()})

            if self.G.lr_scheduler:
                self.G.lr_scheduler.step()
            if self.D.lr_scheduler:
                self.D.lr_scheduler.step()

        else:
            raise NotImplementedError()

        # for metric in metrics:
        #     metric.update(**batch)
