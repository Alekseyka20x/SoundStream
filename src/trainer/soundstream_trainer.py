from typing import Any, Literal

import torch

from src.metrics import BaseMetric
from src.trainer.base_trainer import BaseTrainer
from src.trainer.processors import MultiModelProcessor
from src.utils.train_utils import TrainableModel


class SoundStreamProcessor(MultiModelProcessor):
    def __init__(
        self,
        generator: TrainableModel,
        discriminator: TrainableModel,
        metrics: dict[str, list[BaseMetric]],
    ):
        super().__init__({"generator": generator, "discriminator": discriminator})
        self.G = generator
        self.D = discriminator
        self.metrics = metrics

    def process_batch(self, batch: dict[str, Any], mode: Literal["train", "inference"]):
        real_data: torch.Tensor = batch["audio"]
        metrics = {}

        G_output = self.G.model(real_data)
        fake_data: torch.Tensor = G_output["reconstruction"]
        batch.update(G_output)

        if mode == "train":
            # Disctiminator update
            D_output_fake_detached = self.D.model(fake_data.detach())
            D_output_real = self.D.model(real_data)
            D_loss = self.D.loss_function(
                discriminator_output_real=D_output_real,
                discriminator_output_fake=D_output_fake_detached,
            )["loss"]

            batch["discriminator_output_real"] = D_output_real

            self.D.optimizer.zero_grad()
            D_loss.backward()
            self.D.clip_grad_norm()
            self.D.optimizer.step()

            # Generator update
            D_output_fake = self.D.model(fake_data)
            batch["discriminator_output_fake"] = D_output_fake

            G_losses = self.G.loss_function(**batch)
            G_loss = G_losses["loss"]

            self.G.optimizer.zero_grad()
            G_loss.backward()
            self.G.clip_grad_norm()
            self.G.optimizer.step()

            metrics.update(
                {
                    "loss_discriminator": D_loss.item(),
                    "loss_generator": G_loss.item(),
                    **{
                        loss_name: loss.item()
                        for loss_name, loss in G_losses.items()
                        if loss_name != "loss"
                    },
                }
            )
            metrics.update(self.get_lr_and_make_step())
            metrics.update(self.get_grad_norm())

        with torch.no_grad():
            for metric in self.metrics[mode]:
                metrics[metric.name] = metric(**batch)

        return batch, metrics


class SoundStreamTrainer(BaseTrainer):
    def __init__(
        self,
        generator: TrainableModel,
        discriminator: TrainableModel,
        metrics: dict[str, list[BaseMetric]],
        config,
        device,
        dataloaders,
        logger,
        writer,
        epoch_len=None,
        skip_oom=True,
        batch_transforms=None,
    ):
        super().__init__(
            model_processor=SoundStreamProcessor(generator, discriminator, metrics),
            config=config,
            device=device,
            dataloaders=dataloaders,
            logger=logger,
            writer=writer,
            epoch_len=epoch_len,
            skip_oom=skip_oom,
            batch_transforms=batch_transforms,
        )

    def _log_batch(self, batch_idx, batch, mode="train"):
        raise NotImplementedError
