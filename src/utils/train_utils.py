from logging import Logger
from typing import Any

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.nn.utils import clip_grad_norm_


class TrainableModel:
    def __init__(
        self, config: DictConfig, logger: Logger, device: str, model_name: str
    ):
        self.config = config
        self.logger = logger

        self.logger.info(f"- Instantiating {model_name}")
        self.model: nn.Module = instantiate(config.model).to(device)
        self.logger.info(self.model)

        # get function handles of loss and metrics
        self.loss_function: nn.Module = instantiate(config.loss_function).to(device)

        # build optimizer, learning rate scheduler
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer: torch.optim.Optimizer = instantiate(
            config.optimizer, params=trainable_params
        )
        self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler = instantiate(
            config.lr_scheduler, optimizer=self.optimizer
        )

    def state_dict(self):
        return {
            "arch": type(self.model).__name__,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "config": self.config,
        }

    def load_state_dict(self, checkpoint):
        if checkpoint["config"]["model"] != self.config["model"]:
            self.logger.warning(
                "Warning: Architecture configuration given in the config file is different from that "
                "of the checkpoint. This may yield an exception when state_dict is loaded."
            )
        self.model.load_state_dict(checkpoint["model"])

        if (
            checkpoint["config"]["optimizer"] != self.config["optimizer"]
            or checkpoint["config"]["lr_scheduler"] != self.config["lr_scheduler"]
        ):
            self.logger.warning(
                "Warning: Optimizer or lr_scheduler given in the config file is different "
                "from that of the checkpoint. Optimizer and scheduler parameters "
                "are not resumed."
            )
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    def clip_grad_norm(self):
        """
        Clips the gradient norm by the value defined in
        config.<model name>.max_grad_norm
        """
        if self.config.get("max_grad_norm", None) is not None:
            clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        """
        Calculates the gradient norm for logging.

        Args:
            norm_type (float | str | None): the order of the norm.
        Returns:
            total_norm (float): the calculated norm.
        """
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type,
        )
        return total_norm.item()
