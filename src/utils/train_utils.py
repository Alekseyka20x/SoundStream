from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from hydra.utils import instantiate


@dataclass
class TrainableModel:
    model: nn.Module
    loss_function: nn.Module
    metrics: Any
    optimizer: torch.optim.Optimizer
    lr_scheduler: Any


def setup_trainable_model(config, logger, device, model_name) -> TrainableModel:
    # build model architecture, then print to console
    logger.info(f"- Instantiating {model_name}")
    model = instantiate(config.model).to(device)
    logger.info(model)

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)
    # metrics = instantiate(config.metrics)

    torch.optim.Adam

    # build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    return TrainableModel(
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        metrics=None,
    )
