import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.collate import collate_fn
from src.datasets.data_utils import get_dataloaders
from src.trainer import SoundStreamTrainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging
from src.utils.train_utils import TrainableModel

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="train")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """

    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    # batch_transforms should be put on device

    dataloaders, batch_transforms = get_dataloaders(
        config,
        device,
        logger,
        lambda batch: collate_fn(batch, config.consts.div_length),
    )

    # build model architecture, then print to console
    generator = TrainableModel(config.generator, logger, device, "generator")
    discriminator = TrainableModel(
        config.discriminator, logger, device, "discriminator"
    )

    metrics = instantiate(config.metrics)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = SoundStreamTrainer(
        generator=generator,
        discriminator=discriminator,
        metrics=metrics,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
