import logging
import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.collate import collate_fn
from src.datasets.data_utils import get_dataloaders
from src.logger import setup_logging
from src.trainer import SoundStreamInferencer
from src.utils.init_utils import set_random_seed

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    logger = setup_logging()

    project_config = OmegaConf.to_container(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(
        config,
        device,
        logger,
        lambda batch: collate_fn(batch, config.consts.div_length),
    )

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    logger.info("Model was successfully loaded")

    # get metrics
    metrics = instantiate(config.metrics)

    inferencer = SoundStreamInferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=None,
        writer=writer,
        metrics=metrics,
    )

    logs = inferencer.run_inference()

    for part in logs.keys():
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print(f"    {full_key:15s}: {value}")


if __name__ == "__main__":
    main()
