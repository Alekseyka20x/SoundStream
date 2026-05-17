from typing import Any, Literal

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.metrics import BaseMetric
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.trainer.processors import InferenceModelProcessor


class SoundStreamInferenceProcessor(InferenceModelProcessor):
    def __init__(self, model: nn.Module, metrics: dict[str, list[BaseMetric]]):
        self.model = model
        self.metrics = metrics

    def process_batch(self, batch: dict[str, Any], mode: str):
        real_data: torch.Tensor = batch["audio"]
        metrics = {}

        batch.update(self.model(real_data))

        for metric in self.metrics[mode]:
            result = metric(**batch)
            if isinstance(result, dict):
                metrics.update(result)
            else:
                metrics[metric.name] = result

        return batch, metrics


class SoundStreamInferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        writer,
        metrics=None,
        batch_transforms=None,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model_processor = SoundStreamInferenceProcessor(model, metrics)
        self.batch_transforms = batch_transforms

        self.writer = writer
        self.epoch_len = 0

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition
        self.save_path = save_path

        # define metrics
        self.evaluation_metrics = MetricTracker()

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._evaluation_epoch(0, part, dataloader)
            part_logs[part] = logs
        return part_logs

    def _log_batch(self, batch_idx, batch, mode="train"):
        pass
