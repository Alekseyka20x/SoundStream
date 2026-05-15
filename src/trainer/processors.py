from abc import ABC, abstractmethod
from typing import Literal

from src.utils.train_utils import TrainableModel


class BaseModelProcessor(ABC):
    @abstractmethod
    def set_train_mode(self):
        pass

    @abstractmethod
    def set_eval_mode(self):
        pass

    @abstractmethod
    def process_batch(self, batch, mode: Literal["train", "inference"]):
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, checkpoint):
        pass


class MultiModelProcessor(BaseModelProcessor):
    def __init__(self, models: dict[str, TrainableModel]):
        self._models = models

    def set_train_mode(self):
        for model in self._models.values():
            model.model.train()

    def set_eval_mode(self):
        for model in self._models.values():
            model.model.eval()

    def state_dict(self):
        return {name: model.state_dict() for name, model in self._models.items()}

    def load_state_dict(self, checkpoint):
        for name, model in self._models.items():
            if checkpoint.get(name) is not None:
                model.load_state_dict(checkpoint[name])
            else:
                model.logger.critical(
                    f"Cannot find model '{name}' in saved checkpoint. Skipping"
                )

    def get_grad_norm(self):
        return {
            f"grad_norm_{name}": model.get_grad_norm()
            for name, model in self._models.items()
        }

    def get_lr_and_make_step(self):
        lrs = {}
        for name, model in self._models.items():
            lrs[f"lr_{name}"] = model.lr_scheduler.get_last_lr()[0]
            model.lr_scheduler.step()
        return lrs
