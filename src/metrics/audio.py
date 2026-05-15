import torch
from torchmetrics.audio import (
    NonIntrusiveSpeechQualityAssessment,
    ShortTimeObjectiveIntelligibility,
)

from src.metrics.base_metric import BaseMetric


class STOI(BaseMetric):
    def __init__(self, sample_rate: int, device: str):
        super().__init__(name="STOI")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = ShortTimeObjectiveIntelligibility(fs=sample_rate).to(device)

    def __call__(
        self,
        audio: torch.Tensor,
        reconstruction: torch.Tensor,
        pad_lengths: torch.Tensor,
        **batch
    ):
        result = 0
        for i, length in enumerate(pad_lengths):
            result += self.metric(
                audio[i, ..., : length.item()], reconstruction[i, ..., : length.item()]
            ).item()
        return result / reconstruction.shape[0]


class NISQA(BaseMetric):
    def __init__(self, sample_rate: int, device: str):
        super().__init__(name="NISQA")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = NonIntrusiveSpeechQualityAssessment(fs=sample_rate).to(device)

    def __call__(
        self, reconstruction: torch.Tensor, pad_lengths: torch.Tensor, **batch
    ):
        result = 0
        for i, length in enumerate(pad_lengths):
            result += self.metric(reconstruction[i, ..., : length.item()])[0].item()
        return result / reconstruction.shape[0]
