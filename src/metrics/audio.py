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

    def __call__(self, audio: torch.Tensor, reconstruction: torch.Tensor, **batch):
        return self.metric(audio, reconstruction).item()


class NISQA(BaseMetric):
    def __init__(self, sample_rate: int, device: str):
        super().__init__(name="NISQA")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = NonIntrusiveSpeechQualityAssessment(fs=sample_rate).to(device)

    def __call__(self, reconstruction: torch.Tensor, **batch):
        return self.metric(reconstruction)[0].item()
