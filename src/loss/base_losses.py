from typing import Any

import torch
import torch.nn as nn
import torchaudio


class FeatureLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        discriminator_output_real: dict[Any, dict[str, Any]],
        discriminator_output_fake: dict[Any, dict[str, Any]],
        **batch
    ):
        K = len(discriminator_output_real)
        loss = 0
        for disc_label in discriminator_output_real.keys():
            features_real = discriminator_output_real[disc_label]["features"]
            features_fake = discriminator_output_fake[disc_label]["features"]

            L = len(features_real)
            disc_loss = 0
            for feature_real, feature_fake in zip(features_real, features_fake):
                disc_loss += torch.abs(feature_real.detach() - feature_fake).mean()
            loss += disc_loss / L

        loss /= K
        return {"loss": loss}


class CommitmentLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, embeddings: torch.Tensor, quantized_embeddings: torch.Tensor, **batch
    ):
        return {
            "loss": nn.functional.mse_loss(embeddings, quantized_embeddings.detach())
        }


class ReconstructionLoss(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_mels: int,
        window_lengths: list[int],
        eps: float = 1e-4,
    ):
        super().__init__()
        self.register_buffer("window_lengths", torch.tensor(window_lengths))
        self.eps = eps
        self.mel_specs = nn.ModuleList(
            [
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=s,
                    win_length=s,
                    hop_length=s // 4,
                    n_mels=n_mels,
                )
                for s in self.window_lengths
            ]
        )

    def forward(self, audio: torch.Tensor, reconstruction: torch.Tensor, **batch):
        B = audio.shape[0]
        loss = 0
        for s, mel_spec in zip(self.window_lengths, self.mel_specs):
            mel_orig = mel_spec(audio)
            mel_fake = mel_spec(reconstruction)

            l1 = torch.abs(mel_orig - mel_fake).sum()
            diff = torch.log(mel_orig + self.eps) - torch.log(mel_fake + self.eps)
            l2 = torch.sqrt((diff**2).sum(dim=-2) + self.eps).sum()

            loss += l1 + torch.sqrt(s / 2) * l2

        return {"loss": loss / B}


class CombinedLoss(nn.Module):
    def __init__(self, losses: list[dict[str, Any]]):
        super().__init__()
        self.loss_names = [loss["name"] for loss in losses]
        self.register_buffer(
            "weights", torch.tensor([loss["weight"] for loss in losses])
        )
        self.losses = nn.ModuleList(loss["loss"] for loss in losses)

    def forward(self, **batch):
        losses = {}
        final_loss = 0
        for i, loss_name in enumerate(self.loss_names):
            loss = self.losses[i](**batch)["loss"]
            final_loss += loss * self.weights[i]
            losses[loss_name] = loss
        losses["loss"] = final_loss
        return losses
