import torch
import torch.nn as nn
import torchaudio

from .encoder_decoder import Decoder, Encoder
from .rvq import RVQ


class SoundStream(nn.Module):
    def __init__(
        self, encoder: Encoder, rvq: RVQ, decoder: Decoder, target_sample_rate: int
    ):
        super().__init__()
        self.encoder = encoder
        self.rvq = rvq
        self.decoder = decoder
        self.target_sample_rate = target_sample_rate

    def forward(self, x: torch.Tensor):
        embeddings = self.encoder(x)["embeddings"]
        rvq_output = self.rvq(embeddings)
        quantized_embeddings = rvq_output["quantized_embeddings"]
        reconstruction = self.decoder(quantized_embeddings)["reconstruction"]
        return {
            "embeddings": embeddings,
            "quantized_embeddings": quantized_embeddings,
            "reconstruction": reconstruction,
            "codebook_indices": rvq_output["codebook_indices"],
        }

    @torch.inference_mode()
    def encode(self, audio: torch.Tensor, sample_rate: int):
        need_squeeze = audio.dim() < 3
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)

        if sample_rate != self.target_sample_rate:
            audio = torchaudio.functional.resample(
                audio, sample_rate, self.target_sample_rate
            )

        embeddings = self.encoder(audio)["embeddings"]
        indices = self.rvq(embeddings)["codebook_indices"]
        if need_squeeze:
            indices = indices.squeeze(0)
        return indices

    @torch.inference_mode()
    def decode(self, encoded_audio: torch.Tensor, sample_rate: int):
        need_squeeze = encoded_audio.dim() == 1
        if encoded_audio.dim() == 1:
            encoded_audio = encoded_audio.unsqueeze(0)

        quantized_embeddings = self.rvq.decode(encoded_audio)
        decoded_audio = self.decoder(quantized_embeddings)["reconstruction"]

        if sample_rate != self.target_sample_rate:
            decoded_audio = torchaudio.functional.resample(
                decoded_audio, self.target_sample_rate, sample_rate
            )
        if need_squeeze:
            decoded_audio = decoded_audio.squeeze(0)
        return decoded_audio

    @staticmethod
    def from_pretrained(path: str):
        model_dict = torch.load(path, weights_only=True)
        config = model_dict["config"]

        model = SoundStream(
            Encoder(**config["encoder"]),
            RVQ(**config["rvq"]),
            Decoder(**config["decoder"]),
            target_sample_rate=config["target_sample_rate"],
        )
        model.load_state_dict(model_dict["state_dict"])
        return model
