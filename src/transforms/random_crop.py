import random

import torch


class RandomCrop:
    def __init__(self, target_length: int):
        super().__init__()
        self.target_length = target_length

    def __call__(self, x: torch.Tensor):
        length = x.shape[-1]
        if length < self.target_length:
            return torch.nn.functional.pad(
                x, (0, self.target_length - length), mode="replicate"
            )
        st = random.randint(0, length - self.target_length)
        return x[..., st : st + self.target_length]
