import torch
import torch.nn as nn


def find_nearest(v: torch.Tensor, u: torch.Tensor):
    return torch.cdist(v, u).argmin(dim=1)


def kmeans_cluster_centers(vectors: torch.Tensor, n_clusters: int, max_iter: int = 300):
    rand_indices = torch.randperm(vectors.shape[0], device=vectors.device)
    centers = vectors[rand_indices[:n_clusters]]

    for _ in range(max_iter):
        labels = find_nearest(vectors, centers)
        for cl_index in range(n_clusters):
            cl_labels = labels == cl_index
            if cl_labels.any():
                centers[cl_index] = vectors[cl_labels].mean(dim=0)
    return centers


class VQ(nn.Module):
    def __init__(
        self, codebook_size: int, embedding_dim: int, decay: float, min_stat: float
    ):
        super().__init__()

        # codebook = ema_m / ema_N
        self.register_buffer("codebook", torch.randn(codebook_size, embedding_dim))
        self.register_buffer("ema_m", torch.zeros(codebook_size, embedding_dim))
        self.register_buffer("ema_N", torch.zeros(codebook_size))
        self.register_buffer("_is_initialized", torch.tensor(False))
        self.codebook_size = codebook_size
        self.gamma = decay
        self.min_stat = min_stat

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            if not self._is_initialized.item():
                self._is_initialized = torch.tensor(True)
                self.codebook = kmeans_cluster_centers(x, n_clusters=self.codebook_size)
                nearest = find_nearest(x, self.codebook)
                self.ema_N = torch.bincount(nearest, minlength=self.codebook_size)
                self.ema_m = self.codebook * self.ema_N.unsqueeze(1)

            else:
                nearest = find_nearest(x, self.codebook)

                if self.training:
                    counts = nearest.bincount(minlength=self.codebook_size)
                    x_sum = (
                        nn.functional.one_hot(
                            nearest, num_classes=self.codebook_size
                        ).T.float()
                        @ x
                    )

                    self.ema_N = self.ema_N * self.gamma + counts * (1 - self.gamma)
                    self.ema_m = self.ema_m * self.gamma + x_sum * (1 - self.gamma)

                    replace_indices = self.ema_N < self.min_stat
                    if replace_indices.any():
                        rand_indices = torch.randperm(x.shape[0], device=x.device)
                        ema_N_new = self.min_stat / self.gamma
                        self.ema_N[replace_indices] = ema_N_new
                        self.ema_m[replace_indices] = (
                            x[rand_indices[: replace_indices.sum()]] * ema_N_new
                        )

                    self.codebook = self.ema_m / self.ema_N.unsqueeze(1)

        return {
            "quantized_embeddings": x + (self.codebook[nearest] - x).detach(),
            "indices": nearest,
        }

    def get_vectors(self, indices: torch.Tensor):
        return self.codebook[indices]


class RVQ(nn.Module):
    def __init__(
        self,
        n_quantizers: int,
        embedding_dim: int,
        codebook_size: int,
        decay: float,
        min_stat: float,
    ):
        super().__init__()
        self.quantizers = nn.ModuleList(
            [
                VQ(
                    codebook_size=codebook_size,
                    embedding_dim=embedding_dim,
                    decay=decay,
                    min_stat=min_stat,
                )
                for _ in range(n_quantizers)
            ]
        )
        self.embedding_dim = embedding_dim
        self.Nq = n_quantizers

    def forward(self, x: torch.Tensor):
        B, S, D = x.shape
        x = x.reshape(B * S, D)

        quantized_embeddings = torch.zeros_like(x)
        indices = []

        for quantizer in self.quantizers:
            output = quantizer(x)
            quantized_embeddings += output["quantized_embeddings"]
            x -= output["quantized_embeddings"]
            indices.append(output["indices"])

        quantized_embeddings = quantized_embeddings.reshape(B, S, D)
        # Nq x B*S -> B x Nq*S
        codebook_indices = (
            torch.stack(indices)
            .reshape(self.Nq, B, S)
            .swapaxes(0, 1)
            .reshape(B, self.Nq * S)
        )

        return {
            "quantized_embeddings": quantized_embeddings,
            "codebook_indices": codebook_indices,
        }

    def decode(self, indices: torch.Tensor):
        # B x Nq*S -> Nq x B*S
        B = indices.shape[0]
        S = indices.shape[1] // self.Nq
        indices = indices.reshape(B, self.Nq, S).swapaxes(0, 1).reshape(self.Nq, B * S)

        x = 0
        for q_index, quantizer in enumerate(self.quantizers):
            x += quantizer.get_vectors(indices[q_index])
        return x.reshape(B, S, self.embedding_dim)
