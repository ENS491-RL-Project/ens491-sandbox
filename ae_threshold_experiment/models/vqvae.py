import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=64, embedding_dim=32, commitment_cost=0.25):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(
            -1 / num_embeddings,
            1 / num_embeddings
        )

    def forward(self, z):
        # z: (batch, embedding_dim)
        distances = (
            torch.sum(z ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z, self.embedding.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(encoding_indices)

        codebook_loss = F.mse_loss(z_q, z.detach())
        commitment_loss = F.mse_loss(z, z_q.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        return z_q, vq_loss, encoding_indices


class VQVAE(nn.Module):
    def __init__(
        self,
        embedding_dim=32,
        num_embeddings=64,
        commitment_cost=0.25,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )

        self.vector_quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
        )

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 7 * 7 * 3),
        )

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, encoding_indices = self.vector_quantizer(z)

        reconstruction = self.decoder(z_q)
        reconstruction = reconstruction.view(-1, 3, 7, 7)

        return reconstruction, vq_loss, encoding_indices