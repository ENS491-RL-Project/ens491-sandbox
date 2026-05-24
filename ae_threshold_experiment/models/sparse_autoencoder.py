import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 7 * 7 * 3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        latent = self.encoder(x)

        reconstruction = self.decoder(latent)
        reconstruction = reconstruction.view(-1, 3, 7, 7)

        return reconstruction, latent

    def reconstruction_loss(self, x, reconstruction):
        return F.mse_loss(reconstruction, x)

    def sparsity_loss(self, latent):
        # L1 sparsity penalty
        return torch.mean(torch.abs(latent))

    def total_loss(
        self,
        x,
        reconstruction,
        latent,
        sparsity_weight=1e-3,
    ):
        recon_loss = self.reconstruction_loss(x, reconstruction)

        sparse_loss = self.sparsity_loss(latent)

        total = recon_loss + sparsity_weight * sparse_loss

        return total, recon_loss, sparse_loss