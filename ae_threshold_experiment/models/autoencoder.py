import torch
import torch.nn as nn


class MiniGridAutoencoder(nn.Module):
    """
    Baseline Undercomplete Autoencoder for MiniGrid observations.

    Input:
        (batch, 3, 7, 7)

    Output:
        (batch, 3, 7, 7)
    """

    def __init__(self, latent_dim=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 7 * 7 * 3),
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        reconstructed = reconstructed.view(-1, 3, 7, 7)
        return reconstructed