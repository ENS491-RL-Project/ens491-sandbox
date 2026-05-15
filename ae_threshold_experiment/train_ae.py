import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.autoencoder import MiniGridAutoencoder
from models.sparse_autoencoder import SparseAutoencoder
from models.vqvae import VQVAE


def load_observations(path):
    obs = np.load(path).astype(np.float32)
    obs = obs / 8.0

    # Shape: (N, 7, 7, 3) -> (N, 3, 7, 7)
    obs = torch.tensor(obs).permute(0, 3, 1, 2)

    return obs


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    observations = load_observations(args.data)
    dataset = TensorDataset(observations)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    if args.model == "undercomplete":
        model = MiniGridAutoencoder(latent_dim=args.latent_dim).to(device)
        model_type = "undercomplete"

    elif args.model == "sparse":
        model = SparseAutoencoder(latent_dim=args.latent_dim).to(device)
        model_type = "sparse"

    elif args.model == "vqvae":
        model = VQVAE(
            embedding_dim=args.latent_dim,
            num_embeddings=args.num_embeddings,
            commitment_cost=args.commitment_cost,
        ).to(device)
        model_type = "vqvae"

    else:
        raise ValueError(f"Unknown model type: {args.model}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = 0.0
        total_recon_loss = 0.0
        total_aux_loss = 0.0

        for (batch,) in loader:
            batch = batch.to(device)

            if model_type == "sparse":
                reconstructed, latent = model(batch)
                recon_loss = criterion(reconstructed, batch)
                aux_loss = torch.mean(torch.abs(latent))
                loss = recon_loss + args.sparsity_weight * aux_loss

            elif model_type == "vqvae":
                reconstructed, vq_loss, _ = model(batch)
                recon_loss = criterion(reconstructed, batch)
                aux_loss = vq_loss
                loss = recon_loss + aux_loss

            else:
                reconstructed = model(batch)
                recon_loss = criterion(reconstructed, batch)
                aux_loss = torch.tensor(0.0, device=device)
                loss = recon_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.size(0)
            total_recon_loss += recon_loss.item() * batch.size(0)
            total_aux_loss += aux_loss.item() * batch.size(0)

        avg_loss = total_loss / len(dataset)
        avg_recon_loss = total_recon_loss / len(dataset)
        avg_aux_loss = total_aux_loss / len(dataset)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Loss: {avg_loss:.6f} | "
            f"Recon: {avg_recon_loss:.6f} | "
            f"Aux: {avg_aux_loss:.6f}"
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)

    print(f"Saved trained AE to: {args.out}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="undercomplete",
        choices=["undercomplete", "sparse", "vqvae"],
    )

    parser.add_argument(
        "--data",
        type=str,
        default="ae_threshold_experiment/data/empty_obs.npy",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="ae_threshold_experiment/results/ae_empty.pt",
    )

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)

    # Sparse AE specific
    parser.add_argument("--sparsity-weight", type=float, default=1e-3)

    # VQ-VAE specific
    parser.add_argument("--num-embeddings", type=int, default=64)
    parser.add_argument("--commitment-cost", type=float, default=0.25)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()