import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.autoencoder import MiniGridAutoencoder


def load_observations(path):
    obs = np.load(path).astype(np.float32)

    # MiniGrid değerleri 0-8 arası geldiği için normalize ediyoruz
    obs = obs / 8.0

    # numpy shape: (N, 7, 7, 3)
    # torch shape: (N, 3, 7, 7)
    obs = torch.tensor(obs).permute(0, 3, 1, 2)

    return obs


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    observations = load_observations(args.data)
    dataset = TensorDataset(observations)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = MiniGridAutoencoder(latent_dim=args.latent_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = 0.0

        for (batch,) in loader:
            batch = batch.to(device)

            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{args.epochs} | Loss: {avg_loss:.6f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)

    print(f"Saved trained AE to: {args.out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="ae_threshold_experiment/data/empty_obs.npy")
    parser.add_argument("--out", type=str, default="ae_threshold_experiment/results/ae_empty.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()