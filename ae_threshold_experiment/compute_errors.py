import os
import argparse
import numpy as np
import torch

from models.autoencoder import MiniGridAutoencoder
from models.sparse_autoencoder import SparseAutoencoder
from models.vqvae import VQVAE


def load_observations(path):
    obs = np.load(path).astype(np.float32)
    obs = obs / 8.0

    # numpy: (N, 7, 7, 3)
    # torch: (N, 3, 7, 7)
    obs = torch.tensor(obs).permute(0, 3, 1, 2)

    return obs


def compute_reconstruction_errors(model, observations, batch_size, device):
    model.eval()
    errors = []

    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            batch = observations[start:start + batch_size].to(device)

            output = model(batch)

            # Undercomplete AE: reconstruction
            # Sparse AE: (reconstruction, latent)
            # VQ-VAE: (reconstruction, vq_loss, encoding_indices)
            if isinstance(output, tuple):
                reconstructed = output[0]
            else:
                reconstructed = output

            batch_errors = torch.mean((batch - reconstructed) ** 2, dim=(1, 2, 3))
            errors.append(batch_errors.cpu().numpy())

    return np.concatenate(errors)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="undercomplete",
        choices=["undercomplete", "sparse", "vqvae"],
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="ae_threshold_experiment/results/ae_empty.pt",
    )

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)

    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)

    # VQ-VAE specific
    parser.add_argument("--num-embeddings", type=int, default=64)
    parser.add_argument("--commitment-cost", type=float, default=0.25)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "undercomplete":
        model = MiniGridAutoencoder(latent_dim=args.latent_dim).to(device)

    elif args.model == "sparse":
        model = SparseAutoencoder(latent_dim=args.latent_dim).to(device)

    elif args.model == "vqvae":
        model = VQVAE(
            embedding_dim=args.latent_dim,
            num_embeddings=args.num_embeddings,
            commitment_cost=args.commitment_cost,
        ).to(device)

    else:
        raise ValueError(f"Unknown model type: {args.model}")

    model.load_state_dict(torch.load(args.model_path, map_location=device))

    observations = load_observations(args.data)
    errors = compute_reconstruction_errors(
        model=model,
        observations=observations,
        batch_size=args.batch_size,
        device=device,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, errors)

    print(f"Model type: {args.model}")
    print(f"Loaded model: {args.model_path}")
    print(f"Loaded data: {args.data}")
    print(f"Saved errors to: {args.out}")
    print(f"Number of errors: {len(errors)}")
    print(f"Mean error: {errors.mean():.6f}")
    print(f"Std error: {errors.std():.6f}")
    print(f"Min error: {errors.min():.6f}")
    print(f"Max error: {errors.max():.6f}")


if __name__ == "__main__":
    main()