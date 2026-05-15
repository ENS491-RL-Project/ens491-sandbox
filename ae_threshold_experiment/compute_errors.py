import os
import argparse
import numpy as np
import torch

from models.autoencoder import MiniGridAutoencoder


def load_observations(path):
    obs = np.load(path).astype(np.float32)

    # collect_obs.py çıktısı 0-8 aralığında.
    # train_ae.py ile aynı normalize işlemi yapılmalı.
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

            reconstructed = model(batch)

            # Her observation için ayrı MSE error
            batch_errors = torch.mean((batch - reconstructed) ** 2, dim=(1, 2, 3))

            errors.append(batch_errors.cpu().numpy())

    return np.concatenate(errors)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="ae_threshold_experiment/results/ae_empty.pt")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)

    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MiniGridAutoencoder(latent_dim=args.latent_dim).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))

    observations = load_observations(args.data)
    errors = compute_reconstruction_errors(
        model=model,
        observations=observations,
        batch_size=args.batch_size,
        device=device,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, errors)

    print(f"Loaded model: {args.model}")
    print(f"Loaded data: {args.data}")
    print(f"Saved errors to: {args.out}")
    print(f"Number of errors: {len(errors)}")
    print(f"Mean error: {errors.mean():.6f}")
    print(f"Std error: {errors.std():.6f}")
    print(f"Min error: {errors.min():.6f}")
    print(f"Max error: {errors.max():.6f}")


if __name__ == "__main__":
    main()