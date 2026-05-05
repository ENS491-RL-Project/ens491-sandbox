import os
import argparse
import numpy as np
import gymnasium as gym
import minigrid


def make_env(env_id: str):
    return gym.make(env_id, render_mode=None)


def obs_to_array(obs):
    """
    MiniGrid observation genelde dict gelir:
    {
      "image": 7x7x3 array,
      "direction": int,
      "mission": str
    }
    Biz AE için sadece image kısmını kullanıyoruz.
    """
    if isinstance(obs, dict):
        return obs["image"]
    return obs


def collect_observations(env_id: str, num_steps: int, seed: int):
    env = make_env(env_id)

    obs, info = env.reset(seed=seed)
    observations = []

    for step in range(num_steps):
        obs_arr = obs_to_array(obs)
        observations.append(obs_arr)

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()

    observations = np.array(observations, dtype=np.float32)
    return observations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="ae_threshold_experiment/data/empty_obs.npy")
    args = parser.parse_args()

    observations = collect_observations(
        env_id=args.env,
        num_steps=args.steps,
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, observations)

    print(f"Saved observations to: {args.out}")
    print(f"Shape: {observations.shape}")
    print(f"Dtype: {observations.dtype}")
    print(f"Min value: {observations.min()}, max value: {observations.max()}")


if __name__ == "__main__":
    main()