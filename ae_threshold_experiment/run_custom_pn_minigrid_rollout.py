import argparse

import gymnasium as gym
import minigrid
import torch

from models.custom_progressive_network import CustomProgressiveNetwork


def obs_to_flat_tensor(obs):
    image = obs["image"].astype("float32") / 8.0
    flat = image.reshape(1, -1)
    return torch.tensor(flat)


def run_episode(env, model, column_index, max_steps):
    obs, info = env.reset()

    total_reward = 0.0

    for step in range(max_steps):
        x = obs_to_flat_tensor(obs)

        with torch.no_grad():
            action_logits = model(x, column_index=column_index)
            action = torch.argmax(action_logits, dim=1).item()

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if terminated or truncated:
            return total_reward, step + 1, terminated, truncated

    return total_reward, max_steps, False, False


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=100)

    args = parser.parse_args()

    env = gym.make(args.env)

    input_dim = 7 * 7 * 3
    hidden_dim = 64
    output_dim = env.action_space.n

    model = CustomProgressiveNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )

    model.add_column()

    print("\nCustom PN MiniGrid Rollout")
    print("--------------------------")
    print(f"Environment: {args.env}")
    print(f"Episodes: {args.episodes}")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"Action space size: {output_dim}")

    rewards = []

    for ep in range(args.episodes):
        total_reward, steps, terminated, truncated = run_episode(
            env=env,
            model=model,
            column_index=0,
            max_steps=args.max_steps,
        )

        rewards.append(total_reward)

        print(
            f"Episode {ep + 1}: "
            f"reward={total_reward:.3f}, "
            f"steps={steps}, "
            f"terminated={terminated}, "
            f"truncated={truncated}"
        )

    env.close()

    avg_reward = sum(rewards) / len(rewards)

    print("\nSummary")
    print("-------")
    print(f"Average reward: {avg_reward:.3f}")


if __name__ == "__main__":
    main()