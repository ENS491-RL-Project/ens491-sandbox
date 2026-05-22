import argparse
import gymnasium as gym
import minigrid
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from models.custom_progressive_network import CustomProgressiveNetwork


def obs_to_tensor(obs):
    image = obs["image"].astype("float32") / 8.0
    flat = image.reshape(1, -1)
    return torch.tensor(flat)


def collect_rollout(env, model, column_index, steps_per_rollout, gamma):
    obs, info = env.reset()

    observations = []
    actions = []
    log_probs = []
    rewards = []

    total_reward = 0.0

    for _ in range(steps_per_rollout):
        x = obs_to_tensor(obs)

        logits = model(x, column_index=column_index)
        dist = Categorical(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_obs, reward, terminated, truncated, info = env.step(action.item())

        observations.append(x.squeeze(0))
        actions.append(action.squeeze(0))
        log_probs.append(log_prob.squeeze(0))
        rewards.append(reward)

        total_reward += reward
        obs = next_obs

        if terminated or truncated:
            obs, info = env.reset()

    returns = []
    G = 0.0

    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)

    return (
        torch.stack(observations),
        torch.stack(actions),
        torch.stack(log_probs).detach(),
        torch.tensor(returns, dtype=torch.float32),
        total_reward,
    )


def train_ppo(args):
    env = gym.make(args.env)

    input_dim = 7 * 7 * 3
    hidden_dim = args.hidden_dim
    output_dim = env.action_space.n

    model = CustomProgressiveNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )

    model.add_column()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("\nCustom PN PPO MiniGrid Training")
    print("--------------------------------")
    print(f"Environment: {args.env}")
    print(f"Updates: {args.updates}")
    print(f"Steps per rollout: {args.steps_per_rollout}")

    for update in range(args.updates):
        obs_batch, action_batch, old_log_probs, returns, rollout_reward = collect_rollout(
            env=env,
            model=model,
            column_index=0,
            steps_per_rollout=args.steps_per_rollout,
            gamma=args.gamma,
        )

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for _ in range(args.ppo_epochs):
            logits = model(obs_batch, column_index=0)
            dist = Categorical(logits=logits)

            new_log_probs = dist.log_prob(action_batch)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)

            unclipped = ratio * returns
            clipped = torch.clamp(
                ratio,
                1 - args.clip_eps,
                1 + args.clip_eps,
            ) * returns

            policy_loss = -torch.min(unclipped, clipped).mean()
            loss = policy_loss - args.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (update + 1) % 10 == 0:
            print(
                f"Update {update + 1}/{args.updates} | "
                f"Rollout reward: {rollout_reward:.3f} | "
                f"Loss: {loss.item():.4f} | "
                f"Entropy: {entropy.item():.4f}"
            )

    env.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--steps-per-rollout", type=int, default=512)
    parser.add_argument("--ppo-epochs", type=int, default=4)

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)

    args = parser.parse_args()
    train_ppo(args)


if __name__ == "__main__":
    main()