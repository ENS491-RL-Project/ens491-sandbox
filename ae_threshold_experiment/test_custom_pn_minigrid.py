import gymnasium as gym
import minigrid
import torch

from models.custom_progressive_network import CustomProgressiveNetwork


def obs_to_flat_tensor(obs):
    image = obs["image"].astype("float32") / 8.0
    flat = image.reshape(1, -1)
    return torch.tensor(flat)


def main():
    env = gym.make("MiniGrid-Empty-8x8-v0")

    obs, info = env.reset(seed=42)

    input_dim = 7 * 7 * 3
    hidden_dim = 64
    output_dim = env.action_space.n

    model = CustomProgressiveNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )

    model.add_column()

    x = obs_to_flat_tensor(obs)
    action_logits = model(x, column_index=0)

    action = torch.argmax(action_logits, dim=1).item()

    print("Custom PN MiniGrid Compatibility Test")
    print("------------------------------------")
    print(f"Observation tensor shape: {x.shape}")
    print(f"Action logits shape: {action_logits.shape}")
    print(f"Chosen action: {action}")
    print(f"Env action space size: {env.action_space.n}")

    next_obs, reward, terminated, truncated, info = env.step(action)

    print("\nEnvironment step")
    print("----------------")
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")

    env.close()

    if action_logits.shape == (1, env.action_space.n):
        print("\nTEST PASSED")
    else:
        print("\nTEST FAILED")


if __name__ == "__main__":
    main()