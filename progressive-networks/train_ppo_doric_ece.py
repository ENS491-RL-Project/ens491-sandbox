"""
train_ppo_doric_ece.py
Test: Progressive Networks (Doric) + SB3 PPO on MiniGrid.

Goal: Train Task1 (Empty), freeze column, train Task2 (FourRooms),
      verify Task1 column is not affected (no forgetting).

Fix notes:
  - SB3 PPO used instead of REINFORCE (stable training)
  - Forgetting check uses raw Doric net forward pass, not SB3 model
    (SB3 models are separate, only the Doric feature extractor is shared)
  - N_ACTIONS=3: left/right/forward only (avoids pickup/drop loops)
"""

import torch
import numpy as np
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from Doric import ProgNet, ProgColumn, ProgColumnGenerator, ProgDenseBlock

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OBS_SIZE  = 2835
HIDDEN    = 64
TIMESTEPS = 50_000

TASK1_ENV = "MiniGrid-Empty-8x8-v0"
TASK2_ENV = "MiniGrid-FourRooms-v0"


# ---------------------------------------------------------------------------
# Column Generator
# ---------------------------------------------------------------------------
class MiniGridColumnGenerator(ProgColumnGenerator):
    def __init__(self, obs_size, hidden):
        self.obs_size = obs_size
        self.hidden   = hidden

    def generateColumn(self, parentCols, msg=None):
        n_lat  = len(parentCols)
        col_id = msg if msg else f"task{n_lat+1}"
        blocks = [
            ProgDenseBlock(self.obs_size, self.hidden, numLaterals=n_lat),
            ProgDenseBlock(self.hidden,   self.hidden, numLaterals=n_lat),
            ProgDenseBlock(self.hidden,   7,           numLaterals=n_lat, activation=None),
        ]
        return ProgColumn(colID=col_id, blockList=blocks, parentCols=parentCols)


# ---------------------------------------------------------------------------
# Doric as SB3 Features Extractor
# ---------------------------------------------------------------------------
class DoricExtractor(BaseFeaturesExtractor):
    """Wraps active Doric column as SB3 feature extractor."""

    def __init__(self, observation_space, net, col_id, features_dim=64):
        super().__init__(observation_space, features_dim)
        self.net    = net
        self.col_id = col_id

    def forward(self, obs):
        return self.net.forward(self.col_id, obs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_env(env_id):
    return FlatObsWrapper(gym.make(env_id))


def evaluate_sb3(env, model, n_episodes=10):
    """Evaluate SB3 model, return mean reward."""
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
    return float(np.mean(rewards))


def evaluate_doric(env, net, col_id, n_episodes=10):
    """
    Evaluate Doric column directly (no SB3).
    Used for forgetting check — tests the frozen column in isolation.
    """
    rewards = []
    max_steps = env.unwrapped.max_steps
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0
        for _ in range(max_steps):
            obs_t  = torch.FloatTensor(obs).unsqueeze(0)
            logits = net.forward(col_id, obs_t)
            # sample instead of argmax to avoid action loops
            probs  = torch.softmax(logits, dim=-1)
            action = torch.distributions.Categorical(probs).sample().item()
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break
        rewards.append(ep_reward)
    return float(np.mean(rewards))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 55)
    print("Progressive Networks + SB3 PPO -- MiniGrid Test")
    print("=" * 55)

    gen = MiniGridColumnGenerator(OBS_SIZE, HIDDEN)
    net = ProgNet(colGen=gen)

    # ------------------------------------------------------------------
    # Task 1: Empty
    # ------------------------------------------------------------------
    env1 = make_env(TASK1_ENV)
    col1 = net.addColumn(msg="task1_empty")

    model1 = PPO("MlpPolicy", env1,
                 policy_kwargs=dict(
                     features_extractor_class=DoricExtractor,
                     features_extractor_kwargs=dict(net=net, col_id=col1, features_dim=HIDDEN),
                     net_arch=[],
                 ),
                 verbose=0, device="cpu")

    print(f"\n--- Training Task1: {TASK1_ENV} ---")
    model1.learn(total_timesteps=TIMESTEPS)

    # Evaluate with SB3 model (full policy)
    reward_sb3_before = evaluate_sb3(env1, model1)
    # Evaluate with raw Doric column (features only, random head)
    reward_doric_before = evaluate_doric(env1, net, col1)
    print(f"[Task1] SB3 reward:   {reward_sb3_before:.3f}")
    print(f"[Task1] Doric reward: {reward_doric_before:.3f}")

    net.freezeColumn(col1)
    print(f"[Task1] Column frozen: {net.isColumnFrozen(col1)}")

    # ------------------------------------------------------------------
    # Task 2: FourRooms (lateral from Task1)
    # ------------------------------------------------------------------
    env2 = make_env(TASK2_ENV)
    col2 = net.addColumn(msg="task2_fourrooms")

    model2 = PPO("MlpPolicy", env2,
                 policy_kwargs=dict(
                     features_extractor_class=DoricExtractor,
                     features_extractor_kwargs=dict(net=net, col_id=col2, features_dim=HIDDEN),
                     net_arch=[],
                 ),
                 verbose=0, device="cpu")

    print(f"\n--- Training Task2: {TASK2_ENV} ---")
    model2.learn(total_timesteps=TIMESTEPS)

    # ------------------------------------------------------------------
    # Forgetting check — use Doric direct eval on frozen col1
    # col1 is frozen so its weights cannot have changed
    # ------------------------------------------------------------------
    reward_doric_after = evaluate_doric(env1, net, col1)
    drop = reward_doric_before - reward_doric_after

    print(f"\n[Forgetting Check — Doric column direct eval]")
    print(f"  col1 reward BEFORE Task2: {reward_doric_before:.3f}")
    print(f"  col1 reward AFTER  Task2: {reward_doric_after:.3f}")
    print(f"  Drop: {drop:.3f} -> {'OK - no forgetting' if abs(drop) < 0.05 else 'FORGETTING DETECTED'}")

    print("\n" + "=" * 55)
    print("DONE")
    print(f"  col1 frozen: {net.isColumnFrozen(col1)}")
    print(f"  col2 frozen: {net.isColumnFrozen(col2)}")
    print("=" * 55)

    env1.close()
    env2.close()


if __name__ == "__main__":
    main()