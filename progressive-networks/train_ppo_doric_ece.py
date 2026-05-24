"""
train_ppo_doric_ece.py
Progressive Networks (Doric) + SB3 PPO on MiniGrid.

Goal: Train Task1 (Empty-5x5), freeze column, train Task2 (Empty-6x6),
      verify Task1 performance preserved after Task2 training.

Architecture per column:
    obs -> Dense(obs,64) -> Dense(64,64) -> features(64)
    SB3 PPO adds its own action head on top (64 -> n_actions).

Forgetting check: reload saved model1 after Task2, re-evaluate on Task1.
col1 is frozen so model1 weights are unchanged — reload confirms this.
"""

import numpy as np
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from Doric import ProgNet, ProgColumn, ProgColumnGenerator, ProgDenseBlock

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HIDDEN    = 64
TIMESTEPS = 30_000

TASK1_ENV = "MiniGrid-Empty-5x5-v0"
TASK2_ENV = "MiniGrid-Empty-6x6-v0"

MODEL1_PATH = "progressive-networks/model1_task1"


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
        ]
        return ProgColumn(colID=col_id, blockList=blocks, parentCols=parentCols)


# ---------------------------------------------------------------------------
# Doric as SB3 Features Extractor
# ---------------------------------------------------------------------------
class DoricExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, net, col_id, features_dim=64):
        super().__init__(observation_space, features_dim)
        self.net    = net
        self.col_id = col_id

    def forward(self, obs):
        return self.net.forward(self.col_id, obs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_obs_size(env_id):
    env = FlatObsWrapper(gym.make(env_id))
    size = env.observation_space.shape[0]
    env.close()
    return size

def make_env(env_id):
    return FlatObsWrapper(gym.make(env_id))

def evaluate_sb3(env, model, n_episodes=20):
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        print(f"  eval ep {ep+1}/{n_episodes}: reward={ep_reward:.3f}")
    return float(np.mean(rewards))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 55)
    print("Progressive Networks + SB3 PPO -- MiniGrid Test")
    print("=" * 55)

    # ------------------------------------------------------------------
    # Task 1: Empty-5x5
    # ------------------------------------------------------------------
    obs_size_t1 = get_obs_size(TASK1_ENV)
    print(f"\n[Setup] Task1 obs_size = {obs_size_t1}")

    gen = MiniGridColumnGenerator(obs_size_t1, HIDDEN)
    net = ProgNet(colGen=gen)

    env1 = make_env(TASK1_ENV)
    col1 = net.addColumn(msg="task1_empty5x5")
    print(f"[Setup] Column added: {col1}")

    model1 = PPO(
        "MlpPolicy", env1,
        policy_kwargs=dict(
            features_extractor_class=DoricExtractor,
            features_extractor_kwargs=dict(net=net, col_id=col1, features_dim=HIDDEN),
            net_arch=[],
        ),
        verbose=1, device="auto", seed=0
    )

    print(f"\n--- Training Task1: {TASK1_ENV} ---")
    model1.learn(total_timesteps=TIMESTEPS)

    print(f"\n[Task1] Evaluating before Task2...")
    reward_t1_before = evaluate_sb3(env1, model1)
    print(f"[Task1] Mean reward before Task2: {reward_t1_before:.3f}")

    # Freeze col1 and save model1 — used for forgetting check after Task2
    net.freezeColumn(col1)
    model1.save(MODEL1_PATH)
    print(f"[Task1] Column frozen: {net.isColumnFrozen(col1)}")
    print(f"[Task1] Model saved to {MODEL1_PATH}")

    # ------------------------------------------------------------------
    # Task 2: Empty-6x6
    # ------------------------------------------------------------------
    obs_size_t2 = get_obs_size(TASK2_ENV)
    print(f"\n[Setup] Task2 obs_size = {obs_size_t2}")

    env2 = make_env(TASK2_ENV)
    col2 = net.addColumn(msg="task2_empty6x6")
    print(f"[Setup] Column added: {col2}, lateral connections from: {col1}")

    model2 = PPO(
        "MlpPolicy", env2,
        policy_kwargs=dict(
            features_extractor_class=DoricExtractor,
            features_extractor_kwargs=dict(net=net, col_id=col2, features_dim=HIDDEN),
            net_arch=[],
        ),
        verbose=1, device="auto", seed=0
    )

    print(f"\n--- Training Task2: {TASK2_ENV} ---")
    model2.learn(total_timesteps=TIMESTEPS)

    print(f"\n[Task2] Evaluating...")
    reward_t2 = evaluate_sb3(env2, model2)
    print(f"[Task2] Mean reward: {reward_t2:.3f}")

    # ------------------------------------------------------------------
    # Forgetting check — reload saved model1, re-evaluate on fresh env1
    # model1 was saved after freeze, so weights reflect pre-Task2 state
    # ------------------------------------------------------------------
    env1.close()
    env1 = make_env(TASK1_ENV)
    model1 = PPO.load(MODEL1_PATH, env=env1)
    print(f"[Task1] Model reloaded from {MODEL1_PATH}")

    print(f"\n[Task1] Re-evaluating after Task2...")
    reward_t1_after = evaluate_sb3(env1, model1)
    drop = reward_t1_before - reward_t1_after

    print(f"\n{'='*55}")
    print(f"[Forgetting Check]")
    print(f"  col1 frozen: {net.isColumnFrozen(col1)}")
    print(f"  col2 frozen: {net.isColumnFrozen(col2)}")
    print(f"  Task1 reward BEFORE Task2: {reward_t1_before:.3f}")
    print(f"  Task1 reward AFTER  Task2: {reward_t1_after:.3f}")
    print(f"  Drop: {drop:.3f}")
    print(f"  -> {'OK — no forgetting' if abs(drop) < 0.1 else 'FORGETTING DETECTED'}")
    print(f"  Task2 mean reward: {reward_t2:.3f}")
    print(f"{'='*55}")

    env1.close()
    env2.close()


if __name__ == "__main__":
    main()