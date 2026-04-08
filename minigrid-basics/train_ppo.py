import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from minigrid.wrappers import FlatObsWrapper

# Environment oluştur
env = make_vec_env(
    lambda: FlatObsWrapper(gym.make("MiniGrid-Empty-8x8-v0")),
    n_envs=4
)

# PPO agent - FlatObs için MlpPolicy
model = PPO("MlpPolicy", env, verbose=1, device="cuda")

# Train et
model.learn(total_timesteps=100_000)

# Kaydet
model.save("ppo_empty")
print("Training tamamlandı!")