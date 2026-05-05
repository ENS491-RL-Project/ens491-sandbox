import gymnasium as gym
from stable_baselines3 import PPO
from minigrid.wrappers import FlatObsWrapper

env = gym.make("MiniGrid-FourRooms-v0") 
env = FlatObsWrapper(env)

model = PPO.load("ppo_empty", env=env)

# 3. Train 
print("Starting fine-tuning on FourRooms...")
model.learn(total_timesteps=100000)

model.save("ppo_finetuned")
print("Finished and saved as ppo_finetuned")