import gymnasium as gym
from stable_baselines3 import PPO
from minigrid.wrappers import FlatObsWrapper

# Modeli yükle
model = PPO.load("ppo_empty")

# Environment oluştur - bu sefer render_mode="human"
env = FlatObsWrapper(gym.make("MiniGrid-Empty-8x8-v0", render_mode="human"))

# Test et
for episode in range(5):
    obs, info = env.reset()
    total_reward = 0
    steps = 0
    
    while True:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            print(f"Episode {episode+1}: reward={total_reward:.3f}, steps={steps}")
            break

env.close()