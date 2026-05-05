import gymnasium as gym
from stable_baselines3 import PPO
from minigrid.wrappers import FlatObsWrapper

# 1. Go BACK to the original EMPTY environment
env = gym.make("MiniGrid-Empty-8x8", render_mode="human")
env = FlatObsWrapper(env)

# 2. Load the brain that just learned FourRooms
model = PPO.load("ppo_finetuned")

# 3. Test it!
print("Testing the fine-tuned model on the original Empty environment...")
obs, _ = env.reset()
steps = 0
total_reward = 0

while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    steps += 1
    total_reward += reward
    
    if terminated or truncated:
        print(f"Episode finished! Steps: {steps}, Reward: {total_reward}")
        break