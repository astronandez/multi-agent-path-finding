import gymnasium
import json
import highway_env
from stable_baselines3 import PPO

# load environment
with open("config.json", "r") as f:
    config = json.load(f)
env = gymnasium.make("intersection-v0", render_mode='rgb_array', config=config)

# load model
model_path = "intersection_ppo/model"
try:
    model = PPO.load(model_path, env=env)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("No pre-existing model found. Creating a new model.")


# Test and Visualize
while True:
  done = truncated = False
  obs, info = env.reset()
  while not (done or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()