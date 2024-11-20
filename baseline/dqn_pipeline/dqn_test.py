import gymnasium
import json
import highway_env
from stable_baselines3 import DQN

# Load environment configuration from a JSON file
with open("config.json", "r") as f:
    config = json.load(f)

# Create the environment using the specified configuration
env = gymnasium.make("intersection-v0", render_mode='rgb_array', config=config)

# Specify the path for saving/loading the model
model_path = "intersection_dqn/model"

# Try to load a pre-trained model, if available
try:
    model = DQN.load(model_path, env=env)
    print("Model loaded successfully.")
except FileNotFoundError:
    # If no pre-trained model is found, create a new one with specific hyperparameters
    print("No pre-existing model found. Creating a new model.")


# Test and visualize the trained model
while True:  # Infinite loop for continuous testing
    done = truncated = False  # Reset done and truncated flags
    obs, info = env.reset()  # Reset the environment to the initial state
    
    while not (done or truncated):  # Continue until the episode ends
        # Predict the action using the trained model (deterministic for evaluation)
        action, _states = model.predict(obs, deterministic=True)
        
        # Take the action in the environment and observe the results
        obs, reward, done, truncated, info = env.step(action)
        
        # Render the environment to visualize the agent's behavior
        env.render()
