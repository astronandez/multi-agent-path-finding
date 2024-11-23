import gymnasium
import json
import highway_env
from stable_baselines3 import DQN
from attention_policy import AttentionFeatureExtractor

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
    # If no pre-trained model is found, create a new one
    print("No pre-existing model found. Creating a new model.")

    policy_kwargs = dict(
        features_extractor_class=AttentionFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    #policy_kwargs=dict(net_arch=[256, 256]) #for when you are running the mlp dqn without attention

    model = DQN(
        'MlpPolicy',  # Use a multi-layer perceptron policy
        env,  # The environment to train on
        policy_kwargs=policy_kwargs,            # Define the neural network architecture
        learning_rate=1e-3,                                 # Learning rate for the optimizer
        buffer_size=50000,                                  # Size of the replay buffer
        learning_starts=1000,                               # Number of steps before training starts
        batch_size=64,                                      # Size of mini-batches sampled from the replay buffer
        tau=1.0,                                            # Soft update coefficient for the target network
        gamma=0.99,                                         # Discount factor for future rewards
        train_freq=(4, "step"),                             # Train the model every 4 steps
        target_update_interval=1000,                        # Frequency of target network updates
        exploration_fraction=0.1,                           # Fraction of training during which exploration rate decays
        exploration_final_eps=0.05,                         # Final value of the exploration rate
        verbose=1,                                          # Print detailed training logs
        tensorboard_log="intersection_dqn/experiment1/"     # Path for TensorBoard logs
    )

# Train the model for 10,000 timesteps
model.learn(int(5e3))

# Save the trained model to the specified path
model.save(model_path)
