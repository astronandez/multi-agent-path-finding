# train.py
import os
import json
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import highway_env

def make_env():
    """
    Create and wrap the intersection environment.
    
    Returns:
        gym.Env: Wrapped environment
    """
    env = gym.make('intersection-v0', render_mode='rgb_array')
    
    # Load configuration from JSON file
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    env.configure(config)
    env = Monitor(env)  # Add episode monitoring
    return env

def train_agent(algo="PPO", total_timesteps=2e4):
    """
    Train an RL agent using the specified algorithm.
    
    Args:
        algo (str): Algorithm to use ('PPO', 'DQN')
        total_timesteps (int): Total training timesteps
        
    Returns:
        stable_baselines3.BaseAlgorithm: Trained model
    """
    # Create directories for logs and models
    log_dir = f"./intersection_training/{algo}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create environment
    env = make_env()
    
    # Initialize the appropriate algorithm
    if algo == "PPO":
        model = PPO(
            "MlpPolicy", 
            env,
            # PPO specific parameters
            learning_rate=3e-4,
            n_steps=2048,        # Steps per update
            batch_size=64,       # Minibatch size
            n_epochs=10,         # Number of epochs
            gamma=0.99,          # Discount factor
            gae_lambda=0.95,     # GAE parameter
            clip_range=0.2,      # Clipping parameter
            ent_coef=0.01,       # Entropy coefficient
            # Neural network architecture
            policy_kwargs=dict(
                net_arch=[dict(pi=[256, 256], vf=[256, 256])]
            ),
            tensorboard_log=log_dir,
            verbose=1
        )
    elif algo == "DQN":
        model = DQN(
            "MlpPolicy",
            env,
            # DQN specific parameters
            learning_rate=1e-4,
            buffer_size=50000,   # Replay buffer size
            learning_starts=1000,
            batch_size=64,
            gamma=0.99,
            target_update_interval=250,
            train_freq=4,
            gradient_steps=1,
            exploration_fraction=0.12,
            exploration_final_eps=0.02,
            # Neural network architecture
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=log_dir,
            verbose=1
        )

    # Setup evaluation callback
    eval_env = make_env()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{log_dir}/best_model",
        log_path=f"{log_dir}/eval_results",
        eval_freq=1000,
        deterministic=True,
        render=False
    )
    
    # Train the model
    model.learn(
        total_timesteps=int(total_timesteps),
        callback=eval_callback
    )
    
    # Save the final model
    model.save(f"{log_dir}/final_model")
    
    return model

if __name__ == "__main__":
    # Train models with different algorithms
    algorithms = ["DQN","PPO"]

    for algo in algorithms:
        print(f"\nTraining {algo}...")
        model = train_agent(algo=algo, total_timesteps=2e4)
        print(f"{algo} training completed!")