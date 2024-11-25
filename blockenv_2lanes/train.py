import highway_env
import gymnasium
from highway_env.vehicle.objects import Obstacle
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import numpy as np
import json
from stable_baselines3 import PPO, DQN
import os

env_config = {
    "collision_position_range": [(-5, 5), (-5, 1)],  # Customize the randomization range
    # "action": {
    #     "type": "ContinuousAction",
    #     "acceleration_range": (-3, 3.0),
    #     "steering_range": (-np.pi/6, np.pi/6),
    #     "longitudinal": True,
    #     "lateral": True,
    #     "dynamical": False,
    #     }
    }

class CustomIntersectionEnv(gymnasium.Wrapper):
    def __init__(self, env_config=None):
        env = gymnasium.make('my_env-v0', render_mode='rgb_array')
        # Load configuration from JSON file
        with open('config.json', 'r') as f:
            config = json.load(f)
        env.configure(config)
        super().__init__(env)
        self.env_config = env_config or {
            "collision_position_range": [(-5, -5), (5, 5)],  # Randomization range (x, y)
        }

    def reset(self, *, seed=None, options=None):
        """Reset the environment with support for seed and options parameters.
        
        Args:
            seed (int, optional): The seed for random number generation
            options (dict, optional): Additional options for reset
            
        Returns:
            tuple: (observation, info dictionary)
        """
        # If seed is provided, set it
        if seed is not None:
            np.random.seed(seed)
            
        # Reset the underlying environment
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Clear existing obstacles
        self.env.road.objects.clear()
        
        # Add collision (static vehicles)
        collision_pos = self._randomize_collision_position()
        for offset in [(-1, 0), (0, 0)]:  # Add two static vehicles around the collision
            position = [collision_pos[0] + offset[0], collision_pos[1] + offset[1]]
            obstacle = Obstacle(self.env.road, position)
            self.env.road.objects.append(obstacle)

        return obs, info

    def _randomize_collision_position(self):
        """Randomize collision positions within a specified range."""
        x_range, y_range = self.env_config["collision_position_range"]
        x = np.random.uniform(x_range[0], x_range[1])
        y = np.random.uniform(y_range[0], y_range[1])
        return x, y

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
    
    # Create environment with monitor wrapper
    env = Monitor(CustomIntersectionEnv(env_config))
    
    # Initialize the appropriate algorithm
    if algo == "PPO":
        model = PPO(
            "MlpPolicy", 
            env,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=10,
            gamma=0.8,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
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
            learning_rate=1e-4,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=128,
            gamma=0.99,
            target_update_interval=250,
            train_freq=4,
            gradient_steps=1,
            exploration_fraction=0.12,
            exploration_final_eps=0.02,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=log_dir,
            verbose=1
        )

    # Setup evaluation callback
    eval_env = Monitor(CustomIntersectionEnv(env_config))
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
    algorithms = ["DQN", "PPO"]
    
    for algo in algorithms:
        print(f"\nTraining {algo}...")
        model = train_agent(algo=algo, total_timesteps=5e4)
        print(f"{algo} training completed!")