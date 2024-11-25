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



def evaluate_agent(model, env, n_episodes=20):
    """
    Evaluate a trained agent over multiple episodes.
    
    Args:
        model: Trained RL model
        env: Environment to evaluate in
        n_episodes: Number of evaluation episodes
        
    Returns:
        dict: Evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    collision_count = 0
    success_count = 0
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # Get action from model
            action, _states = model.predict(obs, deterministic=True)
            
            # Execute action in environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
            
            # Track metrics
            if info.get("crashed", False):
                collision_count += 1
            
            # Render the environment
            env.render()
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
    
    # Calculate evaluation metrics
    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "collision_rate": collision_count / n_episodes,
    }

def test_all_models(n_episodes=20):
    """
    Test all trained models and report their performance.
    
    Args:
        n_episodes (int): Number of test episodes per model
    """
    env = Monitor(CustomIntersectionEnv(env_config))

    algorithms = ["PPO","DQN"]
    
    for algo in algorithms:
        print(f"\nEvaluating {algo}...")
        
        # Load the best model for current algorithm
        if algo == "PPO":
            model = PPO.load(f"./intersection_training/{algo}/best_model/best_model")
        elif algo == "DQN":
            model = DQN.load(f"./intersection_training/{algo}/best_model/best_model")
            
        # Run evaluation
        results = evaluate_agent(model, env, n_episodes)
        
        # Print results
        print(f"Results for {algo}:")
        print(f"Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Mean episode length: {results['mean_length']:.2f}")
        print(f"Collision rate: {results['collision_rate']*100:.2f}%")

if __name__ == "__main__":
    test_all_models()