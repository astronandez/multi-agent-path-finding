# test.py
import numpy as np
import json
import gymnasium as gym
import highway_env
from stable_baselines3 import PPO, DQN

def make_env():
    """
    Create and configure the environment.
    
    Returns:
        gym.Env: Configured environment
    """
    env = gym.make('intersection-v0', render_mode='rgb_array')
    
    # Load configuration from JSON file
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    env.configure(config)
    return env

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

def test_all_models(n_episodes=100):
    """
    Test all trained models and report their performance.
    
    Args:
        n_episodes (int): Number of test episodes per model
    """
    env = make_env()

    algorithms = ["PPO", "DQN"]
    
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