import gymnasium
import json
import highway_env
from stable_baselines3 import PPO
from attention_policy import AttentionFeatureExtractor

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

    policy_kwargs = dict(
        features_extractor_class=AttentionFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    #policy_kwargs=dict(net_arch=[256, 256]) #for when you are running the mlp dqn without attention

    model = PPO('MlpPolicy', env,
                policy_kwargs=policy_kwargs,
                learning_rate=5e-4,
                n_steps=256,
                batch_size=64,
                n_epochs=10,
                gamma=0.8,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                verbose=1,
                tensorboard_log="intersection_ppo/experiment1/")
    
# train model
model.learn(int(1e4))
model.save(model_path)
