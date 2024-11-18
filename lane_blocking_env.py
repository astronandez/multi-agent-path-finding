import highway_env
import gymnasium
from highway_env.vehicle.objects import Obstacle
import numpy as np

class CustomIntersectionEnv(gymnasium.Wrapper):
    def __init__(self, env_config=None):
        env = gymnasium.make('intersection-v0', render_mode='rgb_array')
        super().__init__(env)
        self.env_config = env_config or {
            "collision_position_range": [(-5, -5), (5, 5)],  # Randomization range (x, y)
        }

    def reset(self):
        obs = self.env.reset()
        
        # Clear existing obstacles
        self.env.road.objects.clear()
        
        # Add collision (static vehicles)
        collision_pos = self._randomize_collision_position()
        for offset in [(-1, 0), (1, 0)]:  # Add two static vehicles around the collision
            position = [collision_pos[0] + offset[0], collision_pos[1] + offset[1]]
            obstacle = Obstacle(self.env.road, position)
            self.env.road.objects.append(obstacle)

        return obs

    def _randomize_collision_position(self):
        """Randomize collision positions within a specified range."""
        x_range, y_range = self.env_config["collision_position_range"]
        x = np.random.uniform(x_range[0], x_range[1])
        y = np.random.uniform(y_range[0], y_range[1])
        return x, y

# Example usage:
env_config = {
    "collision_position_range": [(-1, -5), (1, 5)],  # Customize the randomization range
}
custom_env = CustomIntersectionEnv(env_config)

# Run an episode
while True:
    obs = custom_env.reset()
    done = truncated = False
    while not (done or truncated):
        action = custom_env.unwrapped.action_type.actions_indexes["IDLE"]
        obs, reward, done, truncated, info = custom_env.step(action)
        custom_env.render()
