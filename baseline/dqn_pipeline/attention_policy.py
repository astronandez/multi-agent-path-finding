import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class AttentionFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, feature_dim=128, encoder_dim=64, attention_dim=32):
        super().__init__(observation_space, feature_dim)

        # Assumes a flat observation space for simplicity; adjust for your observation shape
        obs_dim = observation_space.shape[0]

        # Smaller encoder for each observation component
        self.encoder = nn.Linear(obs_dim, encoder_dim)

        #UPDATE THE ENCODER TO REFLECT OUR OBSERVATION SPACE IF NECESSARY

        # Attention mechanism: query, key, value projections
        self.query_proj = nn.Linear(encoder_dim, attention_dim)
        self.key_proj = nn.Linear(encoder_dim, attention_dim)
        self.value_proj = nn.Linear(encoder_dim, attention_dim)

        # Output layer
        self.output_layer = nn.Linear(attention_dim, feature_dim)

    def forward(self, observations):
        # Encoding step
        x = F.relu(self.encoder(observations))

        # Attention mechanism
        queries = self.query_proj(x)
        keys = self.key_proj(x)
        values = self.value_proj(x)

        # Compute attention weights
        attention_scores = torch.bmm(queries.unsqueeze(1), keys.unsqueeze(2)).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Weighted sum of values
        attention_output = torch.sum(attention_weights.unsqueeze(-1) * values, dim=1)

        # Final output
        return F.relu(self.output_layer(attention_output))