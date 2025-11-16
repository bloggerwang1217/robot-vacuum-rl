"""
DQN Network Architecture for Multi-Robot Environment
Simplified version for Independent DQN training
"""

import torch
import torch.nn as nn


def init_weights(m):
    """
    Initialize network weights using Kaiming initialization

    Args:
        m: Network module
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class DQN(nn.Module):
    """
    Deep Q-Network with MLP architecture

    Network structure:
    - Input layer: observation_dim (29 for our multi-robot env)
    - Hidden layers: 128 -> 256 -> 256 -> 128
    - Output layer: num_actions (5 for our multi-robot env)
    """
    def __init__(self, num_actions, input_dim):
        super(DQN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input observation tensor

        Returns:
            Q-values for each action
        """
        return self.network(x)
