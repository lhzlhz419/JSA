# src/models/components/joint_model.py

import torch
import torch.nn as nn
from src.base.base_jsa_modules import BaseJointModel

class JointModel(BaseJointModel):
    """
    p_theta(x, h), must implement:
        - log_joint_prob(x, h)
    """
    def __init__(self, hidden_dim=256, num_layers=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(200, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 784),
        )

    def log_joint_prob(self, x, h):
        """
        Must compute log p_theta(x, h).
        h shape depends on latent structure (Bernoulli/Categorical)
        """
        # placeholder
        logits_x = self.net(h)
        log_p_x_given_h = -torch.nn.functional.binary_cross_entropy_with_logits(
            logits_x, x, reduction="none"
        ).sum(dim=1)

        # assume p(h) uniform → constant
        return log_p_x_given_h
