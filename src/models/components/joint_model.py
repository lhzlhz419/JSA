# src/models/components/joint_model.py

import torch
import torch.nn as nn

from src.base.base_jsa_modules import BaseJointModel
from src.utils.mlp_utils import  build_mlp

class JointModel(BaseJointModel):
    """
    p_theta(x, h), must implement:
        - log_joint_prob(x, h)

    Using MLP to model p_theta(x|h)
    """

    def __init__(
        self, latent_dim=256, layers=[512, 512], output_dim=784, activation: str = "relu"
    ):
        super().__init__()

        self._latent_dim = latent_dim
        self.output_dim = output_dim
        self.net = build_mlp(
            input_dim=latent_dim, layers=layers, output_dim=output_dim, activation=activation
        )

    @property
    def latent_dim(self):
        return self._latent_dim

    def log_prior_prob(self, h):
        """
        Compute log p(h)
        Assuming uniform prior over Bernoulli latent variables
        """
        # For Bernoulli latent variables with uniform prior,
        # log p(h) = sum over dimensions of log(0.5) = h.size(1) * log(0.5)
        return h.size(1) * torch.log(torch.tensor(0.5, device=h.device))

    def log_joint_prob(self, x, h):
        """ Must compute log p(x, h) = log p(h) + log p(x|h)
        
        Args:
            x: observed data
            h: latent variables
        
        """
        # Prior p(h)
        log_p_h = self.log_prior_prob(h)
        
        # Likelihood p(x|h) 
        # We assume independent Bernoulli distribution for each pixel
        logits_x = self.net(h)
        log_p_x_given_h = -torch.nn.functional.binary_cross_entropy_with_logits(
            logits_x, x, reduction="none"
        ).sum(dim=1)
        return log_p_h + log_p_x_given_h
    
    def sample(self, h=None, num_samples=1):
        """ Sample x ~ p(x|h)
        
        Args:
            h: latent variables
            num_samples: number of samples to generate
        
        Returns:
            x_sample: sampled observed data
        """
        
        if h is None:
            h = torch.bernoulli(0.5 * torch.ones((1, self.latent_dim))).to(
                next(self.parameters()).device
            )
        logits_x = self.net(h)
        probs_x = torch.sigmoid(logits_x)
        
        # Generate samples
        x_samples = torch.bernoulli(probs_x.repeat(num_samples, 1)) # [num_samples, output_dim]
        return x_samples
