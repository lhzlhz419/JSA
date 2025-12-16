# src/models/components/networks.py
# Define some network components here for proposal and joint models

import torch
import torch.nn as nn
import numpy as np
from src.utils.mlp_utils import build_mlp

class MLPNetwork(nn.Module):
    """ A simple Multi-Layer Perceptron network 
    
    """
    def __init__(self, input_dim, output_dim, layers=[512, 512], activation="relu", final_activation=None):
        super().__init__()
        self.net = build_mlp(
            input_dim=input_dim,
            layers=layers,
            output_dim=output_dim,
            activation=activation,
            final_activation=final_activation
        )

    def post_process(self, x):
        """Post-process output x to be in [0, 1] range"""
        if self.net[-1].__class__ == nn.Sigmoid:
            return x  # already in [0, 1]
        elif self.net[-1].__class__ == nn.Tanh:
            return (x + 1) / 2  # scale from [-1, 1] to [0, 1]
        else:
            return torch.clamp(x, 0.0, 1.0)  # clamp to [0, 1]
        
    def forward(self, x):
        feature = self.net(x)
        return self.post_process(feature)
    
    
class CNNEncoder(nn.Module):
    pass

class CNNDecoder(nn.Module):
    pass