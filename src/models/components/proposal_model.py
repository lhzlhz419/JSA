# src/models/components/proposal_model.py
import torch
import torch.nn as nn

class ProposalModel(nn.Module):
    """
    q_phi(h|x), must implement:
        - log_conditional_prob(h, x)
        - sample_latent(x)
    """
    def __init__(self, hidden_dim=256, num_layers=2, h_dim=200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, h_dim),
        )

    def sample_latent(self, x):
        logits = self.net(x)
        probs = torch.sigmoid(logits)
        return torch.bernoulli(probs)

    def log_conditional_prob(self, h, x):
        logits = self.net(x)
        return -torch.nn.functional.binary_cross_entropy_with_logits(
            logits, h, reduction="none"
        ).sum(dim=1)
