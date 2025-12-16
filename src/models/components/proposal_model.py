# src/models/components/proposal_model.py
import torch
import torch.nn as nn

from src.base.base_jsa_modules import BaseProposalModel


class ProposalModelBernoulli(BaseProposalModel):
    """q_phi(h|x)

    Must implement functions:
    - log_conditional_prob(h, x)
    - sample_latent(x)

    We assume Bernoulli distribution for q_phi(h|x)
    """

    def __init__(
        self,
        net: nn.Module = None,
        num_latent_vars=256,
    ):
        super().__init__()

        self.num_latent_vars = num_latent_vars
        self._latent_dim = num_latent_vars

        self._categories = [2] * num_latent_vars  # for compatibility
        self.net = net

    @property
    def latent_dim(self):
        return self._latent_dim

    def forward(self, x):
        """
        Compute distribution parameters (logits) for q(h|x).

        Args:
            x: [B, input_dim]

        Returns:
            logits: [B, num_latent_vars]
        """
        logits = self.net(x)  # [B, latent_dim]
        return logits

    def sample_latent(self, x, num_samples=1):
        """Sample h ~ q(h|x)"""
        logits = self.forward(x)  # [B, latent_dim]
        probs = torch.sigmoid(logits)  # [B, latent_dim]

        # Expand for num_samples
        probs_expanded = probs.unsqueeze(-1).expand(
            -1, -1, num_samples
        )  # [B, latent_dim, num_samples]

        # Bernoulli sampling requires input shape [B, latent_dim, num_samples]
        # But torch.bernoulli expects the last dim to be the one we sample from usually,
        # actually it samples element-wise.
        h_samples = torch.bernoulli(probs_expanded)  # [B, latent_dim, num_samples]

        return h_samples

    def encode(self, x):
        """Deterministic encoding (mode of the distribution)"""
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        h = (probs > 0.5).float()
        return h  # [B, latent_dim]

    def log_conditional_prob(self, h, x):
        """
        Compute log q(h|x)

        Args:
            h: [B, latent_dim] or [B, latent_dim, num_samples]
            x: [B, input_dim]

        Returns:
            log_prob: [B, num_samples] (summed over latent_dim)
        """
        logits = self.forward(x)  # [B, latent_dim]

        # Handle dimensions
        if h.dim() == 2:
            h = h.unsqueeze(-1)  # [B, latent_dim, 1]

        # logits: [B, latent_dim] -> [B, latent_dim, 1]
        logits = logits.unsqueeze(-1)

        # BCE with logits = -log q(h|x)
        # We want log q(h|x), so we negate BCE
        # h must be broadcastable to logits
        log_prob_elementwise = -torch.nn.functional.binary_cross_entropy_with_logits(
            logits.expand_as(h), h, reduction="none"
        )  # [B, latent_dim, num_samples]

        return log_prob_elementwise.sum(dim=1)  # [B, num_samples]

    def get_loss(self, h, x):
        """
        Compute negative log conditional probability as loss

        Args:
            x: observed data
            h: latent variables (targets)

        Returns:
            loss: scalar
        """
        log_cond = self.log_conditional_prob(h, x)  # [B, num_samples]
        return -log_cond.mean()


class ProposalModelCategorical(BaseProposalModel):
    """q_phi(h|x)

    Must implement functions:
    - log_conditional_prob(h, x)
    - sample_latent(x)

    We assume Categorical distribution for q_phi(h|x)
    """

    def __init__(
        self,
        net: nn.Module = None,
        num_latent_vars=10,
        num_categories=256,
    ):
        super().__init__()

        self.num_latent_vars = num_latent_vars

        if isinstance(num_categories, int):
            self._num_categories = [num_categories] * num_latent_vars
        elif len(num_categories) == 1 and num_latent_vars > 1:
            self._num_categories = list(num_categories) * num_latent_vars
        else:
            assert (
                len(num_categories) == num_latent_vars
            ), "num_categories must be an integer or a list of length num_latent_vars"
            self._num_categories = list(num_categories)

        self.total_num_categories = sum(self._num_categories)
        self.net = net  # nn.Module that outputs logits of shape [B, total_num_categories]
        

    @property
    def latent_dim(self):
        return self.total_num_categories

    def forward(self, x):
        """
        Compute distribution parameters (logits) for q(h|x).

        Args:
            x: [B, input_dim]

        Returns:
            split_logits: List of tensors, each shape [B, num_categories_i]
        """
        logits = self.net(x)  # [B, total_num_categories]
        split_logits = torch.split(
            logits, self._num_categories, dim=-1
        )  # List of [B, num_categories_i]
        return split_logits

    def sample_latent(self, x, num_samples=1, encoded=True):
        """Sample h ~ q(h|x)

        Returns:
            h_samples: Tensor of shape [B, num_latent_vars, num_samples] containing sampled latent variable indices.
                dtype=torch.float
        """
        split_logits = self.forward(x)  # List of [B, num_categories_i]

        h_samples_list = []
        for _, logit in enumerate(split_logits):
            probs = torch.softmax(logit, dim=-1)  # [B, num_categories_i]
            h_samples = torch.multinomial(
                probs, num_samples=num_samples, replacement=True
            )  # [B, num_samples]
            h_samples = h_samples.view(-1, 1, num_samples)  # [B, 1, num_samples]
            h_samples_list.append(h_samples)

        h_samples = torch.cat(
            h_samples_list, dim=1
        )  # [B, num_latent_vars, num_samples]

        return h_samples.float()

    def encode(self, x):
        """Deterministic encoding (argmax)"""
        split_logits = self.forward(x)

        idx_list = []
        for logit in split_logits:
            idx = torch.argmax(logit, dim=-1, keepdim=True)  # [B, 1]
            idx_list.append(idx)

        h_idx = torch.cat(idx_list, dim=1)  # [B, num_latent_vars]
        return h_idx.float()

    def log_conditional_prob(self, h, x):
        """Compute log q(h|x)

        Args:
            h: Tensor of latent variable indices, shape [B, num_latent_vars] or [B, num_latent_vars, num_samples].
            x: Input tensor of shape [B, input_dim].

        Returns:
            log_cond: Tensor of shape [B, num_samples] containing log probabilities log q(h|x).
        """
        split_logits = self.forward(x)  # List of [B, num_categories_i]

        if h.dim() == 2:
            h = h.unsqueeze(-1)  # [B, num_latent_vars, 1]

        B, _, num_samples = h.shape
        log_cond = x.new_zeros(B, num_samples)  # [B, num_samples]

        for i, logit in enumerate(split_logits):
            # logit: [B, num_categories_i]
            h_i = h[:, i, :]  # [B, num_samples] (indices)

            log_prob_i = torch.log_softmax(logit, dim=-1)  # [B, num_categories_i]

            # We need to gather the log_prob for each sample
            # log_prob_i needs to be expanded to [B, num_samples, num_categories_i] if we want to gather easily
            # OR we can expand log_prob_i to [B, num_categories_i] and gather along dim 1 if we reshape h_i

            # Let's use gather on dim 1.
            # log_prob_i: [B, num_categories_i]
            # h_i: [B, num_samples]

            # To use gather, input and index must have same number of dimensions.
            # Expand log_prob_i to [B, num_samples, num_categories_i] is expensive.
            # Instead, let's treat samples as batch.

            # Method 1: Expand log_prob
            log_prob_i_expanded = log_prob_i.unsqueeze(1).expand(
                -1, num_samples, -1
            )  # [B, num_samples, num_categories_i]
            gathered = log_prob_i_expanded.gather(2, h_i.long().unsqueeze(-1)).squeeze(
                -1
            )  # [B, num_samples]

            log_cond += gathered

        return log_cond  # [B, num_samples]

    def get_loss(self, h, x):
        """Compute negative log conditional probability as loss"""
        log_cond = self.log_conditional_prob(h, x)
        return -log_cond.mean()
