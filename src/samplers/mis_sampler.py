# src/samplers/mis_sampler.py
import torch
from src.base.base_sampler import BaseSampler


class MISampler(BaseSampler):
    def __init__(
        self,
        joint_model,
        proposal_model,
        use_cache=True,
        dataset_size=None,
        device="cpu",
    ):
        self.joint_model = joint_model
        self.proposal_model = proposal_model
        self.use_cache = use_cache
        self.device = device

        if use_cache:
            assert dataset_size is not None
            self.cache = [None for _ in range(dataset_size)]

    @torch.no_grad()
    def step(self, x, h_old=None, idx=None):
        """
        Perform single MIS step:
            propose h'
            compute acceptance probability
            accept/reject
        """
        # propose from q_phi(h|x)
        h_new = self.proposal_model.sample_latent(x)

        if h_old is None:
            h_old = h_new.clone()

        log_p_new = self.joint_model.log_joint_prob(x, h_new)
        log_p_old = self.joint_model.log_joint_prob(x, h_old)

        log_q_new = self.proposal_model.log_conditional_prob(h_new, x)
        log_q_old = self.proposal_model.log_conditional_prob(h_old, x)

        log_accept = (log_p_new + log_q_old) - (log_p_old + log_q_new)
        accept_prob = torch.exp(torch.clamp(log_accept, max=0.0))

        u = torch.rand_like(accept_prob)
        accept = (u < accept_prob).float().unsqueeze(-1)

        h_next = accept * h_new + (1 - accept) * h_old

        # update cache
        if self.use_cache and idx is not None:
            self.cache[idx] = h_next.detach().cpu()

        return h_next

    @torch.no_grad()
    def sample(self, x, h_init=None, idx=None, num_steps=1):
        h = h_init
        for _ in range(num_steps):
            h = self.step(x, h_old=h, idx=idx)
        return h

    def state_dict(self):
        if not self.use_cache:
            return {}
        return {"cache": self.cache}

    def load_state_dict(self, state):
        if "cache" in state:
            self.cache = state["cache"]
