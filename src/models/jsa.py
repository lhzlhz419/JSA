# src/models/jsa.py
import torch
from lightning.pytorch import LightningModule


class JSA(LightningModule):
    def __init__(self, joint_model, proposal_model, sampler, lr=1e-3, num_mis_steps=1):
        super().__init__()
        self.joint_model = joint_model
        self.proposal_model = proposal_model
        self.sampler = sampler
        self.lr = lr
        self.num_mis_steps = num_mis_steps

    def forward(self, x, idx):
        # MIS sampling
        h = self.sampler.sample(x, idx=idx, num_steps=self.num_mis_steps)
        return h

    def training_step(self, batch, batch_idx):
        x, _, idx = batch
        idx = idx.tolist()

        # obtain latent samples
        h = []
        for i in range(x.size(0)):
            h_i = self.sampler.sample(
                x[i : i + 1], idx=idx[i], num_steps=self.num_mis_steps
            )
            h.append(h_i)
        h = torch.cat(h, dim=0)

        # compute gradients
        log_p = self.joint_model.log_joint_prob(x, h)
        log_q = self.proposal_model.log_conditional_prob(h, x)

        loss = -(log_p + log_q).mean()

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.joint_model.parameters())
            + list(self.proposal_model.parameters()),
            lr=self.lr,
        )
        return optimizer
