# src/models/jsa.py
import torch
from lightning.pytorch import LightningModule
from hydra.utils import instantiate
import torchvision

from src.samplers.misampler import MISampler
from src.models.components.joint_model import JointModel
from src.models.components.proposal_model import ProposalModel


class JSA(LightningModule):
    def __init__(
        self,
        joint_model,
        proposal_model,
        sampler,
        lr_joint=1e-3,
        lr_proposal=1e-3,
        num_mis_steps=3,
    ):
        super().__init__()
        self.joint_model: JointModel = instantiate(joint_model)
        self.proposal_model: ProposalModel = instantiate(proposal_model)
        self.sampler: MISampler = instantiate(
            sampler,
            joint_model=self.joint_model,
            proposal_model=self.proposal_model,
        )

        self.automatic_optimization = False
        self.save_hyperparameters()

    def forward(self, x, idx):
        h = self.sampler.sample(
            x, idx=idx, num_steps=self.hparams.num_mis_steps
        )
        x_hat = self.joint_model.sample(h=h)
        return x_hat

    def setup(self, stage=None):
        device = self.device
        self.sampler.to(device)
    
    def configure_optimizers(self):
        opt_joint = torch.optim.Adam(
            self.joint_model.parameters(), lr=self.hparams.lr_joint
        )
        opt_proposal = torch.optim.Adam(
            self.proposal_model.parameters(), lr=self.hparams.lr_proposal
        )
        return [opt_joint, opt_proposal]

    def training_step(self, batch, batch_idx):
        x, _, idx = batch  # x: [B, D], idx: [B,]
        idx = idx.tolist()

        # MISampling step
        h = self.sampler.sample(
            x, idx=idx, num_steps=self.hparams.num_mis_steps
        )

        # Optimizers
        opt_joint, opt_proposal = self.optimizers()

        # Update joint model
        opt_joint.zero_grad()
        loss_joint = -self.joint_model.log_joint_prob(x, h).mean()
        self.manual_backward(loss_joint)
        opt_joint.step()

        # Update proposal model
        opt_proposal.zero_grad()
        loss_proposal = -self.proposal_model.log_conditional_prob(h, x).mean()
        self.manual_backward(loss_proposal)
        opt_proposal.step()

        self.log("train/loss_joint", loss_joint, prog_bar=True)
        self.log("train/loss_proposal", loss_proposal, prog_bar=True)

    def get_nll(self, x, idx):

        # MISampling step
        h = self.sampler.sample(x, idx=idx, num_steps=self.hparams.num_mis_steps)

        # log p(x) ~ log p(x,h) - log q(h|x)
        log_nll = self.joint_model.log_joint_prob(
            x, h
        ) - self.proposal_model.log_conditional_prob(h, x)

        return log_nll.detach().cpu().numpy()

    def validation_step(self, batch, batch_idx):
        x, _, idx = batch  # x: [B, D], idx: [B,]
        idx = idx.tolist()

        nll = -self.get_nll(x, idx=idx)

        self.log("valid/nll", nll.mean(), prog_bar=True)

    def on_train_epoch_end(self):
        # Show some generated samples
        samples = self.joint_model.sample(num_samples=16)
        grid = torchvision.utils.make_grid(samples.view(-1, 1, 28, 28), nrow=4)
        self.logger.experiment.add_image("generated_samples", grid, self.current_epoch)
        
    def on_train_batch_start(self, batch, batch_idx):
        print(f"[JSA] Starting training batch {batch_idx} at epoch {self.current_epoch}.")
