# src/base/base_jsa_modules.py
from abc import ABC, abstractmethod
from torch.nn import Module

# Base class for Proposal Model
class BaseProposalModel(Module, ABC):
    @abstractmethod
    def sample_latent(self, x):
        """Sample latent variable h given input x.
        
        h ~ q(h|x)"""
        pass

    @abstractmethod
    def log_conditional_prob(self, h, x):
        """Compute log probability log q(h|x)"""
        pass
    
    @property
    @abstractmethod
    def latent_dim(self):
        """Return the dimension of the latent variable h"""
        pass
    
# Base class for Joint Model
class BaseJointModel(Module, ABC):
    @abstractmethod
    def log_joint_prob(self, x, h):
        """Compute log joint probability log p(x,h)"""
        pass
    
    @property
    @abstractmethod
    def latent_dim(self):
        """Return the dimension of the latent variable h"""
        pass
    
