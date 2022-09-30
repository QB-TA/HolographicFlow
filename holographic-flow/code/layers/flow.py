from torch import nn
import torch

from args import args


class Flow(nn.Module):
    def __init__(self, reparametrize=None, prior=None):
        super().__init__()
        self.prior = prior
        self.reparametrize = reparametrize

    def forward(self, x):
        raise NotImplementedError(str(type(self)))

    def inverse(self, z):
        raise NotImplementedError(str(type(self)))

    def sample(self, batch_size, prior=None):
        if prior is None:
            prior = self.prior
        assert prior is not None
        z = prior.sample(batch_size)
        logp = prior.log_prob(z)
        if self.reparametrize != None:
            z, logp_ = self.reparametrize.reparametrize(z)
        x, inv_ldj = self.inverse(z)
        if self.reparametrize != None:
            inv_ldj = logp_+inv_ldj
        if args.complex and x.dtype == torch.float32:
            x = torch.complex(x[:,0,:,:], x[:,1:,:])
        return x, logp, inv_ldj

    def log_prob(self, x):
        z, logp = self.forward(x)
        if self.prior is not None:
            logp = logp + self.prior.log_prob(z)
        return logp
