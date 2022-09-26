import torch
from torch import nn


class Scaling(nn.Module):
    def __init__(self):
        super().__init__()
        # Zero init to avoid value explosion
        self.scale = nn.Parameter(torch.zeros((2,2)))
        # For debug
        #self.register_buffer('saved_mean', torch.zeros(num_features))
        #self.register_buffer('saved_var', torch.ones(num_features))

    def forward(self, x):
        with torch.no_grad():
            self.saved_mean = x.mean(dim=0)
            self.saved_var = x.var(dim=0)
        ldj = x.new_zeros(x.shape[0])
        ldj = ldj - self.scale.sum()
        x = torch.exp(-self.scale) * x
        return x, ldj

    def inverse(self, z):
        with torch.no_grad():
            self.saved_mean = z.mean(dim=0)
            self.saved_var = z.var(dim=0)
        inv_ldj = z.new_zeros(z.shape[0])
        inv_ldj = inv_ldj + self.scale.sum()
        z = torch.exp(self.scale) * z
        return z, inv_ldj
