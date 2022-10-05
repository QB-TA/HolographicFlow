import torch
from torch import nn

from .flow import Flow


class EHM(Flow):
    def __init__(self,
                 scaling,
                 unitary,
                 activation,
                 prior=None):
        super().__init__(prior)
        self.layers = nn.ModuleList([activation, 
                                     unitary, 
                                     scaling
                                    ])

    def forward(self, x):
        ldj = x.new_zeros(x.shape[0], dtype=torch.get_default_dtype())
        for layer in self.layers:
            x, ldj_ = layer(x)
            ldj = ldj + ldj_
        return x, ldj

    def inverse(self, z):
        inv_ldj = z.new_zeros(z.shape[0], dtype=torch.get_default_dtype())
        for layer in reversed(self.layers):
            z, inv_ldj_ = layer.inverse(z)
            inv_ldj = inv_ldj + inv_ldj_
        return z, inv_ldj
        