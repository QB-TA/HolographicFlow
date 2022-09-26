import torch
from torch import nn
import math
import jax.numpy as jnp
from torch.nn.utils import weight_norm

from .flow import Flow

from sources import Gaussian

class EHM(Flow):
    def __init__(self,
                 scaling,
                 orthogonal,
                 activation,
                 prior=None):
        super().__init__(prior)
        self.layers = nn.ModuleList([scaling, 
                                     orthogonal, 
                                     activation
                                    ])

    def forward(self, x):
        ldj = x.new_zeros(x.shape[0], dtype=torch.get_default_dtype())
        for layer in self.layers:
            x, ldj_ = layer(x)
            ldj = ldj + ldj_
        return x, ldj.real

    def inverse(self, z):
        inv_ldj = z.new_zeros(z.shape[0], dtype=torch.get_default_dtype())
        for layer in self.layers:
            z, inv_ldj_ = layer.inverse(z)
            inv_ldj = inv_ldj + inv_ldj_
        return z, inv_ldj.real
        