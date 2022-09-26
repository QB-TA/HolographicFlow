import torch
from torch import nn


class Activation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        ldj = torch.log(torch.cosh(torch.abs(x)))
        ldj = ldj.sum(dim=(1,2,3)).real
        x = torch.sinh(torch.abs(x)) * x / torch.abs(x)
        return x, ldj
    
    def inverse(self, x):
        inv_ldj = -1/2 * torch.log(1 + torch.square(torch.abs(x)))
        inv_ldj = inv_ldj.sum(dim=(1,2,3)).real
        x = torch.asinh(torch.abs(x)) * x / torch.abs(x)
        return x, inv_ldj
