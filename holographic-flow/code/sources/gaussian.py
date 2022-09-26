from math import log, pi

import torch

from .source import Source

from args import args


class Gaussian(Source):
    def __init__(self, nvars, scale=1):
        super().__init__(nvars)
        self.register_buffer(
            'scale', torch.tensor(scale))

    def sample(self, batch_size):
        shape = [batch_size, 1 , self.nvars[1], self.nvars[2]]
        out = self.scale.new_empty(shape, dtype=torch.complex64).normal_()
        out = out * self.scale
        if args.subnet != 'ehm':
            out = torch.cat((out.real, out.imag), dim=1)
        return out

    def log_prob(self, x):
        out = (-0.5 * (x.abs() / self.scale)**2 - torch.log(self.scale) -
               0.5 * log(2 * pi))
        out = out.view(out.shape[0], -1).sum(dim=1)
        return out

    """
    def log_prob(self, x):
        x = x.reshape(x.shape[0], -1)
        prec = self.precision_matrix()
        det_cov = torch.linalg.inv(prec)
        _, logdet = torch.linalg.slogdet(2*pi*det_cov)
        torch.sum(x.conj() * (x @ prec.T), dim=-1)
        out = -0.5 * torch.sum(x.conj() * (x @ prec.T), dim=-1) - 0.5 * logdet
        return out.real
    """
