import torch
from torch import nn
import math
from args import args


class Orthogonal(nn.Module):
    def __init__(self, type):
        super().__init__()
        self.theta1 = nn.Parameter(torch.Tensor(4))
        self.theta2 = nn.Parameter(torch.Tensor(4))
        self.type = type
        if type == 'decimator':
            nn.init.constant_(self.theta1, math.pi/4)
            nn.init.constant_(self.theta2, 0.)
        elif type == 'disentangler':
            nn.init.constant_(self.theta1, 0.)
            nn.init.constant_(self.theta2, 0.)
        else: raise RuntimeError("type has to be 'decimator' or 'disentangler'")
        
        self.mask0 = torch.zeros((2,2), device=args.device)
        self.mask0[0,:] = 1
        self.mask1 = torch.ones_like(self.mask0) - self.mask0
    
    def build_matrix(self, block_type, theta):
        M = torch.zeros((2,2), dtype=torch.complex64)
        if block_type == 'one':
            M[0][0] = torch.sin(theta+0j)
            M[0][1] = torch.cos(theta+0j)
            M[1][0] = torch.cos(theta+0j)
            M[1][1] = - torch.sin(theta+0j)
        if block_type == 'two':
            M[0][0] = torch.cos(theta+0j)
            M[0][1] = - torch.sin(theta+0j)
            M[1][0] = torch.sin(theta+0j)
            M[1][1] = torch.cos(theta+0j)
        M = M.to(args.device)
        return M
    
    def _forward(self, block_type, theta, x):
        matrix0 = self.build_matrix(block_type, theta[0])
        matrix1 = self.build_matrix(block_type, theta[1])
        matrix2 = self.build_matrix(block_type, theta[2])
        matrix3 = self.build_matrix(block_type, theta[3])
        x0 = self.mask0 * x
        x1 = self.mask1 * x

        x0 = x0 @ matrix2
        x1 = x1 @ matrix3
        x0[:,:,[0],[1]], x1[:,:,[1],[0]] = x1[:,:,[1],[0]], x0[:,:,[0],[1]] 
        x0 = x0 @ matrix0
        x1 = x1 @ matrix1

        x = x0 + x1
        return x

    def _inverse(self, block_type, theta, z):
        matrix0 = self.build_matrix(block_type, theta[0])
        matrix1 = self.build_matrix(block_type, theta[1])
        matrix2 = self.build_matrix(block_type, theta[2])
        matrix3 = self.build_matrix(block_type, theta[3])
        z0 = self.mask0 * z
        z1 = self.mask1 * z

        z0 = z0 @ matrix0.T
        z1 = z1 @ matrix1.T
        z0[:,:,[0],[1]], z1[:,:,[1],[0]] = z1[:,:,[1],[0]], z0[:,:,[0],[1]]
        z0 = z0 @ matrix2.T
        z1 = z1 @ matrix3.T

        z = z0 + z1
        return z
    
    def forward(self, x):
        ldj = x.new_zeros(x.shape[0])
        if self.type == 'decimator':
            x = self._forward('one', self.theta1, x)
            x = self._forward('two', self.theta2, x)
        if self.type == 'disentangler':
            x = self._forward('two', self.theta1, x)
            x = self._forward('two', self.theta2, x)
        return x, ldj
    
    def inverse(self, z):
        inv_ldj = z.new_zeros(z.shape[0])
        if self.type == 'decimator':
            z = self._inverse('two', self.theta2, z)
            z = self._inverse('one', self.theta1, z)
        if self.type == 'disentangler':
            z = self._inverse('two', self.theta2, z)
            z = self._inverse('two', self.theta1, z)
        return z, inv_ldj
