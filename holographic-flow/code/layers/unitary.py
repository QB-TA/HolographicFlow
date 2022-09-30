import torch
from torch import nn
import math
from torch.nn.utils.parametrizations import orthogonal
from torch.nn.utils import parametrize

from emlp.nn.pytorch import Linear as EquivarLinear
from emlp.reps import Vector
from emlp.groups import SO, O, SU, U


from args import args


class Skew(nn.Module):
    def forward(self, X):
        A = X.triu(1)
        if args.complex:
            A = A - A.transpose(-1, -2).conj()
        else:
            A = A - A.transpose(-1, -2)
        return A
    
    def right_inverse(self, A):
        # We assume that A is skew-symmetric / skew-hermitian
        # We take the upper-triangular elements, as these are those used in the forward
        return A.triu(1)


class CayleyMap(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.register_buffer("Id", torch.eye(n))

    def forward(self, X):
        # (I + X)(I - X)^{-1}
        return torch.linalg.solve(self.Id - X, self.Id + X)
    
    def right_inverse(self, A):
        # Assume A orthogonal/unitary
        # (X - I)(X + I)^{-1}
        return torch.linalg.solve(A + self.Id, self.Id - A)


class MatrixExponential(nn.Module):
    def forward(self, X):
        return torch.matrix_exp(X)


class Unitary(nn.Module):
    def __init__(self, type):
        super().__init__()
        self.type = type
        n = args.kernel_size ** 2
        
        #Implements a general O(4) trafo.
        if args.unitary == 'linear' or args.unitary == 'cayley' or args.unitary == 'exp':
            self.unitary = nn.Linear(n, n)
            
            if args.unitary == 'cayley':
                parametrize.register_parametrization(self.unitary, "weight", Skew())
                parametrize.register_parametrization(self.unitary, "weight", CayleyMap(n))
            
            if args.unitary == 'exp':
                parametrize.register_parametrization(self.unitary, "weight", Skew())
                parametrize.register_parametrization(self.unitary, "weight", MatrixExponential())
            
            if self.type == 'decimator':
                w_init = torch.tensor([[1.,  0.,  1.,  0.],
                                       [1.,  0., -1.,  0.],
                                       [0.,  1.,  0.,  1.],
                                       [0.,  1.,  0., -1.]])
                w_init = w_init / math.sqrt(2.)
            if self.type == 'disentangler':
                w_init = torch.tensor([[1., 0., 0., 0.],
                                       [0., 0., 1., 0.],
                                       [0., 1., 0., 0.],
                                       [0., 0., 0., 1.]])

            nn.init.zeros_(self.unitary.bias)
            if args.unitary == 'linear':
                self.unitary.weight = nn.Parameter(w_init)
            #if args.unitary == 'cayley' and self.type == 'decimator':
            #    self.unitary.weight = w_init

            if args.complex:
                self.unitary.to(torch.complex64)

        if args.unitary == 'emlp' or args.unitary == 'emlp_sp':
            #uses the EMLP package to make an Equivariant Linear Layer
            if args.unitary == 'emlp': 
                if args.complex: G = U(n)
                else: G = O(n)
            if args.unitary == 'emlp_sp':
                if args.complex: G = SU(n)
                else: G = SO(n)
            self.unitary = EquivarLinear(Vector(G), Vector(G))
            if args.complex:
                self.unitary.to(torch.complex64)

        if args.unitary == 'o2_stack':
        #Implements O(4) as 2 stacked O(2) trafos
            self.theta1 = nn.Parameter(torch.Tensor(4))
            self.theta2 = nn.Parameter(torch.Tensor(4))
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
        if args.unitary == 'o2_stack':
            inv_ldj = z.new_zeros(z.shape[0])
            if self.type == 'decimator':
                z = self._inverse('two', self.theta2, z)
                z = self._inverse('one', self.theta1, z)
            if self.type == 'disentangler':
                z = self._inverse('two', self.theta2, z)
                z = self._inverse('two', self.theta1, z)

        else:
            inv_ldj = z.new_zeros(z.shape[0])
            oldshape = z.shape
            z = z.view(z.shape[0], -1)
            z = self.unitary(z)
            z = z.view(oldshape)

        return z, inv_ldj
    
    #def forward(self, x): #general O(4)
    #    ldj = x.new_zeros(x.shape[0])
    #    oldshape = x.shape
    #    x = x.view(x.shape[0], -1)
    #    x = self.unitary(x) #TODO: invert the linear layer
    #    x = x.view(oldshape)
    #    return x, ldj
    