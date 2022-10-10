from audioop import bias
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
    #applied to skew matrices it produces a subset of SU(n)
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
    #applied to skew matrices it produces matrices of SU(n)
    def forward(self, X):
        return torch.matrix_exp(X)


class LinearMap(nn.Module):
    def __init__(self, type):
        super().__init__()
        self.type = type
        n = args.kernel_size ** 2

        if self.type == 'disentangler':
            self.linear_map = args.disentangler
        elif self.type == 'decimator':
            self.linear_map = args.decimator
        
        #Implements a general O(4) trafo.
        if self.linear_map == 'linear' or self.linear_map == 'cayley' or self.linear_map == 'exp':
            if args.complex: 
                dtype = torch.complex64
            else: dtype = torch.float32
            self.linear = nn.Linear(n, n, bias=False, dtype=dtype)
            
            if self.linear_map == 'cayley':
                parametrize.register_parametrization(self.linear, "weight", Skew())
                parametrize.register_parametrization(self.linear, "weight", CayleyMap(n))
            
            if self.linear_map == 'exp':
                parametrize.register_parametrization(self.linear, "weight", Skew())
                parametrize.register_parametrization(self.linear, "weight", MatrixExponential())
            
            if self.type == 'decimator':
                w_init = torch.tensor([[1.,  0.,  1.,  0.],
                                       [1.,  0., -1.,  0.],
                                       [0.,  1.,  0.,  1.],
                                       [0.,  1.,  0., -1.]], dtype=dtype)
                w_init = w_init / math.sqrt(2.)
            if self.type == 'disentangler':
                w_init = torch.tensor([[1., 0., 0., 0.],
                                       [0., 0., 1., 0.],
                                       [0., 1., 0., 0.],
                                       [0., 0., 0., 1.]], dtype=dtype)
            #nn.init.zeros_(self.linear.bias)
            if self.linear_map == 'linear':
                self.linear.weight = nn.Parameter(w_init)

            #if self.linear_map == 'cayley' and self.type == 'decimator':
            #    self.linear.weight = w_init

        if self.linear_map == 'emlp' or self.linear_map == 'emlp_sp':
            #uses the EMLP package to make an Equivariant Linear Layer
            if self.linear_map == 'emlp': 
                if args.complex: G = U(n)
                else: G = O(n)
            if self.linear_map == 'emlp_sp':
                if args.complex: G = SU(n)
                else: G = SO(n)
            self.linear = EquivarLinear(Vector(G), Vector(G))
            nn.init.zeros_(self.linear.bias)
            self.linear.bias.requires_grad = False
            if args.complex:
                self.linear.to(torch.complex64)

        if self.linear_map == 'o2_stack':
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
        if self.linear_map == 'o2_stack':
            if self.type == 'decimator':
                x = self._forward('one', self.theta1, x)
                x = self._forward('two', self.theta2, x)
            if self.type == 'disentangler':
                x = self._forward('two', self.theta1, x)
                x = self._forward('two', self.theta2, x)
        else:
            oldshape = x.shape
            x = x.view(x.shape[0], -1)
            inv_weight = torch.linalg.inv(self.linear.weight)
            x = x - self.linear.bias
            x = x @ inv_weight.T
            x = x.view(oldshape)
            _, logdet = torch.linalg.slogdet(inv_weight)
            ldj += logdet
        return x, ldj
    
    def inverse(self, z):
        inv_ldj = z.new_zeros(z.shape[0])

        if self.linear_map == 'o2_stack':
            if self.type == 'decimator':
                z = self._inverse('two', self.theta2, z)
                z = self._inverse('one', self.theta1, z)
            if self.type == 'disentangler':
                z = self._inverse('two', self.theta2, z)
                z = self._inverse('two', self.theta1, z)
        else:
            oldshape = z.shape
            z = z.view(z.shape[0], -1)
            z = self.linear(z)
            z = z.view(oldshape)
            _, logdet = torch.linalg.slogdet(self.linear.weight)
            inv_ldj += logdet
        return z, inv_ldj