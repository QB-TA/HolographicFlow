import torch
from torch import nn
from torch.nn.utils import parametrize

from args import args


class LowerTriangular(nn.Module):
    def forward(self, X):
        return X.tril()


class CorrelatedGaussian(nn.Module):
    def __init__(self, nvars, indexI, indexJ, mass=1.):
        super().__init__()
        self.nvars = nvars
        H = self.nvars[1]
        W = self.nvars[2]
        
        self.indexI, self.indexJ = self.seperate_scales(indexI, indexJ)

        if args.complex: 
            self.dtype = torch.complex64

        if args.reparametrize == 'nearest_neighbor':
            #Nearest neighbour approach
            self.mass = nn.Parameter(torch.tensor(mass), requires_grad=False)
            self.kinetic = nn.Parameter(torch.zeros([len(self.indexI), 2]), requires_grad=False)
            
        if args.reparametrize == 'positive_definite':
            #Positive-definite approach
            self.cholesky = nn.Linear(H*W, H*W, bias=False, dtype=self.dtype)
            #nn.init.zeros_(self.cholesky.bias)
            nn.init.eye_(self.cholesky.weight)

            parametrize.register_parametrization(self.cholesky, 'weight', LowerTriangular())

            #self.cholesky.bias.requires_grad = False
            self.cholesky.parametrizations.weight.original.requires_grad = False

        
    def seperate_scales(self, indexI, indexJ):
        """
        returns lists of unique indices sorted according to their RG-scale
        """
        indexI = indexI[1::2]
        indexJ = indexJ[1::2]
        for i in range(len(indexI)):
            indexI[i] = indexI[i].flatten()
            indexJ[i] = indexJ[i].flatten()

        numlayers = len(indexI)
        for l in range(numlayers):
            for m in range(l+1, numlayers):
                for k in range(indexI[m].shape[0]):
                    maski = (indexI[l] != indexI[m][k])
                    maskj = (indexJ[l] != indexJ[m][k])
                    mask = maski.__or__(maskj)
                    indexI[l] = indexI[l][mask]
                    indexJ[l] = indexJ[l][mask]

        return indexI, indexJ
    
    def is_in_layer(self, i, j, layer):
        """
        checks if the bulk variable at (i,j) is a specific RG layer
        """
        if layer >= 0 and layer < len(self.indexI):
            maski = (self.indexI[layer] == i)
            maskj = (self.indexJ[layer] == j)
            mask = maski.__and__(maskj)
            in_layer = mask.any()
        else:
            in_layer = False
        return in_layer

    def precision_matrix(self):
        """
        Implements only nearest neighbour interactions.
        The resulting matrix might not be positive definite.
        """
        C = self.nvars[0]
        H = self.nvars[1]
        W = self.nvars[2]

        prec = torch.zeros([H, W, H, W], device=args.device, dtype=self.dtype)
        
        for l, (indexI, indexJ) in enumerate(zip(self.indexI, self.indexJ)):
            d = int(2**l)
            d_uv = int(2**(l-1))
            d_ir = int(2**(l+1))
            for k in range(indexI.shape[0]):
                i = indexI[k]
                j = indexJ[k]
                jright = (j + d) % W
                idown = (i + d) % H
                jleft = (j + W - d) % W
                iup = (i + H - d) % H

                jright_uv = (j + d_uv) % W
                idown_uv = (i + d_uv) % H
                jleft_uv = (j + W - d_uv) % W
                iup_uv = (i + H - d_uv) % H

                jright_ir = (j + d_ir) % W
                idown_ir = (i + d_ir) % H
                jleft_ir = (j + W - d_ir) % W
                iup_ir = (i + H - d_ir) % H

                l_uv = l - 1

                l_ir = l + 1
                
                if self.is_in_layer(i, jright, l) or self.is_in_layer(i, jright, l_uv) or self.is_in_layer(i, jright, l_ir):
                    prec [i][j] [i][jright] = -self.kinetic[l][1].abs()

                if self.is_in_layer(idown, j, l) or self.is_in_layer(idown, j, l_uv) or self.is_in_layer(idown, j, l_ir):
                    prec [i][j] [idown][j] = -self.kinetic[l][1].abs()

                if self.is_in_layer(i, jleft, l) or self.is_in_layer(i, jleft, l_uv) or self.is_in_layer(i, jleft, l_ir):
                    prec [i][j] [i][jleft] = -self.kinetic[l][1].abs()

                if self.is_in_layer(iup, j, l) or self.is_in_layer(iup, j, l_uv) or self.is_in_layer(iup, j, l_ir):
                    prec [i][j] [iup][j] = -self.kinetic[l][1].abs()
                

                if self.is_in_layer(i, jright_uv, l_uv):
                    prec [i][j] [i][jright_uv] = -self.kinetic[l_uv][1].abs()

                if self.is_in_layer(idown_uv, j, l_uv):
                    prec [i][j] [idown_uv][j] = -self.kinetic[l_uv][1].abs()

                if self.is_in_layer(i, jleft_uv, l_uv):
                    prec [i][j] [i][jleft_uv] = -self.kinetic[l_uv][1].abs()

                if self.is_in_layer(iup_uv, j, l_uv) or self.is_in_layer(iup_uv, j, l):
                    prec [i][j] [iup_uv][j] = -self.kinetic[l_uv][1].abs()


                if self.is_in_layer(i, jright_ir, l_ir):
                    prec [i][j] [i][jright_ir] = -self.kinetic[l_ir][1].abs()

                if self.is_in_layer(idown_ir, j, l_ir):
                    prec [i][j] [idown_ir][j] = -self.kinetic[l_ir][1].abs()

                if self.is_in_layer(i, jleft_ir, l_ir):
                    prec [i][j] [i][jleft_ir] = -self.kinetic[l_ir][1].abs()

                if self.is_in_layer(iup_ir, j, l_ir):
                    prec [i][j] [iup_ir][j] = -self.kinetic[l_ir][1].abs()


                prec [i][j] [i][j] = self.kinetic[l][0].abs() + self.mass.abs()

        prec = torch.reshape(prec, (H*W, H*W))

        return prec
    
    def covariance(self):
        if args.reparametrize == 'nearest_neighbor':
            cov = self.precision_matrix()
            cov = torch.linalg.inv(cov)
        if args.reparametrize == 'positive_definite':
            cov = self.cholesky.weight @ self.cholesky.weight.T.conj()
        return cov

    def forward(self, x):
        H = self.nvars[1]
        W = self.nvars[2]
        ldj = x.new_zeros(x.shape[0]).real
        oldshape = x.shape
        x = x.reshape((x.shape[0], -1))

        if args.reparametrize == 'nearest_neighbor':
        #Cholesky decomposition in nearest neighbour approach
            L = self.precision_matrix()
            L = torch.linalg.inv(L)
            L = torch.linalg.cholesky(L)
            L = torch.linalg.inv(L)
            x = x @ L.T
            _, logdet = torch.linalg.slogdet(L)
            
        if args.reparametrize == 'positive_definite':
        #General reparametrization with positive definite covariance
            x = x.reshape((x.shape[0], -1))
            chol_inv = torch.linalg.inv(self.cholesky.weight)
            x = x @ chol_inv.T
            _, logdet = torch.linalg.slogdet(chol_inv)

        x = x.reshape(oldshape)
        if args.complex and self.nvars[0] == 2:
            x = torch.cat([x.real, x.imag], dim=1)

        ldj = ldj + logdet
        return x, ldj

    def inverse(self, z):
        H = self.nvars[1]
        W = self.nvars[2]
        inv_ldj = z.new_zeros(z.shape[0]).real
        oldshape = z.shape
        z = z.reshape((z.shape[0], -1))

        if args.reparametrize == 'nearest_neighbor':
        #Cholesky decomposition in nearest neighbour approach
            L = self.precision_matrix()
            L = torch.linalg.inv(L)
            L = torch.linalg.cholesky(L)
            z = z @ L.T
            _, logdet = torch.linalg.slogdet(L)
            
        if args.reparametrize == 'positive_definite':
        #General reparametrization with positive definite covariance
            z = z.reshape((z.shape[0], -1))
            self.cholesky(z)
            _, logdet = torch.linalg.slogdet(self.cholesky.weight)

        z = z.reshape(oldshape)
        if args.complex and self.nvars[0] == 2:
            z = torch.cat([z.real, z.imag], dim=1)

        inv_ldj = inv_ldj + logdet
        return z, inv_ldj