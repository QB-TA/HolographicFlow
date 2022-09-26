from turtle import down, left, right
import torch
from torch import nn
from math import log2
import numpy as np

from args import args


class CorrelatedGaussian(nn.Module):
    def __init__(self, nvars, indexI, indexJ, mass=1.):
        super().__init__()
        self.nvars = nvars
        #kinetic = torch.zeros(self.nvars + [3])
        #kinetic[:,:,:,0] = 0.5
        #self.kinetic = nn.Parameter(kinetic, requires_grad=False)        
        #self.offdiag = nn.ParameterList()
        #for _ in range(int(log2(min(H,W)))):
        #    self.offdiag.append(nn.Parameter(torch.zeros([C,H,W,2]), requires_grad=False))
        #    H = int(H/2)
        #    W = int(W/2)
        self.indexI, self.indexJ = self.seperate_scales(indexI, indexJ)
        self.mass = nn.Parameter(torch.tensor(mass), requires_grad=False)
        self.kinetic = nn.Parameter(torch.zeros([len(self.indexI), 2]), requires_grad=False)

        
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
        checks if node (i,j) is in the local neighbourhood of layer
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
        C = self.nvars[0]
        H = self.nvars[1]
        W = self.nvars[2]
        
        if args.subnet == 'ehm':
            prec = torch.zeros([C, H, W, C, H, W], device=args.device, dtype=torch.complex64)
        else: 
            prec = torch.zeros([C, H, W, C, H, W], device=args.device, dtype=torch.float32)
        
        for l, (indexI, indexJ) in enumerate(zip(self.indexI, self.indexJ)):
            d = int(2**l)
            d_uv = int(2**(l-1))
            d_ir = int(2**(l+1))
            for c in range(C):
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
                        prec [c][i][j] [c][i][jright] = -self.kinetic[l][1].abs()

                    if self.is_in_layer(idown, j, l) or self.is_in_layer(idown, j, l_uv) or self.is_in_layer(idown, j, l_ir):
                        prec [c][i][j] [c][idown][j] = -self.kinetic[l][1].abs()

                    if self.is_in_layer(i, jleft, l) or self.is_in_layer(i, jleft, l_uv) or self.is_in_layer(i, jleft, l_ir):
                        prec [c][i][j] [c][i][jleft] = -self.kinetic[l][1].abs()

                    if self.is_in_layer(iup, j, l) or self.is_in_layer(iup, j, l_uv) or self.is_in_layer(iup, j, l_ir):
                        prec [c][i][j] [c][iup][j] = -self.kinetic[l][1].abs()
                    

                    if self.is_in_layer(i, jright_uv, l_uv):
                        prec [c][i][j] [c][i][jright_uv] = -self.kinetic[l_uv][1].abs()

                    if self.is_in_layer(idown_uv, j, l_uv):
                        prec [c][i][j] [c][idown_uv][j] = -self.kinetic[l_uv][1].abs()

                    if self.is_in_layer(i, jleft_uv, l_uv):
                        prec [c][i][j] [c][i][jleft_uv] = -self.kinetic[l_uv][1].abs()

                    if self.is_in_layer(iup_uv, j, l_uv) or self.is_in_layer(iup_uv, j, l):
                        prec [c][i][j] [c][iup_uv][j] = -self.kinetic[l_uv][1].abs()


                    if self.is_in_layer(i, jright_ir, l_ir):
                        prec [c][i][j] [c][i][jright_ir] = -self.kinetic[l_ir][1].abs()

                    if self.is_in_layer(idown_ir, j, l_ir):
                        prec [c][i][j] [c][idown_ir][j] = -self.kinetic[l_ir][1].abs()

                    if self.is_in_layer(i, jleft_ir, l_ir):
                        prec [c][i][j] [c][i][jleft_ir] = -self.kinetic[l_ir][1].abs()

                    if self.is_in_layer(iup_ir, j, l_ir):
                        prec [c][i][j] [c][iup_ir][j] = -self.kinetic[l_ir][1].abs()


                    prec [c][i][j] [c][i][j] = self.kinetic[l][0].abs() + self.mass.abs()
                    
                    """
                    prec [c][i][j] [c][i][jright] = -self.kinetic[l][1].abs()
                    prec [c][i][j] [c][idown][j] = -self.kinetic[l][2].abs()
                    prec [c][i][j] [c][i][jleft] = -self.kinetic[l][1].abs()
                    prec [c][i][j] [c][iup][j] = -self.kinetic[l][2].abs()
                    
                    
                    if self.is_neighbour(l, idown, jright):
                        prec [c][i][j] [c][idown][jright] = -self.kinetic[l][3].abs()
                    if self.is_neighbour(l, iup, jright):
                        prec [c][i][j] [c][iup][jright] = -self.kinetic[l][3].abs()
                    if self.is_neighbour(l, iup, jleft):
                        prec [c][i][j] [c][iup][jleft] = -self.kinetic[l][3].abs()
                    if self.is_neighbour(l, idown, jleft):
                        prec [c][i][j] [c][iup][jleft] = -self.kinetic[l][3].abs()
                    
        
        # all bonds without translation invariance
        for l, (indexI, indexJ) in enumerate(zip(self.indexI[::2], self.indexJ[::2])):
            d = int(2**l)
            for c in range(C):
                for m in range(indexI.shape[0]):
                    for n in range(indexI.shape[1]):
                        i = indexI[m][n]
                        j = indexJ[m][n]
                        if d == 1:
                            prec [c][i][j] [c][i][j] = self.diag[c][i][j].abs() + self.mass.abs()

                        prec [c][i][j] [c][i][(j+d)%W] = -self.offdiag[l] [c][int(i/d)][int(j/d)][0].abs()
                        prec [c][i][j] [c][(i+d)%H][j] = -self.offdiag[l] [c][int(i/d)][int(j/d)][1].abs()

                        prec [c][i][j] [c][i][(j+W-d)%W] = -self.offdiag[l] [c][int(i/d)][int(((j+W-d)%W)/d)][0].abs()
                        prec [c][i][j] [c][(i+H-d)%H][j] = -self.offdiag[l] [c][int(((i+H-d)%H)/d)][int(j/d)][1].abs()
       
        for c in range(C):
            for i in range(H):
                for j in range(W):
                    prec [c][i][j] [c][i][j] = self.kinetic[c][i][j][0].abs() + self.mass.abs()
                    prec [c][i][j] [c][i][(j+1)%W] = -self.kinetic[c][i][j][1].abs()
                    prec [c][i][j] [c][(i+1)%H][j] = -self.kinetic[c][i][j][2].abs()
                    prec [c][i][j] [c][i][(j+W-1)%W] = -self.kinetic[c][i][(j+W-1)%W][1].abs()
                    prec [c][i][j] [c][(i+H-1)%H][j] = -self.kinetic[c][(i+H-1)%H][j][2].abs()
        """

        prec = torch.reshape(prec, (C*H*W, C*H*W))

        return prec

    def reparametrize(self, z):
        oldshape = z.shape
        inv_ldj = z.new_zeros(z.shape[0]).real
        L = self.precision_matrix()
        L = torch.linalg.inv(L)
        L = torch.linalg.cholesky(L)
        z = z.reshape((z.shape[0], -1))
        z = z @ L.T
        z = z.reshape(oldshape)
        _, logdet = torch.linalg.slogdet(L)
        inv_ldj = inv_ldj + logdet
        return z, inv_ldj