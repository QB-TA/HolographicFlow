from math import log
import numpy as np
import torch
from args import args

class HolographicDistance():
    def __init__(self, flow):
        self.flow = flow

    def mutual_info(self, i1, j1, i2, j2, cov):
        W = self.flow.reparametrize.nvars[2]
        k1 = i1 * W + j1
        k2 = i2 * W + j2
        info = - 0.5 * log(1 - ((cov[k1][k2] * cov[k2][k1]) / (cov[k1][k1] * cov[k2][k2])).real)
        return info

    def point_distance(self, i1, j1, i2, j2, cov, corr=1., offset=1.):
        info = self.mutual_info(i1, j1, i2, j2, cov)
        d = - corr * log(info / offset)
        return d

    def decimator_distance(self, indexI1, indexJ1, indexI2, indexJ2, cov):
        """
        index list are of length 4. There should be 4 index pairs for each of the 2 decimators.
        """
        dist_list = []
        for k1 in range(len(indexI1)):
            for k2 in range(len(indexI2)):
                dist_list.append(self.point_distance(indexI1[k1], indexJ1[k1], indexI2[k2], indexJ2[k2], cov))
        return np.mean(dist_list)


    def angular_distance(self, layer):
        """
        The boundary is layer 0.
        x1, x2 is the center position of the decimator projected to the boundary
        """
        cov = self.flow.reparametrize.precision_matrix()
        cov = torch.linalg.inv(cov)
        ang_dist = []
        I1 = None
        J1 = None
        r_range = np.arange(1, args.L / 2**(layer+1))

        I1 = self.flow.indexI[layer*2+1][0]
        J1 = self.flow.indexJ[layer*2+1][0]

        for r in r_range:
            I2 = None
            J2 = J1

            for indexI2 in self.flow.indexI[layer*2+1]:
                if np.all(indexI2 == (I1 + r * 2**(layer+1)) % args.L):
                    I2 = indexI2

            print(I1, J1, I2, J2)
            print(self.decimator_distance(I1, J1, I2, J2, cov))
            ang_dist.append(self.decimator_distance(I1, J1, I2, J2, cov))

        return ang_dist, r_range


    def radial_distance():
        #TODO: implement
        return

    def two_point_fct(self, i1, j1, i2, j2):
        qft_config = self.flow.sample(1)[0]
        two_point = qft_config[:, 0, i1, j1].conj() * qft_config[:, 0, i2, j2]
        return two_point.mean()