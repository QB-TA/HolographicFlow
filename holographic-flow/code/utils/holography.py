from math import log
import numpy as np
import torch
from args import args

class HolographicDistance():
    def __init__(self, flow):
        self.flow = flow
        cov = self.flow.reparametrize.covariance()

    def mutual_info(self, i1, j1, i2, j2, cov):
        W = self.flow.reparametrize.nvars[2]
        k1 = i1 * W + j1
        k2 = i2 * W + j2
        info = - 0.5 * log(1 - ((cov[k1][k2] * cov[k2][k1]) / (cov[k1][k1] * cov[k2][k2])).real)
        return info

    def geodesic_distance(self, i1, j1, i2, j2, cov, corr_length=1., offset=1.):
        info = self.mutual_info(i1, j1, i2, j2, cov)
        d = - corr_length * log(info / offset)
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
        Calculates the distances between decimators within one layer 
        as the average of geodesic distances of its associated bulk variables,
        if the variables belong to that layer.
        """
        cov = self.flow.reparametrize.covariance()
        r_range = np.arange(1, args.L / 2**(layer+1))
        ang_dist = []

        for r in r_range:
            I1 = self.flow.indexI [2*layer+1] [0]
            J1 = self.flow.indexJ [2*layer+1] [0]
            I2 = self.flow.indexI [2*layer+1] [int(r * args.L / 2**(layer+1))]
            J2 = self.flow.indexJ [2*layer+1] [int(r * args.L / 2**(layer+1))]
            dist = []

            for k in range(args.kernel_size**2):
                if not self.flow.reparametrize.is_in_layer(I1[k], J1[k], layer):
                    I1 = np.delete(I1, k)
                    J1 = np.delete(J1, k)
                if not self.flow.reparametrize.is_in_layer(I2[k], J2[k], layer):
                    I2 = np.delete(I2, k)
                    J2 = np.delete(J2, k)

            print(I1, I2)
            for m in range(len(I1)):
                for n in range(len(I2)):
                        dist.append(self.geodesic_distance(I1[m],J1[m], I2[n],J2[n], cov))
 
            if dist:
                ang_dist.append(np.mean(dist))

        return r_range, ang_dist


    def radial_distance(self):
        """
        The boundary is layer 0.
        Calculates the distances between decimators of different layers
        as the average of geodesic distances of its associated bulk variables,
        if the variables belong to that layer.
        """
        cov = self.flow.reparametrize.covariance()
        r_range = np.arange(1, args.depth / 2)
        rad_dist = []

        for r in r_range:
            print('r =', r)
            l1 = 0
            l2 = int(l1 + r)
            I1 = self.flow.indexI [int(l1 + 1)] [0]
            J1 = self.flow.indexJ [int(l1 + 1)] [0]
            I2 = self.flow.indexI [int(l1 + 1 + r*2)] [0]
            J2 = self.flow.indexJ [int(l1 + 1 + r*2)] [0]

            dist = []

            for k in range(args.kernel_size**2):
                if not self.flow.reparametrize.is_in_layer(I1[k], J1[k], l1):
                    I1 = np.delete(I1, k)
                    J1 = np.delete(J1, k)
                if not self.flow.reparametrize.is_in_layer(I2[k], J2[k], l2):
                    I2 = np.delete(I2, k)
                    J2 = np.delete(J2, k)

            print(I1, I2)
            print(J1, J2)
            for m in range(len(I1)):
                for n in range(len(I2)):
                        dist.append(self.geodesic_distance(I1[m],J1[m], I2[n],J2[n], cov))

            if dist:
                rad_dist.append(np.mean(dist))

        return r_range, rad_dist

    def two_point(self, i1, j1, i2, j2):
        qft_config = self.flow.sample(args.L**2)[0]
        two_point = qft_config[:, 0, i1, j1].conj() * qft_config[:, 0, i2, j2]
        return two_point.mean()
    
    def one_point(self):
        qft_config = self.flow.sample(args.L**2)[0]
        return qft_config.mean().abs()