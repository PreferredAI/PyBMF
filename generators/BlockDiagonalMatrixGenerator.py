import numpy as np
from .BaseGenerator import BaseGenerator
from scipy.sparse import lil_matrix


class BlockDiagonalMatrixGenerator(BaseGenerator):
    '''Block diagonal Boolean matrix generator.

    This generation procedure produces factor matrices U and V with C1P (contiguous-1 property).
    The factors form a block diagonal matrix with overlap configuration (when overlap < 0, there's no overlap).
    The matrix is sorted by nature upon generation.
    '''
    def __init__(self, m, n, k, overlap=[0.0, 0.0]):
        '''
        Parameters
        ----------
        overlap : length 2 list, (-inf, 1.0)
            Overlap ratio for factor U (overlap among columns) and factor V (overlap among rows).
        '''
        super().__init__()
        self.check_params(m=m, n=n, k=k, overlap=overlap)

        assert self.overlap[0] < 1 and self.overlap[1] < 1


    def generate(self, seed=None):
        self.check_params(seed=seed)
        self.generate_factors()
        self.boolean_matmul()
        # self.sorted_index()
        # self.set_factor_info()
        self.to_sparse(type='csr')


    def generate_factors(self):
        self.U = self.generate_factor(self.m, self.k, overlap=self.overlap[0])
        self.V = self.generate_factor(self.n, self.k, overlap=self.overlap[1])


    def generate_factor(self, n, k, overlap):
        L = n / (k - (k - 1) * overlap)
            
        points_start = np.linspace(start=0, stop=n-L, num=k, endpoint=True, dtype=int)
        points_end = np.linspace(start=L, stop=n, num=k, endpoint=True, dtype=int)
        
        # build the C1P factor sparse matrix
        X = lil_matrix((n, k))
        for c in range(k):
            X[points_start[c]:points_end[c], c] = 1

        return X