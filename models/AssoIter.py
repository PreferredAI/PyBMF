from .Asso import Asso
import numpy as np
from multiprocessing import Pool
import time
from utils import matmul, add
from copy import deepcopy
from scipy.sparse import issparse, lil_matrix
from functools import reduce


class AssoIter(Asso):
    '''The AssoIter algorithm using iterative search
    
    Reference:
        The discrete basis problem
    '''
    def fit(self, X_train, X_val=None, **kwargs):
        super().fit(X_train, X_val, **kwargs)
        self.iterative_search()
        self.show_matrix(title="after iterative search")


    def iterative_search(self):
        '''Using iterative search to refine U

        In the paper, the algorithm uses cover (with w=[1, 1]) as updating criteria, and uses error as stopping criteria.
        '''
        best_cover = 0
        best_error = 0
        this_cover = 0
        this_error = 0

        self.Xs = [] # build k overlapping matrices
        for k in range(self.k):
            X = matmul(self.U[:, k], self.V[:, k].T, boolean=True, sparse=True)
            self.Xs.append(X)

        while True:
            best_error = self.error()

            for i in range(self.k):
                best_cover = self.cover()

                score, column = self.get_refined_column(i)

                # update factors
                self.U[:, i] = column
                X = matmul(self.U[:, i], self.V[:, i].T, boolean=True, sparse=True)
                self.Xs[i] = X

                this_cover = score
                this_error = self.error()
                print("[I] Refined column i = {}, cover: {} -> {}, error: {} -> {}.".format(i, best_cover, this_cover, best_error, this_error))

            if this_error < best_error:
                best_error = this_error
            else:
                print("[I] Error stops decreasing.")
                break


    def get_refined_column(self, i):
        '''Return the optimal column given i-th basis
        
        The other k-1 basis remains unchanged.
        '''
        X_before = lil_matrix(np.zeros((self.m, self.n)))
        for idx in range(self.k):
            if idx != i: # without i-th basis
                X_before = add(X_before, self.Xs[idx])
        
        U = lil_matrix(np.ones([self.m, 1]))
        V = self.V[:, i]

        X_after = matmul(U, V.T, sparse=True, boolean=True)
        X_after = add(X_before, X_after)

        cover_before = self.cover(Y=X_before, axis=1, w=[1, 1])
        cover_after = self.cover(Y=X_after, axis=1, w=[1, 1])

        U = lil_matrix(np.array(cover_after > cover_before, dtype=int)).T

        X_after = matmul(U, V.T, sparse=True, boolean=True)
        X_after = add(X_before, X_after)

        cover = self.cover(Y=X_after)

        return cover, U