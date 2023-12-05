from .Asso import Asso
import numpy as np
from multiprocessing import Pool
import time
from utils import matmul, add
from copy import deepcopy
from scipy.sparse import issparse, lil_matrix
from functools import reduce


class AssoIter(Asso):
    '''The AssoIter Algorithm using iterative search
    
    From the paper 'The discrete basis problem'.
    '''
    def __init__(self, k, tau=None, w=None):
        super().__init__(k=k, tau=tau, w=w)


    def fit(self, train_set, val_set=None, display=False):
        super().fit(train_set=train_set, val_set=val_set, display=display)
        self.iterative_search()
        self.show_matrix(title="tau: {}, w: {}".format(self.tau, self.w))


    def iterative_search(self):
        '''Using iterative search to refine U

        In the paper, the algorithm uses cover (with w=[1, 1]) as updating criteria, and uses error as stopping criteria.
        '''
        best_cover = 0
        best_error = 0
        this_cover = 0
        this_error = 0

        self.Xs = []
        for i in range(self.k):
            X = matmul(self.U[:, i], self.V[i, :], boolean=True)
            self.Xs.append(X)

        while True:
            best_error = self.error()

            for i in range(self.k):
                best_cover = self.cover()

                score, column = self.get_optimal_column(i)

                # update factors
                self.U[:, i] = column
                X = matmul(self.U[:, i], self.V[i, :], boolean=True)
                self.Xs[i] = X

                this_cover = score
                this_error = self.error()
                print("[I] Refined column i = {}. cover: {} -> {}. error: {} -> {}.".format(i, best_cover, this_cover, best_error, this_error))

            if this_error < best_error:
                best_error = this_error
            else: # error stops decreasing
                break


    def get_optimal_column(self, i):
        '''Return the optimal column given i-th basis, while the other k-1 basis remains unchanged
        '''
        idx = [j for j in range(self.k) if i != j]
        before = reduce(add, self.Xs[idx]) # without j-th basis

        U = lil_matrix(np.ones((self.m, 1)))
        V = self.V[i, :]
        after = matmul(U, V, sparse=True, boolean=True)
        after = add(before, after) # with j-th basis

        before_cover = self.cover(Y=before, axis=1, w=[1, 1])
        after_cover = self.cover(Y=after, axis=1, w=[1, 1])
        optimal_col = (after_cover > before_cover) * 1
        optimal_col = lil_matrix(optimal_col).T

        U = optimal_col
        after = matmul(U, V, sparse=True, boolean=True)
        after = add(before, after)
        cover = self.cover(Y=after)

        return cover, optimal_col


    #     for i in range(self.m):
    #         current_row_in_X = self.X[i].astype(int) # j-th row in X
    #         score_0 = self.vec_cover(current_row_in_X, self.Xwoj[i]) # score when U[i, j] = 0
    #         score_1 = self.vec_cover(current_row_in_X, self.Xwj[i]) # score when U[i, j] = 1
    #         self.U[i, j] = 0 if score_0 >= score_1 else 1


    # def vec_cover(self, x, y):
    #     '''Vector version for cover function (with modified weights)
    #     x: ground truth
    #     y: reconstruction
    #     '''
    #     covered = np.sum(np.bitwise_and(x, y))
    #     overcovered = np.sum(np.maximum(y-x, 0))
    #     return 1 * covered - 1 * overcovered # w = [1, 1] in iterative search
    
    
    # def X_without_j_th_basis(self, j):
    #     '''UV without j-th basis
    #     '''
    #     idx = [i for i in range(self.k) if j != i]
    #     U = self.U[:, idx]
    #     V = self.V[idx, :]
    #     X = matmul(U, V, sparse=True, boolean=True)
    #     return X
    
    # def X_with_j_th_basis(self, j):
    #     '''UV with j-th basis
    #     '''
    #     U = deepcopy(self.U)
    #     U[:, j] = 1
    #     V = self.V
    #     X = matmul(U, V, boolean=True)
    #     return X
