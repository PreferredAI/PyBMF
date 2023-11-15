from .Asso import Asso
import numpy as np
from multiprocessing import Pool
import time
from utils import matmul
from copy import deepcopy
from scipy.sparse import issparse, lil_matrix


class AssoIter(Asso):
    '''The AssoIter Algorithm
    
    From the paper 'The discrete basis problem', using iterative search.
    '''
    def __init__(self, k, tau=None, w=None):
        super().__init__(k=k, tau=tau, w=w)


    def fit(self, train_set, val_set=None, display=False):
        super().fit(train_set=train_set, val_set=val_set, display=display)
        self.iterative_search()
        self.show_matrix(title="tau: {}, w: {}".format(self.tau, self.w))


    def iterative_search(self):
        '''Using iterative search to refine U
        '''
        best_cover = 0
        best_error = 0
        this_cover = 0
        this_error = 0
        while True:
            for j in range(self.k):
                last_cover = self.cover()
                last_error = self.error()

                self.Xnoj = self.X_without_j_th_basis(j=j)
                self.Xwithj = self.X_with_j_th_basis(j=j)
                self.find_optimal_Uj(j=j)

                this_cover = self.cover()
                this_error = self.error()
                print("[I] Refined column j = {}. cover: {} -> {}. error: {} -> {}.".format(j, last_cover, this_cover, last_error, this_error))

            if this_error < best_error:
                best_error = this_error
            else:
                break


    def find_optimal_Uj(self, j):
        '''Bit-by-bit comparison for finding optimal column Uj
        '''
        for i in range(self.m):
            current_row_in_X = self.X[i].astype(int) # j-th row in X
            score_0 = self.vec_cover(current_row_in_X, self.Xnoj[i]) # score when U[i, j] = 0
            score_1 = self.vec_cover(current_row_in_X, self.Xwithj[i]) # score when U[i, j] = 1
            self.U[i, j] = 0 if score_0 >= score_1 else 1


    def vec_cover(self, x, y):
        '''Vector version for cover function (with modified weights)
        x: ground truth
        y: reconstruction
        '''
        covered = np.sum(np.bitwise_and(x, y))
        overcovered = np.sum(np.maximum(y-x, 0))
        return 1 * covered - 1 * overcovered # w = [1, 1] in iterative search
    
    
    def X_without_j_th_basis(self, j):
        '''UV without j-th basis)
        '''
        idx = [i for i in range(self.k) if i != j]
        U = self.U[:, idx]
        V = self.V[idx, :]
        X = matmul(U, V, boolean=True)
        return X
    
    def X_with_j_th_basis(self, j):
        '''UV with j-th basis
        '''
        U = deepcopy(self.U)
        U[:, j] = 1
        V = self.V
        X = matmul(U, V, boolean=True)
        return X
