from .Asso import Asso
import numpy as np
import time
from utils import matmul, ERR
from scipy.sparse import csr_matrix, lil_matrix
from tqdm import tqdm
from typing import Union
from multiprocessing import Pool, cpu_count


class AssoOpt(Asso):
    '''The Asso algorithm using exhaustive search
    
    From the paper 'The discrete basis problem'.
    '''
    def __init__(self, k, tau=None, w=None):
        super().__init__(k=k, tau=tau, w=w)

    
    def fit(self, train_set, val_set=None, display=False):
        super().fit(train_set=train_set, val_set=val_set, display=display)
        self.exhaustive_search()


    def exhaustive_search(self):
        '''Using exhaustive search to refine U
        '''
        self.Us = self.get_candidate_rows(dim=0) # all candidates for a row in U
        self.Xs = matmul(U=self.Us, V=self.V, sparse=True, boolean=True) # corresponding rows in X

        start_time = time.perf_counter()
        with Pool() as pool:
            result = pool.map(self.get_optimal_row, range(self.m))
        finish_time = time.perf_counter()
        print("[I] Exhaustive search finished in {} seconds with parallelism.".format(finish_time-start_time))

        # debug
        # print(result)
        self.U = self.Us[result, :] # refine U


    def get_candidate_rows(self, dim):
        '''Return all possible choices for a row in U (dim = 0), or for a column in V (dim = 1)

        E.g, when dim == 0 and m == 2, the output will be
            0   0
            0   1
            1   0
            1   1
        '''
        # rows, cols = bits, 2 ** bits
        # candidates = np.zeros((rows, cols), dtype=int)
        # for i in range(cols):
        #     binary = bin(i)[2:].zfill(rows)  # convert i to binary representation
        #     for j in range(rows):
        #         candidates[i, j] = int(binary[j])  # fill each cell with binary digit
        # return candidates if dim == 0 else candidates.T
        bits = self.k
        rows, cols = 2 ** bits, bits
        candidates = lil_matrix(shape=(rows, cols), dtype=np.uint8)
        for i in range(rows):
            binary = bin(i)[2:].zfill(cols) # convert i to binary strings with length k
            for j in range(cols):
                candidates[i, j] = np.uint8(binary[j]) # fill each row with binary digits
        return candidates if dim == 0 else candidates.T


    def get_optimal_row(self, i):
        '''Returns the index of the optimal row for i-th row
        '''
        trials = 2 ** self.k
        scores = np.zeros(trials)
        X = self.X_train[i, :].astype(int)
        for t in range(trials):
            Y = self.Xs[t, :]
            scores[t] = self.cover(X, Y)
        return np.argmax(scores)


    # def vec_cover(self, x, y):
    #     '''Vector version for cover function
    #     x: ground truth
    #     y: reconstruction
    #     '''
    #     covered = np.sum(np.bitwise_and(x, y))
    #     overcovered = np.sum(np.maximum(y-x, 0))
    #     return self.w[0] * covered - self.w[1] * overcovered
    