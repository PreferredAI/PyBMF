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
    
    Reference:
        The discrete basis problem
    '''
    def fit(self, X_train, X_val=None, **kwargs):
        super().fit(X_train, X_val, **kwargs)
        self.exhaustive_search()
        self.show_matrix(title="after iterative search")


    def exhaustive_search(self):
        '''Using exhaustive search to refine U
        '''
        self.Us = self.get_candidate_rows(k=self.k) 
        self.Xs = matmul(U=self.Us, V=self.V.T, sparse=True, boolean=True)

        start_time = time.perf_counter()
        with Pool() as pool:
            result = pool.map(self.get_optimal_row, range(self.m))
        finish_time = time.perf_counter()
        print("[I] Exhaustive search finished in {} seconds with parallelism.".format(finish_time - start_time))

        self.U = self.Us[result, :] # refine U


    @staticmethod
    def get_candidate_rows(k):
        '''Return all possible choices for a latent factor

        E.g: when k == 2, the output will be
            0   0
            0   1
            1   0
            1   1
        '''
        n_rows, n_cols = 2 ** k, k
        candidates = lil_matrix(np.zeros((n_rows, n_cols)))
        for i in range(n_rows):
            # convert i to binary strings with length k
            binary = bin(i)[2:].zfill(n_cols)
            for j in range(n_cols):
                # fill each row with binary digits
                candidates[i, j] = np.uint8(binary[j])
        return candidates


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

