from .Asso import Asso
import numpy as np
import time
from utils import matmul, cover
from scipy.sparse import csr_matrix, lil_matrix
from p_tqdm import p_map
from typing import Union
from multiprocessing import Pool, cpu_count


class AssoOpt(Asso):
    '''The Asso algorithm with exhaustive search over each row of U.
    
    Reference
    ---------
    The discrete basis problem.
    '''
    def fit(self, X_train, X_val=None, **kwargs):
        super().fit(X_train, X_val, **kwargs)
        self.exhaustive_search()

        display(self.logs['refinements'])
        self.show_matrix(colorbar=True, discrete=True, clim=[0, 1], title="exhaustive search results")


    def exhaustive_search(self):
        '''Using exhaustive search to refine U.
        '''
        tic = time.perf_counter()

        # with Pool() as pool:
        #     pool.map(self.set_optimal_row, range(self.m))

        results = p_map(self.set_optimal_row, range(self.m))

        toc = time.perf_counter()
        print("[I] Exhaustive search finished in {}s.".format(toc-tic))

        for i in range(self.m):
            self.U[i] = self.int2bin(results[i], self.k)

        self.predict()
        score = cover(gt=self.X_train, pd=self.X_pd, w=self.w)
        self.evaluate(names=['score'], values=[score], df_name='refinements')


    @staticmethod
    def int2bin(i, bits):
        '''Turn `i` into (1, `bits`) binary sparse matrix.
        '''
        return csr_matrix(list(bin(i)[2:].zfill(bits)), dtype=int)


    def set_optimal_row(self, i):
        '''Update the i-th row in U.
        '''
        trials = 2 ** self.k
        scores = np.zeros(trials)
        X_gt = self.X_train[i, :]
        for j in range(trials):
            U = self.int2bin(j, self.k)
            X_pd = matmul(U, self.V.T, sparse=True, boolean=True)
            scores[j] = cover(gt=X_gt, pd=X_pd, w=self.w)
        idx = np.argmax(scores)
        return idx
