from .Asso import Asso
import numpy as np
import time
from utils import matmul, cover
from scipy.sparse import csr_matrix, lil_matrix
from p_tqdm import p_map
from typing import Union
from multiprocessing import Pool, cpu_count
from .BaseModel import BaseModel


class AssoOpt(Asso):
    '''The Asso algorithm with exhaustive search over each row of U.

    This implementation may be slow but is able to deal with large `k` or huge dimension of `X_train`.
    
    Reference
    ---------
    The discrete basis problem. Zhang et al. 2007.
    '''
    def __init__(self, model, w):
        self.check_params(model=model, w=w)


    def check_params(self, **kwargs):
        super().check_params(**kwargs)
        
        # weight to use for refinement
        self.set_params(['w'], **kwargs)
        # model to import
        if 'model' in kwargs:
            model = kwargs.get('model')
            self.import_model(k=model.k, U=model.U, V=model.V, logs=model.logs)


    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        super(Asso, self).fit(X_train, X_val, X_test, **kwargs)

        self._fit()
        self.finish()


    def init_model(self):
        self.import_model(k=self.model.k, U=self.model.U, V=self.model.V, logs=self.model.logs)


    def _fit(self):
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

        self.predict_X()
        score = cover(gt=self.X_train, pd=self.X_pd, w=self.w)
        self.evaluate(df_name='refinements', train_info={'score': score})


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


    @staticmethod
    def int2bin(i, bits):
        '''Turn `i` into (1, `bits`) binary sparse matrix.
        '''
        return csr_matrix(list(bin(i)[2:].zfill(bits)), dtype=int)