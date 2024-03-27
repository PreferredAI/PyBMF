from .Asso import Asso
import numpy as np
from multiprocessing import Pool
import time
from utils import matmul, add, ERR, cover
from copy import deepcopy
from scipy.sparse import issparse, lil_matrix
from functools import reduce
from .BaseModel import BaseModel


class AssoIter(Asso):
    '''The Asso algorithm with iterative search over each column of U.
    
    Reference
    ---------
    The discrete basis problem. Zhang et al. 2007.
    '''
    def __init__(self, model, w):
        self.check_params(model=model, w=w)


    def check_params(self, **kwargs):
        super().check_params(**kwargs)
        self.set_params(['model', 'w'], **kwargs)


    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        super(Asso, self).fit(X_train, X_val, X_test, **kwargs)
        self._fit()
        self._finish()


    def init_model(self):
        self.import_model(k=self.model.k, U=self.model.U, V=self.model.V, logs=self.model.logs)


    def _fit(self):
        '''Using iterative search to refine U

        In the paper, the algorithm uses cover function with the same weight for coverage and over-coverage (w = [0.5, 0.5]) as updating criteria, and uses error function as stopping criteria. This will not lead to the optimal solution. Change them to improve the performance.
        '''
        self.predict_X()
        best_score = cover(gt=self.X_train, pd=self.X_pd, w=self.w)
        best_error = ERR(gt=self.X_train, pd=self.X_pd)
        counter = 0

        while True:
            for k in range(self.k):
                score, col = self.get_refined_column(k)
                self.U[:, k] = col.T

                self.predict_X()
                error = ERR(gt=self.X_train, pd=self.X_pd)
                if error < best_error:
                    print("[I] Refined column i: {}, error: {:.4f} ---> {:.4f}, score: {:.2f} ---> {:.2f}.".format(k, best_error, error, best_score, score))
                    best_error = error
                    best_score = score

                    self.evaluate(df_name='refinements', head_info={'k': k}, train_info={'score': best_score, 'error': best_error})
                    counter = 0
                else:
                    counter += 1
                    print("[I] Skipped column i: {}.".format(k))
                    if counter == self.k:
                        break
            if counter == self.k:
                print("[I] Error stops decreasing.")
                break


    def get_refined_column(self, k):
        '''Return the optimal column given i-th basis
        
        The other k-1 basis remains unchanged.
        '''
        idx = [i for i in range(self.k) if k != i]
        X_old = matmul(self.U[:, idx], self.V[:, idx].T, sparse=True, boolean=True)
        s_old = cover(gt=self.X_train, pd=X_old, w=self.w, axis=1)
        basis = self.V[:, k].T

        score, col = self.get_vector(X_gt=self.X_train, X_old=X_old, s_old=s_old, basis=basis, basis_dim=1, w=self.w)

        return score, col
