from .Asso import Asso, get_vector
import numpy as np
from multiprocessing import Pool
import time
from ..utils import matmul, get_prediction, ERR, coverage_score
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
    def __init__(self, model, w_fp=0.5, w_fn=None):
        self.check_params(model=model, w_fp = w_fp, w_fn = w_fn)


    def check_params(self, **kwargs):
        super().check_params(**kwargs)

        if 'model' in kwargs:
            model = kwargs.get('model')
            self.import_model(k=model.k, U=model.U, V=model.V, logs=model.logs)


    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        super(Asso, self).fit(X_train, X_val, X_test, **kwargs)

        self._fit()
        self.finish(show_logs=self.show_logs, save_model=self.save_model, show_result=self.show_result)


    def init_model(self):
        self._init_logs()
        

    def _fit(self):
        '''Using iterative search to refine U

        In the paper, the algorithm uses cover function with the same weight for coverage and over-coverage as updating criteria, and uses error function as stopping criteria.
        Changing them may improve the performance.
        '''
        self.X_pd = get_prediction(U=self.U, V=self.V, boolean=True)
        best_score = coverage_score(gt=self.X_train, pd=self.X_pd, w_fp=self.w_fp, w_fn=self.w_fn)
        best_error = ERR(gt=self.X_train, pd=self.X_pd)

        n_stop = 0
        is_improving = True
        while is_improving:
            for k in range(self.k):
                score, col = self.get_refined_column(k)
                self.U[:, k] = col.T

                self.X_pd = get_prediction(U=self.U, V=self.V, boolean=True)
                error = ERR(gt=self.X_train, pd=self.X_pd)
                if error < best_error:
                    print("[I] Refined column i: {}, error: {:.4f} -> {:.4f}, score: {:.2f} -> {:.2f}.".format(k, best_error, error, best_score, score))
                    best_error = error
                    best_score = score

                    self.evaluate(df_name='refinements', head_info={'k': k}, train_info={'score': best_score, 'error': best_error})
                    n_stop = 0
                else:
                    n_stop += 1
                    print("[I] Skipped column i: {}.".format(k))
                    if n_stop == self.k:
                        print("[I] Error stops decreasing.")
                        is_improving = False
                        break


    def get_refined_column(self, k):
        '''Return the optimal column given i-th basis
        
        The other k-1 basis remains unchanged.
        '''
        idx = [i for i in range(self.k) if k != i]
        X_old = matmul(self.U[:, idx], self.V[:, idx].T, sparse=True, boolean=True)
        s_old = coverage_score(gt=self.X_train, pd=X_old, w_fp=self.w_fp, w_fn=self.w_fn, axis=1)
        basis = self.V[:, k].T

        score, col = get_vector(
            X_gt=self.X_train, 
            X_old=X_old, 
            s_old=s_old, 
            basis=basis, 
            basis_dim=1, 
            w_fp=self.w_fp, 
            w_fn=self.w_fn,
        )

        return score, col
