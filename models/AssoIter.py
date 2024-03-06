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
        if 'model' in kwargs:
            model = kwargs.get('model')
            assert isinstance(model, BaseModel), "[E] Import a BaseModel."
            self.k = model.k
            self.U = model.U
            self.V = model.V
            self.logs = model.logs
            print("[I] k from model :", self.k)


    def fit(self, X_train, X_val=None, **kwargs):
        self.check_params(**kwargs)
        self.load_dataset(X_train=X_train, X_val=X_val)

        self.iterative_search()

        display(self.logs['refinements'])
        self.show_matrix(colorbar=True, discrete=True, clim=[0, 1], title="assoiter results")


    def iterative_search(self):
        '''Using iterative search to refine U

        In the paper, the algorithm uses cover (with w=[1, 1]) as updating criteria, and uses error as stopping criteria.
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

                    self.evaluate(names=['k', 'error', 'score'], values=[k, best_error, best_score], df_name='refinements')

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
